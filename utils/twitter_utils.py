import json
import math
import time
from datetime import datetime, timedelta
from typing import Callable, Iterator
from urllib.parse import parse_qs, urlparse

import twitter
from tqdm import trange

from utils.database import tweets_handler
from utils.logging_utils import logger


def oauth_login() -> twitter.Twitter:
    """
    Taken from cookbook
    Create a twitter API object using keys stored in config/api_keys.config
    """
    with open("config/api_keys.config", "r") as f:
        config: dict = json.load(f)

    auth = twitter.oauth.OAuth(**config)

    twitter_api = twitter.Twitter(auth=auth)
    return twitter_api


num_api_calls = 0


def rate_limit_safe(twitter_func: Callable) -> Callable:
    """
    A function decorator that will automatically retry the function if encountering rate limit
    Modified from cookbook function make_twitter_request - Decorators are much cleaner than partial application
    """

    def decorated(*args, **kwargs):
        # Run the attempted function, but retry when encoutering rate limit
        # Print all errors to a logger and exit if its not an error we can fix
        global num_api_calls
        # Keep attempting the program
        while True:
            try:
                # Track the number of API calls for debugging purposes
                num_api_calls += 1
                if num_api_calls % 5 == 0:
                    logger.info(f"Twitter call #{num_api_calls}")
                # Attempt to call the underlying function
                return twitter_func(*args, **kwargs)
            except twitter.api.TwitterHTTPError as e:
                # If we get a rate limit, sleep and try again
                if e.e.code == 429:
                    logger.error(
                        f"Rate Limit Exceeded at {num_api_calls}, performing database maintenance"
                    )
                    # Mark the start time
                    limit_start = datetime.now()
                    # Perform database maintence and mark tweets already gotten
                    tweets_handler.update_all_oldest_newest()
                    elapsed = int((datetime.now() - limit_start).total_seconds())
                    # Sleep (15 minutes - maintence time)
                    for i in trange(
                        elapsed,
                        (15 * 60) + 5,
                        desc="sleep time",
                    ):
                        # Sleep until the next whole second
                        next_wake = limit_start + timedelta(seconds=i + 1)
                        duration = (next_wake - datetime.now()).total_seconds()
                        if duration > 0:
                            time.sleep(duration)
                    logger.info("Retrying after rate limit")
                else:
                    # If we get another error, retry
                    logger.error(
                        f"HTTP error code {e.e.code} encountered, retrying \nError:\n{repr(e)}"
                    )
            except Exception as e:
                logger.error(f"Other error encountered, retrying: {repr(e)}")

    # Return the new wrapped function
    return decorated


twitter_api = oauth_login()
_search_tweets = rate_limit_safe(twitter_api.search.tweets)


# def get_users(
#     handles: Optional[str] = None,
#     ids: Optional[str | int] = None,
#     attributes: Optional[list[str]] = None,
# ) -> list[dict]:
#     """
#     Return a list of user dicts for the users with their handle in handles or id in ids
#     Optionally specify a list of strings that represent the attributes to return in the list
#     Ex: get_users(handles=["Oprah", "NBA"], attributes=["id_str", "followers_count"])
#         Returns [{"id_str": "....", "followers_count": 1000000}, ...]
#     """
#     # We must have either handles or ids, but not both
#     assert (handles and not ids) or (not handles and ids)

#     if handles:
#         screen_name_str = ",".join([i for i in handles])
#         user_objects = _users_lookup(screen_name=screen_name_str)
#         return [
#             {key: val for key, val in user.items() if key in attributes}
#             for user in user_objects
#         ]
#     screen_name_str = ",".join([i for i in ids])
#     user_objects = _users_lookup(user_id=ids)
#     return [
#         {key: val for key, val in user.items() if key in attributes}
#         for user in user_objects
#     ]


def get_search_results(
    query: str,
    oldest_tweet_id: int | None = None,
    newest_tweet_id: int | None = None,
) -> Iterator[list[dict]]:
    """
    Generator that yields a batch of tweet objects from the search of query
    Only search more recently then the newest_tweet_id we have, and older than the oldest_tweet_id
    Tweets are yielded in a way so that they are contiguous from the tweets already gathered
    """

    def search_helper(
        query: str, since_id: int | None = None, max_id: int | float | None = None
    ) -> Iterator[list[dict]]:
        """
        Perform a search of query from the max_id to the since_id
        Yield tweets in a batche when doing so doesn't break the continuous interval
        """
        # Store the tweets in a list
        since_id = since_id if since_id else 0
        max_id = max_id - 1 if max_id else math.inf
        tweets = []
        # Iterate until no more tweets
        while True:
            # Make the API call
            tweet_data = _search_tweets(
                q=query,
                count=100,
                result_type="recent",
                since_id=since_id,
                max_id=max_id,
            )
            new_tweets = tweet_data["statuses"]
            # If we are searching before the current oldest tweet, its okay to yield a batch as it comes
            # If we are searching after the current newest tweet, we must wait until we have the whole interval from now until then
            if new_tweets:
                if since_id:
                    tweets += new_tweets
                else:
                    yield new_tweets
            else:
                # If we are out of tweets, return the ones we haven't yet
                if since_id:
                    yield tweets
                    return
                else:
                    return
            # Find what twitter says the next page of results is by identifying the max_id
            parsed_args = parse_qs(
                urlparse(tweet_data["search_metadata"]["next_results"]).query
            )
            # Set the new max_id, thus the next iteration will be older tweets
            max_id = int(parsed_args["max_id"][0])

    # Perform a search from the oldest tweet available to the oldest_tweet_id in the db
    logger.info(f"Searching: {query} from the oldest available to {oldest_tweet_id}")
    # Yield tweet batches from the generator
    yield from search_helper(query, max_id=oldest_tweet_id)

    if newest_tweet_id:
        # Perform a search from the newest_tweet_id in db to the present
        logger.info(f"Searching: {query} from {newest_tweet_id} to the present")
        # Yield a single mega batch from the generator to preserve the contiguous interval
        yield from search_helper(query, since_id=newest_tweet_id)
