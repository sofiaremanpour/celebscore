import json
import math
import time
from typing import Dict, List
from urllib.parse import parse_qs, urlparse

import twitter

from utils.logging_utils import logger


def oauth_login() -> twitter.Twitter:
    """
    Taken from cookbook
    Create a twitter API object using keys stored in config
    """
    with open("config/api_keys.config", "r") as f:
        config = json.load(f)

    auth = twitter.oauth.OAuth(**config)

    twitter_api = twitter.Twitter(auth=auth)
    return twitter_api


num_api_calls = 0


def rate_limit_safe(twitter_func):
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
                        f"Rate Limit Exceeded at {num_api_calls}, sleeping 15 mins"
                    )
                    time.sleep(60 * 15 + 5)
                    logger.info("Retrying after rate limit")
                else:
                    # If we get another error, retry
                    logger.error(
                        f"HTTP error code {e.e.code} encountered, retrying \nError:\n{repr(e)}"
                    )
            except Exception as e:
                logger.error(f"Other error encountered, retrying: {repr(e)}")

    return decorated


twitter_api = oauth_login()
_user_lookup = rate_limit_safe(twitter_api.users.show)
_users_lookup = rate_limit_safe(twitter_api.users.lookup)
_followers_list = rate_limit_safe(twitter_api.followers.list)
_followers_ids = rate_limit_safe(twitter_api.followers.ids)
_search_tweets = rate_limit_safe(twitter_api.search.tweets)


def get_users(handles=None, ids=None, attributes=None) -> List[Dict]:
    """
    Return a list of user dicts for the users with their handle in handles or id in ids
    Optionally specify a list of strings that represent the attributes to return in the list
    Ex: get_users(handles=["Oprah", "NBA"], attributes=["id_str", "followers_count"])
        Returns [{"id_str": "....", "followers_count": 1000000}, ...]
    """
    # We must have either handles or ids, but not both
    assert (handles and not ids) or (not handles and ids)

    if handles:
        screen_name_str = ",".join([i for i in handles])
        user_objects = _users_lookup(screen_name=screen_name_str)
        return [
            {key: val for key, val in user.items() if key in attributes}
            for user in user_objects
        ]
    screen_name_str = ",".join([i for i in ids])
    user_objects = _users_lookup(user_id=ids)
    return [
        {key: val for key, val in user.items() if key in attributes}
        for user in user_objects
    ]


def get_search_results(query, oldest_tweet_id=None, newest_tweet_id=None) -> List:
    """
    Returns a list of tweets objects from the search of string
    Only search more recently then the newest_tweet_id we have, and older than the oldest_tweet_id
    """

    def search_helper(query, since_id=None, max_id=None):
        """
        Perform a search of query from the max_id to the since_id
        Return the tweets
        """
        # Store the tweets in a list
        tweets = []
        since_id = since_id if since_id else 0
        max_id = max_id if max_id else math.inf
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

            # If we are out of tweets, stop
            new_tweets = tweet_data["statuses"]
            if not new_tweets:
                break
            tweets += new_tweets
            logger.info(
                f"# Tweets: {len(tweets)}\tCurrent tweet time: {new_tweets[-1]['created_at']}"
            )
            # Find what twitter says the next page of results is by identifying the max_id
            parsed_args = parse_qs(
                urlparse(tweet_data["search_metadata"]["next_results"]).query
            )
            # Set the new max_id, thus the next iteration will be older tweets
            max_id = int(parsed_args["max_id"][0])
        return tweets

    assert (newest_tweet_id and oldest_tweet_id) or (
        not newest_tweet_id and not oldest_tweet_id
    )
    # Perform a search from the newest_tweet_id in db to the present
    logger.info(f"Searching tweets: {query} from {newest_tweet_id} to the present")
    newest_to_current = search_helper(query, since_id=newest_tweet_id)
    oldest_to_older = []
    if oldest_tweet_id:
        # Perform a search from the oldest tweet available to the oldest_tweet_id in the db
        logger.info(
            f"Searching tweets: {query} from the oldest available to {oldest_tweet_id}"
        )
        oldest_to_older = search_helper(query, max_id=oldest_tweet_id)
    tweets = newest_to_current + oldest_to_older
    # Find the current oldest and newest tweet ids for the db
    tweet_ids = [i["id"] for i in tweets]
    oldest_tweet_id = min(tweet_ids)
    newest_tweet_id = max(tweet_ids)
    return tweets, oldest_tweet_id, newest_tweet_id
