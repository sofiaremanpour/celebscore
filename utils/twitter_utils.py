from typing import Dict, List
import twitter
from utils.logging_utils import logger
import time
import json


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
        errors_left = 2
        global num_api_calls
        # Keep attempting the program
        while errors_left > 0:
            try:
                # Track the number of API calls for debugging purposes
                num_api_calls += 1
                if num_api_calls % 5 == 0:
                    logger.info(f"Twitter call #{num_api_calls}")
                # Attempt to call the underlying function
                return twitter_func(*args, **kwargs)
            except twitter.api.TwitterHTTPError as e:
                # If we get a rate limit, sleep and try again
                errors_left -= 1
                if e.e.code == 429:
                    logger.error(
                        f'Rate Limit Exceeded at {num_api_calls}, sleeping 15 mins')
                    time.sleep(60*15 + 5)
                    logger.info("Retrying after rate limit")
                else:
                    # If we get another error, retry
                    errors_left -= 1
                    logger.error(
                        f"HTTP error code {e.e.code} encountered, retrying \nError:\n{repr(e)}")
            except Exception as e:
                # If its not an HTTP error, then just exit because something is wrong
                logger.error(
                    f"Other error encountered, exiting: {repr(e)}")
                break
        logger.error("Out of retries, exiting")
    return decorated


twitter_api = oauth_login()
_user_lookup = rate_limit_safe(twitter_api.users.show)
_users_lookup = rate_limit_safe(twitter_api.users.lookup)
_followers_list = rate_limit_safe(twitter_api.followers.list)
_followers_ids = rate_limit_safe(twitter_api.followers.ids)


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
        return [{key: val for key, val in user.items() if key in attributes} for user in user_objects]
    screen_name_str = ",".join([i for i in ids])
    user_objects = _users_lookup(user_id=ids)
    return [{key: val for key, val in user.items() if key in attributes} for user in user_objects]


def get_followers(id, limit=5000):
    """
    Returns the followers of the user with id
    """
    pass


def get_followers_ids(id, limit=5000):
    """
    Returns the ids of followers of the user with id
    """
    pass
