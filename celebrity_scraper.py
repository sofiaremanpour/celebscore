import pandas as pd

from utils.database import celebrity_handler, tweets_handler
from utils.logging_utils import logger
from utils.twitter_utils import get_search_results, get_users


def update_celebrities_from_file():
    """
    Reads celebrities to add from the file
    Finds the ones not in the db, looks them up, and adds to db
    """
    # Open the list of celebrities from the csv file
    celeb_data = pd.read_csv("most_followed_accounts.csv")
    celeb_handles = [i.lower() for i in celeb_data["handle"].astype("string")]
    logger.info(f"Found {len(celeb_handles)} celebrities in file")
    # Find the celebrities already in the database
    current_handles = [
        i["handle"] for i in celebrity_handler.get_celebrities(["handle"])
    ]
    # Calculate celebrities not already in the db
    new_handles = [i for i in celeb_handles if i not in current_handles]
    logger.info(f"Looking up data for {len(new_handles)} new celebrities")
    if not new_handles:
        return
    logger.info(f"{new_handles}")
    # Gather the user objects for the celebrities
    celebrities = get_users(
        handles=new_handles, attributes=["id_str", "name", "screen_name"]
    )
    # Add each celebrity to the database
    for c in celebrities:
        celebrity_handler.add_celebrity(
            id=c["id_str"],
            handle=c["screen_name"],
            name=c["name"],
        )


def get_celebrities_tweets():
    """
    Gets tweets from searching the handle of each celebrity and adds to database
    Checks what tweets are already in the database,
    and searches only for tweets older and what we have and newer than what we have
    """
    # Get the id and handle of each celebrity
    celebrities = celebrity_handler.get_celebrities(["_id", "handle"])
    # For each celebrity
    for celebrity in celebrities:
        id = celebrity["_id"]
        handle = celebrity["handle"]
        # Find the current bounds of the tweets we have
        oldest_tweet_id = celebrity_handler.get_oldest_tweet_id(id)
        newest_tweet_id = celebrity_handler.get_newest_tweet_id(id)
        # Gather the tweets, and update the list we got
        tweets, oldest_tweet_id, newest_tweet_id = get_search_results(
            handle, oldest_tweet_id=oldest_tweet_id, newest_tweet_id=newest_tweet_id
        )
        tweets_handler.add_tweets(tweets, id)


def main():
    # update_celebrities_from_file()
    get_celebrities_tweets()


if __name__ == "__main__":
    main()
