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
    celeb_handles = list(celeb_data["handle"].astype("string"))
    logger.info(f"Found {len(celeb_handles)} celebrities in file")
    # Find the celebrities not already in the database
    new_handles = celebrity_handler.get_missing_celebrities_handles(celeb_handles)
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
    """
    celebrities = celebrity_handler.get_celebrities(["_id", "handle"])
    for celebrity in celebrities:
        # TODO - record what tweets we already have and stop once we reach overlapping tweets
        id = celebrity["_id"]
        handle = celebrity["handle"]
        logger.info(f"Searching for tweets about {handle}")
        tweets = get_search_results(handle, limit=5000)
        logger.info("Adding tweets to database")
        tweets_handler.add_tweets(tweets, id)


def main():
    # update_celebrities_from_file()
    get_celebrities_tweets()


if __name__ == "__main__":
    main()
