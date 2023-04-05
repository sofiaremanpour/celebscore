from utils.logging_utils import logger
from utils.twitter_utils import get_users
from utils.database import celebrity_handler
import pandas as pd


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
    new_handles = celebrity_handler.get_missing_celebrities_handles(
        celeb_handles)
    logger.info(f"Looking up data for {len(new_handles)} new celebrities")
    if not new_handles:
        return
    logger.info(f"{new_handles}")
    # Gather the user objects for the celebrities
    celebrities = get_users(handles=new_handles, attributes=[
        "id_str", "name", "screen_name"])
    # Add each celebrity to the database
    for c in celebrities:
        celebrity_handler.add_celebrity(
            handle=c["screen_name"], name=c["name"], id=c["id_str"])


def update_celebrities_followers():
    # Get a list of remaining celebrities
    celeb_ids = celebrity_handler.get_celebrities_ids_incomplete_followers()
    # For each, add (all) of their followers to the database
    for celeb_id in celeb_ids:
        pass


def main():
    update_celebrities_from_file()
    #update_celebrities_followers()


if __name__ == "__main__":
    main()
