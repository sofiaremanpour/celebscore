import random

import pandas as pd

from utils.database import terms_handler, tweets_handler
from utils.logging_utils import logger
from utils.twitter_utils import get_search_results


def update_terms_from_file():
    """
    Reads search terms to add from the file
    Finds the ones not in the db, and adds to db
    """
    # Open the list of search terms from the csv file
    search_data = pd.read_csv("search_terms.csv", index_col=False)
    main_terms = list(search_data["term"].astype("string"))
    alt_terms = list(search_data["alt_term"].astype("string"))
    search_terms = main_terms + alt_terms
    logger.info(f"Found {len(search_terms)} search terms in file")
    # Add to database
    for i in search_terms:
        terms_handler.add_term(i)
    logger.info(f"{search_terms}")


def search_terms_get_tweets():
    """
    In a round robin fasion, gather a batch of tweets searching for each term
    Add tweets to the database, and store what tweets we've already gotten
    """
    # Get the data for each term
    terms = terms_handler.get_search_terms(["oldest_tweet_id", "newest_tweet_id"])
    random.shuffle(terms)
    # For each term, create a tweet iterator that will get batches of tweets
    for t in terms:
        # Find the current bounds of the tweets we have
        oldest_tweet_id = t.get("oldest_tweet_id")
        newest_tweet_id = t.get("newest_tweet_id")

        # Get a batch of tweets at a time, and add to database
        t["tweet_iterator"] = get_search_results(
            t["_id"],
            oldest_tweet_id=oldest_tweet_id,
            newest_tweet_id=newest_tweet_id,
        )
        t["iterator_exhausted"] = False
    # Continue getting batches until there are no more
    while True:
        gotten = False
        num_complete = sum((1 for i in terms if i["iterator_exhausted"]))
        logger.info(f"Currently completed with {num_complete} terms")
        for t in terms:
            try:
                term = t["_id"]
                tweet_batch = next(t["tweet_iterator"])
                if tweet_batch:
                    logger.info(
                        f"{term}\t#Tweets: {len(tweet_batch)}\tCur tweet time: {tweet_batch[-1]['created_at']}"
                    )
                    gotten = True
                    tweets_handler.add_tweets(tweet_batch, term)
            except StopIteration:
                # If the generator for the celebrity is out, then just move on
                t["iterator_exhausted"] = True
                continue
        if not gotten:
            break


def main():
    update_terms_from_file()
    search_terms_get_tweets()


if __name__ == "__main__":
    main()
