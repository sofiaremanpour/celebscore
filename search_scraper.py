import random

from utils import team_utils
from utils.database import terms_handler, tweets_handler
from utils.logging_utils import logger
from utils.twitter_utils import get_search_results


def update_terms_from_file():
    """
    Add the search terms that we loaded from the file into the database
    """
    # Add to database
    for i in team_utils.all_terms:
        terms_handler.add_term(i)
    logger.info(f"{team_utils.all_terms}")


def search_terms_get_tweets():
    """
    In a round robin fasion, gather a batch of tweets searching for each term
    Add tweets to the database, and store what tweets we've already gotten
    """
    # Get the data for each term
    terms = terms_handler.get_search_terms(["oldest_tweet_id", "newest_tweet_id"])
    # Randomly shuffle the order of who to get each program run
    random.shuffle(terms)
    # For each term, create the tweet iterator that will get batches of tweets
    for t in terms:
        # Find the current bounds of the tweets we have
        oldest_tweet_id = t.get("oldest_tweet_id")
        newest_tweet_id = t.get("newest_tweet_id")

        # Get the iterator from the search function searching for the term
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
                # Get the next batch for the term
                tweet_batch = next(t["tweet_iterator"])
                if tweet_batch:
                    logger.info(
                        f"{term}\t#Tweets: {len(tweet_batch)}\tCur tweet time: {tweet_batch[-1]['created_at']}"
                    )
                    gotten = True
                    tweets_handler.add_tweets(tweet_batch, term)
            except StopIteration:
                # If the generator for the term is out, then just move on
                t["iterator_exhausted"] = True
                continue
        if not gotten:
            break


def main():
    """
    Use Twitter search to scrape all visible tweets for all search terms and add to a database
    Automatically managed resume functionality to not gather duplicate tweets
    Perform batching to ensure an similar number of searches for all of the terms
    """
    update_terms_from_file()
    search_terms_get_tweets()


if __name__ == "__main__":
    main()
