from utils.database import tweets_handler, followers_handler, celebrity_handler

# Sentiment analysis stuff goes here


def calculate_sentiment(tweet):
    """
    Given the tweet object,
    Return the sentiment of the tweet
    """
    pass


def main():
    """
    For each celebrity, iterate over all of their tweets and calculate the sentiment
    Add sentiment score to the db
    """
    # Get a list of all the celebrities
    celebrities = celebrity_handler.get_celebrities("id_str", "handle")
    # Iterate through each one
    for celebrity in celebrities:
        id = celebrity["id_str"]
        handle = celebrity["handle"]
        # Define an iterator that will return tweets in batches
        tweet_iterator = tweets_handler.get_celebrities_tweets(id)
        # Define vars to average sentiment for all tweets
        celebrity_sentiment = None
        # Iterate a batch at a time to prevent an overload
        for batch in tweet_iterator:
            # Iterate each tweet within the batch
            for tweet in batch:
                tweet_sentiment = calculate_sentiment(tweet)
                # Add sentiment to the total sentiment calculation

        # Add the score to the database
        celebrity_handler.set_sentiment(celebrity_sentiment, id)


if __name__ == "__main__":
    main()
