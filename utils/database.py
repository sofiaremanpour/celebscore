from typing import Iterator, Optional

from pymongo import MongoClient, UpdateOne
from pymongo.collection import Collection
from pymongo.database import Database

from utils.logging_utils import logger


class TweetsHandler:
    def __init__(self, db: Database):
        """
        Creates an instance. Perform add and get operations on tweets.
        """
        self.tweets: Collection = db["Tweets"]

    def add_tweets(self, tweets: list[dict], search_term: str) -> None:
        """
        Given a list of tweets,
        Add or update the tweets to the db being associated with the search term
        """
        if not tweets:
            return
        # Define the list of operations to perform
        bulk_ops: list[UpdateOne] = []
        # For each tweet, create an operation to insert/update it
        for tweet in tweets:
            # Set the tweet data, and add the celebrity to the list of celebrities for that tweet
            tweet_update = UpdateOne(
                {"_id": tweet["id_str"]},
                {
                    "$set": {"tweet": tweet, "author": tweet["user"]["id_str"]},
                    "$addToSet": {"search_terms": search_term},
                },
                upsert=True,
            )
            bulk_ops.append(tweet_update)
            # Perform the bulk dump of tweets
        self.tweets.bulk_write(bulk_ops)
        # Update our new bounds of tweet ids gathered
        self._update_oldest_newest(search_term)

    def get_tweets(
        self, search_term: str, attributes: Optional[list[str]] = None
    ) -> Iterator[list[dict]]:
        """
        Given the id, return a generator of tweets associated with the search_term
        """
        if attributes is None:
            attributes = []
        if "_id" not in attributes:
            attributes += "_id"
        yield from self.tweets.find(
            {"search_terms": {"$in": [search_term]}}, {key: 1 for key in attributes}
        )

    def set_sentiment(
        self, sentiment, tweet_id: str
    ) -> bool:  # TODO add sentiment type
        """
        Given a sentiment and tweet_id, set the sentiment score in the database
        """
        result = self.tweets.update_one(
            {"_id": tweet_id}, {"$set": {"sentiment": sentiment}}
        )
        return result.modified_count > 0

    def get_sentiment(self, tweet_id: str):  # TODO add sentiment return type
        """
        Given a celebrity id and tweet_id, return the current sentiment score in the database
        If it doesn't exist, return None
        """
        exists = self.tweets.find_one({"_id": tweet_id}, {"_id": 0, "sentiment": 1})
        if exists:
            return exists["sentiment"]
        return None

    def _update_oldest_newest(self, search_term: str) -> None:
        """
        Update the celebrities database with our current oldest and newest tweet ids
        """
        # Get all tweets with the celebrity in their ids list
        pipeline = [
            {"$match": {"search_terms": search_term}},
            {"$project": {"tweet.id": 1}},
            {"$unwind": "$tweet"},
            {
                "$group": {
                    "_id": None,
                    "oldest_tweet_id": {"$min": "$tweet.id"},
                    "newest_tweet_id": {"$max": "$tweet.id"},
                }
            },
        ]
        # Execute the aggregation pipeline and retrieve the result
        result = list(self.tweets.aggregate(pipeline))
        if result:
            data = {
                "$set": {
                    "oldest_tweet_id": result[0]["oldest_tweet_id"],
                    "newest_tweet_id": result[0]["newest_tweet_id"],
                }
            }
        else:
            data = {"$unset": {"oldest_tweet_id": "", "newest_tweet_id": ""}}
        # Store the minimum and maximum
        terms_handler.search_terms.update_one(
            {"_id": search_term},
            data,
            upsert=True,
        )


class SearchTermsHandler:
    def __init__(self, db: Database):
        self.search_terms: Collection = db["SearchTerms"]

    def add_term(self, search_term: str, **kwargs) -> bool:
        """
        Add a term to the database. If entry already exists with the term, update information
        """
        # Store whatever we have available
        data = {"_id": search_term}
        data.update(kwargs)
        logger.info(f"Adding/updating search term: {data}")
        # Update the database entry by setting the new data
        res = self.search_terms.update_one(
            {"_id": search_term}, {"$set": data}, upsert=True
        )
        return bool(res.modified_count)

    def get_search_terms(self, attributes: Optional[list[str]] = None) -> list[dict]:
        """
        Returns a list of whatever attributes for all terms in the db
        """
        if attributes is None:
            attributes = []
        if "_id" not in attributes:
            attributes += "_id"
        return [i for i in self.search_terms.find({}, {key: 1 for key in attributes})]

    def get_newest_tweet_id(self, search_term: str) -> Optional[int]:
        """
        Given a search term, return the newest tweet id in the database
        If it doesn't exist, return None
        """
        exists = self.search_terms.find_one(
            {"_id": search_term}, {"_id": 0, "newest_tweet_id": 1}
        )
        if exists:
            return exists["newest_tweet_id"]
        return None

    def get_oldest_tweet_id(self, search_term: str) -> Optional[int]:
        """
        Given a search term, return the oldest tweet id in the database
        If it doesn't exist, return None
        """
        exists = self.search_terms.find_one(
            {"_id": search_term}, {"_id": 0, "oldest_tweet_id": 1}
        )
        if exists:
            return exists["oldest_tweet_id"]
        return None


def connect_to_db() -> MongoClient:
    """
    Initializes the MongoDB connection and provides API"s for interaction
    """
    logger.info("Reading database info from config")
    # with open("config/mongodb.config", "r") as f:
    #     config = json.load(f)
    #     db_username = config["username"]
    #     db_password = config["password"]
    #     # Connect to database
    #     client = MongoClient(
    #         f"mongodb+srv://{db_username}:{db_password}@celebscore.inxw4wt.mongodb.net",
    #         tlsCAFile=certifi.where(),
    #     )
    client = MongoClient("mongodb://localhost:27017")
    logger.info("Pinging database")
    result = client.admin.command("ping")
    if not result["ok"]:
        logger.error("Failed database connection")
    else:
        logger.info("Database ping successful")
    return client


client = connect_to_db()
db = client["CelebScoreData"]
terms_handler = SearchTermsHandler(db)
tweets_handler = TweetsHandler(db)
