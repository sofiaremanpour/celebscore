from typing import Optional

from pymongo import MongoClient, UpdateOne
from pymongo.collection import Collection
from pymongo.cursor import Cursor
from pymongo.database import Database
from tqdm import tqdm

from utils.logging_utils import logger


class TweetsHandler:
    def __init__(self, db: Database):
        """
        Creates an instance. Perform add and get operations on tweets.
        """
        self.tweets: Collection = db["Tweets"]
        # uncomment to clear sentiments
        self.tweets.update_many(
            {"sentiment": {"$exists": True}}, {"$unset": {"sentiment": None}}
        )

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

    def get_tweet_count(self, search_terms: tuple[str, str]) -> Optional[int]:
        """
        Return the total number of tweets for a search term, otherwise none
        """
        return self.tweets.count_documents({"search_terms": {"$in": search_terms}})

    def get_tweets(
        self, search_terms: tuple[str, str], attributes: Optional[list[str]] = None
    ) -> Cursor:
        """
        Given the id, return a generator of tweet documents associated with a list of search_terms
        Optionally specify other info in the document to return as well
        """
        if attributes is None:
            attributes = []
        if "tweet" not in attributes:
            attributes.append("tweet")
        return self.tweets.find(
            {"search_terms": {"$in": search_terms}}, {key: 1 for key in attributes}
        )

    def set_sentiment(self, sentiment: dict, tweet_id: str) -> bool:
        """
        Given a sentiment and tweet_id, set the sentiment score in the database
        """
        result = self.tweets.update_one(
            {"_id": tweet_id}, {"$set": {"sentiment": sentiment}}
        )
        return result.modified_count > 0

    def get_sentiment(self, tweet_id: str) -> Optional[dict]:
        """
        Given a tweet_id, return the current sentiment score in the database
        If it doesn't exist, return None
        """
        exists = self.tweets.find_one({"_id": tweet_id}, {"_id": 0, "sentiment": 1})
        if exists:
            return exists["sentiment"]
        return None

    def update_all_oldest_newest(self) -> None:
        """
        Update all terms with their oldest and newest tweet ids
        """
        for term in tqdm(
            terms_handler.get_search_terms(), desc="Finding tweets already in database"
        ):
            # Update our new bounds of tweet ids gathered
            self._update_oldest_newest(term["_id"])

    def _update_oldest_newest(self, search_term: str) -> None:
        """
        Update the terms database with our current oldest and newest tweet ids
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
        """
        Create an instance. Add and retrieve search terms and associated data
        """
        self.search_terms: Collection = db["SearchTerms"]

    def add_term(self, search_term: str) -> bool:
        """
        Add a term to the database. If entry already exists with the term, update information
        """
        # Store whatever we have available
        data = {"_id": search_term}
        # Update the database entry by setting the new data
        res = self.search_terms.update_one(
            {"_id": search_term}, {"$set": data}, upsert=True
        )
        return bool(res.modified_count)

    def get_search_terms(self, attributes: Optional[list[str]] = None) -> list[dict]:
        """
        Returns a list of terms in the database, optionally getting other values from the document
        """
        if attributes is None:
            attributes = []
        if "_id" not in attributes:
            attributes += "_id"
        return list(self.search_terms.find({}, {key: 1 for key in attributes}))


def connect_to_db() -> MongoClient:
    """
    Initializes the MongoDB connection and provides API"s for interaction
    """
    # logger.info("Reading database info from config")
    # with open("config/mongodb.config", "r") as f:
    #     config = json.load(f)
    #     db_username = config["username"]
    #     db_password = config["password"]
    #     # Connect to database
    #     client = MongoClient(
    #         f"mongodb+srv://{db_username}:{db_password}@celebscore.inxw4wt.mongodb.net",
    #         tlsCAFile=certifi.where(),
    #     )
    client = MongoClient("mongodb://127.0.0.1:27017")
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
