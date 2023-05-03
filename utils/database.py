from typing import Sequence

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
        # self.tweets.update_many(
        #     {"sentiment": {"$exists": True}}, {"$unset": {"sentiment": None}}
        # )

    def add_tweets(self, tweets: list[dict], search_term: str) -> None:
        """
        Given a list of tweets,
        Add or update the tweets to the database
        Add the search_term to the list of serach_terms for each tweet
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

    def get_tweet_count(self, search_terms: Sequence[str]) -> int | None:
        """
        Return the total number of tweets for the tweets associated with the search terms, otherwise None
        """
        return self.tweets.count_documents({"search_terms": {"$in": search_terms}})

    def get_tweets(
        self,
        search_terms: Sequence[str] | None = None,
        attributes: list[str] | None = None,
    ) -> Cursor:
        """
        Given the id, return a generator that yields tweet documents associated with a term from the list of search_terms
        By default, returns the tweet object under the key "tweet", but optionally specify other attributes in the document to return as well
        """
        # If search_terms not provided, get all the tweets
        if search_terms is None:
            search_terms = [i["_id"] for i in terms_handler.get_search_terms()]
        # If attributes isn't provided, add the actual tweet object to be returned by default
        if attributes is None:
            attributes = []
        if "tweet" not in attributes:
            attributes.append("tweet")
        return self.tweets.find(
            {"search_terms": {"$in": search_terms}},
            {key: 1 for key in attributes},
        )

    def set_sentiment(self, sentiment: dict, tweet_id: str) -> bool:
        """
        Given a sentiment and tweet_id, set the sentiment score for the tweet with tweet_id in the database
        """
        result = self.tweets.update_one(
            {"_id": tweet_id}, {"$set": {"sentiment": sentiment}}
        )
        return result.modified_count > 0

    def get_sentiment(self, tweet_id: str) -> dict | None:
        """
        Given a tweet_id, return the current sentiment in the database
        If it doesn't exist, return None
        """
        exists = self.tweets.find_one({"_id": tweet_id}, {"_id": 0, "sentiment": 1})
        if exists:
            return exists["sentiment"]
        return None

    def update_all_oldest_newest(self) -> None:
        """
        For each term in the database, update the oldest and newest tweet ids associated with that term
        """
        for term in tqdm(
            terms_handler.get_search_terms(), desc="Finding tweets already in database"
        ):
            # Update our new bounds of tweet ids gathered
            self._update_oldest_newest(term["_id"])

    def _update_oldest_newest(self, search_term: str) -> None:
        """
        For a search term, update the oldest_tweet_id and newest_tweet_id in the database for that term
        """
        # Get all tweets with the search_term in their search_terms list
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
        # Set the data to update from aggregation results
        if result:
            data = {
                "$set": {
                    "oldest_tweet_id": result[0]["oldest_tweet_id"],
                    "newest_tweet_id": result[0]["newest_tweet_id"],
                }
            }
        else:
            data = {"$unset": {"oldest_tweet_id": "", "newest_tweet_id": ""}}
        # Update the db document
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
        Add a term to the database. If it already exists, do nothing
        Return True if it is a new term
        """
        data = {"_id": search_term}
        # Update the database entry by setting the new data
        res = self.search_terms.update_one(
            {"_id": search_term}, {"$set": data}, upsert=True
        )
        return bool(res.modified_count)

    def get_search_terms(self, attributes: list[str] | None = None) -> list[dict]:
        """
        Return a list of search_term documents from the database
        By default, returns the term name under the key "_id", but optionally specify other attributes in the document to return as well
        """
        if attributes is None:
            attributes = []
        if "_id" not in attributes:
            attributes += "_id"
        return list(
            self.search_terms.find(
                {key: {"$exists": True} for key in attributes},
                {key: 1 for key in attributes},
            )
        )


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
    # Use local database because we ran out of room remotely
    client = MongoClient("mongodb://127.0.0.1:27017")
    logger.info("Pinging database")
    result = client.admin.command("ping")
    if not result["ok"]:
        logger.error("Failed database connection")
    else:
        logger.info("Database ping successful")
    return client


# Initialize database connection and handlers
client = connect_to_db()
db = client["CelebScoreData"]
terms_handler = SearchTermsHandler(db)
tweets_handler = TweetsHandler(db)
