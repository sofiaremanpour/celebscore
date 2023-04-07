import json

import certifi
from pymongo import MongoClient, UpdateOne

from utils.logging_utils import logger


class TweetsHandler:
    def __init__(self, db):
        self.tweets = db["Tweets"]

    def add_tweets(self, tweets, celebrity_id):
        """
        Given a list of tweets,
        Add the tweets to the db
        """
        if not tweets:
            return
        # Define the list of operations to perform
        bulk_ops = []
        # For each tweet, create an operation to insert/update it
        for tweet in tweets:
            # Set the tweet data, and add the celebrity to the list of celebrities for that tweet
            tweet_update = UpdateOne(
                {"_id": tweet["id_str"]},
                {
                    "$set": {"tweet": tweet, "author": tweet["user"]["id_str"]},
                    "$addToSet": {"celebrity_ids": celebrity_id},
                },
                upsert=True,
            )
            bulk_ops.append(tweet_update)
            # Perform the bulk dump of tweets
        self.tweets.bulk_write(bulk_ops)
        # Update our new bounds of tweet ids gathered
        self._update_oldest_newest(celebrity_id)

    def get_celebrities_tweets(self, celebrity_id):
        """
        Given the id, return a list of tweets
        """
        return [
            i["tweet"]
            for i in self.tweets.find(
                {"celebrity_ids": {"$in": [celebrity_id]}}, {"tweet": 1}
            )
        ]

    def _update_oldest_newest(self, celebrity_id):
        """
        After inserting tweets, update the celebrities database with our current oldest and newest tweet ids
        """
        # Get all tweets with the celebrity in their ids list
        pipeline = [
            {"$match": {"celebrity_ids": celebrity_id}},
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
        result = list(self.tweets.aggregate(pipeline))[0]
        # Store the minimum and maximum
        celebrity_handler.celebrities.update_one(
            {"_id": celebrity_id},
            {
                "$set": {
                    "oldest_tweet_id": result["oldest_tweet_id"],
                    "newest_tweet_id": result["newest_tweet_id"],
                }
            },
            upsert=True,
        )


class CelebrityHandler:
    def __init__(self, db):
        self.celebrities = db["Celebrities"]

    def add_celebrity(self, id, handle=None, name=None):
        """
        Add a celebrity to the database. If one already exists with the same handle, update information
        """
        # Store whatever we have available
        # Store handles in lowercase always
        data = {"_id": id}
        if handle:
            data["handle"] = handle.lower()
        if name:
            data["name"] = name
        logger.info(f"Adding/updating celebrity: {data}")
        # Update the database entry by setting the new data
        res = self.celebrities.update_one({"_id": id}, {"$set": data}, upsert=True)
        return True if res.modified_count else False

    def get_celebrities(self, attributes):
        """
        Returns a list of whatever attributes for all celebrities in the db
        """
        if "_id" not in attributes:
            attributes += "_id"
        return sorted(
            [i for i in self.celebrities.find({}, {key: 1 for key in attributes})],
            key=lambda x: x["_id"],
        )

    def get_newest_tweet_id(self, celebrity_id):
        """
        Given a celebrity id, return the newest tweet id in the database
        If it doesn't exist, return None
        """
        exists = self.celebrities.find_one(
            {"_id": celebrity_id}, {"_id": 0, "newest_tweet_id": 1}
        )
        if exists:
            return exists["newest_tweet_id"]
        return None

    def get_oldest_tweet_id(self, celebrity_id):
        """
        Given a celebrity id, return the oldest tweet id in the database
        If it doesn't exist, return None
        """
        exists = self.celebrities.find_one(
            {"_id": celebrity_id}, {"_id": 0, "oldest_tweet_id": 1}
        )
        if exists:
            return exists["oldest_tweet_id"]
        return None


def connect_to_db():
    """
    Initializes the MongoDB connection and provides API"s for interaction
    """
    logger.info("Reading database info from config")
    with open("config/mongodb.config", "r") as f:
        config = json.load(f)
        db_username = config["username"]
        db_password = config["password"]
        # Connect to database
        client = MongoClient(
            f"mongodb+srv://{db_username}:{db_password}@celebscore.inxw4wt.mongodb.net",
            tlsCAFile=certifi.where(),
        )
        logger.info("Pinging database")
        result = client.admin.command("ping")
        if not result["ok"]:
            logger.error("Failed database connection")
        else:
            logger.info("Database ping successful")
        return client


client = connect_to_db()
db = client["CelebScoreData"]
tweets_handler = TweetsHandler(db)
celebrity_handler = CelebrityHandler(db)
