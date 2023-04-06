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
        bulk_ops = []
        for tweet in tweets:
            tweet_update = UpdateOne(
                {"_id": tweet["id_str"]},
                {
                    "$set": {"tweet": tweet, "author": tweet["user"]["id_str"]},
                    "$addToSet": {"celebrity_ids": celebrity_id},
                },
                upsert=True,
            )
            bulk_ops.append(tweet_update)
        result = self.tweets.bulk_write(bulk_ops)
        return result.matched_count == len(tweets)

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

    # TODO write function to return the id of the newest tweet for a celebrity


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
        return [i for i in self.celebrities.find({}, {key: 1 for key in attributes})]


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
