from pymongo import MongoClient
import json
import re
from utils.logging_utils import logger


class TweetsHandler:
    def __init__(self, db):
        self.tweets = db["tweets"]

    def add_tweet(self, id, tweet):
        """
        Given a tweet by a user,
        Add the tweet to the db as authored by that user
        """
        result = self.tweets.update_one(
            {"_id", id}, {"$addToSet": {"tweets": tweet}}, upsert=True)
        return result.matched_count > 0

    def get_tweets(self, id):
        """
        Given the id, return a list of tweets
        """
        return [i["tweets"] for i in self.followers.find({"_id"}, {"tweets", 1})]


class FollowersHandler:
    def __init__(self, db):
        self.followers = db["Followers"]

    def add_follower(self, follower_id, celebrity_id):
        """
        Given a user's id an the id of a celebrity,
        Add the user to the db as a follower of that celebrity
        """
        result = self.followers.update_one(
            {"_id", follower_id}, {"$addToSet": {"celebrity_ids": celebrity_id}}, upsert=True)
        return result.matched_count > 0

    def add_followers(self, follower_ids, celebrity_id):
        """
        Given a list of user ids and the id of a celebrity,
        Add the users to the db as followers of that celebrity
        """
        result = self.followers.update_many(
            {"_id": {"$in": follower_ids}},
            {"$addToSet": {"celebrity_ids": celebrity_id}},
            upsert=True)
        return result.matched_count == len(follower_ids)

    def get_followers_ids(self, celebrity_id):
        """
        Given the id of a celebrity, return a list of ids of their followers
        """
        return [i["_id"] for i in self.followers.find({"celebrity_ids": {"$in": [celebrity_id]}}, {"_id", 1})]


class CelebrityHandler:
    def __init__(self, db):
        self.celebrities = db["Celebrities"]

    def add_celebrity(self, handle, name=None, id=None, **kwargs):
        """
        Add a celebrity to the database. If one already exists with the same handle, update information
        """
        # Store whatever we have available
        # Store handles in lowercase always
        data = {"handle": handle.lower()}
        if name:
            data["name"] = name
        if id:
            data["_id"] = id
        # Store any extra information passed as kwargs
        data.update(kwargs)
        logger.info(f"Adding/updating celebrity: {data}")
        # Update the database entry by setting the new data
        res = self.celebrities.update_one(
            {"_id": id}, {"$set": data}, upsert=True)
        return True if res.modified_count else False

    def get_celebrities_ids(self):
        """
        Returns a list of ids of all celebrities in the db
        """
        return [i["_id"] for i in self.celebrities.find({}, {"_id": 1})]

    def get_celebrities_ids_incomplete_followers(self):
        """
        Returns a list of ids of all celebrities without a complete list of followers
        """

    def get_missing_celebrities_handles(self, handles):
        """
        Given a list of celebrity handles, return only the handles not in the database
        """
        lowercase_handles = [i.lower() for i in handles]
        existing = [i["handle"] for i in self.celebrities.find(
            {"handle": {"$in": lowercase_handles}}, {"handle": 1})]
        return [i for i in lowercase_handles if i not in existing]


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
            f"mongodb+srv://{db_username}:{db_password}@celebscore.inxw4wt.mongodb.net")
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
followers_handler = FollowersHandler(db)
celebrity_handler = CelebrityHandler(db)
