from pymongo import MongoClient
import json
import re
from utils.logging_utils import logger


class FollowersHandler:
    def __init__(self, db):
        self.followers = db["Followers"]


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

    def get_missing_celebrities(self, handles):
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
celebrity_handler = CelebrityHandler(db)
