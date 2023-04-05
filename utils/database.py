from pymongo import MongoClient
import json
from utils.logging_utils import logger


class CelebrityHandler:
    def __init__(self, db):
        self.celebrities = db["Celebrities"]

    def add_celebrity(self, handle, name=None, id=None, **kwargs):
        """
        Add a celebrity to the database. If one already exists with the same handle, update information
        """
        # Store whatever we have available
        data = {"handle": handle}
        if name:
            data["name"] = name
        if id:
            data["id"] = id
        # Store any extra information passed as kwargs
        data.update(kwargs)
        logger.info(f"Adding or updating celebrity: {data}")
        res = self.celebrities.update_one(
            {"handle": handle}, {"$set": data}, upsert=True)
        return True if res.modified_count else False

    def get_handles_without_id(self):
        """
        Returns the handle of all celebrities in the celebrities database currently without an id
        """
        return [i["handle"] for i in self.celebrities.find({"id": {"$exists": False}})]


class DatabaseHandler:
    def __init__(self):
        """
        Initializes the MongoDB connection and provides API's for interaction
        """
        logger.info("Reading database info from config")
        with open("config/mongodb.config", "r") as f:
            config = json.load(f)
            db_username = config["username"]
            db_password = config["password"]
            # Connect to database
            self.client = MongoClient(
                f"mongodb+srv://{db_username}:{db_password}@celebscore.inxw4wt.mongodb.net")
            self.db = self.client["CelebScoreData"]
            logger.info("Pinging database")
            result = self.client.admin.command('ping')
            if not result["ok"]:
                logger.error("Failed database connection")
            else:
                logger.info("Database ping successful")

        self.celebrity_handler = CelebrityHandler(self.db)
