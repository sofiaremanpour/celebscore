from utils.logging_utils import logger
from utils.twitter_utils import get_users
from utils.database import celebrity_handler


def update_celebrities_info():
    """
    Reads celebrities in the celebrities database connection
    Finds the ones without any id, looks them up, and adds to db
    """
    handles = celebrity_handler.get_handles_without_id()
    if not handles:
        return
    celebrities = get_users(handles=handles, attributes=[
        "id_str", "name", "screen_name"])
    for c in celebrities:
        celebrity_handler.add_celebrity(
            handle=c["screen_name"], name=c["name"], id=c["id_str"])


def main():
    update_celebrities_info()


if __name__ == "__main__":
    main()
