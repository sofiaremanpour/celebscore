import pickle
import random
import time

import branca
import folium
import numpy as np
import pandas as pd
from folium.plugins import HeatMap
from selenium import webdriver
from tqdm import tqdm

from utils.database import terms_handler, tweets_handler
from utils.logging_utils import logger
from utils import team_utils


def create_maps(coordinates: dict[str, pd.DataFrame]) -> None:
    for team_name, df in tqdm(coordinates.items(), desc="Maps"):
        # Define the data for the heatmap
        df = df[["lat", "long", "sentiment"]]
        heatmap_data = df.to_numpy(dtype=np.float64)
        # Define the gradient
        colormap = branca.colormap.LinearColormap(["red", "blue"])
        gradient = {x: colormap(x) for x in np.linspace(-1, 1, 20)}
        # Create the map centered at the US
        heatmap_map = folium.Map(location=[39.8283, -98.5795], zoom_start=4)
        # Create a heatmap layer using the lat and long coordinates
        heatmap_layer = HeatMap(
            data=heatmap_data,
            name="Heatmap",
            min_opacity=0.3,
            radius=20,
            blur=15,
            max_zoom=1,
            overlay=False,
            gradient=gradient,
        )
        # colormap.add_to(heatmap_map)
        heatmap_layer.add_to(heatmap_map)
        heatmap_map.save(f"maps/{team_name}_map.html")

        # Open in browser and save
        browser = webdriver.Chrome()
        browser.get(
            f"file://C:/Users/Chris/Documents/GitHub/celebscore/maps/{team_name}_map.html"
        )
        browser.fullscreen_window()
        time.sleep(5)
        browser.save_screenshot(f"maps/{team_name}_screenshot.png")
        browser.quit()


def find_center(coords: list[list[float]]) -> tuple[float, float]:
    return (
        sum(i[0] for i in coords) / len(coords),
        sum(i[1] for i in coords) / len(coords),
    )


def get_coordinates() -> dict[str, pd.DataFrame]:
    """
    Return a list of dataframes with the coordinates of each tweet in each one
    """
    coordinates: dict[str, pd.DataFrame] = {}
    # Get the list of data for each team
    terms = terms_handler.get_search_terms()
    terms = [
        i
        for i in terms
        if i["_id"]
        in [
            "Milwaukee Bucks",
            "Bucks",
            "Miami Heat",
            "Heat",
            "Memphis Grizzlies",
            "Grizzlies",
            "LA Lakers",
            "Lakers",
        ]
    ]
    for term in tqdm(terms, desc="Gather Coords"):
        term_name = term["_id"]
        tweet_iterator = tweets_handler.get_tweets(term_name, ["sentiment"])
        # Create a dataframe of coordinates for each tweet
        team_coords = []
        # Iterate through all tweets
        for tweet in tweet_iterator:
            tweet_id = tweet["_id"]
            # Extract the place from the tweet
            place = tweet["tweet"]["place"]
            sentiment = tweet.get("sentiment", {"roberta": "neutral"})
            roberta = sentiment["roberta"]
            mapping = {"positive": 1, "neutral": 0, "negative": -1}
            roberta = mapping[roberta]
            # If there is none, skip it
            if not place:
                continue
            long, lat = find_center(place["bounding_box"]["coordinates"][0])
            team_coords.append(
                {"tweet_id": tweet_id, "lat": lat, "long": long, "sentiment": roberta}
            )
        coordinates[term_name] = pd.DataFrame(data=team_coords)
    return coordinates


def terms_to_teams(terms_coordinates: dict[str, pd.DataFrame]) -> None:
    # Combine the coordinates of the alt term to the main term dataframe
    for main_term, alt_term in team_utils.teams:
        if main_term not in terms_coordinates:
            continue
        terms_coordinates[main_term] = pd.concat(
            [terms_coordinates[main_term], terms_coordinates[alt_term]],
            ignore_index=True,
            sort=True,
        )
        # Drop duplicates from tweets picked up by both terms
        terms_coordinates[main_term].drop_duplicates(inplace=True)
    # Delete terms we just added to the main terms
    for alt_term in alt_terms:
        try:
            del terms_coordinates[alt_term]
        except KeyError:
            pass


def save_coordinates(terms_coordinates) -> None:
    with open("map_coords.pickle", "wb") as f:
        pickle.dump(terms_coordinates, f)


def load_coordinates() -> dict[str, pd.DataFrame]:
    with open("map_coords.pickle", "rb") as f:
        return pickle.load(f)


def main():
    # Select to regather data from db or load from file for faster modification
    terms_coordinates = None
    choice = int(input("Select:\n1. Load coordinates from db\n2. Load from file\n"))
    if choice == 1:
        # Load from database, convert, then save
        terms_coordinates = get_coordinates()
        terms_to_teams(terms_coordinates)
        save_coordinates(terms_coordinates)
        logger.info(terms_coordinates)
    elif choice == 2:
        # Load directly from file
        terms_coordinates = load_coordinates()
    if terms_coordinates is None:
        return
    # Create the maps
    create_maps(terms_coordinates)


if __name__ == "__main__":
    main()
