import json
import time

# import branca
import folium
import numpy as np
import pandas as pd
from folium.plugins import HeatMap
from selenium import webdriver
from tqdm import tqdm

from utils import team_utils
from utils.database import tweets_handler
from utils.logging_utils import logger


def create_maps(coordinates: dict[str, pd.DataFrame]) -> None:
    """
    For each team, create the map and save to the maps folder
    """
    for team_name, df in tqdm(coordinates.items(), desc="Maps"):
        # Define the data for the heatmap
        df = df[["lat", "long"]]
        heatmap_data = df.to_numpy(dtype=np.float64)
        # Define the gradient
        # colormap = branca.colormap.LinearColormap(["red", "blue"])
        # gradient = {x: colormap(x) for x in np.linspace(0, 1, 20)}
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
            # gradient=gradient,
        )
        # Stack layers and save
        # colormap.add_to(heatmap_map)
        heatmap_layer.add_to(heatmap_map)
        heatmap_map.save(f"maps/{team_name}_map.html")

        # Open in browser and take screenshot
        browser = webdriver.Chrome()
        browser.get(
            f"file://C:/Users/Chris/Documents/GitHub/celebscore/maps/{team_name}_map.html"
        )
        browser.fullscreen_window()
        # Wait for loading
        time.sleep(5)
        # Save screenshot
        browser.save_screenshot(f"maps/{team_name}_screenshot.png")
        browser.quit()


def find_center(coords: list[list[float]]) -> tuple[float, float]:
    """
    Given a list of (lat, long) coordinates, return the average of each component
    """
    return (
        sum(i[0] for i in coords) / len(coords),
        sum(i[1] for i in coords) / len(coords),
    )


def get_coordinates() -> dict[str, pd.DataFrame]:
    """
    Return a map of team_name to a dataframe with columns (lat, long, tweet_id)
    """
    coordinates: dict[str, pd.DataFrame] = {}
    # Get the list of data for each team
    team_tuples = team_utils.team_tuples
    for team_tuple in tqdm(team_tuples, desc="Gather Coords", total=16):
        tweet_iterator = tweets_handler.get_tweets(team_tuple)
        # Create a list of row dictionaries of (lat, long, tweet_id) for each tweet
        team_coords = []
        # Iterate through all tweets
        for tweet in tqdm(
            tweet_iterator,
            desc="Tweets",
            total=tweets_handler.get_tweet_count(team_tuple),
            position=1,
            leave=False,
        ):
            tweet_id = tweet["_id"]
            # Extract the place from the tweet
            place = tweet["tweet"]["place"]
            # If there is none, skip it
            if not place:
                continue
            long, lat = find_center(place["bounding_box"]["coordinates"][0])
            team_coords.append({"tweet_id": tweet_id, "lat": lat, "long": long})
        coordinates[team_tuple[0]] = pd.DataFrame(data=team_coords)
    return coordinates


def save_coordinates(terms_coordinates: dict[str, pd.DataFrame]) -> None:
    """
    Save the coordinate mapping of team_name to coordinate dataframe as a json file
    """
    with open("map_coords.json", "w") as f:
        terms_coordinates_json = {
            name: df.to_dict() for name, df in terms_coordinates.items()
        }
        json.dump(terms_coordinates_json, f)


def load_coordinates() -> dict[str, pd.DataFrame]:
    """
    Load the coordinate mapping of team_name to coordinate dataframe from json file
    """
    with open("map_coords.json", "r") as f:
        return {
            name: pd.DataFrame.from_dict(coords)
            for name, coords in json.load(f).items()
        }


def main():
    """
    Presents a menu on how to gather the data for the maps
    Creates the maps and saves the page and a screenshot for each team to the map folder
    """
    # Select to regather data from db or load from file for faster modification
    choice = int(input("Select:\n1. Load coordinates from db\n2. Load from file\n"))
    coordinates = None
    if choice == 1:
        # Load from database, then save to file
        coordinates = get_coordinates()
        save_coordinates(coordinates)
    elif choice == 2:
        # Load directly from file
        coordinates = load_coordinates()
    if coordinates is None:
        logger.error("Error loading coordinates, exiting")
        return
    # Create the maps
    create_maps(coordinates)


if __name__ == "__main__":
    main()
