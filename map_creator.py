import pickle
import random
import branca

import folium
import numpy as np
import pandas as pd
from folium.plugins import HeatMap
from tqdm import tqdm

from utils.database import terms_handler, tweets_handler
from utils.logging_utils import logger


def create_maps(coordinates: dict[str, pd.DataFrame]) -> None:
    for team_name, df in tqdm(coordinates.items(), desc="Maps"):
        # Define the data for the heatmap
        df = df[["lat", "long", "sentiment"]]
        heatmap_data = df.to_numpy(dtype=np.float64)
        # Define the gradient
        colormap = branca.colormap.LinearColormap(["red", "blue"])
        gradient = {x: colormap(x) for x in np.linspace(0, 1, 20)}
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
        colormap.add_to(heatmap_map)
        heatmap_layer.add_to(heatmap_map)
        heatmap_map.save(f"maps/{team_name}_map.html")
        # Add a legend to the map

        # legend_html = '<div style="position: fixed; bottom: 50px; left: 50px; width: 120px; height: 100px; border:2px solid grey; z-index:9999; font-size:14px; background-color:white; opacity: .9;">&nbsp; Color Legend <br>'
        # for key in gradient:
        #     legend_html += '&nbsp; <i class="fa fa-circle fa-1x" style="color:{}"></i> {} <br>'.format(
        #         gradient[key], str(key)
        #     )
        # legend_html += "</div>"
        # legend = folium.Element(legend_html)
        # legend.add_to(heatmap_map)


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
    for term in tqdm(terms, desc="Gather Coords"):
        term_name = term["_id"]
        tweet_iterator = tweets_handler.get_tweets(term_name)
        # Create a dataframe of coordinates for each tweet
        team_coords = []
        # Iterate through all tweets
        for tweet in tweet_iterator:
            tweet_id = tweet["_id"]
            sentiment = random.random()  # tweets_handler.get_sentiment()
            # Extract the place from the tweet
            place = tweet["tweet"]["place"]
            # If there is none, skip it
            if not place:
                continue
            long, lat = find_center(place["bounding_box"]["coordinates"][0])
            team_coords.append(
                {"tweet_id": tweet_id, "lat": lat, "long": long, "sentiment": sentiment}
            )
        coordinates[term_name] = pd.DataFrame(team_coords)
    return coordinates


def terms_to_teams(terms_coordinates: dict[str, pd.DataFrame]) -> None:
    # Open the file of terms for each team
    terms_df = pd.read_csv("search_terms.csv", index_col=False)
    main_terms = list(terms_df["term"].astype("string"))
    alt_terms = list(terms_df["alt_term"].astype("string"))
    # Combine the coordinates of the alt term to the main term dataframe
    for main_term, alt_term in zip(main_terms, alt_terms):
        terms_coordinates[main_term] = pd.concat(
            [terms_coordinates[main_term], terms_coordinates[alt_term]],
            ignore_index=True,
            sort=True,
        )
        # Drop duplicates from tweets picked up by both terms
        terms_coordinates[main_term].drop_duplicates(inplace=True)
    # Delete terms we just added to the main terms
    for alt_term in alt_terms:
        del terms_coordinates[alt_term]


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
    elif choice == 2:
        # Load directly from file
        terms_coordinates = load_coordinates()
    if terms_coordinates is None:
        return
    # Create the maps
    logger.info(terms_coordinates)
    create_maps(terms_coordinates)


if __name__ == "__main__":
    main()
