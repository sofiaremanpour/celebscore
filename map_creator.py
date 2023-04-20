import folium
import numpy as np
import pandas as pd
from folium.plugins import HeatMap
from tqdm import tqdm

from utils.database import terms_handler, tweets_handler
from utils.logging_utils import logger


def create_maps(coordinates: dict[str, pd.DataFrame]):
    for team_name, df in tqdm(coordinates.items(), desc="Maps"):
        # Create the map centered at the US
        heatmap_map = folium.Map(location=[39.8283, -98.5795], zoom_start=4)
        # Create a heatmap layer using the lat and long coordinates
        heatmap_layer = HeatMap(
            data=df.to_numpy(dtype=np.float32),
            name="Heatmap",
            min_opacity=0.2,
            radius=15,
            blur=10,
            max_zoom=1,
        )
        heatmap_layer.add_to(heatmap_map)
        heatmap_map.save(f"maps/{team_name}_map.html")
    # # Add a legend to the map
    # colormap = {-1: "red", 0: "white", 1: "blue"}
    # legend_html = '<div style="position: fixed; bottom: 50px; left: 50px; width: 120px; height: 100px; border:2px solid grey; z-index:9999; font-size:14px; background-color:white; opacity: .9;">&nbsp; Color Legend <br>'
    # for key in colormap:
    #     legend_html += (
    #         '&nbsp; <i class="fa fa-circle fa-1x" style="color:{}"></i> {} <br>'.format(
    #             colormap[key], str(key)
    #         )
    #     )
    # legend_html += "</div>"
    # heatmap_map.get_root().html.add_child(folium.Element(legend_html))


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
    teams = terms_handler.get_search_terms()
    for team in tqdm(teams, desc="Gather Coords"):
        team_name = team["_id"]
        tweet_iterator = tweets_handler.get_tweets(team_name)
        # Create a dataframe of coordinates for each tweet
        team_coords = []
        # Iterate through all tweets
        for tweet in tweet_iterator:
            # tweet_id = tweet["_id"]
            # sentiment = tweets_handler.get_sentiment()
            # Extract the place from the tweet
            place = tweet["tweet"]["place"]
            # If there is none, skip it
            if not place:
                continue
            long, lat = find_center(place["bounding_box"]["coordinates"][0])
            team_coords.append((lat, long))
        coordinates[team_name] = pd.DataFrame(team_coords, columns=["lat", "long"])

    return coordinates


def main():
    coordinates = get_coordinates()
    logger.info(coordinates)
    create_maps(coordinates)


if __name__ == "__main__":
    main()
