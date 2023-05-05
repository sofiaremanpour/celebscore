
# NBA Playoff Predictor

## Features

Our NBA team playoff predictor scrapes tweets from Twitter that are found from searching for the team names. From there, we create a heatmap by extracting the locations of the tweets, and perform sentiment analysis on the tweets to score each team by the proportion that are positive, negative, and neutral.
  
## Project Setup

### Packages and MongoDB
This project makes use of a MongoDB server and various libraries. To install all of the required libraries, run ```pip -r requirements.txt```. The MongoDB server is expected to be running locally, however it can be modified through the code commented out in the ```connect_to_db()``` function in ```database.py```.

### Config files

To avoid uploading sensitive information to github, I've moved the configuration for twitter and mongodb to the config folder, where it won't be uploaded. For Twitter, it is easy, just look at the template config file, and add your keys to the correct locations, and remove the ```.template``` from the name. For MongoDB, the process is the same (although not required when using a unauthenticated local server)

## Project Structure

### Utils

The program is broken up into interacting components that allows code reuse for the different phases of the project.

The files under the ```utils``` directory: ```twitter_utils.py```, ```logging_utils.py```, ```team_utils.py``` and ```database.py``` contain functions that are helpful for multiple aspects of the project

#### ```twitter_utils.py```
This file sets up the connection with the Twitter API, and provides the function get_search_results(...), which allows you to input a particular query to search for, as well as the bounds of time you want tweets from using the optional parameters oldest_tweet_id and newest_tweet_id. It returns a generator that yields tweets in batches.

#### ```logging_utils.py```
This file just sets up the global logger, so that it can be imported and used in all files.

#### ```team_utils.py```
This file loads the teams we want to search for from the csv file ```search_terms.csv```, and creates various lists for the main team name, and an alternate name

#### ```database.py```
This file initializes a MongoDB database connection to a local MongoDB server, and provides two handler objects for interfacing with database information

### Main files
There are 3 main files that serve as the entrypoint for different phases of the project.

#### ```search_scraper.py```
This file initializes terms from the file into the database, and then starts scraping tweets by searching the queries. Every batch that the tweet generator yields is immediately put into the database, and will only search from tweets not within the currently gather bounds for any term.

#### ```map_creator.py```
This file gathers the locations from tweets in the database, and creates a heatmap for each team using the tweets associated with it. It also saves the locations to a json file ```map_coords.json``` to cache the locations instead of regathering from the database each time, chosen through a selection menu.

#### ```sentiment_analysis.py```
This file runs sentiment analysis through 3 sentiment analyzers on all of the tweets in the database, and then performs calculations to count the number of labels assigned for each team. It then generates scores which are written to the csv files under the ```/scores``` directory.