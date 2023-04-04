
# CelebScore

## Features

Our celebrity obsession analysis tool, “CelebScores” is a project aimed at determining how much the followers of a celebrity discuss that celebrity. Within our project, we will look at tweets that have been tweeted by followers of popular celebrities, such as actors, artists, and politicians, and create a score based on how “obsessed” they are with that celebrity, measured by how often they tweet about them. In addition, we will perform sentiment analysis on tweets regarding that celebrity, creating a score that measures how well liked the celebrity is by their followers.

  

In the end, we will create a front end display through a website that will display this data visually. Celebrities will be ranked by their followers' obsession score, as well as how well liked they are.

  

## Project Structure

### Python

The program is broken up into interacting components that allows code reuse for the different phases of the project.

The files ```twitter_utils.py```, ```logging_utils.py```, and ```database.py``` contain functions meant to be used whenever you need to access the twitter api, the logger for debugging, or database data respectively.

The first section, ```celebrity_scraper.py``` finds the twitter handles of celebrities in MongoDB collection, and update any missing info by looking up their twitter user. This allows us to add things to the database and have them automatically become part of the system. See below for more info on connecting with MongoDB.

  

## Config files

To avoid uploading sensitive information to github, I've moved the configuration for twitter and mongodb to the config folder, where it won't be uploaded. For twitter, it is easy, just look at the template config file, and add your keys to the correct locations, and remove the ```.template``` from the name. For MongoDB, you need an account to access the database, which I can make for you if you contact me.
