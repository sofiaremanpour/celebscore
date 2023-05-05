import pandas as pd
import json
from collections import defaultdict
from typing import Literal, Sequence

import nltk
import numpy as np
import torch
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from scipy.special import softmax
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from utils import team_utils
from utils.database import tweets_handler
from utils.logging_utils import logger

# SENTIMENT ANALYSIS MODELS AND OBJECTS
logger.info("Loading sentiment analysis resources")
# Roberta model sentiment analysis
ROBERTA_URL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
roberta_tokenizer = AutoTokenizer.from_pretrained(ROBERTA_URL)

roberta_model = AutoModelForSequenceClassification.from_pretrained(ROBERTA_URL)
roberta_config = AutoConfig.from_pretrained(ROBERTA_URL)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
logger.info(f"Roberta model compute device {device}")
# Move the model to the GPU for faster compute
roberta_model = roberta_model.to(device)

# VADER model sentiment analysis
nltk.download("vader_lexicon")
nltk.download("punkt")
nltk.download("stopwords")
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words("english"))

# Sentiwordnet model sentiment analysis
nltk.download("sentiwordnet")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")
lemmatizer = WordNetLemmatizer()


def calculate_sentiment(tweet) -> dict:
    """
    Given the tweet object,
    Return the sentiment of the tweet from the different analyzers
    """
    sentiment = {name: func(tweet) for name, func in sentiment_funcs.items()}
    return sentiment


def get_roberta_sentiment(tweet) -> dict:
    """
    Run roberta sentiment analysis on a tweet
    Return the dict of positive, neutral, negative scores
    """
    # Tokenize the text using the roberta model tokenizer and send tensor to gpu before running through model
    encoded_tweet = roberta_tokenizer(tweet["tweet"]["text"], return_tensors="pt").to(
        device
    )
    # Run tokenized text through roberta model
    model_output = roberta_model(**encoded_tweet)
    tweet_scores = model_output[0][0].detach().cpu().numpy()
    # Softmax and get the label
    scores = softmax(tweet_scores)
    labeled_scores = {
        roberta_config.id2label[i]: scores[i].astype(float)
        for i in range(scores.shape[0])
    }
    # Return the scores labelled
    return labeled_scores


def get_vader_sentiment(tweet) -> dict:
    """
    Run VADER sentiment analysis on a tweet
    Return the dict of pos, neu, neg scores
    """
    # Tokenize using nltk word_tokenizer
    tokenized = word_tokenize(tweet["tweet"]["text"])
    clean_tokens = []
    # Remove stop words and put to lowercase
    for token in tokenized:
        if token.lower() not in stop_words and token.isalpha():
            clean_tokens.append(token.lower())
    # Get polarity scores from cleaned input using SIA
    sentiment_scores = sia.polarity_scores(" ".join(clean_tokens))
    return sentiment_scores


def convert_tags(pos_tag) -> str | None:
    """
    Convert the part of speech tag from nltk to the sentiwordnet tag
    Return None if there is no conversion
    """
    if pos_tag.startswith("JJ"):
        return wn.ADJ
    elif pos_tag.startswith("NN"):
        return wn.NOUN
    elif pos_tag.startswith("RB"):
        return wn.ADV
    elif pos_tag.startswith("VB") or pos_tag.startswith("MD"):
        return wn.VERB
    return None


def get_swn_sentiment(tweet) -> dict:
    """
    Run sentiwordnet sentiment analysis on a tweet
    Return the dict of pos, neu, objectivity scores
    """
    # Tokenize with nltk work_tokenize
    word_tokenized_sentences = (
        nltk.word_tokenize(sentence)
        for sentence in nltk.sent_tokenize(tweet["tweet"]["text"])
    )
    # Add part of speech tags to each of the words for each sentence in the list
    tagged_sentences = nltk.pos_tag_sents(word_tokenized_sentences)
    # Convert to a 1d list of all words, tagged
    tagged_words = (j for i in tagged_sentences for j in i)
    pos_scores = []
    neg_scores = []
    obj_scores = []
    # Average the sentiment of each word
    for word, part_of_speech in tagged_words:
        # Remove stop words
        if word in stop_words:
            continue
        # Get the correct part of speech
        swn_part_of_speech = convert_tags(part_of_speech)
        if swn_part_of_speech is None:
            continue
        # Find all the sentiment scores of the word
        synsets = list(swn.senti_synsets(word, pos=swn_part_of_speech))
        if not synsets:
            continue
        # Take the most common
        synset = synsets[0]
        if not synset:
            continue
        # Add the sentiment to the total to be averaged
        obj_scores.append(synset.obj_score())
        pos_scores.append(synset.pos_score())
        neg_scores.append(synset.neg_score())
    try:
        positive = np.mean(pos_scores)
        negative = np.mean(neg_scores)
        objectivity = np.mean(obj_scores)
    except ZeroDivisionError:
        positive = 0
        negative = 0
        objectivity = 0
    return {
        "positive": float(positive),
        "negative": float(negative),
        "objectivity": float(objectivity),
    }


def calculate_sentiments(terms: Sequence[str] | None) -> None:
    """
    For a list of terms, calculate the sentiment for each tweet associated with all analyzers
    Add to database for that tweet
    """
    if terms is None:
        terms = team_utils.all_terms
    # Define an iterator that will return tweets associated with a term in terms
    tweet_iterator = tweets_handler.get_tweets(terms, ["sentiment"])
    # Iterate a tweet at a time
    for tweet in tqdm(
        tweet_iterator,
        total=tweets_handler.get_tweet_count(terms),
        desc="Tweets",
        position=1,
        leave=False,
    ):
        # If the sentiment is already in the document, skip it
        if "sentiment" in tweet:
            continue
        # Calculate the sentiment using all analyzers
        tweet_sentiment = calculate_sentiment(tweet)
        # Add sentiment to db for that tweet
        tweets_handler.set_sentiment(tweet_sentiment, tweet["_id"])


# Define the mapping of sentiment analyzer names to the functions that run them on a tweet object
sentiment_funcs = {
    "roberta": get_roberta_sentiment,
    "vader": get_vader_sentiment,
    "sentiwordnet": get_swn_sentiment,
}


def get_max_label(sentiment: dict) -> Literal["pos", "neu", "neg"]:
    label = max(sentiment.items(), key=lambda x: x[1])[0]
    if label == "pos" or label == "positive":
        return "pos"
    if label == "neg" or label == "negative":
        return "neg"
    return "neu"


def get_averaged_label(sentiment: dict) -> str:
    """
    Given a sentiment object with all analyzers, return the label from the average of scores from each analyzer
    """
    # Find the label with the highest score for each analyzer
    averaged_scores = {
        "pos": (
            sum(
                (
                    sentiment["roberta"]["positive"],
                    sentiment["vader"]["pos"],
                    sentiment["sentiwordnet"]["positive"],
                )
            )
            / 3
        ),
        "neg": (
            sum(
                (
                    sentiment["roberta"]["negative"],
                    sentiment["vader"]["neg"],
                    sentiment["sentiwordnet"]["negative"],
                )
            )
            / 3
        ),
        "neu": (
            sum(
                (
                    sentiment["roberta"]["neutral"],
                    sentiment["vader"]["neu"],
                    sentiment["sentiwordnet"]["objectivity"],
                )
            )
            / 3
        ),
    }
    return max(averaged_scores.items(), key=lambda x: x[1])[0]


def find_counts() -> dict[str, dict[str, dict[str, int]]]:
    """
    Return a dict mapping each team_name to a dict of analyzers to dict of counts for each label
    Include a "average" entry that record the counts of labels averaged for all analyzers
    Include a "agreed" entry that records the counts of labels that all the analyzers agree on
    """
    # Create dict of team_name to counts
    counts: dict[str, dict[str, dict[str, int]]] = {}
    for team_tuple in tqdm(team_utils.team_tuples, desc="Teams", total=16):
        # Define a dictionary of counts for the team
        team_counts: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(lambda: 0)
        )
        # Iterate a tweet at a time
        for tweet in tqdm(
            tweets_handler.get_tweets(team_tuple, ["sentiment"]),
            total=tweets_handler.get_tweet_count(team_tuple),
            desc=f"{team_tuple}",
            position=1,
            leave=False,
        ):
            # If the tweet document has no sentiment, skip it
            if "sentiment" not in tweet:
                continue
            sentiment: dict[str, dict[str, float]] = tweet["sentiment"]
            labels = {name: get_max_label(scores) for name, scores in sentiment.items()}
            # Add the count for each of the analyzers
            for analyzer_name in sentiment:
                team_counts[analyzer_name][labels[analyzer_name]] += 1
            # Add the count for the average
            team_counts["average"][get_averaged_label(sentiment)] += 1
            # Add the count for the agreed
            if all((labels["roberta"] == label for label in labels.values())):
                team_counts["agreed"][labels["roberta"]] += 1

        # Add the count dictionary to be mapped to the team_name
        counts[team_tuple[0]] = team_counts
    # Return the final counts
    return counts


def save_counts(filename: str, counts: dict[str, dict[str, dict[str, int]]]) -> None:
    """
    Save the counts of pos, neu, neg for each team to the file
    """
    with open(filename, "w") as f:
        json.dump(counts, f)


def save_scores(scores: dict[str, dict[str, dict[str, int]]]) -> None:
    """
    Save the scores into a csv for each analyer to the file scores.json
    """
    # Modify scores to be a mapping from the analyzer first, and the team mapped to the label scores
    new_scores = defaultdict(lambda: {})
    for team_name, analyzer_scores in scores.items():
        for analyzer_name, label_scores in analyzer_scores.items():
            new_scores[analyzer_name][team_name] = label_scores

    # Save a CSV for each using pandas
    for analyzer_name, team_scores in new_scores.items():
        df = pd.DataFrame.from_dict(team_scores, orient="index")
        df.to_csv(f"scores/{analyzer_name}.csv")


def find_scores(
    counts: dict[str, dict[str, dict[str, int]]]
) -> dict[str, dict[str, dict[str, int]]]:
    """
    For each team, calculate the pos, neu, neg score using the counts
    """
    scores = defaultdict(lambda: {})
    for team_name, analyzer_counts in counts.items():
        for analyzer_name, label_counts in analyzer_counts.items():
            # Find the total count for the team
            total = sum(label_counts.values())
            # Calculate the score using their sentiment counts
            analyzer_score = {
                label: score / total for label, score in label_counts.items()
            }
            scores[team_name][analyzer_name] = analyzer_score
    # Return the final mapping
    return scores


def main():
    """
    Present a menu to analyze the sentiment for tweets in the database first
    Or skip that and just count and save the scores for each team to the file counts.json
    """
    choice = int(
        input("1. Analyze tweets first\n2. Just count sentiment already analyzed\n")
    )
    if choice == 1:
        # Calculate the sentiment for all of the teams
        calculate_sentiments(team_utils.all_terms)
    # Tally the sentiments for each team, save, score, save
    logger.info("Finding counts")
    counts = find_counts()
    save_counts("counts.json", counts)
    scores = find_scores(counts)
    save_scores(scores)
    logger.info(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
