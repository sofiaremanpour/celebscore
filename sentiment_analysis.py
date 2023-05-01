import json
from typing import Optional

import nltk
import numpy as np
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

# Sentiment analysis stuff goes here
logger.info("Loading tokenizer and model")

# Roberta model sentiment analysis
ROBERTA_URL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
roberta_tokenizer = AutoTokenizer.from_pretrained(ROBERTA_URL)
roberta_model = AutoModelForSequenceClassification.from_pretrained(ROBERTA_URL)
roberta_config = AutoConfig.from_pretrained(ROBERTA_URL)

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
    Run roberta sentiment on a tweet
    Return the dict of scores
    """
    # Tokenize the text
    encoded_tweet = roberta_tokenizer(tweet["tweet"]["text"], return_tensors="pt")
    # Run through model
    model_output = roberta_model(**encoded_tweet)
    tweet_scores = model_output[0][0].detach().numpy()
    # Softmax and get the label
    scores = softmax(tweet_scores)
    labeled_scores = {
        roberta_config.id2label[i]: scores[i].astype(float)
        for i in range(scores.shape[0])
    }
    # Return the label
    return labeled_scores


def get_vader_sentiment(tweet) -> dict:
    """
    Run VADER sentiment analysis on a tweet
    Return the dict of scores
    """
    tokenized = word_tokenize(tweet["tweet"]["text"])
    clean_tokens = []
    for token in tokenized:
        if token.lower() not in stop_words and token.isalpha():
            clean_tokens.append(token.lower())
    sentiment_scores = sia.polarity_scores(" ".join(clean_tokens))
    return sentiment_scores


def convert_tags(pos_tag) -> Optional[str]:
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
    Return the dict of scores
    """
    # Tokenize by word within each sentence list
    word_tokenized_sentences = (
        nltk.word_tokenize(sentence)
        for sentence in nltk.sent_tokenize(tweet["tweet"]["text"])
    )
    # Add tags to each of the words for each sentence in the list
    tagged_sentences = nltk.pos_tag_sents(word_tokenized_sentences)
    # Convert to a 1d list of all words, tagged
    tagged_words = (j for i in tagged_sentences for j in i)
    pos_scores = []
    neg_scores = []
    obj_scores = []
    # Average the sentiment of each word
    for word, part_of_speech in tagged_words:
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
        positive = np.average(pos_scores)
        negative = np.average(neg_scores)
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


def calculate_sentiments(term_tuple: tuple[str, str]) -> None:
    """
    For a team, calculate the sentiment for each tweet and add to db
    """
    # Define an iterator that will return tweets in batches
    tweet_iterator = tweets_handler.get_tweets(term_tuple, ["sentiment"])
    count = 0
    # Calculate the sentiment for each tweet
    for tweet in tqdm(
        tweet_iterator,
        total=tweets_handler.get_tweet_count(term_tuple),
        desc=f"{term_tuple} tweets",
        position=1,
        leave=False,
    ):
        if "sentiment" in tweet:
            continue
        # Calculate the sentiment using all analyzers
        tweet_sentiment = calculate_sentiment(tweet)
        # Add sentiment to db
        tweets_handler.set_sentiment(tweet_sentiment, tweet["_id"])
        count += 1
        if count == 1000:
            break


sentiment_funcs = {
    "roberta": get_roberta_sentiment,
    "vader": get_vader_sentiment,
    "sentiwordnet": get_swn_sentiment,
}

# def tabulate_sentiment() -> dict[str, dict[str, pd.DataFrame]]:
#     """
#     Return a dict mapping a analyzer name to a dataframe of tweet sentiment scores
#     """
#     # For each team, go through all their tweets and create a dataframe for each analyzer
#     for team_tuple in tqdm(team_utils.team_tuples):
#         for tweet in tweets_handler.get_tweets(team_tuple):
#             tweet_id = tweet["_id"]
#             sentiment: dict = tweet["sentiment"]
#             for analyzer_name, entry in sentiment.items():
#                 if analyzer_name not in analyzer_data:
#                     analyzer_data[analyzer_name] = []
#                 entry["_id"] = tweet_id
#                 analyzer_data[analyzer_name].append(entry)
#         analyzer_frames = {
#             name: pd.DataFrame(scores) for name, scores in analyzer_data.items()
#         }
#         return analyzer_frames


def save_counts(counts: dict[str, dict[str, int]]) -> None:
    with open("counts.json", "w") as f:
        json.dump(counts, f)


def main():
    choice = int(input("1. Calculate Sentiment\n2. Count Sentiment\n"))
    if choice == 1:
        # Define a pool to work in parallel
        while True:
            for term_tuple in tqdm(team_utils.team_tuples, desc="Teams", total=16):
                calculate_sentiments(term_tuple)
    else:
        pass
        # counts = find_counts()
        # save_counts(counts)


if __name__ == "__main__":
    main()
