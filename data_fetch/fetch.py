#!/usr/bin/python3

import sys
import os
import GetOldTweets3 as got
import pandas as pd
import csv
import re

CORONA_WORDS = [
    'social distancing', 'quarantine', 'self-isolation', 'covid19',
    'coronavirus'
]
SYMBOLS = ['@', '#', '"', ':', '-', ')', '(', ',', '.']
POSITIVE_EMOTICONS = [':)', ':-)']
NEGATIVE_EMOTICONS = [':(', ':-(']
SINCE = '2020-02-15'
UNTIL = '2020-05-01'
OUTPUT_FILE = 'data_fetch/validation_set_raw.csv'
GOOD_FILE = 'data/good_validation.csv'
BAD_FILE = 'data/bad_validation.csv'
LOCATION = 'US'
MAX = 10000


def build_query(main_word):
    q = main_word + ' (' + ' OR '.join(POSITIVE_EMOTICONS +
                                       NEGATIVE_EMOTICONS) + ')'
    return q


def run_download():
    all_tweets = pd.DataFrame(columns=['text'])
    for main_query_word in CORONA_WORDS:
        tweetCriteria = got.manager.TweetCriteria().setQuerySearch(
            build_query(main_query_word)).setSince(SINCE).setUntil(
                UNTIL).setMaxTweets(MAX).setNear(LOCATION).setWithin('3881mi')
        tweets = pd.DataFrame(map(
            lambda tweet: [tweet.id, tweet.text, tweet.date],
            got.manager.TweetManager.getTweets(tweetCriteria)),
                              columns=['tweet_id', 'text', 'date'])
        tweets = tweets.drop('tweet_id', axis=1).drop('date', axis=1)
        all_tweets = all_tweets.append(tweets, ignore_index=True)
    all_tweets.to_csv(OUTPUT_FILE, index=False)


def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


def run_preprocess():
    # 1. load csv
    data = pd.read_csv(OUTPUT_FILE)
    data['label'] = 0
    # 2. add label
    for i, row in data.iterrows():
        if any(ele in row['text'] for ele in POSITIVE_EMOTICONS):
            data.at[i, 'label'] = 1
    # 3. remove URLs
    for i, row in data.iterrows():
        data.at[i, 'text'] = re.sub(r"http\S+", "", data.at[i, 'text'])
    # 4. remove emoticons and symbols
    for i, row in data.iterrows():
        for e in (POSITIVE_EMOTICONS + NEGATIVE_EMOTICONS + SYMBOLS):
            data.at[i, 'text'] = data.at[i, 'text'].replace(e, '')
    # # 5. remove non-english tweets
    # for i, row in data.iterrows():
    #     if not isEnglish(row['text']):
    #         data.drop(i, inplace=True)

    # 6. split data into good_validation.csv bad_validation.csv
    good = data[data['label'] == 1].drop('label', axis=1)
    bad = data[data['label'] == 0].drop('label', axis=1)
    # 7. output stats
    print(f'outputting {len(good)} good tweets and {len(bad)} bad tweets')
    if not os.path.exists('../data'):
        os.makedirs('../data')
    good.to_csv(GOOD_FILE, index=False, header=False, quoting=csv.QUOTE_NONE)
    bad.to_csv(BAD_FILE, index=False, header=False, quoting=csv.QUOTE_NONE)


def main(argv):
    print('downloading data')
    run_download()
    print('running preprocessing')
    run_preprocess()


if __name__ == "__main__":
    main(sys.argv[1:])
