#!/usr/bin/python3
import sys
import pandas as pd
import concurrent.futures
import os

FILENAME = 'preprocess/training_set_raw.csv'
OUTPUT_POS = 'data/good_tweets.csv'
OUTPUT_NEG = 'data/bad_tweets.csv'


def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


def filter_non_english(start, end, data):
    print(f'processing {start} - {end-1}')
    droplist = []
    for i, row in data.iloc[start:end].iterrows():
        if not isEnglish(row['text']):
            droplist.append(i)
    return droplist


def main():
    # 1. load data
    data = pd.read_csv(
        FILENAME,
        encoding='latin_1',
        header=None,
        names=['label', 'tweet_id', 'date', 'idk', 'user', 'text'])
    # 2. drop unneeded columns
    data.drop(['tweet_id', 'date', 'idk', 'user'], axis=1, inplace=True)
    # 3. remove non-english tweets
    executor = concurrent.futures.ThreadPoolExecutor(8)
    step = 10000
    futures = [
        executor.submit(filter_non_english, s, min(s + step, len(data)), data)
        for s in range(0, len(data), step)
    ]
    done, pending = concurrent.futures.wait(futures)
    for f in done:
        data.drop(f.result(), inplace=True)

    # 4. split by positive/negative tweets
    pos = data[data['label'] == 4].drop('label', axis=1)
    neg = data[data['label'] == 0].drop('label', axis=1)
    # 4. output
    print(f'output {len(pos)} positive tweets and {len(neg)} negative tweets')
    if not os.path.exists('../data'):
        os.makedirs('../data')
    pos.to_csv(OUTPUT_POS, header=False, index=False)
    neg.to_csv(OUTPUT_NEG, header=False, index=False)


if __name__ == '__main__':
    main()
