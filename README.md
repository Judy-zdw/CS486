# cs486-project
## First Time Setup
In the root directory, run
```
bash setup.sh
```
which unzips the 1.6m training data and runs preprocessing as well as fetches covid-19 validation data from twitter

After the script finishes, you should get the following files in the `data` folder:
good_tweets.csv - positively labelled training data
bad_tweets.csv - negatively labelled training data
good_validation.csv - positively labelled Covid-19 validation data
bad_validation.csv - negatively labelled Covid-19 validation data
