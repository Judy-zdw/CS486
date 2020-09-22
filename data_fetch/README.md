# Usage
Fetches historical Covid-19 social distancing tweets between 2020-02-15 to 2020-05-01 in the United States

## Requirements
- python3
- pip3
## Prerequisites
``
pip3 install -r requirements.txt
``
## Usage
``
python3 fetch.py [max tweets to fetch] [start date] [end date] [output_file]
``

- default max tweets to fetch is 10
- start/end dates format "yyyy-mm-dd"
- output_file defaults to 'tweets-out.csv'
