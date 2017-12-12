#!/usr/bin/env bash

# Create a virtual environment
virtualenv -p python3 ~/venvs/bhabana
source ~/venvs/bhabana/bin/activate

# Install all the dev dependencies
pip install -r requirements/requirements.txt

# Install the Spacy models for DE and Eng
python -m spacy download en_vectors_web_lg
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_lg

python -m spacy link en_core_web_lg en
python -m spacy link de_core_news_sm de
python -m spacy link en_vectors_web_lg en_vectors

# Download the datasets to start training ASAP
# You can alternatively use the downloader API in Bhabana, but the following
# scripts will definitely make it faster

# This will create the default directory (~/.bhabana) to keep data for Bhabana
# Alternatively, you can create a symlink for this directory and keep the
# data wherever you want, or keep the data in the same structure as below and
# set an environment variable BHABANA_DATA to that directory
# export BHABANA_DATA=<your data directory>
# Make sure to maintain the same directory structure

if [ ! -d "~/.bhabana" ]; then
    mkdir ~/.bhabana ~/.bhabana/datasets
fi


if [ ! -d "~/.bhabana" ]; then
    mkdir ~/.bhabana ~/.bhabana/datasets
fi

mkdir ~/.bhabana ~/.bhabana/datasets ~/.bhabana/models
cd ~/.bhabana/datasets

if [ ! -f /tmp/foo.txt ]; then
    echo "File not found!"
fi
wget http://52.29.250.192:3000/datasets/hotel_reviews.tar.gz
tar -xvzf hotel_reviews.tar.gz
rm hotel_reviews.tar.gz

wget http://52.29.250.192:3000/datasets/amazon_reviews_de.tar.gz
tar -xvzf amazon_reviews_de.tar.gz
rm amazon_reviews_de.tar.gz

# cd ~/


# Clone the repository. You can keep this repo wherever you want.
git clone https://github.com/dashayushman/bhabana.git
cd bhabana
pip install -r requirements dev_requirements.txt

# If you want to test if everything is fine then you can run the following
# script

nosetests
