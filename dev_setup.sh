#!/usr/bin/env bash

virtualenv -p python3 ~/venvs/bhabana
source ~/venvs/bhabana/bin/activate

pip install -r requirements/dev_requirements.txt
python -m spacy download en_vectors_web_lg
python -m spacy download en_vectors_web_lg