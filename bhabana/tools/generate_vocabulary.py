# Copied from
# https://github.com/google/seq2seq/blob/master/bin/tools/generate_vocab.py
#
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import codecs
import argparse
import collections
import logging
import progressbar

from bhabana.processing import *

parser = argparse.ArgumentParser(
    description="Generate vocabulary for a tokenized text file.")
parser.add_argument(
    "--min_frequency",
    dest="min_frequency",
    type=int,
    default=0,
    help="Minimum frequency of a word to be included in the vocabulary.")
parser.add_argument(
    "--max_vocab_size",
    dest="max_vocab_size",
    type=int,
    help="Maximum number of tokens in the vocabulary")
parser.add_argument(
    "--indir",
    dest="indir",
    type=str,
    help="directory containing the text files that you want to process")
parser.add_argument(
    "--mode",
    dest="mode",
    type=str,
    default="word",
    help="word, semhash, char"
)
parser.add_argument(
    "--line_processor",
    dest="line_processor",
    type=str,
    default="json",
    help="json, tsv, text"
)
parser.add_argument(
    "--separator",
    dest="separator",
    type=str,
    default="\t",
    help="\\t, $%%^% or anything else"
)
parser.add_argument(
    "--keys",
    dest="keys",
    type=str,
    default="text",
    help="comma separated for more than one, and column indeces if line "
         "processor is of type tsv or separator"
)
parser.add_argument(
    "--lang",
    dest="lang",
    type=str,
    default="en",
    help="language"
)

args = parser.parse_args()

# Counter for all tokens in the vocabulary
cnt = collections.Counter()
keys = args.keys.split(",")
fields = []


def get_line_processor(lp_name):
    if lp_name.lower() == "json":
        return JSONLineProcessor
    elif lp_name.lower() == "tsv":
        return TSVLineProcessor
    elif lp_name.lower() == "text":
        return TextLineProcessor

LP = get_line_processor(args.line_processor)
tokenizer = Tokenizer(args.lang, mode=args.mode, process_batch=True)
if args.line_processor.lower() == "json":
    for key in keys:
        fields.append({
            "key": key,
            "dtype": str
        })
elif args.line_processor.lower() == "tsv":
    keys = [int(key) for key in keys]
    max_index = 0
    for key in keys:
        if key > max_index:
            max_index = key
    for key in range(max_index):
        fields.append({
            "key": key,
            "dtype": str
        })
if args.line_processor.lower() == "text":
    line_processor = LP(fields, args.separator)
else:
    line_processor = LP(fields)
file_list = os.listdir(args.indir)
bar = progressbar.ProgressBar(max_value=len(file_list),
                              redirect_stdout=True)
n_line = 0
for file_name in file_list:
    file_path = os.path.join(args.indir, file_name)
    with codecs.open(file_path, "r", "utf-8") as fp:
        line = fp.read()
        valid_line = line_processor.is_valid_data(line)
        if valid_line["is_valid"]:
            values = line_processor.process(line)
            if args.line_processor == "tsv":
                values = [values[i] for i in keys]
            tokens = tokenizer.process(values)
            tokens = [y for x in tokens for y in x]
            if args.mode == "semhash":
                tokens = [y for x in tokens for y in x]
            cnt.update(tokens)
            n_line += 1
        else:
            print("Skipping line '{}' because of the following "
                  "error:".format(line))
            print(valid_line["error"])
    bar.update(n_line)
bar.finish()

logging.info("Found %d unique tokens in the vocabulary.", len(cnt))

# Filter tokens below the frequency threshold
if args.min_frequency > 0:
  filtered_tokens = [(w, c) for w, c in cnt.most_common()
                     if c > args.min_frequency]
  cnt = collections.Counter(dict(filtered_tokens))

logging.info("Found %d unique tokens with frequency > %d.",
             len(cnt), args.min_frequency)

# Sort tokens by 1. frequency 2. lexically to break ties
word_with_counts = cnt.most_common()
word_with_counts = sorted(
    word_with_counts, key=lambda x: (x[1], x[0]), reverse=True)

# Take only max-vocab
if args.max_vocab_size is not None:
  word_with_counts = word_with_counts[:args.max_vocab_size]


with open('{}_{}_vocab.txt'.format(args.mode, args.lang), 'w') as vf:
  for word, count in word_with_counts:
    vf.write("{}\t{}\n".format(word, count))
