import os
import re
import sys
import math
import spacy
import codecs
import tarfile
import logging
import requests
import collections
import progressbar

import torch as th
import numpy as np
from sklearn.cluster.k_means_ import k_means

import bhabana.utils as utils
import bhabana.utils.generic_utils as gu

from bhabana.utils import wget
from bhabana.utils import constants
from torch.autograd import Variable

logger = logging.getLogger(__name__)
spacy_nlp_collection = {}


def url_exists(url):
    try:
        request = requests.get(url, timeout=20)
    except Exception as e:
        raise Exception(str(e))
    if request.status_code == 200:
        logger.info('URL: {} exists'.format(url))
        return True
    else:
        logger.warning('URL: {} does not exists or is not '
                       'responding'.format(url))
        return False


def user_wants_to_download(name, type='model', force=False):
    if force:
        return True
    sys.stdout.write("Could not find {} {}. Do you want to download it "
                     "([Y]/n)?".format(type, name))
    sys.stdout.flush()
    user_response = sys.stdin.readline()

    if user_response is None:
        user_response = True
    elif user_response is '':
        user_response = True
    elif user_response is 'n' or user_response is 'N':
        user_response = False
    else:
        user_response = True
    return user_response


def download_from_url(url, output_dir):
    if not url_exists(url):
        raise FileNotFoundError('{} was not found in our data '
                                'repository'.format(url))
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    filename = wget.download(url, out=output_dir)
    return filename


def extract_tar_gz(file_path, output_dir="."):
    logger.info('Untaring {}'.format(file_path))
    if not tarfile.is_tarfile(file_path):
        raise ValueError("{} is not a valid tar file".format(file_path))
    tar = tarfile.open(file_path, "r:gz")
    tar.extractall(path=output_dir)
    tar.close()


def delete_file(file_path):
    os.remove(file_path)


def download_and_extract_tar(file_url, output_dir):
    tar_file_path = download_from_url(file_url, output_dir)
    #extract_tar_gz(tar_file_path, output_dir)
    delete_file(tar_file_path)


def maybe_download(name, type='model', force=False):
    if type == 'dataset':
        subdir = 'datasets'
        output_dir = utils.DATASET_DIR
    elif type == 'model':
        subdir = 'models'
        output_dir = utils.MODELS_DIR
    else:
        raise ValueError('downloadable data of type {} is not '
                         'supported.'.format(type))

    output_path = os.path.join(output_dir, name)
    tar_file_path = output_path + '.tar.gz'
    file_url = utils.BASE_URL + subdir + '/' + name + '.tar.gz'
    if not os.path.exists(output_path):
        if not os.path.exists(tar_file_path):
            if user_wants_to_download(name, type, force):
                logger.info('Trying to download files from {}'.format(file_url))
                try:
                    download_and_extract_tar(file_url, output_dir)
                except:
                    raise FileNotFoundError('Could not find {} {}. Please download '
                                        'the files to successfully run the '
                                        'script'.format(type, name))
            else:
                return None
        else:
            try:
                extract_tar_gz(tar_file_path, output_dir)
                delete_file(tar_file_path)
            except:
                download_and_extract_tar(file_url, output_dir)
    else:
        logger.info('{} {} already exists'.format(name, type))
    return output_path


def get_spacy(lang='en', model=None):
    """
    Returns the spaCy pipeline for the specified language.

    Keyword arguments:
    lang -- the language whose pipeline will be returned.
    """
    if model is not None:
        if lang not in model:
            raise ValueError("There is no correspondence between the Languge "
                     "({})and the Model ({}) provided.".format(lang, model))
    global spacy_nlp_collection
    spacy_model_name = model if model is not None else lang
    model_key = "{}_{}".format(lang, spacy_model_name)
    if model_key not in spacy_nlp_collection:
        spacy_nlp_collection[model_key] = spacy.load(spacy_model_name)

    return spacy_nlp_collection[model_key]


def pad_sentences(data_batch, pad=0, raw=False):
    """
    Given a sentence, returns the sentence padded with the 'PAD' string. If
    `pad` is smaller than the size of the sentence, the sentence is trimmed
    to `pad` elements. If `pad` is 0, the function just returns the original
    `data`. If raw is False, then the sentence is padded with 0 instead of
    the 'PAD' string.

    Keyword arguments:
    pad -- The number of elements to which the sentence should be padded.
    raw -- If True, the padding character will be a string 'PAD'; else, 0.
    """
    padded_batch = []
    for data in data_batch:
        if pad == 0:
            return data
        if pad <= len(data):
            return data[:pad]

        pad_vec = [0 if not raw else 'PAD' for _ in range(len(data[-1]))]
        for i in range(pad - len(data)):
            padded_batch.append(pad_vec)

    return padded_batch


def pad_int_sequences(sequences, maxlen=None, dtype='int32',
                      padding='post',
                      truncating='post', value=0.):
    """ pad_sequences.

    Pad each sequence to the same length: the length of the longest sequence.
    If maxlen is provided, any sequence longer than maxlen is truncated to
    maxlen. Truncation happens off either the beginning or the end (default)
    of the sequence. Supports pre-padding and post-padding (default).

    Arguments:
        sequences: list of lists where each element is a sequence.
        maxlen: int, maximum length.
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than
            maxlen either in the beginning or in the end of the sequence
        value: float, value to pad the sequences to the desired value.

    Returns:
        x: `numpy array` with dimensions (number_of_sequences, maxlen)

    Credits: From Keras `pad_sequences` function.
    """
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    x = (np.ones((nb_samples, maxlen)) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % padding)

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return x


def pad_vector_sequences(data, maxlen, value):
    last_dim_max = 0
    for d in data:
        for f in d:
            if len(f) > last_dim_max: last_dim_max = len(f)
    last_dim_padded_batch = []
    for d in data:
        padded_features = []
        for f in d:
            f = np.array(f)
            diff = last_dim_max - f.shape[0]
            padded_features.append(np.pad(f, (0, diff),
                                       'constant', constant_values=value).tolist())
        last_dim_padded_batch.append(padded_features)
    last_dim_padded_batch = np.array(last_dim_padded_batch)

    padded_batch = []
    for d in last_dim_padded_batch:
        d = np.array(d)
        if d.shape[0] > maxlen:
            padded_batch.append(d[:maxlen, :].tolist())
        else:
            diff = maxlen - d.shape[0]
            padded_batch.append(np.pad(d, [(0, diff), (0, 0)],
                               'constant', constant_values=value).tolist())
    return padded_batch


def get_batch_depth(batch):
    n_dims = 1
    for d in batch:
        if type(d) == list:
            for d_i in d:
                if type(d_i) == list:
                    for d_i_j in d_i:
                        if type(d_i_j) == list:
                            raise Exception("Currently padding for 3 "
                                            "dimensions is supported")
                        else:
                            n_dims = 3
                else:
                    n_dims = 2
                break
        else:
            n_dims = 1
        break
    return n_dims


def pad_sequences(data, padlen=0, padvalue=0, raw=False):
    padded_data = []
    if padlen == 0:
        return []
    elif raw:
        for d in data:
            diff = padlen - len(d)
            if diff > 0:
                pads = ['PAD'] * diff
                d = d + pads
            padded_data.append(d[:padlen])
    else:
        #vec_data = np.array(data)
        len_n_dims = get_batch_depth(data)
        #len_n_dims = len(n_dims)
        if len_n_dims == 2:
            padded_data = pad_int_sequences(data, maxlen=padlen, dtype="int32",
                        padding='post', truncating='post', value=padvalue).tolist()
        elif len_n_dims == 3:
            padded_data = pad_vector_sequences(data, maxlen=padlen,
                                               value=padvalue)
        else:
            raise NotImplementedError("Padding for more than 3 dimensional vectors has "
                            "not been implemented")
    return padded_data


def pad_1dconv_input(input, kernel_size, mode="same"):
    """
    This method pads the input for "same" and "full" 
    convolutions. Currently just Same  and full padding modes have been 
    implemented
    :param input: Input Tensor with shape BATCH_SIZE X TIME_STEPS X FEATURES  
    :param mode: 
    :return: Padded Input Tensor with shape BATCH_SIZE X TIME_STEPS X FEATURES
    """
    input_size = list(input.size())
    if len(input_size) != 3:
        raise ValueError("The Shape of the input is invalid."
         " The Shape of the current input is {}, but ideally a 3D "
             "vector is expected with shape in the following format:"
                 " BATCH_SIZE X TIME_STEPS X FEATURES".format(input_size))

    n_time_steps = input_size[1]
    if mode == "same":
        n_padding = n_time_steps - (n_time_steps - kernel_size + 1)
    elif mode == "full":
        n_padding = 2 * (kernel_size -1)
    else:
        raise NotImplementedError("Other modes for padding have not been "
                                  "implemented. Valid and Full are coming "
                                  "soon")
    if n_padding == 0:
        padded_input = input
    elif (n_padding % 2) == 0:
        pad_len = int(n_padding / 2)
        pad_tensor = Variable(th.zeros(input_size[0], pad_len, input_size[-1]))
        padded_input = th.cat([pad_tensor, input, pad_tensor], dim=1)
    else:
        pad_len = n_padding / 2
        l_pad = int(math.ceil(pad_len))
        r_pad = int(math.floor(pad_len))
        l_pad_tensor = Variable(th.zeros(input_size[0], l_pad,
                                         input_size[-1]))
        r_pad_tensor = Variable(th.zeros(input_size[0], r_pad,
                                         input_size[-1]))
        padded_input = th.cat([l_pad_tensor, input, r_pad_tensor], dim=1)
    return padded_input


def id2seq(data, i2w):
    """
    `data` is a list of sequences. Each sequence is a list of numbers. For
    example, the following could be an example of data:

    [[1, 10, 4, 1, 6],
     [1, 2,  5, 1, 3],
     [1, 8,  4, 1, 2]]

    Each number represents the ID of a word in the vocabulary `i2w`. This
    function transforms each list of numbers into the corresponding list of
    words. For example, the list above could be transformed into:

    [['the', 'dog',  'chased', 'the', 'cat' ],
     ['the', 'boy',  'kicked', 'the', 'girl'],
     ['the', 'girl', 'chased', 'the', 'boy' ]]

    For a function that transforms the abovementioned list of words back into
    IDs, see `seq2id`.
    """
    buff = []
    for seq in data:
        w_seq = []
        for term in seq:
            if term in i2w:
                if term == 0 or term == 1 or term == 2:
                    continue
                w_seq.append(i2w[term])
        sent = ' '.join(w_seq)
        buff.append(sent)
    return buff


def seq2id(data, w2i, seq_begin=False, seq_end=False):
    """
    `data` is a list of sequences. Each sequence is a list of words. For
    example, the following could be an example of data:

    [['the', 'dog',  'chased', 'the', 'cat' ],
     ['the', 'boy',  'kicked', 'the', 'girl'],
     ['the', 'girl', 'chased', 'the', 'boy' ]]

    Each number represents the ID of a word in the vocabulary `i2w`. This
    function transforms each list of numbers into the corresponding list of
    words. For example, the list above could be transformed into:

    [[1, 10, 4, 1, 6],
     [1, 2,  5, 1, 3],
     [1, 8,  4, 1, 2]]

    For a function that transforms the abovementioned list of IDs back into
    words, see `id2seq`.

    Keyword arguments:
    seq_begin -- If True, insert the ID corresponding to 'SEQ_BEGIN' in the
                 beginning of each sequence
    seq_end   -- If True, insert the ID corresponding to 'SEQ_END' in the end
                 of each sequence
    """
    buff = []
    for seq in data:
        id_seq = []

        if seq_begin:
            id_seq.append(w2i[constants.BOS_WORD])

        for term in seq:
            try:
                id_seq.append(w2i[term] if term in w2i else w2i[constants.UNK_WORD])
            except Exception as e:
                print(str(e))

        if seq_end:
            id_seq.append(w2i[constants.EOS_WORD])

        buff.append(id_seq)
    return buff


def semhashseq2id(data, w2i):

    buff = []
    for seq in data:
        id_seq = []

        for term_hash_seq in seq:
            k_hot = []
            for hash in term_hash_seq:
                k_hot.append(gu.to_categorical(w2i[hash] if hash in w2i \
                             else w2i[constants.UNK_WORD], len(w2i)))
            k_hot = np.sum(k_hot, axis=0).tolist()
            id_seq.append(k_hot)
        buff.append(id_seq)
    return buff


def sentence2id(data, w2i):

    buff = []
    for sentences in data:
        id_seq = []

        for sentence in sentences:
            id_sentence = seq2id(sentence, w2i)
            id_seq.append(id_sentence)
        buff.append(id_seq)
    return buff


def onehot2seq(data, i2w):
    buff = []
    for seq in data:
        w_seq = []
        for term in seq:
            arg = np.argmax(term)
            if arg in i2w:
                if arg == 0 or arg == 1 or arg == 2:
                    continue
                w_seq.append(i2w[arg])
        sent = ' '.join(w_seq)
        buff.append(sent)
    return buff


def append_seq_markers(data, seq_begin=True, seq_end=True):
    """
    `data` is a list of sequences. Each sequence is a list of numbers. For
    example, the following could be an example of data:

    [[1, 10, 4, 1, 6],
     [1, 2,  5, 1, 3],
     [1, 8,  4, 1, 2]]

    Assume that 0 and 11 are IDs corresponding to 'SEQ_BEGIN' and 'SEQ_END',
    respectively. This function adds 'SEQ_BEGIN' and 'SEQ_END' to all lists,
    depending on the values of `seq_begin` and `seq_end`. For example, if
    both are true, then, for the input above, this function will return:

    [[0, 1, 10, 4, 1, 6, 11],
     [0, 1, 2,  5, 1, 3, 11],
     [0, 1, 8,  4, 1, 2, 11]]

    Keyword arguments:
    seq_begin -- If True, add the ID corresponding to 'SEQ_BEGIN' to each sequence
    seq_end   -- If True, add the ID corresponding to 'SEQ_END' to each sequence
    """
    data_ = []
    for d in data:
        if seq_begin:
            d = ['SEQ_BEGIN'] + d
        if seq_end:
            d = d + ['SEQ_END']
        data_.append(d)
    return data_


def mark_entities(data, lang='en'):
    """
    `data` is a list of text lines. Each text line is a string composed of one
    or more words. For example:

    [['the dog chased the cat' ],
     ['the boy kicked the girl'],
     ['John kissed Mary']]

    The function uses the spaCy pipeline in each line of text, finds Named
    Entities, and tags them with their type. For example, for the example above,
    the output will be:

    [['the dog chased the cat' ],
     ['the boy kicked the girl'],
     ['BOE John PERSON EOE kissed BOE Mary PERSON EOE']]

    where:
    BOE indicates the beginning of an Entity
    PERSON indicates the type of the Entity
    EOE indicates the beginning of an Entity

    Keyword arguments:
    lang -- The language in which the sentences are (used to choose which spaCy
            pipeline to call).
    """
    marked_data = []
    spacy_nlp = get_spacy(lang=lang)
    for line in data:
        marked_line = []
        for token in line:
            tok = spacy_nlp(token)[0]
            if tok.ent_type_ != '':
                marked_line.append('BOE')
                marked_line.append(token)
                marked_line.append(tok.ent_type_)
                marked_line.append('EOE')
            else:
                marked_line.append(token)
        marked_data.append(marked_line)
    return marked_data


def sentence_tokenize(line, lang='en'):
    """
    `line` is a string containing potentially multiple sentences. For each
    sentence, this function produces a list of tokens. The output of this
    function is a list containing the lists of tokens produced. For example,
    say line is:

    'I ate chocolate. She ate cake.'

    This function produces:
    [['I',   'ate', 'chocolate'],
     ['She', 'ate', 'cake']]
    """
    sentences = []
    doc = get_spacy(lang=lang)(line)
    for sent in doc.sents:
        sentence_tokens = []
        for token in sent:
            if token.ent_type_ == '':
                sentence_tokens.append(token.text.lower())
            else:
                sentence_tokens.append(token.text)
        sentences.append(sentence_tokens)
    return sentences


def default_tokenize(sentence):
    """
    Returns a list of strings containing each token in `sentence`
    """
    return [i for i in re.split(r"([-.\"',:? !\$#@~()*&\^%;\[\]/\\\+<>\n=])",
                                sentence) if i != '' and i != ' ' and i != '\n']


def tokenize(line, tokenizer='spacy', lang='en', spacy_model=None):
    """
    Returns a list of strings containing each token in `line`.

    Keyword arguments:
    tokenizer -- Possible values are 'spacy', 'split' and 'other'.
    lang      -- Possible values are 'en' and 'de'
    """
    tokens = []
    if tokenizer == 'spacy':
        doc = get_spacy(lang=lang, model=spacy_model).tokenizer(line)
        for token in doc:
            if token.ent_type_ == '':
                if lang == 'de':
                    text = token.text
                else:
                    text = token.text.lower()
                tokens.append(text)
            else:
                tokens.append(token.text)
    elif tokenizer == 'split':
        tokens = line.split(' ')
    else:
        tokens = default_tokenize(line)
    return tokens


def pos_tokenize(line, lang='en'):
    tokens = []
    doc = get_spacy(lang=lang)(line)
    for token in doc:
        tokens.append(token.tag_)
    return tokens


def dep_tokenize(line, lang='en'):
    tokens = []
    doc = get_spacy(lang=lang)(line)
    for token in doc:
        tokens.append(token.dep_)
    return tokens


def ent_tokenize(line, lang='en'):
    tokens = []
    doc = get_spacy(lang=lang)(line)
    for token in doc:
        tokens.append(token.ent_type_ if token.ent_type_ != "" else
                      constants.PAD_WORD)
    return tokens


def semhash_tokenize(text, tokenizer="spacy", lang="en"):
    tokens = tokenize(text, tokenizer=tokenizer, lang=lang)
    hashed_tokens = ["#{}#".format(token) for token in tokens]
    sem_hash_tokens = [["".join(gram)
                        for gram in find_ngrams(list(hash_token), 3)]
                           for hash_token in hashed_tokens]
    return sem_hash_tokens


def char_tokenize(text):
    chars = list(text)
    for i_c, char in enumerate(chars):
        if char == " ":
            chars[i_c] = constants.SPACE
    return chars


def find_ngrams(input_list, n):
  return zip(*[input_list[i:] for i in range(n)])


def vocabulary_builder(data_paths, min_frequency=5, tokenizer='spacy',
                       downcase=True, max_vocab_size=None, line_processor=None,
                       lang='en'):
    print('Building a new vocabulary')
    cnt = collections.Counter()
    for data_path in data_paths:
        bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength,
                                      redirect_stdout=True)
        n_line = 0
        for line in codecs.open(data_path, 'r', 'utf-8'):
            line = line_processor(line)
            if downcase:
                line = line.lower()
            tokens = tokenize(line, tokenizer, lang)
            tokens = [_ for _ in tokens if len(_) > 0]
            cnt.update(tokens)
            n_line += 1
            bar.update(n_line)
        bar.finish()

    print("Found %d unique tokens in the vocabulary.", len(cnt))

    # Filter tokens below the frequency threshold
    if min_frequency > 0:
        filtered_tokens = [(w, c) for w, c in cnt.most_common()
                           if c > min_frequency]
        cnt = collections.Counter(dict(filtered_tokens))

    print("Found %d unique tokens with frequency > %d.",
          len(cnt), min_frequency)

    # Sort tokens by 1. frequency 2. lexically to break ties
    vocab = cnt.most_common()
    vocab = sorted(
        vocab, key=lambda x: (x[1], x[0]), reverse=True)

    # Take only max-vocab
    if max_vocab_size is not None:
        vocab = vocab[:max_vocab_size]

    return vocab


def new_vocabulary(files, dataset_path, min_frequency, tokenizer,
                   downcase, max_vocab_size, name,
                   line_processor=lambda line: " ".join(line.split('\t')[:2]),
                   lang='en'):
    vocab_path = os.path.join(dataset_path,
                              '{}_{}_{}_{}_{}_vocab.txt'.format(
                                  name.replace(' ', '_'), min_frequency,
                                  tokenizer, downcase, max_vocab_size))
    metadata_path = os.path.join(dataset_path,
                                 '{}_{}_{}_{}_{}_metadata.txt'.format(
                                     name.replace(' ', '_'), min_frequency,
                                     tokenizer, downcase, max_vocab_size))
    w2v_path = os.path.join(dataset_path,
                            '{}_{}_{}_{}_{}_w2v.npy'.format(
                                name.replace(' ', '_'),
                                min_frequency, tokenizer, downcase,
                                max_vocab_size))

    if os.path.exists(vocab_path) and os.path.exists(w2v_path) and \
            os.path.exists(metadata_path):
        print("Files exist already")
        return vocab_path, w2v_path, metadata_path

    word_with_counts = vocabulary_builder(files,
                                          min_frequency=min_frequency,
                                          tokenizer=tokenizer,
                                          downcase=downcase,
                                          max_vocab_size=max_vocab_size,
                                          line_processor=line_processor,
                                          lang=lang)

    entities = ['PERSON', 'NORP', 'FACILITY', 'ORG', 'GPE', 'LOC' +
                'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LANGUAGE',
                'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY',
                'ORDINAL', 'CARDINAL', 'BOE', 'EOE']

    with codecs.open(vocab_path, 'w', 'utf-8') as vf, codecs.open(metadata_path, 'w', 'utf-8') as mf:
        mf.write('word\tfreq\n')
        mf.write('PAD\t1\n')
        mf.write('SEQ_BEGIN\t1\n')
        mf.write('SEQ_END\t1\n')
        mf.write('UNK\t1\n')

        vf.write('PAD\t1\n')
        vf.write('SEQ_BEGIN\t1\n')
        vf.write('SEQ_END\t1\n')
        vf.write('UNK\t1\n')

        for ent in entities:
            vf.write("{}\t{}\n".format(ent, 1))
            mf.write("{}\t{}\n".format(ent, 1))
        for word, count in word_with_counts:
            vf.write("{}\t{}\n".format(word, count))
            mf.write("{}\t{}\n".format(word, count))

    return vocab_path, w2v_path, metadata_path


def write_spacy_vocab(path, lang="en", model_name=None):
    if not os.path.exists(path):
        spacy_nlp = get_spacy(lang=lang, model=model_name)
        vocab_size = 0
        with codecs.open(path, 'w', 'utf-8') as f:
            for tok in spacy_nlp.vocab:
                vocab_size += 1
                f.write("{}\n".format(tok.text))


def load_classes(classes_path):
    """
    Loads the classes from file `classes_path`.
    """
    c2i = {}
    i2c = {}
    c_id = 0
    with codecs.open(classes_path, 'r', 'utf-8') as cf:
        for line in cf:
            label = line.strip()
            c2i[label] = c_id
            i2c[c_id] = label
            c_id += 1
    return c2i, i2c


def load_vocabulary(vocab_path):
    """
    Loads the vocabulary from file `vocab_path`.
    """
    w2i = {constants.PAD_WORD: constants.PAD, constants.UNK_WORD: constants.UNK,
           constants.BOS_WORD: constants.BOS, constants.EOS_WORD: constants.EOS,
           constants.SPACE_WORD: constants.SPACE}
    i2w = {constants.PAD: constants.PAD_WORD, constants.UNK: constants.UNK_WORD,
           constants.BOS: constants.BOS_WORD, constants.EOS: constants.EOS_WORD,
           constants.SPACE: constants.SPACE_WORD}
    with codecs.open(vocab_path, 'r', 'utf-8') as vf:
        wid = 5
        dup_id = 0
        for line in vf:
            term = line.strip().split('\t')[0]
            if term not in w2i:
                w2i[term] = wid
                i2w[wid] = term
                wid += 1
            else:
                w2i["{}{}".format(term, dup_id)] = wid
                i2w[wid] = "{}{}".format(term, dup_id)
                wid += 1
                dup_id += 1

    return w2i, i2w


def preload_w2v(w2i, lang='en', model=None):
    '''
    Loads the vocabulary based on spaCy's vectors.

    Keyword arguments:
    initialize -- Either 'random' or 'zeros'. Indicate the value of the new
                    vectors to be created (if a word is not found in spaCy's
                    vocabulary
    lang       -- Either 'en' or 'de'.
    '''
    logger.info('Preloading a w2v matrix')
    spacy_nlp = get_spacy(lang, model)
    vec_size = get_spacy_vector_size(lang, model)
    w2v = np.zeros((len(w2i), vec_size))
    bar = progressbar.ProgressBar(max_value=len(w2i),
                                  redirect_stdout=True)
    for i_t, term in enumerate(w2i):
        if spacy_nlp(term).has_vector:
            w2v[w2i[term]] = spacy_nlp(term).vector
            bar.update(i_t)
    bar.finish()
    return w2v


def get_spacy_vector_size(lang="en", model=None):
    spacy_nlp = get_spacy(lang, model)
    for lex in spacy_nlp.vocab:
        tok = spacy_nlp(lex.text)
        if tok.has_vector:
            return tok.vector.shape[0]


def get_spacy_pos_tags(lang="en"):
    get_spacy(lang)
    mod = sys.modules["spacy.lang.{}.tag_map".format(lang)]
    tag_list = []
    for k in mod.TAG_MAP:
        tag_list.append(k)
    #del mod
    return list(set(tag_list))


def get_spacy_dep_tags(lang="en"):
    if lang == "en":
        return constants.EN_DEP_TAGS
    elif lang == "en":
        return constants.DE_DEP_TAGS
    else:
        return constants.UNIVERSAL_DEP_TAGS


def get_spacy_ner_tags(lang="en"):
    if lang == "en":
        return constants.ONE_NOTE_NER_TAGS
    else:
        return constants.WIKI_NER_TAGS


def write_spacy_aux_vocab(path, lang, type="pos"):
    if not os.path.exists(path):
        if type == "pos":
            vocab = get_spacy_pos_tags(lang)
        elif type == "ent":
            vocab = get_spacy_ner_tags(lang)
        elif type == "dep":
            vocab = get_spacy_dep_tags(lang)
        else:
            raise Exception("Type {} is not supported or is an invalid type of "
                            "vocab.".format(type))
        with codecs.open(path, 'w', 'utf-8') as f:
            for tok in vocab:
                f.write("{}\n".format(tok))


def load_w2v(path):
    return np.load(path)


def save_w2v(path, w2v):
    return np.save(path, w2v)


def validate_rescale(range):
    if range[0] > range[1]:
        raise ValueError('Incompatible rescale values. rescale[0] should '
                         'be less than rescale[1]. An example of a valid '
                         'rescale is (4, 8).')


def rescale(values, new_range, original_range):
    """
    `values` is a list of numbers. Rescale the numbers in `values` so that
    they are always between `new_range` and `original_range`.
    """
    if new_range is None:
        return values

    if new_range == original_range:
        return values

    rescaled_values = []
    for value in values:
        original_range_size = (original_range[1] - original_range[0])
        if (original_range_size == 0):
            new_value = new_range[0]
        else:
            new_range_size = (new_range[1] - new_range[0])
            new_value = (((value - original_range[
                0]) * new_range_size) / original_range_size) + \
                        new_range[0]
        rescaled_values.append(new_value)
    return rescaled_values


def is_supported_data(name):
    if name in utils.DATA_REGISTER:
        return True
    else:
        return False

if __name__ == '__main__':
    data_1 = [[[1, 2, 3], [2, 2, 3], [2, 3], [2, 3, 4, 5]],
              [[1, 2, 3], [2, 2, 3], [2, 3]]]
    data_2 = [[1, 2, 3, 4, 5 ,6], [1, 2, 3, 4]]
    padded_data2 = pad_sequences(data_2, 2)
    padded_data1 = pad_sequences(data_1, 2)

    print(np.array(padded_data1))
    print(padded_data2)

