"""
intro2nlp, assignment 4, 2020
In this assignment you will implement a Hidden Markov model and an LSTM model
to predict the part of speech sequence for a given sentence.
(Adapted from Nathan Schneider)
"""
import re
import torch
import torch.nn as nn
from nltk.tokenize import word_tokenize
import torchtext
from torchtext import data, vocab
import torch.optim as optim
from math import log, isfinite
from collections import Counter
import numpy as np
import pandas as pd
import sys, os, time, platform, nltk, random
import string
from nltk.corpus import stopwords
from RNN import RNN

# With this line you don't need to worry about the HW  -- GPU or CPU
# GPU cuda cores will be used if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# You can call use_seed with other seeds or None (for complete randomization)
# but DO NOT change the default value.
def use_seed(seed=1512021):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_deterministic(True)
    # torch.backends.cudnn.deterministic = True


# utility functions to read the corpus
def who_am_i():  # this is not a class method
    """Returns a ductionary with your name, id number and email. keys=['name', 'id','email']
        Make sure you return your own info!
    """
    # TODO Ori add your details plz
    # if work is submitted by a pair of students, add the following keys: name2, id2, email2
    return {'name1': 'John Doe', 'id1': '012345678', 'email1': 'jdoe@post.bgu.ac.il',
            'name2': 'Jonathan Martinez', 'id2': '201095569', 'email2': 'martijon@post.bgu.ac.il'}


def read_annotated_sentence(f):
    line = f.readline()
    if not line:
        return None
    sentence = []
    while line and (line != "\n"):
        line = line.strip()
        word, tag = line.split("\t", 2)
        sentence.append((word, tag))
        line = f.readline()
    return sentence


def load_annotated_corpus(filename):
    sentences = []
    with open(filename, 'r', encoding='utf-8') as f:
        sentence = read_annotated_sentence(f)
        while sentence:
            sentences.append(sentence)
            sentence = read_annotated_sentence(f)
    return sentences


START = "<DUMMY_START_TAG>"
END = "<DUMMY_END_TAG>"
UNK = "<UNKNOWN>"
stop_words = set(stopwords.words('english'))

suffix_list = ['ation', 'ible', 'ious', 'ment', 'ness', 'sion', 'ship', 'able', 'less', 'ward', 'wise', 'eer', 'cian',
               'able', 'ion', 'ies', 'ity', 'off', 'ous', 'ive', 'ant', 'ary', 'ful', 'ing', 'ize', 'ise', 'est', 'ess',
               'ate', 'al', 'er', 'or', 'ic', 'ed', 'es', 'th', 'en', 'ly', 'y', 's']
suffixes = {}
all_capital = Counter()
all_words_count = Counter()
start_with_capital = Counter()
singelton_tt = {}
singelton_tw = {}
allTagCounts = Counter()
# use Counters inside these
perWordTagCounts = {}
transitionCounts = {}
emissionCounts = {}
# log probability distributions: do NOT use Counters inside these because
# missing Counter entries default to 0, not log(0)
A = {}  # transisions probabilities
B = {}  # emmissions probabilities

WORD_IDX = 0
TAG_IDX = 1


def update_dic_counter(dic, key, value):
    key = key[0]
    if key not in dic:
        dic[key] = Counter()
    dic[key].update(value)


def get_value_from_transition(transition_count, previous_tag, current_tag):
    """
        calc the transition log prob, use One-Count Smoothing to smooth to prob for open groups of
        tags and closed groups of tags.
    :return: transition log prob
    """
    ptt_backof = (allTagCounts[current_tag] / sum(all_words_count.values()))
    singelton_count = 1 + singelton_tt[previous_tag]
    transition_log_prob = log(
        ((transition_count + ptt_backof * singelton_count) / (allTagCounts[previous_tag] + singelton_count)))
    return transition_log_prob


def get_word_suffix(word):
    """
    :param word:
    :return:  return the suffix for the given word
    """
    for suffix in suffix_list:
        if word.endswith(suffix) and word != suffix:
            return suffix
    return ''


def get_word_features_weight(word, current_tag, prev_tag):
    """
        for oov words- calc word features for suffix and capital letters
    :param word:
    :param current_tag:
    :param prev_tag: used for calc weight for first letter as capital letter, don't add weight if prev_tag is START
    :return: word features sum
    """
    word_suffix = get_word_suffix(word.lower())
    word_suffix_weight = 0
    word_all_capital_weight = 0
    word_start_with_capital_weight = 0
    if word_suffix != '':
        word_suffix_weight = max(suffixes[word_suffix][current_tag], word_suffix_weight)
    if word.isupper():
        word_all_capital_weight = max(all_capital[current_tag], word_all_capital_weight)
    else:
        if word[0].isupper() and prev_tag != START:
            word_start_with_capital_weight = max(start_with_capital[current_tag], word_start_with_capital_weight)
    return word_suffix_weight + word_start_with_capital_weight + word_all_capital_weight


def get_value_from_emmission(emmission_count, current_tag, word, prev_tag=None):
    """
        return the emission logged prob for the given tag and word, if the word are oov then add to consideration
        word features as suffix and capital letters, I used One-Count Smoothing to smooth to prob for open groups of
        tags and closed groups of tags.
    :param emmission_count: number of appearances of word with tag in training set
    :param current_tag: tag to evaluate
    :param word: word to evaluate
    :param prev_tag: used only for oov words, for word feature weight
    :return: the emission logged prob
    """
    word_features_weight = 1
    if prev_tag is not None:
        word_features_weight = max(word_features_weight, get_word_features_weight(word, current_tag, prev_tag))
    ptw_backof = ((all_words_count[word] + 1) / (sum(all_words_count.values()) + len(perWordTagCounts)))
    singelton_count = 1 + singelton_tw[current_tag]
    value = log(((emmission_count + ptw_backof * singelton_count * word_features_weight) / (
            allTagCounts[current_tag] + singelton_count)))
    return value


def learn_params(tagged_sentences):
    """Populates and returns the allTagCounts, perWordTagCounts, transitionCounts,
     and emissionCounts data-structures.
    allTagCounts and perWordTagCounts should be used for baseline tagging and
    should not include pseudocounts, dummy tags and unknowns.
    The transisionCounts and emmisionCounts
    should be computed with pseudo tags and shoud be smoothed.
    A and B should be the log-probability of the normalized counts, based on
    transisionCounts and  emmisionCounts
    Args:
      tagged_sentences: a list of tagged sentences, each tagged sentence is a
       list of pairs (w,t), as retunred by load_annotated_corpus().
   Return:
      [allTagCounts,perWordTagCounts,transitionCounts,emissionCounts,A,B] (a list)
  """
    # init a counter for each suffix
    for suffix in suffix_list:
        suffixes[suffix] = Counter()
    for tagged_sentence in tagged_sentences:
        previous_tag = [START]
        allTagCounts.update(previous_tag)
        for tagged_word in tagged_sentence:
            current_word = [tagged_word[WORD_IDX]]
            current_tag = [tagged_word[TAG_IDX]]

            # count capital letters stats
            # distribution of words of all capital letters:
            if current_word[0].isupper():
                all_capital.update(current_tag)
            # distribution of words that start with capital letters:
            else:
                if current_word[0][0].isupper() and previous_tag[0] != START:
                    start_with_capital.update(current_tag)
            allTagCounts.update(current_tag)
            all_words_count.update(current_word)
            update_dic_counter(perWordTagCounts, current_word, current_tag)
            update_dic_counter(transitionCounts, previous_tag, current_tag)
            update_dic_counter(emissionCounts, current_tag, current_word)
            #  count word suffix for emission probs
            for suffix in suffix_list:
                if current_word[0].endswith(suffix):
                    suffixes[suffix].update(current_tag)
                    break
            previous_tag = current_tag
        current_tag = [END]
        allTagCounts.update(current_tag)
        update_dic_counter(transitionCounts, previous_tag, current_tag)

    # count singeltones
    for tag in allTagCounts.keys():
        if tag != END:
            singelton_tt[tag] = [i for i in transitionCounts[tag].values()].count(1)
        if tag != START and tag != END:
            singelton_tw[tag] = [i for i in emissionCounts[tag].values()].count(1)

    for previous_tag in allTagCounts.keys():
        if previous_tag != END:
            for current_tag in transitionCounts[previous_tag].keys():
                A[(previous_tag, current_tag)] = get_value_from_transition(transitionCounts[previous_tag][current_tag],
                                                                           previous_tag,
                                                                           current_tag)
            if previous_tag != START:
                tag = previous_tag
                for current_word in emissionCounts[tag].keys():
                    B[(tag, current_word)] = get_value_from_emmission(emissionCounts[tag][current_word], tag,
                                                                      current_word)

    return [allTagCounts, perWordTagCounts, transitionCounts, emissionCounts, A, B]


def baseline_tag_sentence(sentence, perWordTagCounts, allTagCounts):
    """Returns a list of pairs (w,t) where each w corresponds to a word
    (same index) in the input sentence. Each word is tagged by the tag most
    frequently associated with it. OOV words are tagged by sampling from the
    distribution of all tags.
    Args:
        sentence (list): a list of tokens (the sentence to tag)
        perWordTagCounts (Counter): tags per word as specified in learn_params()
        allTagCounts (Counter): tag counts, as specified in learn_params()
        Return:
        list: list of pairs
    """
    tagged_sentence = []
    for word in sentence:
        word_tag_counts = perWordTagCounts.get(word, None)
        if word_tag_counts:
            best_tag = word_tag_counts.most_common(1)[0][0]
        else:
            best_tag = random.choices(list(allTagCounts.keys()), weights=list(allTagCounts.values()), k=1)[0]
        tagged_sentence.append((word, best_tag))

    return tagged_sentence


# ===========================================
#       POS tagging with HMM
# ===========================================


def hmm_tag_sentence(sentence, A, B):
    """Returns a list of pairs (w,t) where each w corresponds to a word
    (same index) in the input sentence. Tagging is done with the Viterby
    algorithm.
    Args:
        sentence (list): a list of tokens (the sentence to tag)
        A (dict): The HMM Transition probabilities
        B (dict): tthe HMM emmission probabilities.
    Return:
        list: list of pairs
    """

    tags = viterbi(sentence, A, B)
    tagged_sentence = []
    for index, word in enumerate(sentence):
        tagged_sentence.append([word, tags[index]])
    return tagged_sentence


def get_tags_list(allTagCounts):
    return list(set(allTagCounts.keys()).difference(set([END, START])))


def get_value_from_B(B, current_tag, word, prev_tag):
    """
        wrapper the access to B, if the emission is known from training then return value from B,
        else calc the emission online
    :param B: matrix B
    :param current_tag:
    :param word:
    :param prev_tag:
    :return: log emission prob
    """
    if (current_tag, word) in B:
        return B[(current_tag, word)]
    return get_value_from_emmission(0, current_tag, word, prev_tag)


def get_value_from_A(A, previous_tag, current_tag):
    """
        wrapper the access to A, if the transition is known from training then return value from A,
        else calc the transition online
    :param A: matrix A
    :param previous_tag:
    :param current_tag:
    :return: log transition prob
    """
    if (previous_tag, current_tag) in A:
        return A[(previous_tag, current_tag)]
    else:
        return get_value_from_transition(0, previous_tag, current_tag)


def get_most_probable_path(viterbi, A, B, prev_word, word, current_tag):
    possible_tags_prob = {}
    all_possible_previous_states = viterbi[viterbi[prev_word] != 0]
    for previous_tag, row in all_possible_previous_states.iterrows():
        possible_tags_prob[previous_tag] = row[prev_word][-1] + get_value_from_A(A, previous_tag, current_tag) + \
                                           get_value_from_B(B, current_tag, word, previous_tag)
    value = max(possible_tags_prob.items(), key=lambda k: k[1])
    return value


def viterbi(sentence, A, B):
    """Creates the Viterbi matrix, column by column. Each column is a list of
    tuples representing cells. Each cell ("item") is a tupple (t,r,p), were
    t is the tag being scored at the current position,
    r is a reference to the corresponding best item from the previous position,
    and p is a log probabilityof the sequence so far).
    The function returns the END item, from which it is possible to
    trace back to the beginning of the sentence.
    Args:
        sentence (list): a list of tokens (the sentence to tag)
        A (dict): The HMM Transition probabilities
        B (dict): tthe HMM emmission probabilities.
    Return:
        obj: the last item, tagged with END. should allow backtraking.
        """
    # Hint 1: For efficiency reasons - for words seen in training there is no
    #      need to consider all tags in the tagset, but only tags seen with that
    #      word. For OOV you have to consider all tags.
    # Hint 2: start with a dummy item  with the START tag (what would it log-prob be?).
    #         current list = [ the dummy item ]
    # Hint 3: end the sequence with a dummy: the highest-scoring item with the tag END

    number_of_rows = len(get_tags_list(allTagCounts))
    number_of_columns = len(sentence)
    data = [[0 for i in range(number_of_columns)] for j in range(number_of_rows)]
    viterbi = pd.DataFrame(data, index=[k for k in get_tags_list(allTagCounts)],
                           columns=[word + "_{}".format(index) for index, word in enumerate(sentence)], dtype=object)
    # initialization step
    intialize_states = get_tags_list(allTagCounts)
    first_word = sentence[0]
    if first_word in perWordTagCounts:
        intialize_states = perWordTagCounts[first_word].keys()

    for state in intialize_states:
        backpointer = START
        prob = get_value_from_B(B, state, first_word, START) + get_value_from_A(A, START, state)
        viterbi.loc[state, sentence[0] + "_{}".format(0)] = (state, backpointer, prob)
    # recursion step
    for index, word in enumerate(sentence[1:]):
        possible_states = get_tags_list(allTagCounts)
        if word in perWordTagCounts:
            possible_states = perWordTagCounts[word].keys()

        for state in possible_states:
            backpointer, prob = get_most_probable_path(viterbi, A, B, sentence[index] + "_{}".format(index), word,
                                                       state)
            viterbi.loc[state, word + "_{}".format(index + 1)] = (state, backpointer, prob)
    word = sentence[-1]
    word_index = len(sentence) - 1
    current_tag = END
    possible_tags_prob = {}
    all_possible_previous_states = viterbi[viterbi[word + "_{}".format(word_index)] != 0]
    for previous_tag, row in all_possible_previous_states.iterrows():
        possible_tags_prob[previous_tag] = row[word + "_{}".format(word_index)][-1] + \
                                           get_value_from_A(A, previous_tag, current_tag)
    backpointer, prob = max(possible_tags_prob.items(), key=lambda k: k[1])

    tags = []
    tags.append(END)
    for index, word in enumerate(reversed(viterbi.columns)):
        tags.append(backpointer)
        # tags.append((words_in_reversed_order[index], backpointer))
        word_possible_tags = viterbi[word]
        backpointer = word_possible_tags[backpointer][1]
    tags.reverse()
    return tags

    # return v_last


# a suggestion for a helper function. Not an API requirement
def retrace(end_item):
    """Returns a list of tags (retracing the sequence with the highest probability,
        reversing it and returning the list). The list should correspond to the
        list of words in the sentence (same indices).
    """


# a suggestion for a helper function. Not an API requirement
def predict_next_best(word, tag, predecessor_list):
    """Returns a new item (tupple)
    """


def joint_prob(sentence, A, B):
    """Returns the joint probability of the given sequence of words and tags under
     the HMM model.
     Args:
         sentence (pair): a sequence of pairs (w,t) to compute.
         A (dict): The HMM Transition probabilities
         B (dict): tthe HMM emmission probabilities.
     """
    p = 0  # joint log prob. of words and tags
    previous_tag = START
    for pair in sentence:
        word = pair[0]
        tag = pair[1]
        p += get_value_from_B(B, tag, word, previous_tag) + get_value_from_A(A, previous_tag, tag)
        previous_tag = tag
    tag = END
    p += get_value_from_A(A, previous_tag, tag)
    assert isfinite(p) and p < 0  # Should be negative. Think why!
    return p


def normalize_text(text):
    """
    This function takes as input a text on which several
    NLTK algorithms will be applied in order to preprocess it
    """
    pattern = r'''(?x)          # set flag to allow verbose regexps
           (?:[A-Z]\.)+          # abbreviations, e.g. U.S.A.
           | \w+(?:-\w+)*        # words with optional internal hyphens
           | \$?\d+(?:\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
           | \.\.\.              # ellipsis
           | [][.,;"'?():_`-]    # these are separate tokens; includes ], [
           '''
    text = text.lower().translate(string.punctuation)
    regexp = re.compile(pattern)
    tokens = regexp.findall(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)


def remove_web_links(token):
    token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|' \
                   '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', token)
    return token


def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = []
    for item in tokens:
        if item not in stop_words:
            stems.append(item)
    return stems


def evaluate_hmm(model, test_data):
    epoch_acc = 0

    A = model[4]
    B = model[5]

    for test_samp in test_data:
        text = test_samp.text
        tags = test_samp.tags

        predictions = hmm_tag_sentence(text, A, B)
        golden = [(w, t) for w, t in zip(text, tags)]
        correct, correctOOV, OOV = count_correct(golden, predictions)
        pred_tags = [pair[1] for pair in predictions]
        acc = correct / len(pred_tags)
        epoch_acc += acc

    return epoch_acc / len(test_data)


def test_hmm(model, dataset):
    test_acc = evaluate_hmm(model, dataset)
    print(f'Test Acc: {test_acc * 100:.2f}%')


# ===========================================
#       POS tagging with BiLSTM
# ===========================================

""" You are required to support two types of bi-LSTM:
    1. a vanilla biLSTM in which the input layer is based on simple word embeddings
    2. a case-based BiLSTM in which input vectors combine a 3-dim binary vector
        encoding case information, see
        https://arxiv.org/pdf/1510.06168.pdf
"""


# Suggestions and tips, not part of the required API
#
#  1. You can use PyTorch torch.nn module to define your LSTM, see:
#     https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM
#  2. You can have the BLSTM tagger model(s) implemented in a dedicated class
#     (this could be a subclass of torch.nn.Module)
#  3. Think about padding.
#  4. Consider using dropout layers
#  5. Think about the way you implement the input representation
#  6. Consider using different unit types (LSTM, GRU,LeRU)


def initialize_rnn_model(params_d):
    """Returns a dictionary with the objects and parameters needed to run/train_rnn
       the lstm model. The LSTM is initialized based on the specified parameters.
       thr returned dict is may have other or additional fields.

    Args:
        params_d (dict): a dictionary of parameters specifying the model. The dict
                        should include (at least) the following keys:
                        {'max_vocab_size': max vocabulary size (int),
                        'min_frequency': the occurence threshold to consider (int),
                        'input_rep': 0 for the vanilla and 1 for the case-base (int),
                        'embedding_dimension': embedding vectors size (int),
                        'num_of_layers': number of layers (int),
                        'output_dimension': number of tags in tagset (int),
                        'pretrained_embeddings_fn': str,
                        'data_fn': str
                        }
                        max_vocab_size sets a constraints on the vocab dimention.
                            If the its value is smaller than the number of unique
                            tokens in data_fn, the words to consider are the most
                            frequent words. If max_vocab_size = -1, all words
                            occuring more that min_frequency are considered.
                        min_frequency privides a threshold under which words are
                            not considered at all. (If min_frequency=1 all words
                            up to max_vocab_size are considered;
                            If min_frequency=3, we only consider words that appear
                            at least three times.)
                        input_rep (int): sets the input representation. Values:
                            0 (vanilla), 1 (case-base);
                            <other int>: other models, if you are playful
                        The dictionary can include other keys, if you use them,
                             BUT you shouldn't assume they will be specified by
                             the user, so you should spacify default values.
    Return:
        a dictionary with the at least the following key-value pairs:
                                       {'lstm': torch.nn.Module object,
                                       input_rep: [0|1]}
        #Hint: you may consider adding the embeddings and the vocabulary
        #to the returned dict
    """
    field_text, field_tags = data.Field(lower=True), data.Field(unk_token=None)
    fields = [('text', field_text), ('tags', field_tags)]
    if params_d['input_rep'] == 1:
        field_case = data.Field(use_vocab=False, sequential=True, pad_token=0)
        fields.append(('case', field_case))

    train_examples = get_examples_from_data(load_annotated_corpus(params_d['data_fn']), fields)
    train_data = data.Dataset(train_examples, fields)
    vectors = load_pretrained_embeddings(params_d['pretrained_embeddings_fn'], params_d['max_vocab_size'])

    field_text.build_vocab(train_data, min_freq=params_d['min_frequency'], vectors=vectors,
                           unk_init=torch.Tensor.normal_)
    field_tags.build_vocab(train_data)
    padding_idx = field_text.vocab.stoi[field_text.pad_token]

    global GLOBAL_VOCAB
    GLOBAL_VOCAB = field_text.vocab

    model = RNN(len(field_text.vocab), vectors.dim, params_d['num_of_layers'],
                len(field_tags.vocab), padding_idx=padding_idx, input_rep=params_d['input_rep'])

    return {'lstm': model, 'params_d': params_d, 'vectors': vectors, 'fields': fields}


def load_pretrained_embeddings(path, max_vocab_size=None, vocab=None):
    """ Returns an object with the the pretrained vectors, loaded from the
        file at the specified path. The file format is the same as
        https://www.kaggle.com/danielwillgeorge/glove6b100dtxt
        You can also access the vectors at:
         https://www.dropbox.com/s/qxak38ybjom696y/glove.6B.100d.txt?dl=0
         (for efficiency (time and memory) - load only the vectors you need)
        The format of the vectors object is not specified as it will be used
        internaly in your code, so you can use the datastructure of your choice.

    Args:
        path (str): full path to the embeddings file
        max_vocab_size: max vocabulary size (int)
        vocab (list): a list of words to have embeddings for. Defaults to None.

    """
    vectors = torchtext.vocab.Vectors(name=path, max_vectors=max_vocab_size if max_vocab_size != -1 else None)
    return vectors if vocab is None else torchtext.vocab.Vectors(vectors.get_vecs_by_tokens(vocab))


def train_rnn(model, train_data, val_data=None, epochs=10, batch_size=128):
    """Trains the BiLSTM model on the specified data.

    Args:
        model (dict): the model dict as returned by initialize_rnn_model()
        train_data (list): a list of annotated sentences in the format returned
                            by load_annotated_corpus()
        val_data (list): a list of annotated sentences in the format returned
                            by load_annotated_corpus() to be used for validation.
                            Defaults to None
        epochs: num of epochs to train model
        batch_size: size of train batches
    """
    model, params_d, vectors, fields = model['lstm'], model['params_d'], model['vectors'], model['fields']
    # field_text = data.Field(lower=True)
    # field_tags = data.Field(unk_token=None)
    # fields = [('text', field_text), ('tags', field_tags)]
    # if model.input_rep == 1:
    #     field_case = data.Field(use_vocab=False, sequential=True, pad_token=0)
    #     fields.append(('case', field_case))

    train_examples = get_examples_from_data(train_data, fields)
    train_dataset = data.Dataset(train_examples, fields)
    val_dataset = None
    if val_data:
        val_examples = get_examples_from_data(val_data, fields)
        val_dataset = data.Dataset(val_examples, fields)

    # field_text.build_vocab(train_dataset, min_freq=params_d['min_frequency'], vectors=vectors,
    #                        unk_init=torch.Tensor.normal_)
    # field_tags.build_vocab(train_dataset)

    field_text, field_tags = fields[0][1], fields[1][1]

    padding_idx = field_text.vocab.stoi[field_text.pad_token]
    padding_tag_idx = field_tags.vocab.stoi[field_tags.pad_token]

    model.init_weights(field_text.vocab.vectors, padding_idx)
    criterion = nn.CrossEntropyLoss(ignore_index=padding_tag_idx)
    model = model.to(device)
    criterion = criterion.to(device)

    train_iterator = data.BucketIterator(train_dataset, batch_size=batch_size, device=device)
    valid_iterator = None
    if val_data:
        valid_iterator = data.BucketIterator(val_dataset, batch_size=batch_size, device=device)
    model.fit(train_iterator, criterion, epochs, padding_tag_idx, valid_iterator)
    if val_data:
        saved_path = f'model_input_rep_{model.input_rep}.sav'
        model.load_state_dict(torch.load(saved_path))


def test_rnn(model: RNN, dataset, padding_tag_idx=1):
    """
    Test the model.
    Args:
        model: to test
        dataset: to test on
    """
    iterator = data.BucketIterator(dataset, batch_size=128, device=device)
    criterion = nn.CrossEntropyLoss(ignore_index=padding_tag_idx)
    model = model.to(device)
    criterion = criterion.to(device)
    loss, acc = model.evaluate(iterator, criterion, padding_tag_idx)
    print(f'loss:{loss:.4f} acc:{acc:.4f}')


def rnn_tag_sentence(sentence, model):
    """ Returns a list of pairs (w,t) where each w corresponds to a word
        (same index) in the input sentence and t is the predicted tag.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        model (dict):  a dictionary with the trained BiLSTM model and all that is needed
                        to tag a sentence.

    Return:
        list: list of pairs
    """
    model, fields = model['lstm'], model['fields']
    field_text, field_tags = fields[0][1], fields[1][1]
    x_embeddings = [field_text.vocab.stoi[token.lower()] for token in sentence]
    x_embeddings = torch.LongTensor(x_embeddings).unsqueeze(-1).to(device)
    if model.input_rep == 0:
        x = x_embeddings
    else:
        x_case = [get_case_type(token) for token in sentence]
        x_case = torch.LongTensor(x_case).unsqueeze(-1).to(device)
        x = [x_embeddings, x_case]

    model.eval()
    y_pred = model(x)
    tags = [field_tags.vocab.itos[t.item()] for t in y_pred.argmax(-1)]

    tagged_sentence = [(w, t) for w, t in zip(sentence, tags)]
    return tagged_sentence


def get_best_performing_model_params():
    """Returns a disctionary specifying the parameters of your best performing
        BiLSTM model.
        IMPORTANT: this is a *hard coded* dictionary that will be used to create
        a model and train a model by calling
               initialize_rnn_model() and train_lstm()
    """
    return {'max_vocab_size': -1,
            'min_frequency': 2,
            'input_rep': 1,
            'num_of_layers': 2}


# ===========================================================
#       Wrapper function (tagging with a specified model)
# ===========================================================

def tag_sentence(sentence, model):
    """Returns a list of pairs (w,t) where pair corresponds to a word (same index) in
    the input sentence. Tagging is done with the specified model.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        model (dict): a dictionary where key is the model name and the value is
           an ordered list of the parameters of the trained model (baseline, HMM)
           or the model isteld and the input_rep flag (LSTMs).

        Models that must be supported (you can add more):
        1. baseline: {'baseline': [perWordTagCounts, allTagCounts]}
        2. HMM: {'hmm': [A,B]}
        3. Vanilla BiLSTM: {'blstm':[model_dict]}
        4. BiLSTM+case: {'cblstm': [model_dict]}
        5. (NOT REQUIRED: you can add other variations, agumenting the input
            with further subword information, with character-level word embedding etc.)

        The parameters for the baseline model are:
        perWordTagCounts (Counter): tags per word as specified in learn_params()
        allTagCounts (Counter): tag counts, as specified in learn_params()

        The parameters for the HMM are:
        A (dict): The HMM Transition probabilities
        B (dict): tthe HMM emmission probabilities.

        Parameters for an LSTM: the model dictionary (allows tagging the given sentence)


    Return:
        list: list of pairs
    """
    model_name, values = list(model.items())[0]
    if model_name == 'baseline':
        return baseline_tag_sentence(sentence, values[0], values[1])
    if model_name == 'hmm':
        return hmm_tag_sentence(sentence, values[0], values[1])
    if model_name == 'blstm':
        return rnn_tag_sentence(sentence, values[0])
    if model_name == 'cblstm':
        return rnn_tag_sentence(sentence, values[0])


def get_examples_from_data(texts, fields):
    """
    Build a list of Examples to make the dataset with
    Args:
        data_fn: path of dataset (tsv file)
        fields (List[Tuple[str, Field]]): fields to extract from data

    Returns:
        list of torchtext.data.Example objects
    """
    examples = []
    case_based = len(fields) == 3
    for text in texts:
        values = [[] for _ in fields]
        for token, tag in text:
            values[0].append(token)
            values[1].append(tag)
            if case_based:
                values[2].append(get_case_type(token))
        example = data.Example.fromlist(values, fields)
        examples.append(example)
    return examples


def get_case_type(token):
    """
    Get the case type of a token
    Args:
        token: to analyze

    Returns:
        1 if is all uppercase, 2 if first char is uppercase, 3 if is all lowercase, 0 otherwise
    """
    if token.replace('.', '').isalpha():  # ignore period, for token like 'U.S.'
        if token.isupper():
            return 1
        if token[0].isupper():
            return 2
        if token.islower():
            return 3
    return 0


# GLOBAL FUNCTION #

def count_correct(gold_sentence, pred_sentence):
    """Return the total number of correctly predicted tags,the total number of
    correcttly predicted tags for oov words and the number of oov words in the
    given sentence.
    Args:
        gold_sentence (list): list of pairs, assume to be gold labels
        pred_sentence (list): list of pairs, tags are predicted by tagger
    """
    assert len(gold_sentence) == len(pred_sentence)
    vocabulary = GLOBAL_VOCAB.stoi if GLOBAL_VOCAB is not None else perWordTagCounts
    correct, correctOOV, OOV = 0, 0, 0
    for gold_token, pred_token in zip(gold_sentence, pred_sentence):
        is_correct = gold_token[1] == pred_token[1]
        is_oov = gold_token[0] not in vocabulary
        correct += 1 if is_correct else 0
        correctOOV += 1 if is_correct and is_oov else 0
        OOV += 1 if is_oov else 0
    return correct, correctOOV, OOV

GLOBAL_VOCAB = None

if __name__ == "__main__":

    """ JONI'S MAIN: """
    #
    # # input_rep = 0
    # input_rep = 1
    #
    # train_data_fn = 'en-ud-train.upos.tsv'
    # test_data_fn = 'en-ud-dev.upos.tsv'
    # pretrained_embeddings_fn = 'glove.6B.100d.txt'
    #
    # # train_data_fn = 'train_small.tsv'
    # # test_data_fn = 'test_small.tsv'
    # # pretrained_embeddings_fn = 'some_vectors.txt'
    #
    # train_texts = load_annotated_corpus(train_data_fn)
    # test_texts = load_annotated_corpus(test_data_fn)
    # params_d = get_best_performing_model_params()
    # params_d['pretrained_embeddings_fn'] = pretrained_embeddings_fn
    # params_d['data_fn'] = train_data_fn
    # params_d['input_rep'] = input_rep
    #
    # # TRAIN MODEL
    # model = initialize_rnn_model(params_d)
    # train_rnn(model, train_texts, test_texts)
    #
    # # # LOAD MODEL
    # # saved_path = f'model_input_rep_{input_rep}.sav'
    # # model.load_state_dict(torch.load(saved_path))
    #
    # # GET TEST PERFORMANCE
    # # test_rnn(model, test_data)
    #
    # # EVALUATE
    # model_name = 'blstm' if input_rep == 0 else 'cblstm'
    # d = {model_name: [model]}
    # correct_count, word_count = 0, 0
    # oov_correct_count, oov_count = 0, 0
    # for gold_sentence in test_texts:
    #     words = [w[0] for w in gold_sentence]
    #     pred_sentence = tag_sentence(words, d)
    #     correct, correctOOV, OOV = count_correct(gold_sentence, pred_sentence)
    #     correct_count += correct
    #     word_count += len(words)
    #     oov_correct_count += correctOOV
    #     oov_count += OOV
    # test_acc = correct_count / word_count
    # oov_ratio = oov_count / word_count
    # oov_acc = oov_correct_count / oov_count
    # print(f'\n{model_name} test_acc={test_acc:.4f} oov_ratio={oov_ratio:.4f} oov_acc={oov_acc:.4f}')

    """ ORI'S MAIN: """

    # train_data_fn = 'en-ud-train.upos.tsv'
    # test_data_fn = 'en-ud-dev.upos.tsv'

    train_data_fn = 'train_small.tsv'
    test_data_fn = 'test_small.tsv'

    TEXT, TAGS = data.Field(lower=True), data.Field(unk_token=None)
    fields = [('text', TEXT), ('tags', TAGS)]

    # TRAIN HMM MODEL
    corpus = load_annotated_corpus(train_data_fn)
    hmm_model = learn_params(corpus)
    test = "You are such a good boy!"
    A = hmm_model[4]
    B = hmm_model[5]
    tagged_base = baseline_tag_sentence(word_tokenize(test), hmm_model[1], hmm_model[0])
    tagged_hmm = hmm_tag_sentence(word_tokenize(test), A, B)
    print(tagged_base)
    print(tagged_hmm)

    # GET TEST PERFORMANCE
    test_examples = get_examples_from_data(load_annotated_corpus(test_data_fn), fields)
    test_data = data.Dataset(test_examples, fields)
    test_hmm(hmm_model, test_data)

    print('\ndone')
