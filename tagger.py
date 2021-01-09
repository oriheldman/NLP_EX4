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
from torchtext import data, vocab
import torch.optim as optim
from math import log, isfinite
from collections import Counter
import numpy as np
import sys, os, time, platform, nltk, random
import string
from nltk.corpus import stopwords
import pandas as pd
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
    # TODO edit the dictionary to have your own details
    # if work is submitted by a pair of students, add the following keys: name2, id2, email2
    return {'name1': 'John Doe', 'id1': '012345678', 'email1': 'jdoe@post.bgu.ac.il',
            'name2': 'John Doe', 'id2': '012345678', 'email2': 'jdoe@post.bgu.ac.il'}


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
    """Returns an lstm model based on the specified parameters.

    Args:
        params_d (dict): an dictionary of parameters specifying the model. The dict
                        should include (at least) the following keys:
                        {'input_dimension': int,
                        'embedding_dimension': int,
                        'num_of_layers': int,
                        'output_dimension': int}
                        The dictionary can include other keys, if you use them,
                             BUT you shouldn't assume they will be specified by
                             the user, so you should spacify default values.
    Return:
        torch.nn.Module object
    """

    # TODO update the padding index
    model = RNN(**params_d)
    return model


def get_model_params(model):
    """Returns a dictionary specifying the parameters of the specified model.
    This dictionary should be used to create another instance of the model.

    Args:
        model (torch.nn.Module): the network architecture

    Return:
        a dictionary, containing at least the following keys:
        {'input_dimension': int,
        'embedding_dimension': int,
        'num_of_layers': int,
        'output_dimension': int}
    """
    return {
        'input_dimension': model.input_dimension,
        'embedding_dimension': model.embedding_dimension,
        'num_of_layers': model.num_of_layers,
        'output_dimension': model.output_dimension,
        'hidden_dim': model.hidden_dim,
        'dropout': model.dropout,
        'padding_idx': model.padding_idx
    }


def load_pretrained_embeddings(path):
    """ Returns an object with the the pretrained vectors, loaded from the
        file at the specified path. The file format is the same as
        https://www.kaggle.com/danielwillgeorge/glove6b100dtxt
        The format of the vectors object is not specified as it will be used
        internaly in your code, so you can use the datastructure of your choice.
    """
    return vocab.Vectors(name=path)


def train_rnn(model, train_data_fn, pretrained_embeddings_fn, input_rep=0, train_dataset=None, fields=None,
              epochs=10, batch_size=128):
    """Trains the BiLSTM model on the specified data.

    Args:
        model (torch.nn.Module): the model to train
        train_data_fn (string): full path to the file with training data (in the provided format)
        pretrained_embeddings_fn (string): full path to the file with pretrained embeddings
        input_rep (int): sets the input representation. Defaults to 0 (vanilla),
                         1: case-base; <other int>: other models, if you are playful
    """
    # Tips:
    # 1. you have to specify an optimizer
    # 2. you have to specify the loss function and the stopping criteria
    # 3. consider loading the data and preprocessing it
    # 4. consider using batching
    # 5. some of the above could be implemented in helper functions (not part of
    #    the required API)

    if train_dataset is None:
        pass  # todo: load data and fields here!

    TEXT, TAGS = fields
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

    iterator = data.BucketIterator(train_dataset, batch_size=batch_size, device=device)

    model.apply(init_weights)

    vectors = TEXT.vocab.vectors
    model.embedding.weight.data.copy_(vectors)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(vectors.shape[1])

    optimizer = optim.Adam(model.parameters())

    TAG_PAD_IDX = TAGS.vocab.stoi[TAGS.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index=TAG_PAD_IDX)

    model = model.to(device)
    criterion = criterion.to(device)

    for epoch in range(epochs):
        start_time = time.time()

        loss, acc = train(model, iterator, optimizer, criterion, TAG_PAD_IDX)
        # valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, TAG_PAD_IDX)

        ep_time = int(time.time() - start_time)
        print(f'epoch:{epoch + 1} time:{ep_time}(secs) loss:{loss:.4f} acc:{acc:.2f}')

    torch.save(model.state_dict(), 'model.sav')


def init_weights(m):  # todo: change this since it came from ref
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.1)


def categorical_accuracy(preds, y, tag_pad_idx):  # todo: change this since it came from ref
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim=1, keepdim=True)  # get the index of the max probability
    non_pad_elements = (y != tag_pad_idx).nonzero()
    correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
    return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]])


def train(model, iterator, optimizer, criterion, tag_pad_idx):  # todo: change this since it came from ref
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        text = batch.text
        tags = batch.tags

        optimizer.zero_grad()

        # text = [sent len, batch size]

        predictions = model(text)

        # predictions = [sent len, batch size, output dim]
        # tags = [sent len, batch size]

        predictions = predictions.view(-1, predictions.shape[-1])
        tags = tags.view(-1)

        # predictions = [sent len * batch size, output dim]
        # tags = [sent len * batch size]

        loss = criterion(predictions, tags)

        acc = categorical_accuracy(predictions, tags, tag_pad_idx)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, tag_pad_idx):  # todo: change this since it came from ref
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            text = batch.text
            tags = batch.tags

            predictions = model(text)

            predictions = predictions.view(-1, predictions.shape[-1])
            tags = tags.view(-1)

            loss = criterion(predictions, tags)

            acc = categorical_accuracy(predictions, tags, tag_pad_idx)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate_hmm(model, test_data):
    epoch_loss = 0
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


def test_hmm(model, dataset, fields=None, saved_path=None):
    test_acc = evaluate_hmm(model, dataset)
    print(f'Test Acc: {test_acc * 100:.2f}%')


def test_rnn(model, dataset, fields=None, saved_path=None):
    """Trains the BiLSTM model on the specified data.

    Args:
        model (torch.nn.Module): the model to train
        train_data_fn (string): full path to the file with training data (in the provided format)
        pretrained_embeddings_fn (string): full path to the file with pretrained embeddings
        input_rep (int): sets the input representation. Defaults to 0 (vanilla),
                         1: case-base; <other int>: other models, if you are playful
    """
    iterator = data.BucketIterator(dataset, batch_size=128, device=device)
    TAG_PAD_IDX = TAGS.vocab.stoi[TAGS.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index=TAG_PAD_IDX)
    model = model.to(device)
    criterion = criterion.to(device)

    # if saved_path:
    #     model.load_state_dict(torch.load(saved_path))

    test_loss, test_acc = evaluate(model, iterator, criterion, TAG_PAD_IDX)
    print(f'Test Loss: {test_loss:.3f} |  Test Acc: {test_acc * 100:.2f}%')


def rnn_tag_sentence(sentence, model, input_rep=0):
    """ Returns a list of pairs (w,t) where each w corresponds to a word
        (same index) in the input sentence. Tagging is done with the Viterby
        algorithm.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        model (torch.nn.Module):  a trained BiLSTM model
        input_rep (int): sets the input representation. Defaults to 0 (vanilla),
                         1: case-base; <other int>: other models, if you are playful

    Return:
        list: list of pairs
    """
    if TEXT.lower:
        sentence = [token.lower() for token in sentence]
    sentence_idxs = [TEXT.vocab.stoi[token] for token in sentence]
    sentence_tensor = torch.LongTensor(sentence_idxs).unsqueeze(-1).to(device)

    model.eval()
    predictions = model(sentence_tensor)
    tags = [TAGS.vocab.itos[t.item()] for t in predictions.argmax(-1)]

    tagged_sentence = [(w, t) for w, t in zip(sentence, tags)]
    return tagged_sentence


def get_best_performing_model_params():
    """Returns a disctionary specifying the parameters of your best performing
        BiLSTM model.
        IMPORTANT: this is a *hard coded* dictionary that will be used to create
        a model and train a model by calling
               initialize_rnn_model() and train_lstm()
    """
    # TODO complete the code

    return model_params


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
        3. Vanilla BiLSTM: {'blstm':[Torch.nn.Module, input_rep]}
        4. BiLSTM+case: {'cblstm': [Torch.nn.Module, input_rep]}
        5. (NOT REQUIRED: you can add other variations, agumenting the input
            with further subword information, with character-level word embedding etc.)

        The parameters for the baseline model are:
        perWordTagCounts (Counter): tags per word as specified in learn_params()
        allTagCounts (Counter): tag counts, as specified in learn_params()

        The parameters for the HMM are:
        A (dict): The HMM Transition probabilities
        B (dict): tthe HMM emmission probabilities.

        Parameters for an LSTM:
        the neural network model
        input_rep (int) - must support 0 and 1 (vanilla and case-base, respectively)


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


def count_correct(gold_sentence, pred_sentence):
    """Return the total number of correctly predicted tags,the total number of
    correcttly predicted tags for oov words and the number of oov words in the
    given sentence.

    Args:
        gold_sentence (list): list of pairs, assume to be gold labels
        pred_sentence (list): list of pairs, tags are predicted by tagger

    """
    assert len(gold_sentence) == len(pred_sentence)
    correct = 0
    correctOOV = 0
    OOV = 0
    for index, pair in enumerate(pred_sentence):
        word = pair[0]
        predicted_tag = pair[1]
        label_tag = gold_sentence[index][1]
        # correct prediction
        if predicted_tag == label_tag:
            correct += 1
        #  oov count
        if word not in perWordTagCounts:
            OOV += 1
        if predicted_tag == label_tag and word not in perWordTagCounts:
            correctOOV += 1

    return correct, correctOOV, OOV


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


def get_examples_from_data(data_fn, fields):
    examples = []
    texts = load_annotated_corpus(data_fn)
    for text in texts:
        values = [[], []]
        for token, tag in text:
            values[0].append(token)
            values[1].append(tag)
        example = data.Example.fromlist(values, fields)
        examples.append(example)
    return examples


if __name__ == "__main__":
    train_data_fn = 'en-ud-train.upos.tsv'
    # data_fn = 'train_small.tsv'
    pretrained_embeddings_fn = 'glove.6B.100d.txt'
    # pretrained_embeddings_fn = 'some_vectors.txt'

    TEXT, TAGS = data.Field(lower=True), data.Field(unk_token=None)
    fields = [('text', TEXT), ('tags', TAGS)]
    train_examples = get_examples_from_data(train_data_fn, fields)
    train_data = data.Dataset(train_examples, fields)
    # vectors = load_pretrained_embeddings(pretrained_embeddings_fn)

    # TEXT.build_vocab(train_data, min_freq=2, vectors=vectors, unk_init=torch.Tensor.normal_)
    # TAGS.build_vocab(train_data)

    # params_d = {
    #     'input_dimension': len(TEXT.vocab),
    #     'embedding_dimension': 100,
    #     'num_of_layers': 2,
    #     'output_dimension': len(TAGS.vocab),
    # }

    # TRAIN MODEL
    # model = initialize_rnn_model(params_d)
    # train_rnn(model, data_fn, pretrained_embeddings_fn, train_dataset=train_data, fields=[TEXT, TAGS])

    # TRAIN HMM MODEL
    corpus = load_annotated_corpus('en-ud-train.upos.tsv')
    hmm_model = learn_params(corpus)
    test = "You are such a good boy!"
    A = hmm_model[4]
    B = hmm_model[5]
    tagged_base = baseline_tag_sentence(word_tokenize(test), hmm_model[1], hmm_model[0])
    tagged_hmm = hmm_tag_sentence(word_tokenize(test), A, B)

    # LOAD MODEL
    # saved_path = 'model.sav'
    # model.load_state_dict(torch.load(saved_path))

    # GET TEST PERFORMANCE
    test_data_fn = 'en-ud-dev.upos.tsv'
    test_examples = get_examples_from_data(test_data_fn, fields)
    test_data = data.Dataset(test_examples, fields)
    test_hmm(hmm_model, test_data, [TEXT, TAGS])
    # test_rnn(model, test_data, [TEXT, TAGS])

    # TAG A SENTENCE
    # sentence = ['[', 'this', 'killing', 'of', 'a', 'respected', 'cleric', 'will', 'be', 'causing', 'us',
    #             'trouble', 'for', 'years', 'to', 'come', '.', ']']
    # for tag in rnn_tag_sentence(sentence, model):
    #     print(tag)

    # print(tagged_base)
    # print(tagged_hmm)

    print('\ndone')
