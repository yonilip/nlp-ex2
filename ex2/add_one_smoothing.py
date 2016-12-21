'''
(d)
Using Add-one smoothing
i. Training phase: Compute the emission probabilities of a bigram HMM tagger directly on the
training set using Add-one smoothing.
ii. Using the new probabilities, run the algorithm from c)ii) on the test set. Compute the error
rate and compare it to the results from b)ii) and c)iii).
'''

import math
import pprint
import collections
from nltk.corpus import brown

from tags_and_words import TAGS, TAG2INDEX, WORDS, WORDS2INDEX
from tags_and_words import START, STOP

data = brown.tagged_sents(categories="news")
train = data[:int(0.9 * len(data))]
test = data[int(0.9 * len(data)):]


def safe_log(x):
    if x == 0:
        return -float("inf")
    else:
        return math.log(x)


def add_start_and_stop(sent):
    return [(START, u"START")] + sent + [(STOP, u"STOP")]


def estimate_transition(train):
    """estimate the transition matrix
    row i column j represents to log-probability of transitions
    from the i'th tag to the j'th tag"""
    trans = [[0 for i in xrange(len(TAGS))] for j in xrange(len(TAGS))]
    for sent in train:
        new_sent = add_start_and_stop(sent)
        for i in xrange(len(new_sent) - 1):
            tag1, tag2 = new_sent[i][1], new_sent[i + 1][1]  # index 1 is the tag in the tuple of (word, tag)
            trans[TAG2INDEX[tag1]][TAG2INDEX[tag2]] += 1  # tag1 is the i-1'th tag, tag2 is the i'th tag

    # to artificially add some exit from STOP tag
    trans[TAG2INDEX[u"STOP"]][TAG2INDEX[u"STOP"]] = 1

    # normalize so each row has sum of 1
    for row in trans:
        row_sum = sum(row)
        for i in xrange(len(row)):
            row[i] = safe_log(
                float(row[i]) / row_sum)  # notice log of division for avoiding truncation of floating points
    return trans


def estimate_emission_and_minprob(train):
    """ estimate the emission matrix
    row i column j represents the log-prob of the i'th tag
    emitting j'th word"""
    min_probs = [0] * len(TAGS)
    # note that the initial count of each word is 1 (+1 smoothing)
    emission = [[1 for i in xrange(len(WORDS))] for j in xrange(len(TAGS))]
    for sent in train:
        new_sent = add_start_and_stop(sent)
        for word, tag in new_sent:
            emission[TAG2INDEX[tag]][WORDS2INDEX[word]] += 1

    # normalize so each row has sum of 1
    for i, row in enumerate(emission):
        row_sum = sum(row)
        # - len to compensate for added ones
        min_probs[i] = safe_log(1. / (row_sum - len(WORDS)))
        for j in xrange(len(row)):
            row[j] = safe_log(float(row[j]) / row_sum)
    return emission, min_probs


def viterbi(trans, emission, min_probs, sentence):
    # sentence should start with START and end with STOP
    pi = [[(-float("inf"), None) for i in xrange(len(TAGS))] for i in xrange(len(sentence))]
    pi[0][TAG2INDEX[u"START"]] = (1, TAG2INDEX[u"START"])
    for k in xrange(1, len(sentence)):
        word = sentence[k]
        for tag in xrange(len(TAGS)):
            emission_log_prob = emission[tag][WORDS2INDEX[word]] if word in WORDS2INDEX else min_probs[tag]
            tuples = [(pi[k - 1][prev_tag][0] + trans[prev_tag][tag] + emission_log_prob, prev_tag) \
                      for prev_tag in xrange(len(TAGS))]
            pi[k][tag] = max(tuples)

    # now use backpointers to collect the best tags
    current_tag = TAG2INDEX[u"STOP"]
    best_tags = []
    for k in xrange(len(sentence) - 1, -1, -1):
        best_tags.insert(0, current_tag)
        current_tag = pi[k][current_tag][1]

    return best_tags


def calc_error_rate(train, test):
    trans = estimate_transition(train)
    emission, min_probs = estimate_emission_and_minprob(train)
    num_words = 0
    num_errors = 0
    for sent in test:
        print num_words, num_errors
        words_sent = [START] + [word for word, tag in sent] + [STOP]
        v_best_tags = viterbi(trans, emission, min_probs, words_sent)
        for i in xrange(1, len(v_best_tags) - 1):  # to avoid comparing start and stop tags
            num_words += 1
            if TAGS[v_best_tags[i]] != sent[i - 1][1]:
                num_errors += 1
    return num_words, num_errors


def build_confusion_matrix(train, test):
    conf = collections.defaultdict(lambda: 0)
    trans = estimate_transition(train)
    emission, min_probs = estimate_emission_and_minprob(train)
    for sent in test:
        words_sent = [START] + [word for word, tag in sent] + [STOP]
        v_best_tags = viterbi(trans, emission, min_probs, words_sent)
        for i in xrange(1, len(v_best_tags) - 1):  # to avoid comparing start and stop tags
            conf[(TAGS[v_best_tags[i]], sent[i - 1][1])] += 1
    return conf

# if __name__ == "__main__":
# print calc_error_rate(train, test)
