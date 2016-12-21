'''
a) Use the NLTK toolkit for importing the Brown corpus.  This corpus contains text from 500
sources, and the sources have been categorized by genre. Here we will use a portion of the corpus:
the \news" category. Load the tagged sentences for this portion of the corpus. Then, divide the
obtained corpus into training set and test set such that the test set is formed by the last 10% of
the sentences.


(b)
Implementation of the most likely tag baseline
i. Using the training set, compute for each word the tag that maximizes
p(tag|word). Assume that all the unknown words are annotated as \NN".
ii. Using the test set, compute the error rate for known words, for unknown words, as well as
the total error rate.
'''

from nltk.corpus import brown

data = brown.tagged_sents(categories="news")
train = data[:int(0.9 * len(data))]
test = data[int(0.9 * len(data)):]


def make_counts_dicts(data):
    counts_dict = dict()
    for sent in data:
        for word, tag in sent:
            if word not in counts_dict:
                counts_dict[word] = dict()
            if tag not in counts_dict[word]:
                counts_dict[word][tag] = 0
            counts_dict[word][tag] += 1
    return counts_dict


def tag_word(word, counts_dict):
    if word not in counts_dict:
        return u'NN'
    else:
        tags = counts_dict[word]
        return max(tags.keys(), key=lambda k: tags[k])


def calc_error_rate(train, test, only_unknown=False, only_known=False):
    counts = make_counts_dicts(train)
    num_words = 0
    num_errors = 0
    for sent in test:
        for word, tag in sent:
            if only_unknown and word in counts:
                continue
            if only_known and word not in counts:
                continue
            num_words += 1
            if tag_word(word, counts) != tag:
                num_errors += 1
    return num_words, num_errors


# print calc_error_rate(train, test)
