from nltk.corpus import brown
from collections import Counter

TAGS = set([])
WORDS = set([])

THRESHOLD = 1


def contains_digits(w):
    return any(d.isdigit() for d in w)


def contains_alpha(w):
    return any(x.isalpha() for x in w)


class Other(object):
    pass


class DigitsAndAlpha(object):
    pass


class TwoDigit(object):
    pass


class FourDigit(object):
    pass


class OtherNum(object):
    pass


class DigitsAndComma(object):
    pass


class DigitsAndSlash(object):
    pass


class DigitsAndDash(object):
    pass


class AllCapsPeriod(object):
    pass


class AllCaps(object):
    pass


class FirstWord(object):
    pass


class InitCap(object):
    pass


class LowerCase(object):
    pass


def word2pseudo(w, is_first_word=False):
    if contains_digits(w) and contains_alpha(w):
        return DigitsAndAlpha
    if w.isdigit():
        if len(w) == 2:
            return TwoDigit
        elif len(w) == 4:
            return FourDigit
        else:
            return OtherNum
    elif "," in w and contains_digits(w):
        return DigitsAndComma
    elif "/" in w and contains_digits(w):
        return DigitsAndSlash
    elif "-" in w and contains_digits(w):
        return DigitsAndDash
    elif w.upper() == w:
        if "." in w:
            return AllCapsPeriod
        else:
            return AllCaps
    if is_first_word:
        return FirstWord
    elif w[0] == w[0].upper() and w.isalpha():
        return InitCap
    elif w.isalpha():
        return LowerCase
    else:
        return Other


def transform_sentences(train):
    all_words = []
    for sent in train:
        all_words += [word for word, tag in sent]
    c = Counter(all_words)
    for sent in train:
        for i in xrange(len(sent)):
            if c[sent[i][0]] <= THRESHOLD:
                sent[i] = (word2pseudo(sent[i][0], i == 0), sent[i][1])


data = brown.tagged_sents(categories="news")
train = list(data[:int(0.9 * len(data))])
test = data[int(0.9 * len(data)):]

transform_sentences(train)


class START(object):
    pass


class STOP(object):
    pass


WORDS.add(STOP)
WORDS.add(START)
TAGS.add(u"START")
TAGS.add(u"STOP")

for sent in train:
    for word, tag in sent:
        TAGS.add(tag)
        WORDS.add(word)
TAGS = list(TAGS)
WORDS = list(WORDS)

TAG2INDEX = {tag: i for i, tag in enumerate(TAGS)}
WORDS2INDEX = {word: i for i, word in enumerate(WORDS)}
