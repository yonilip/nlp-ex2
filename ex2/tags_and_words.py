from nltk.corpus import brown

TAGS = set([])
WORDS = set([])

data = brown.tagged_sents(categories="news")
train = data[:int(0.9 * len(data))]


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
