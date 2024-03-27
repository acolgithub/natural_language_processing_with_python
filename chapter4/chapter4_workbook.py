import copy
import random
from nltk.tokenize import word_tokenize
import nltk
import re
from urllib import request
from bs4 import BeautifulSoup
from nltk.corpus import wordnet as wn



# 4.1 Back to the Basics

# modifying alters all entries
nested = [[]]*3
nested[1].append("not empty")
print(nested)
for i in range(len(nested)):
    print(id(nested[i]))
print("\n")

# assigning does not alter all enties
nested = [[]]*3
nested[1].append("Python")  # first modify
nested[1] = ["Monty"]  # then assign
print(nested)  # second entry is different due to assignment
for i in range(len(nested)):
    print(id(nested[i]))  # first and last are the same but second entry is different
print("\n")

# copy items from a list (copies object references)
foo = ["A", "B", "C"]
bar = foo[:]
print(bar)
print(foo)
print(id(bar))  # different ids for array
print(id(foo))
for i in range(3):
    print(id(bar[i]), id(foo[i]))
bar[1] = "D"
print(bar)  # different value for second entry
print(foo)
for i in range(3):
    print(id(bar[i]), id(foo[i]))  # different id on second entry due to assignment
print("\n")

# copy items without copying object references
foo2 = ["A", "B", "C"]
bar2 = copy.deepcopy(foo2)
print(foo2)
print(bar2)
print(id(foo2))
print(id(bar2))
for i in range(3):
    print(id(foo2[i]), id(bar2[i]))
print("\n")



# Equality

size = 5
python = ["Python"]
snake_nest = [python] * size
# check values are equal
print(snake_nest[0] == snake_nest[1] == snake_nest[2] == snake_nest[3] == snake_nest[4])

# check object identity is equal
print(snake_nest[0] is snake_nest[1] is snake_nest[2] is snake_nest[3] is snake_nest[4])
print("\n")

position = random.choice(range(size))
snake_nest[position] = ["Python"]  # mody idenity of one entry
print(snake_nest)  # looks the same as before

# check eqaulity of entries
print(snake_nest[0] == snake_nest[1] == snake_nest[2] == snake_nest[3] == snake_nest[4])

# check equality of identity
print(snake_nest[0] is snake_nest[1] is snake_nest[2] is snake_nest[3] is snake_nest[4])  # false due to reassignment
print("\n")

# check for differing id
print([id(snake) for snake in snake_nest], "\n")



# Conditionals
sent = ['No', 'good', 'fish', 'goes', 'anywhere', 'without', 'a', 'porpoise', '.']

# check if all elements satisfy condition
print(all(len(w)>4 for w in sent))

# check if any elements satsify condition
print(any(len(w)>4 for w in sent), "\n")



# 4.2 Sequences

t = "walk", "fem", 3  # form tuple
print(t)
print(t[0])
print(t[1:])
print(len(t))

# single element tuple
s = "snark",

# empty tuple
e = ()
print(s, e, "\n")


# convert frequency distribution to different types of sequences
raw = 'Red lorry, yellow lorry, red lorry, yellow lorry.'
text = word_tokenize(raw)
fdist = nltk.FreqDist(text)
print(sorted(fdist))  # print sorted frequency distribution
for key in fdist:  # print key and count
    print(f"{key}: {fdist[key]}; ", "\n")


# zip two sequences together (creates tuples by pairing)
words = ["I", "turned", "off", "the", "spectroroute"]
tags = ["noun", "verb", "prep", "det", "noun"]
print(list(zip(words,tags)))  # print as list

# enumerate pairs words with index
print(list(enumerate(words)), "\n")



# split data into training and test data
text = nltk.corpus.nps_chat.words()
cut = int(0.9 * len(text))  # break text into 90% size
training_data, test_data = text[:cut], text[cut:]  # split data
print(text == training_data + test_data)  # recombines correctly
print(len(training_data)/len(test_data), "\n")  # get size of training to test



# Combining Different Sequence Types
words = "I turned off the spectroroute".split()
lengths = [len(word) for word in words]
sorted_zipped_words = sorted(list(zip(words, lengths)), key=lambda x: x[1])
words_increasing_length = [x[0] for x in sorted_zipped_words]
print(" ".join(words_increasing_length), "\n")  # join back into sentence


# tuple is immutable so cannot modify
lexicon = tuple((('the', 'det', ['Di:', 'D@']), ('off', 'prep', ['Qf', 'O:f'])))


# Generator Expressions

text = '''"When I use a word," Humpty Dumpty said in rather a scornful tone,
"it means just what I choose it to mean - neither more nor less."'''

# use generator expression
print(max(w.lower() for w in word_tokenize(text)), "\n")


# 4.4 Functions: The Foundation of Structured Programming

def get_text(file):
    """Read text from a file, normalizing whitespace and stripping HTML markup."""
    text = open(file).read()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub("\s+", " ", text)
    return text

# help(get_text)  # provides help through docstring



# Function Inputs and Outputs

def repeat(msg, num):
    return ' '.join([msg] * num)
monty = "Monty Python"
print(repeat(monty, 3))


# Parameter Passing

def set_up(word, properties):
    word = "lolcat"
    properties.append("noun")
    properties = 5
w = ""
p = []
set_up(w,p)
print(w, p, "\n")



# Checking Parameter Types

def tag(word):
    assert isinstance(word, basestring), "argument to tag() must be a string"
    if word in ["a", "the", "all"]:
        return "det"
    else:
        return "noun"


# Functional Decomposition
 
def freq_words(url, freqdist, n):
    html = request.urlopen(url).read().decode("utf8")  # get raw html
    raw = BeautifulSoup(html, "html.parser").get_text()  # parse text

    # tokenize text and create frequency distribution of words
    for word in word_tokenize(raw):
        freqdist[word.lower()] += 1

    # get most common words
    result = []
    for word, count in freqdist.most_common(n):
        result = result + [word]
    print(result)

# get 30 most common words in constitution
constitution = "http://www.archives.gov/exhibits/charters/constitution_transcript.html"
fd = nltk.FreqDist()
print(freq_words(constitution, fd, 30), "\n")

def freq_words_refactored(url, n):
    html = request.urlopen(url).read().decode("utf8")
    text = BeautifulSoup(html, "html.parser").get_text()
    freqdist = nltk.FreqDist(word.lower() for word in word_tokenize(text))
    return [word for (word, _) in fd.most_common(n)]


print(freq_words_refactored(constitution, 30))



# Documenting Functions

def accuracy(reference, test):
    """
    Calculate the fraction of test items that equal the corresponding reference items.

    Given a list of reference values and a corresponding list of test values,
    return the fraction of corresponding values that are equal.
    In particular, return the fraction of indexes
    {0<i<=len(test)} such that C{test[i] == reference[i]}

        >>> accuracy(["ADJ", "N", "V", "N"], ["N", "N", "V", "ADJ"])
        0.5

    :param reference: An ordered list of reference values
    :type reference: list
    :param test: A list of values to compare against the corresponding
        reference values
    :type test: list
    :return: the accuracy score
    :rtype: float
    :raises ValueError: If reference and length do not have the same length
    """

    if len(reference) != len(test):
        raise ValueError("Lists must have the same length.")
    num_correct = 0
    for x, y in zip(reference, test):
        if x == y:
            num_correct += 1
    return float(num_correct) / len(reference)



# 4.5 Doing More with Functions

sent = ["Take", "care", "of", "the", "sense", ",", "and", "the",
        "sounds", "will", "take", "care", "of", "themselves", "."]

def extract_property(prop):
    return [prop(word) for word in sent]

print(extract_property(len), "\n")

def last_letter(word):
    return word[-1]

print(extract_property(last_letter), "\n")

print(extract_property(lambda w: w[-1]), "\n")

print(sorted(sent), "\n")



# Accumulative Functions

def search1(substring, words):
    result = []
    for word in words:
        if substring in word:
            result.append(word)
    return result

def search2(substring, words):
    for word in words:
        if substring in word:
            yield word


for item in search1("zz", nltk.corpus.brown.words()):
    print(item, end=" ")

print("\n")

for item in search2("zz", nltk.corpus.brown.words()):
    print(item, end=" ")

print("\n")



def permutations(seq):
    if len(seq) <= 1:
        yield seq
    else:
        for perm in permutations(seq[1:]):
            for i in range(len(perm)+1):
                yield perm[:i] + seq[0:1] + perm[i:]


print(list(permutations(["police", "fish", "buffalo"])), "\n")



# Higher-Order Functions

def is_content_word(word):
    return word.lower() not in ["a", "of", "the", "and", "will", ",", "."]

sent = ["Take", "care", "of", "the", "sense", ",", "and", "the",
        "sounds", "will", "take", "care", "of", "themselves", "."]
print(list(filter(is_content_word, sent)), "\n")  # filter by given function

print([w for w in sent if is_content_word(w)], "\n")


lengths = list(map(len, nltk.corpus.brown.sents(categories="news")))
print(sum(lengths)/len(lengths), "\n")

lengths = [len(sent) for sent in nltk.corpus.brown.sents(categories="news")]
print(sum(lengths)/len(lengths), "\n")

# filter by given function
print(list(map(lambda w: len(list(filter(lambda c: c.lower() in "aeiou", w))), sent)), "\n")

print([len([c for c in w if c.lower() in "aeiou"]) for w in sent], "\n")



# Named Arguments

# define function that takes an arbitrary number of unnamed and named parameters
def generic(*args, **kwargs):
    print(args)
    print(kwargs)

generic(1, "African swallow", monty="python")
print("\n")

song = [["four", "calling", "birds"],
        ["three", "French", "hens"],
        ["two", "turtle", "doves"]]

print(list(zip(song[0], song[1], song[2])))
print("\n")
print(list(zip(*song)))
print("\n")


def freq_words_args(file, min=1, num=10):
    text = open(file).read()
    tokens = word_tokenize(text)
    freqdist = nltk.FreqDist(t for t in tokens if len(t) >= min)
    return freqdist.most_common(num)

# same calls
# fw1 = freq_words_args("ch01.rst", 4, 10)
# fw2 = freq_words_args("ch01.rst", min=4, num=10)
# fw3 = freq_words_args("ch01.rst", num=10, min=4)



# 4.6 Program Development

# Structure of a Python Module
from nltk.metrics import distance
print(distance.__file__)
print("\n")

# help(distance)
import pdb



# 4.7 Algorithm Design

def size1(s):
    return 1 + sum(size1(child) for child in s.hyponyms())


def size2(s):
    layer = [s]
    total = 0
    while layer:
        total += len(layer)
        layer = [h for c in layer for h in c.hyponyms()]
    return total


dog = wn.synset("dog.n.01")
print(size1(dog))
print(size2(dog), "\n")


















