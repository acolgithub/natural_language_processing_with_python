import nltk
import re
import pprint
from nltk import word_tokenize
from urllib import request
from bs4 import BeautifulSoup  # imported to extract text out of html
import feedparser  # imported to parse blogs
import os
from nltk.corpus import gutenberg
import unicodedata  # inspect properties of unicode characters
from nltk.corpus import nps_chat  # for chat words
from nltk.corpus import brown
from random import randint
import math
from textwrap import fill


# 3.1 Accessing Text from the Web and from Disk

# Electronic Books

# get raw text
url = "http://www.gutenberg.org/files/2554/2554-0.txt" # get url
response = request.urlopen(url)  # open url
raw = response.read().decode("utf8")  # read raw txt file and decode utf8
print(f"type of raw: {type(raw)}")
print(f"length of raw: {len(raw)}")
print(f"First 75 words:\n{raw[:75]}\n")

# tokenize
tokens = word_tokenize(raw)  # tokenize raw text
print(f"type of tokens: {type(tokens)}")
print(f"length of tokens: {len(tokens)}")
print(f"first 10 tokens:\n{tokens[:10]}\n")

# form text
text = nltk.Text(tokens)
print(f"get type of text: {type(text)}")
print(f"get subset of text:\n{text[1024:1062]}")
text.collocations()  # get collocations
print("\n")

# find headers and information
print(raw.find("Part I"))  # get index of Part I using find (could not find it)
print(raw.rfind("End of Project Gutenberg's Crime"), "\n")  # get index of Gutenberg statement using reverse find (could not find it)

# slice string to remove undesirable parts
raw = raw[5338:1157743]  # remove unwanted portion
print(raw.find("Part I"), "\n")  # should have Part I at the beginning


# Dealing with HTML

# open url
url = "http://news.bbc.co.uk/2/hi/health/2284783.stm"  # get url
html = request.urlopen(url).read().decode("utf8")  # open url, read, then decode to html
print(html[:60])  # print first 60 characters of html

# extract text from html
raw = BeautifulSoup(html, "html.parser").get_text()  # extract raw text from html
tokens = word_tokenize(raw)  # tokenize raw text
print(tokens)  # print all tokens

# get relevant tokens and convert to text
tokens = tokens[110:390]
text = nltk.Text(tokens)
text.concordance("gene")  # get concordance of gene



# Processing RSS Feeds

llog = feedparser.parse("http://languagelog.ldc.upenn.edu/nll/?feed=atom")  # parse language log blog
print(llog["feed"]["title"])  # access feed and get title
print(f"length of lanague log blog entries: {len(llog.entries)}")
post = llog.entries[2]  # get first two entries
print(f"get title of post: {post.title}")
content = post.content[0].value  # get content
print(content[:70])  # get first 70 characters
raw = BeautifulSoup(content, "html.parser").get_text()  # get raw text content
print(word_tokenize(raw), "\n")  # tokenize raw content


# Reading Local Files

# to open text file and load its contents follow this:
f = open("data/document.txt")
raw = f.read()  # reading f empties contents into raw

# # inspect contents
print(raw, "\n")

# examine current directory
print(os.listdir("."), "\n")

# read contents of file
print(f.read(), "\n")  # f is empty after read

# read file one line at a time
for line in f:
    print(line.strip(), "\n")  # strip off new line characters


# read in nltk corpus files
path = nltk.data.find('corpora/gutenberg/melville-moby_dick.txt')  # get file name for corpus item
raw = open(path).read()  # get raw file



# Capturing User Input

s = input("Enter some text: ")
print(f"You typed {len(word_tokenize(s))} words.\n")



# The NLP Pipeline

# gutenberg raw text file is string
print(type(raw))


# tokenize raw text
tokens = word_tokenize(raw)

# tokenized raw file is list
print(type(tokens))

# lower case verion is still list
words = [w.lower() for w in tokens]
print(type(words))

# vocabulary is still list
vocab = sorted(set(words))
print(type(vocab), "\n")


# append to list
vocab.append("blog")

# concatenate string with strings
query = "Who knows?"
query + "john paul george ringo"



# 3.2 Strings: Text Processing at the Lowest Level

a = [1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1]
b = [' '*2*(7-i) + 'very'*i for i in a]
for line in b:
    print(line)


# get raw moby dick text file
raw = gutenberg.raw("melville-moby_dick.txt")

# get frequency distribution of alphabetical characters in text
fdist = nltk.FreqDist(ch.lower() for ch in raw if ch.isalpha())

# get five most common characters in count
print(fdist.most_common(5))

# get each character is descending order of frequency
print([char for (char, count) in fdist.most_common()])
print("\n")



# 3.3 Text Processing with Unicode

# locate file using find function
path = nltk.data.find('corpora/unicode_samples/polish-lat2.txt')

# read data from latin2 into unicode
f = open(path, encoding="latin2")

# inspect contents of the file
for line in f:
    line = line.strip()
    print(line)
print("\n")


# view underlying numerical values of encoding
for line in f:  # f is empty if previous loop runs
    line = line.strip()
    print(line.encode("unicode_escape"))
print("\n")
f.close()  # close file


# get integer ordinal for a character
print(f"integer ordinal for nacute: {ord('ń')}", "\n")


# get character corresponding to hexadecimal sequence
nacute = "\u0144"
print(nacute)
print(f"0144 is hexadecimal for: {hex(324)}")

# print sequence of bytes representing character in text file
print(nacute.encode("utf8"))

# print UTF-8 byte sequence followed by code point integer using standard Unicode convention
lines = open(path, encoding="latin2").readlines()  # get polish text
line = lines[2]  # get third line
print(line.encode("unicode_escape"))  # print unicode escape

for c in line:
    if ord(c) > 127:
        print("{} U+{:04x} {}".format(c.encode("utf8"), ord(c), unicodedata.name(c)))
print("\n")

# find unicode string within another string
print(line.find("zosta\0142y"))

# convert line to lowercase
path = nltk.data.find('corpora/unicode_samples/polish-lat2.txt')
f = open(path, encoding="latin2")
lines = open(path, encoding="latin2").readlines()
line = lines[2]
line = line.lower()
print(line)

# get unicode escape characters
print(line.encode("unicode_escape"))

# search for string starting with \u015b
m = re.search("\u015b\w*", line)
print(m)
print(m.group())

# tokenize unicode string
print(word_tokenize(line), "\n")


# 3.4 Regular Expressions for Detecting Word Patterns

wordlist = [w for w in nltk.corpus.words.words("en") if w.islower()]


# Using Basic Meta-Characters

# get words ending in 'ed'
print([w for w in wordlist if re.search("ed$", w)], "\n")

# get 8 letter words with 'j' in third position and 't' in the sixth position
print([w for w in wordlist if re.search("^..j..t..$", w)], "\n")

# get words which have 'j' and 't', in that order, separated by two characters as well as two characters in front of 'j' and two letters after 't'
print([w for w in wordlist if re.search("..j..t..", w)], "\n")



# Ranges and Closures

# get textonyms of 'hole' and 'golf'
print([w for w in wordlist if re.search("^[ghi][mno][jkl][def]$", w)], "\n")

# search for 'finger twisters'using only keys 4, 5, and 6
print([w for w in wordlist if re.search("^[ghijklmno]+$", w)], "\n")
print([w for w in wordlist if re.search("^[g-o]+$", w)], "\n")

# search for words using only keys 2, 3, 5, 6
print([w for w in wordlist if re.search("^[a-fj-o]+$", w)], "\n")


# get chat words
chat_words = sorted(set(w for w in nltk.corpus.nps_chat.words()))

# print chat words that only contain 'm''s followed by 'i''s followed by 'n''s followed by 'e''s
print([w for w in chat_words if re.search("^m+i+n+e+$", w)], "\n")

# print words that contain 'h' and 'a'
print([w for w in chat_words if re.search("^[ha]+$", w)], "\n")

# find chat words with no vowels
print([w for w in chat_words if re.search("^[^aeiouAEIOU]+$", w)], "\n")



# get wsj treebank words
wsj = sorted(set(nltk.corpus.treebank.words()))

# get decimal numbers
print([w for w in wsj if re.search("^[0-9]+\.[0-9]+$", w)], "\n")

# get capital letters followed by '$'
print([w for w in wsj if re.search("^[A-Z]+\$$", w)], "\n")

# print 4 digit numbers
print([w for w in wsj if re.search("^[0-9]{4}$", w)], "\n")

# digits followed by '-' followed by three to five letters
print([w for w in wsj if re.search('^[0-9]+-[a-z]{3,5}$', w)], "\n")

# print 5 or more letter words followed by '-' followed by 2 or 3 letters followed by '-' followed by at most 6 letters
print([w for w in wsj if re.search("^[a-z]{5,}-[a-z]{2,3}-[a-z]{,6}$", w)], "\n")

# prints words ending in 'ed' or 'ing'
print([w for w in wsj if re.search("(ed|ing)$", w)], "\n")



# 3.5 Useful Applications of Regular Expressions

# Extracting Word Pieces

# get word
word = "supercalifragilisticexpialidocious"

# find all vowels
print(re.findall(r"[aeiou]", word))

# get number of vowels in word
print(len(re.findall(r"[aeiou]", word)), "\n")


# get text
wsj = sorted(set(nltk.corpus.treebank.words()))

# get frequency distribution of two or more vowels in text
fd = nltk.FreqDist(vs for word in wsj
                      for vs in re.findall(r"[aeiou]{2,}", word))

# print most common
print(fd.most_common(12), "\n")


# extract numbers from date
print([int(n) for n in re.findall("[0-9]+", "2009-12-31")], "\n")



# Doing More with Word Pieces

# regex which looks for leading vowels or ending vowels or vowel sequence in the middle (in that order)
regexp = r"^[AEIOUaeiou]+|[AEIOUaeiou]+$|[^AEIOUaeiou]"

# define compress word function
def compress(word):
    pieces = re.findall(regexp, word)
    return "".join(pieces)

# get udhr words in english-latin1
english_udhr = nltk.corpus.udhr.words("English-Latin1")

# compress first 75 words 
print(nltk.tokenwrap(compress(w) for w in english_udhr[:75]))


# get Rotokas words
rotokas_words = nltk.corpus.toolbox.words("rotokas.dic")

# get consonant vowel pairs
cvs = [cv for w in rotokas_words for cv in re.findall(r"[ptksvr][aeiou]", w)]

# get conditional frequency distribution
cfd = nltk.ConditionalFreqDist(cvs)

# tabulate
cfd.tabulate()


# get consonant vowel word pairs
cv_word_pairs = [(cv, w) for w in rotokas_words
                         for cv in re.findall(r"[ptksvr][aeiou]", w)]


# get indices
cv_index = nltk.Index(cv_word_pairs)

# get words corresponding to 'su' and 'po'
print(cv_index["su"], "\n")
print(cv_index["[po]"], "\n")



# Finding Word Stems

# build up disjunction of all suffixes
print(re.findall(r"^(.*?)(ing|ly|ed|ious|ies|ive|es|s|ment)?$", "processing"))


# define stemming function
def stem(word):
    regexp = r"^(.*?)(ing|ly|ed|ious|ies|ive|es|s|ment)?$"
    stem, suffix = re.findall(regexp, word)[0]
    return stem

# raw text
raw = """DENNIS: Listen, strange women lying in ponds distributing swords
is no basis for a system of government.  Supreme executive power derives from
a mandate from the masses, not from some farcical aquatic ceremony."""

# tokenize raw text
tokens = word_tokenize(raw)

# stem text
print([stem(t) for t in tokens])




# Searching Tokenized Text

# get text
moby = nltk.Text(gutenberg.words("melville-moby_dick.txt"))

# find all token between "a" and "man"
print(moby.findall(r"<a> (<.*>) <man>"), "\n")


# get chat words
chat = nltk.Text(nps_chat.words())

# find three token phrases ending in "bro"
print(chat.findall(r"<.*> <.*> <bro>"), "\n")

# find phrases with three or more tokens all starting in "l"
print(chat.findall(r"<l.*>{3,}"), "\n")


#
word = "This is a test sentence to learn the properties of the re_show function."

# annotate string to show all places where pattern is matched
print(nltk.re_show("is", word), "\n")


# get hobbies and learned sections
hobbies_learned = nltk.Text(brown.words(categories=["hobbies", "learned"]))

# find all tokens of the form "x and other y's"
print(hobbies_learned.findall(r"<\w*> <and> <other> <\w*s>"), "\n")


print(hobbies_learned.findall(r"<as> <\w*> <as> <\w*>"), "\n")



# 3.6 Normalizing Text

# get data
raw = """DENNIS: Listen, strange women lying in ponds distributing swords
is no basis for a system of government.  Supreme executive power derives from
a mandate from the masses, not from some farcical aquatic ceremony."""

# get tokens
tokens = word_tokenize(raw)


# Stemmers

# get porter stemmer
porter = nltk.PorterStemmer()

# get lancaster stemmer
lancaster = nltk.LancasterStemmer()

# apply porter stemmer to tokens
print([porter.stem(t) for t in tokens], "\n")

# apply lancaster stemmer to tokens
print([lancaster.stem(t) for t in tokens], "\n")



# Lemmatization

# get WordNet lemmatizer
wnl = nltk.WordNetLemmatizer()

# print lemmatized words
print([wnl.lemmatize(t) for t in tokens])



# 3.7 Regular Expressions for Tokenizing Text

# Simple Approaches to Tokenization

raw = """'When I'M a Duchess,' she said to herself, (not in a very hopeful tone
though), 'I won't have any pepper in my kitchen AT ALL. Soup does very
well without--Maybe it's always pepper that makes people hot-tempered,'..."""

# tokenize on all whitespace characters
print(re.split(r"[ \t\n]+", raw), "\n")
print(re.split(r"\s+", raw), "\n")
print(re.split(r"\W+", raw), "\n")

# same tokens no empty string
print(re.findall(r"\w+", raw), "\n")

# find all tokens with words or non-space character followed by words
print(re.findall(r"\w+|\S\w*", raw), "\n")

# match hypendated and apostrophed words
print(re.findall(r"\w+(?:[-']\w+)*|'|[-.(]+|\S\w*", raw), "\n")


# new text
text = "That U.S.A. poster-print costs $12.40..."
pattern = r"""(?x)   # set flag to allow verbose regexps, string out embedded whitespace and comments
(?:[A-Z]\.)+         # abbreviations, e.g. U.S.A.
| \w+(?:-\w+)*       # words with optional internal hypens
| \$?\d+(?:\.\d+)?%? # currency and percentages, e.g. $12.40, 82%
| \.\.\.             # ellipsis
| [][.,;"'?():-_`]   # these are separate tokens; includes ], [
"""

# apply regexp_tokenize
print(nltk.regexp_tokenize(text, pattern), "\n")




# 3.8 Segmentation

# Sentence Segmentation

# average number of words per sentence in the Brown Corpus
print(len(nltk.corpus.brown.words())/len(nltk.corpus.brown.sents()), "\n")


# get text
text = nltk.corpus.gutenberg.raw("chesterton-thursday.txt")

# tokenize into sentences (using Punkt sentence segmentizer)
sents = nltk.sent_tokenize(text)

# print
pprint.pprint(sents[79:89])



# Word Segmentation

text = "doyouseethekittyseethedoggydoyoulikethekittylikethedoggy"

# segmentation of text (1 indicates utterance break after)
seg1 = "0000000000000001000000000010000000000000000100000000000"
seg2 = "0100100100100001001001000010100100010010000100010010000"

# segment text using segs
def segment(text, segs):
    words = []
    last = 0
    for i in range(len(segs)):
        if segs[i] == "1":
            words.append(text[last:i+1])  # append word determined by utterance break
            last = i+1
    words.append(text[last:])
    return words


# segment using seg1
print(segment(text, seg1), "\n")

# segment using seg2
print(segment(text, seg2), "\n")



# implement objective function

def evaluate(text, segs):
    words = segment(text, segs)  # segment text
    text_size = len(words)  # get number of words in segmented text
    lexicon_size = sum(len(word) + 1 for word in set(words))  # add length of word + 1 for each word in words
    return text_size + lexicon_size


 	

text = "doyouseethekittyseethedoggydoyoulikethekittylikethedoggy"
seg1 = "0000000000000001000000000010000000000000000100000000000"
seg2 = "0100100100100001001001000010100100010010000100010010000"
seg3 = "0000100100000011001000000110000100010000001100010000001"

# segment text using seg3
print(segment(text, seg3), "\n")

# evaluate on seg3
print(evaluate(text, seg3), "\n")

# evaluate on seg2
print(evaluate(text, seg2), "\n")

# evaluate on seg1
print(evaluate(text, seg1), "\n")



# flip value at position pos
def flip(segs, pos):
    return segs[:pos] + str(1-int(segs[pos])) + segs[pos+1:]


# flip n positions randomly
def flip_n(segs, n):
    for i in range(n):
        segs = flip(segs, randint(0, len(segs)-1))
    return segs


def anneal(text, segs, iterations, cooling_rate):

    # starting temperature
    temperature = float(len(segs))
    while temperature > 0.5:
        best_segs, best = segs, evaluate(text, segs)  # store best segmentation and best evaluation
        for i in range(iterations):
            guess = flip_n(segs, round(temperature))  # make random flip to search for improvement, number of flips proportional to temperature
            score = evaluate(text, guess)  # get score
            if score < best:
                best, best_segs = score, guess  # if improved store result
        score, segs = best, best_segs  # get score and segmentation
        temperature = temperature / cooling_rate  # update temperature
        print(evaluate(text, segs), segment(text, segs))  # get score and segmentation
    print()
    return segs




print(anneal(text, seg1, 5000, 1.2))





# 3.9 Formatting: From Lists to Strings

# From Lists to Strings
silly = ['We', 'called', 'him', 'Tortoise', 'because', 'he', 'taught', 'us', '.']

print(" ".join(silly))
print(";".join(silly))
print("".join(silly), "\n")


# Strings and Formats

fdist = nltk.FreqDist(["dog", "cat", "dog", "cat", "dog", "snake", "dog", "cat"])
for word in sorted(fdist):
    print("{}->{};".format(word, fdist[word]), end=" ")
print("\n")


print("from {1} to {0}".format("A", "B"), "\n")



# Lining Things Up

# numbers are right justified by default
# right-align format string given number of spaces
print("{:6}".format(41), "\n")

# left-align format string with given padding to left
print("{:<6}".format(41), "\n")

# strings are left justified by default
# add 6 spaces to the right of left-justified
print("{:6}".format("dog"), "\n")

# make word right-justified
print("{:>6}".format("dog"), "\n")

# format number to four digits and as float
print("{:.4f}".format(math.pi), "\n")

# format decimal as percentage
count, total = 3205, 9375
print("accuracy for {} words: {:.4%}".format(total, count/total), "\n")



# define tabulation function
def tabulate(cfdist, words, categories):
    print("{:16}".format("Category"), end=" ")  # print category and format space followed by " "
    for word in words:
        print("{:>6}".format(word), end=" ")   # print all words right aligned followd by " "
    print()  # print new line
    for category in categories:
        print("{:16}".format(category), end=" ")  # print category left aligned with spaces followed by " "
        for word in words:
            print("{:6}".format(cfdist[category][word]), end=" ")  # print conditional distribution values right aligned followed by " "
        print()  # print new line



cfd = nltk.ConditionalFreqDist(
    (genre, word)
    for genre in brown.categories()
    for word in brown.words(categories=genre)
)

genres = ["news", "religion", "hobbies", "science_fiction", "romance", "humor"]
modals = ["can", "could", "may", "might", "must", "will"]

tabulate(cfd, modals, genres)
print("\n")


# add with parameter in format
print("{:{width}}".format("Monty Python", width=15), "\n")




# Text Wrapping

# get list of words
saying = ['After', 'all', 'is', 'said', 'and', 'done', ',',
          'more', 'is', 'said', 'than', 'done', '.']

# get tokenized pieces
pieces = ["{} ({}),".format(word, len(word)) for word in saying]

# rejoin the pieces with space between
output = " ".join(pieces)

# wrap around to fit to line
wrapped = fill(output)

print(wrapped, "\n")

# print sring which does not separate counts by having no space in pieces
pieces2 = ["{}_({})".format(word, len(word)) for word in saying]
output2 = " ".join(pieces2)
wrapped2 = fill(output2)
print(wrapped2.replace("_", " "))










