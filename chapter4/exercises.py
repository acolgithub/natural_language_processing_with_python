



# 1.
help(str)
help(list)
help(tuple)



# 2.



# 3.
single = (1,)
single2 = tuple([1])
print(single)
print(single2, "\n")



# 4.
words = ["is", "NLP", "fun", "?"]
temp = words[0]
words[0] = words[1]
words[1] = temp
words[3] = "!"
print(words)

words2 = ["is", "NLP", "fun", "?"]
words2[0], words2[1], words2[3] = words2[1], words2[0], "!"
print(words2, "\n")



# 5.



# 6.
sent = ['The', 'dog', 'gave', 'John', 'the', 'newspaper']
[print([sent[i:i+n] for i in range(len(sent)-n + 1)]) for n in [1, len(sent)]]
print("\n")



# 7.
if 1:
    if 2.0:
        if "a":
            if ("3",):
                if print:
                    print("test", "\n")



# 8.
string1, string2 = "Monty", "Python"
print(string1 < string2)
string1, string2 = "Z", "a"
print(string1 < string2)
string1, string2 = "Monty", "Montague"
print(string1 < string2)
struc_string1, struc_string2 = ("Monty", 1), ("Monty", 2)
print(struc_string1 < struc_string2, "\n")



# 9.
input_string = "   test string   "
input_string2 = "\n\n   \r\ranother\rtest    "

# a.
print(" ".join(input_string.split()))
print(" ".join(input_string2.split()), "\n")

# b.
import re

print(re.findall(r"\s*(\S.*\S)\s*", input_string))
print(re.findall(r"\s*(\S.*\S)\s*", input_string2), "\n")



# 10.

def cmp_len(word1, word2):
    return True if len(word1) < len(word2) else False

def sort_words(word_arr):
    return sorted(word_arr, key=lambda x: len(x))

word_arr = ["a", "list", "of", "words", "for", "testing"]

print(sort_words(word_arr))
print(cmp_len(word_arr[1], word_arr[3]), "\n")



# 11.
sent1 = ["this", "is", "an", "array", "of", "words"]
sent2 = sent1
sent1[4] = "modified"
print(sent1)
print(sent2)

# a.
sent2 = sent1[:]
sent1[4] = "again"
print(sent1)
print(sent2)

# b.
text1 = [["This", "is", "a", "list", "to", "simulate", "text", "."], ["It", "has", "multiple", "sentences", "."]]
text2 = text1[:]
text1[1][1] = "Monty"
print(text1)
print(text2)
print(f"""text2 has its second component list modified since the inner lists have their object reference copied.
Only the outer arrays have entries copied.\n""")

# c.
from copy import deepcopy

sent1_new = ["this", "is", "an", "array", "of", "words"]
sent2_new = deepcopy(sent1_new)
sent1_new[2] = "new"
print(sent1_new)
print(sent2_new)



# 12.
n = 5
m = 6
word_table = [[''] * n] * m
word_table[1][2] = "hello"
print(word_table)
word_test = [""] * n
print(f"""Assigning to a specific entry replaces that with the replacement.
However, since we constructed the list of lists by multiplication then the corresponding entries of all inner lists are modified as they share an object reference.""")

new_word_table = [["" for i in range(n)] for j in range(m)]
new_word_table[1][2] = "hello"
print(new_word_table, "\n")



# 13.
def vowel_count(word):
    return len(re.sub("[^aeiou]", "", word))

n_max = 15
m_max = 15
word_vowels = [[set() for i in range(n_max)] for j in range(m_max)]
list_of_words = ["this", "is", "a", "list", "of", "words", "to", "process"]

for word in list_of_words:
    word_vowels[len(word)][vowel_count(word)].add(word)
print(word_vowels)
print("\n")


# 14.
def novel10(text):
    # get vocabulary of start and end portion of text
    text_start = set(text[:-int(0.1*len(text))])
    text_end = set(text[-int(0.1*len(text)):])

    # print set difference
    print(text_end.difference(text_start))

test = ["this", "is", "a", "test", "text", "to", "see", "if", "the", "function", "works"]
novel10(test)



# 15.
from nltk.probability import FreqDist

def word_freq_func(sent):
    # split sentence
    split_sent = sent.split()

    # remove punctuation
    split_sent = [word for word in split_sent if word.isalpha()]

    # initialize frequency distribution
    fdist = FreqDist()

    # add count to word
    for word in split_sent:
        fdist[word] += 1

    # print word and word frequency alphabetically
    print(*[(word, fdist[word]) for word in sorted(fdist)], sep="\n")


test_words = "this is a sentence to test the function on"
word_freq_func(test_words)
print("\n")



# 16.

# a.
letter_vals = {'a':1, 'b':2, 'c':3, 'd':4, 'e':5, 'f':80, 'g':3, 'h':8,
'i':10, 'j':10, 'k':20, 'l':30, 'm':40, 'n':50, 'o':70, 'p':80, 'q':100,
'r':200, 's':300, 't':400, 'u':6, 'v':6, 'w':800, 'x':60, 'y':10, 'z':7}

def gematria(word):
    return sum([letter_vals[letter] for letter in word])

print(gematria("word"), "\n")

# b.
from nltk.corpus import state_union

def is_ascii(word):
    return all(ord(c)<128 for c in word)

# iterate over file ids
for fileid in state_union.fileids():
    # get texts
    text = state_union.words(fileid)

    # remove punctuation, make lowercase, check if ascii
    text = [word.lower() for word in text if word.isalpha() and is_ascii(word)]

    print(f"{fileid} has {sum([1 for word in text if gematria(word)==666])} words of gematria 666\n")
print("\n")

# c.
import numpy as np

def decode(text, p=0.01):
    # remove punctuation, make lowercase, check if ascii
    text = [word.lower() for word in text if word.isalpha() and is_ascii(word)]

    # randomly replace a word
    text = [word if np.random.binomial(1, p=1-p)>0  else str(gematria(word)) for word in text]

    # rejoin text
    return " ".join(text)

print(decode(state_union.words("2006-GWBush.txt")), "\n")



# 17.

def shorten(text, n):
    # remove punctuation, make lowercase, check if ascii
    text = [word.lower() for word in text if word.isalpha() and is_ascii(word)]

    # make frequency distribution
    fdist = FreqDist()

    # add word counts
    for word in text:
        fdist[word] += 1

    # get common items
    print([word for word in fdist if (word, fdist[word]) not in fdist.most_common(n)])

test = ["text", "to", "test", "stuff"]
shorten(test,2)
shorten(state_union.words("2006-GWBush.txt"), 100)
print("\n")



# 18.



# 19.
from nltk.corpus import wordnet as wn

syns = [wn.synset("minke_whale.n.01"), wn.synset("orca.n.01"), wn.synset("novel.n.01"), wn.synset("tortoise.n.01")]
print(sorted(syns, key=lambda x: x.path_similarity(wn.synset("right_whale.n.01")), reverse=True))
print("\n")



# 20.
def dec_freq(word_list):
    # make frequency distribution
    fdist = FreqDist()

    for word in word_list:
        fdist[word] += 1

    return [word for word in sorted(fdist, key=lambda x: fdist[x], reverse=True)]

test_array = ["this", "is", "some", "words", "to", "test", "if", "this", "function", "is", "working"]
print(dec_freq(test_array), "\n")



# 21.
def text_not_vocab(text, vocab):
    return set(text).difference(set(vocab))

test_text = ["this", "is", "some", "test", "text"]
test_vocab = ["is", "some"]
print(text_not_vocab(test_text, test_vocab), "\n")



# 22.
from operator import itemgetter

words = ["some", "words", " to", "experiment", "with"]
print(sorted(words, key=itemgetter(1)))
print(sorted(words, key=itemgetter(-1)))
print(f"itemgetter is selecting a specific element of each word (i.e. like the second letter or last letter)\n")



# 23.
# def is_unique(trie, pref):
#     for c in pref:
#         trie = trie[c]
#     while len(trie)==1:
#         key = list(trie)[0]
#         if "value" in trie:
#             return trie["value"]
#         trie = trie[key]
#     return {}

def lookup(trie, key):
    if len(key) > 1:
        first = key[0]
        rest = key[1:]
        return lookup(trie[first], rest)
    elif len(key) > 0:
        return lookup(trie[key], {})
    else:
        while len(trie)==1:
            key = list(trie)[0]
            if "value" in trie:
                return trie["value"]
            trie = trie[key]
        return "Ambiguous key"

def insert(trie, key, value):
    if key:
        first, rest = key[0], key[1:]
        if first not in trie:
            trie[first] = {}
        insert(trie[first], rest, value)
    else:
        trie['value'] = value

trie = {}
insert(trie, 'chat', 'cat')
insert(trie, 'chien', 'dog')
insert(trie, 'chair', 'flesh')
insert(trie, 'chic', 'stylish')
trie = dict(trie)

print(lookup(trie,"chie"))
print("\n")



# 24.
import networkx as nx
import matplotlib.pyplot as plt
from nltk.corpus import shakespeare


play = shakespeare.xml("hamlet.xml")
for p in play:
    print("%s: %s" % (p.tag, list(p.itertext())))
    break




# 25.
from nltk.metrics.distance import edit_distance

print(edit_distance("the", "tehm"))
print(edit_distance("mill", "mite"))
print(f"""The Levenshtein Algorithm uses dynamic programming since it calculates distances on smaller initial substrings and saves the result.
The algorithm specifically makes use of the bottom-up approach since it stores all distances for smaller strings.""")
print("\n")



# 26.

# a.
def rec_cat_num(n):
    if n==0:
        return 1
    else:
        return_number = 0
        for i in range(1,n+1):
            return_number += rec_cat_num(i-1)*rec_cat_num(n-i)
        return return_number

# b.
def dyn_cat_num(n):
    lookup = {0:1}
    if n==0:
        return 1
    else:
        return_number = 0
        for i in range(1,n+1):
            if (i-1) not in lookup:
                lookup[i-1] = dyn_cat_num(i-1)
            if (n-i) not in lookup:
                lookup[n-i] = dyn_cat_num(n-i)
            return_number += lookup[i-1]*lookup[n-i]
        return return_number

# c.
# import time

# print(f"Recursive method")
# for i in range(20):
#     tick = time.time()
#     print(f"Catalan number {i}: {rec_cat_num(i)}, time: {time.time()-tick}")
# print("\n")

# Recursive method
# Catalan number 0: 1, time: 9.5367431640625e-07
# Catalan number 1: 1, time: 2.1457672119140625e-06
# Catalan number 2: 2, time: 4.5299530029296875e-06
# Catalan number 3: 5, time: 4.0531158447265625e-06
# Catalan number 4: 14, time: 1.0967254638671875e-05
# Catalan number 5: 42, time: 3.218650817871094e-05
# Catalan number 6: 132, time: 0.00013399124145507812
# Catalan number 7: 429, time: 0.0003490447998046875
# Catalan number 8: 1430, time: 0.0009024143218994141
# Catalan number 9: 4862, time: 0.0027430057525634766
# Catalan number 10: 16796, time: 0.008011579513549805
# Catalan number 11: 58786, time: 0.02374577522277832
# Catalan number 12: 208012, time: 0.06528377532958984
# Catalan number 13: 742900, time: 0.18634486198425293
# Catalan number 14: 2674440, time: 0.5939390659332275
# Catalan number 15: 9694845, time: 1.75821852684021
# Catalan number 16: 35357670, time: 5.290074110031128
# Catalan number 17: 129644790, time: 15.718015909194946
# Catalan number 18: 477638700, time: 48.73279523849487
# Catalan number 19: 1767263190, time: 146.44709038734436

# print(f"Dynamic method")
# for i in range(20):
#     tick = time.time()
#     print(f"Catalan number {i}: {dyn_cat_num(i)}, time: {time.time()-tick}")
# print("\n")

# Dynamic method
# Catalan number 0: 1, time: 9.298324584960938e-06
# Catalan number 1: 1, time: 8.106231689453125e-06
# Catalan number 2: 2, time: 9.5367431640625e-06
# Catalan number 3: 5, time: 1.0728836059570312e-05
# Catalan number 4: 14, time: 8.106231689453125e-06
# Catalan number 5: 42, time: 1.049041748046875e-05
# Catalan number 6: 132, time: 2.5987625122070312e-05
# Catalan number 7: 429, time: 3.409385681152344e-05
# Catalan number 8: 1430, time: 5.8650970458984375e-05
# Catalan number 9: 4862, time: 0.0001232624053955078
# Catalan number 10: 16796, time: 0.00025081634521484375
# Catalan number 11: 58786, time: 0.0005092620849609375
# Catalan number 12: 208012, time: 0.0015804767608642578
# Catalan number 13: 742900, time: 0.0017583370208740234
# Catalan number 14: 2674440, time: 0.003872394561767578
# Catalan number 15: 9694845, time: 0.006776332855224609
# Catalan number 16: 35357670, time: 0.016745567321777344
# Catalan number 17: 129644790, time: 0.046208858489990234
# Catalan number 18: 477638700, time: 0.10988211631774902
# Catalan number 19: 1767263190, time: 0.1560530662536621



# 27.



# 28.
# unable to find website



# 29.
# insert(trie, 'balle', 'ball')
# insert(trie, 'chat', 'cat')
# insert(trie, 'chat', 'cat')
# insert(trie, 'chien', 'dog')
# insert(trie, 'chair', 'flesh')
# insert(trie, 'chic', 'stylish')
# trie = dict(trie)

# def trie_print(trie):
#     dashes = ""
#     sorted_keys = sorted(trie)
#     for key in sorted_keys:
#         dashes += "-"
#         return key + trie_print(trie[key], initial)





# print(trie_print(trie))



# 30.
def uniq_pnt(trie, text):
    # process text
    text = [word.lower() for word in text.split() if word.isalpha()]

    # return list
    unique_points = []

    for word in text:
        if len(word) == 1:
            unique_points.append(word)
            continue
        for i in range(1, len(word)+1):
            if lookup(trie, word[:i]) != "Ambiguous key":
                unique_points.append(word[:i])
                break

    return unique_points

print(uniq_pnt(trie, "chair chat chic chien"))
print("\n")



# 31.



# 32.
from nltk.tokenize import word_tokenize, sent_tokenize

def highest_freq(text, n):
    # get words and sentences
    text_sent = sent_tokenize(text)

    # max sum sentence
    sent_freq = FreqDist()

    for i in range(len(text_sent)):
        # get sentence words
        sent_words = word_tokenize(text_sent[i])

        # get word frequencies
        fdist = FreqDist()

        for word in sent_words:
            if word.isalpha():
                fdist[word.lower()] += 1
        cur_sum = sum([fdist[word.lower()] for word in sent_words if word.isalpha()])

        # assign frequency sum and document position starting from 1
        sent_freq[text_sent[i]] = (cur_sum, i+1)
    
    # get highest ranking sentences in document order (ordered first by freq sum, extract only top n, then rearrange to document order)
    ranked_sents = sorted(sent_freq.most_common(n), key=lambda x: x[1][1])
    
    # return only sentences
    return [s for s, _ in ranked_sents]


print(*highest_freq("This is a test sentence. Perhaps one more sentence to be certain. So is this. This is a second sentence. Just one more sentence to test.", 3), sep="\n")
print("\n")



# 33.

























