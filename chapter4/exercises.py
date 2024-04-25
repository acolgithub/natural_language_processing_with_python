



# Q1: Find out more about sequence objects using Python's help facility. In the interpreter, type help(str),
#     help(list), and help(tuple). This will give you a full list of the functions supported by each type. Some
#     functions have special names flanked with underscore; as the help documentation shows, each such
#     function corresponds to something more familiar. For example x.__getitem__(y) is just a long-winded way
#     of saying x[y].

help(str)
help(list)
help(tuple)



# Q3: Find out how to create a tuple consisting of a single item. There are at least two ways to do this.

single = (1,)
single2 = tuple([1])
print(single)
print(single2, "\n")



# Q4: Create a list words = ['is', 'NLP', 'fun', '?']. Use a series of assignment statements (e.g.
#     words[1] = words[2]) and a temporary variable tmp to transform this list into the list
#     ['NLP', 'is', 'fun', '!']. NOw do the same transformation using tuple assignment.

words = ["is", "NLP", "fun", "?"]
temp = words[0]
words[0] = words[1]
words[1] = temp
words[3] = "!"
print(words)

words2 = ["is", "NLP", "fun", "?"]
words2[0], words2[1], words2[3] = words2[1], words2[0], "!"
print(words2, "\n")



# Q6: Does the method for creating a sliding window of n-grams behave correctly for the two limiting cases: n
#     = 1, and n=len(sent)?

sent = ['The', 'dog', 'gave', 'John', 'the', 'newspaper']
[print([sent[i:i+n] for i in range(len(sent)-n + 1)]) for n in [1, len(sent)]]
print("\n")



# Q7: We pointed out that when empty strings and empty lists occur in the condition part of an if clause, they
#     evaluate to False. In this case, they are said to be occurring in a Boolean context. Experiment with different
#     kind of non-Boolean expressions in Boolean contexts, and see whether they evaluate as True or False.

if 1:
    if 2.0:
        if "a":
            if ("3",):
                if print:
                    print("test", "\n")



# Q8: Use the inequality operators to compare strings, e.g. 'Monty' < 'Python'. What happens when you do
#     'Z' < 'a'? Try pairs of strings which have a common prefix, e.g. 'Monty' < 'Montague'. Read up on
#     "lexicographical sort" in order to understand what is going on here. Try comparing structured objects, e.g.
#     ('Monty', 1) < ('Monty', 2). Does this behave as expected?

string1, string2 = "Monty", "Python"
print(string1 < string2)
string1, string2 = "Z", "a"
print(string1 < string2)
string1, string2 = "Monty", "Montague"
print(string1 < string2)
struc_string1, struc_string2 = ("Monty", 1), ("Monty", 2)
print(struc_string1 < struc_string2, "\n")



# Q9: Write code that removes whitespace at the beginning and end of a string, and normalizes whitespace
#     between words to be a single space character.

input_string = "   test string   "
input_string2 = "\n\n   \r\ranother\rtest    "

# 1. do this task using split() and join()

print(" ".join(input_string.split()))
print(" ".join(input_string2.split()), "\n")

# 2. do this task using regular expression substitutions

import re

print(re.findall(r"\s*(\S.*\S)\s*", input_string))
print(re.findall(r"\s*(\S.*\S)\s*", input_string2), "\n")



# Q10: Write a program to sort words by length. Define a helper function cmp_len which uses the cmp
#      comparison function on word lengths.

def cmp_len(word1, word2):
    return True if len(word1) < len(word2) else False

def sort_words(word_arr):
    return sorted(word_arr, key=lambda x: len(x))

word_arr = ["a", "list", "of", "words", "for", "testing"]

print(sort_words(word_arr))
print(cmp_len(word_arr[1], word_arr[3]), "\n")



# Q11: Create a list of words and store it in a variable sent1. Now assign sent2 = sent1. Modify one of the
#      items in sent1 and verify that sent2 has changed.

sent1 = ["this", "is", "an", "array", "of", "words"]
sent2 = sent1
sent1[4] = "modified"
print(sent1)
print(sent2)

# a: Now try the same exercise but instead assign sent 2 = sent1[:]. Modify sent1 again and see what
#    happens to sent2. Explain

sent2 = sent1[:]
sent1[4] = "again"
print(sent1)
print(sent2)

# b: Now define text1 to be a list of lists of strings (e.g. to represent a text consisting of multiple
#    sentences.) Now assign text2 = text1[:], assign a new value to one of the words, e.g.
#    text[1][1] = 'Monty'. Check what this did to text2. Explain.

text1 = [["This", "is", "a", "list", "to", "simulate", "text", "."], ["It", "has", "multiple", "sentences", "."]]
text2 = text1[:]
text1[1][1] = "Monty"
print(text1)
print(text2)
print(f"""text2 has its second component list modified since the inner lists have their object reference copied.
Only the outer arrays have entries copied.\n""")

# c: Load Python's deepcopy() function (i.e. from copy import deepcopy), consult its documentation, and
#    test that it makes a fresh copy of any object.

from copy import deepcopy

sent1_new = ["this", "is", "an", "array", "of", "words"]
sent2_new = deepcopy(sent1_new)
sent1_new[2] = "new"
print(sent1_new)
print(sent2_new)



# Q12: Initialize an n-by-m list of lists of empty strings using list multiplication, e.g.
#      word_table = [[''] * n] * m. What happens when you set one of its values, e.g.
#      word_table[1][2] = "hello"? Explain why this happens. Now write an expression using range() to
#      construct a list of lists, and show that it does not have this problem.

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



# Q13: Write code to initialize a two-dimensional array of sets called word_vowels and process a list of words,
#      adding each word to word_vowels[l][v] where l is the length of the word and v is the number of vowels it
#      contains.

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


# Q14: Write a function novel1o(text) that prints any word that appeared in the last 10% of a text that had not
#      been encountered earlier.

def novel10(text):
    # get vocabulary of start and end portion of text
    text_start = set(text[:-int(0.1*len(text))])
    text_end = set(text[-int(0.1*len(text)):])

    # print set difference
    print(text_end.difference(text_start))

test = ["this", "is", "a", "test", "text", "to", "see", "if", "the", "function", "works"]
novel10(test)



# Q15: Write a program that takes a sentence expressed as a single string, splits it and counts up the words. Get
#      it to print out each word and the word's frequency, one per line, in alphabetical order.

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



# Q16: Read up on Gematria, a method for assigning numbers to words, and for mapping between words having
#      the same number to discover the hidden meaning of texts (http://en.wikipedia.org/wiki/Gematria,
#      http://essenes.net/gemcal.htm).

# a: Write a function gematria() that sums the numerical values of the letters of a word, according to the
#    letter values in letter_vals:
#
#    >>> letter_vals = {'a':1, 'b':2, 'c':3, 'd':4, 'e':5, 'f':80, 'g':3, 'h':8,
#    ... 'i':10, 'j':10, 'k':20, 'l':30, 'm':40, 'n':50, 'o':70, 'p':80, 'q':100,
#    ... 'r':200, 's':300, 't':400, 'u':6, 'v':6, 'w':800, 'x':60, 'y':10, 'z':7}

letter_vals = {'a':1, 'b':2, 'c':3, 'd':4, 'e':5, 'f':80, 'g':3, 'h':8,
'i':10, 'j':10, 'k':20, 'l':30, 'm':40, 'n':50, 'o':70, 'p':80, 'q':100,
'r':200, 's':300, 't':400, 'u':6, 'v':6, 'w':800, 'x':60, 'y':10, 'z':7}

def gematria(word):
    return sum([letter_vals[letter] for letter in word])

print(gematria("word"), "\n")

# b: Process a corpus (e.g. nltk.corpus.state_union) and for each document, count how many of its
#    words have the number 666.

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

# c: Write a function decode() to process a text, randomly replacing words with their Gematria
#    equivalents, in order to discover the "hidden meaning" of the text.

import numpy as np

def decode(text, p=0.01):
    # remove punctuation, make lowercase, check if ascii
    text = [word.lower() for word in text if word.isalpha() and is_ascii(word)]

    # randomly replace a word
    text = [word if np.random.binomial(1, p=1-p)>0  else str(gematria(word)) for word in text]

    # rejoin text
    return " ".join(text)

print(decode(state_union.words("2006-GWBush.txt")), "\n")



# Q17: Write a function shorten(text, n) to process a text, omitting the n most frequently occuring words of
#      the text. How readable is it?

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



# Q19: Write a list comprehension that sorts a list of WordNet synsets for proximity to a given synset. For
#      example, given the synsets minke_whale.n.01, orca.n.01, novel.n.01, and tortoise.n.01, sort them
#      according to their shortest_path_distance() from right_whale.n.01.

from nltk.corpus import wordnet as wn

syns = [wn.synset("minke_whale.n.01"), wn.synset("orca.n.01"), wn.synset("novel.n.01"), wn.synset("tortoise.n.01")]
print(sorted(syns, key=lambda x: x.path_similarity(wn.synset("right_whale.n.01")), reverse=True))
print("\n")



# Q20: Write a function that takes a list of words (containing duplicates) and returns a list of words (with no
#      duplicates) sorted by decreasing frequency. E.g. if the input list contained 10 instances of the word table
#      and 9 instances of the word chair, then table would appear before chair in the output list.

def dec_freq(word_list):
    # make frequency distribution
    fdist = FreqDist()

    for word in word_list:
        fdist[word] += 1

    return [word for word in sorted(fdist, key=lambda x: fdist[x], reverse=True)]

test_array = ["this", "is", "some", "words", "to", "test", "if", "this", "function", "is", "working"]
print(dec_freq(test_array), "\n")



# Q21: Write a function that takes a text and a vocabulary as its arguments and returns the set of words that
#      appear in the text but not in the vocabulary. Both arguments can be represented as lists of strings. Can you
#      do this in a single line, using set.difference()?

def text_not_vocab(text, vocab):
    return set(text).difference(set(vocab))

test_text = ["this", "is", "some", "test", "text"]
test_vocab = ["is", "some"]
print(text_not_vocab(test_text, test_vocab), "\n")



# Q22: Import the itemgetter() function from the operator module in Python's standard library (i.e.
#      from operator Import itemgetter). Create a list words containing several words. Now try calling:
#      sorted(words, key=itemgetter(1)), and sorted(words, key=itemgetter(-1)). Explain what itemgetter()
#      is doing.

from operator import itemgetter

words = ["some", "words", " to", "experiment", "with"]
print(sorted(words, key=itemgetter(1)))
print(sorted(words, key=itemgetter(-1)))
print(f"itemgetter is selecting a specific element of each word (i.e. like the second letter or last letter)\n")



# Q23: Write a recursive function lookup(trie, key) that looks up a key in a trie, and returns the value it finds.
#      Extend the function to return a word when it is uniquely determined by its prefix (e.g. vanguard is the only
#      word that starts with vang-, so lookup(trie, 'vang') should return the same thing as
#      lookup(trie, 'vanguard')).

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



# Q24: Read up on "keyword linkage" (chapter 5 of (Scott & Tribble, 2006)). Extract keywords from NLTK's
#      Shakespeare Corpus and using the NetworkX package, plot keyword linkage networks.

import networkx as nx
import matplotlib.pyplot as plt
from nltk.corpus import shakespeare


play = shakespeare.xml("hamlet.xml")
for p in play:
    print("%s: %s" % (p.tag, list(p.itertext())))
    break




# Q25: Read about string edit distance and the Levenshtein Algorithm. Try the implementation provided in
#      nltk.edi_distance(). In what way is this using dynamic programming? Does it use the bottom-up or top-
#      down approach? [See also http://norvig.com/spell-correct.html]

from nltk.metrics.distance import edit_distance

print(edit_distance("the", "tehm"))
print(edit_distance("mill", "mite"))
print(f"""The Levenshtein Algorithm uses dynamic programming since it calculates distances on smaller initial substrings and saves the result.
The algorithm specifically makes use of the bottom-up approach since it stores all distances for smaller strings.""")
print("\n")



# Q26: The Catalan numbers arise in many applications of combinatorial mathematics, including the counting of
#      parse trees (6). The series can be defined as follows: C0 = 1, and Cn+1 = Σ0..n (CiCn-i).

# a: Write a recursive function to compute nth Catalan number Cn.

def rec_cat_num(n):
    if n==0:
        return 1
    else:
        return_number = 0
        for i in range(1,n+1):
            return_number += rec_cat_num(i-1)*rec_cat_num(n-i)
        return return_number

# b: Now write another function that does this computation using dynamic programming.

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

# c: Use the timeit module to compare the performance of these functions as n increases

# import time

# print(f"Recursive method")
# for i in range(20):
#     tick = time.time()
#     print(f"Catalan number {i}: {rec_cat_num(i)}, time: {time.time()-tick}")
# print("\n")


# Results:

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



# Q29: Write a recursive function that pretty prints a trie in alphabetically sorted order, e.g.:
#
#      chair: 'flesh'
#      ---t: 'cat'
#      --ic: 'stylish'
#      ---en: 'dog'

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



# Q30: With the help of the trie data structure, write a recursive function that processes text, locating the
#      uniqueness point in each word, and discarding the remainder of each word. How much compression does
#      this give? How readable is the resulting text?

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



# Q32: Develop a simple extractive summarization tool, that prints the sentence of a document which contain
#      the highest total word frequency. Use FreqDist() to count word frequencies, and use sum to sum the
#      frequencies of the words in each sentence. Rank the sentences according to their score. Finally, print the n
#      highest-scoring sentences in document order. Carefully review the design of your program, especially your
#      approach to this double sorting. Make sure the program is written as clearly as possible.

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



