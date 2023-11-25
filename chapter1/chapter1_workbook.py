# nltk
import nltk
from nltk.book import *

# matplotlib
import matplotlib.pyplot as plt


# functions
def lexical_diversity(text, as_percent=False):
	return 100 * len(set(text)) / len(text) if as_percent else len(set(text))/len(text)

def word_proportion(text, word, as_percentage=False):
	return 100 * text.count(word)/len(text) if as_percentage else text.count(word)/len(text)


# get data from nltk
# nltk.download()  # use this interface to download online corpora for book
# in order to download you type "d" and then indicate the identifier you are interested in 


# 1 Computing with Language: Texts and Words

# 1.3 Searching Text

# information about text1
print(f"text 1 information: {text1}")

# information about text2
print(f"text 2 information: {text2}")
print("\n")

# get all occurrences of a word with some context from a text
text1.concordance('monstrous')
print("\n")

# get all occurrences of "affection" in the text "Sense and Sensibility"
text2.concordance("affection")
print("\n")

# get all occurrences of "lived" from "the book of Genesis" (helps check how long people lived)
text3.concordance("lived")
print("\n")

# get all occurrences of "nation", "terror", and "god" from "Inaugural Address Corpus"
words = ["nation", "terror", "god"]
for w in words:
	text4.concordance(w)
	print("\n")


# get all occurrences of "im", "ur", and "lol" from "NPS Chat Corpus"
words = ["im", "ur", "lol"]
for w in words:
	text5.concordance(w)
	print("\n")


# get words that appear in a similar range of contexts
text1.similar("monstrous")
print("\n")
text2.similar("monstrous")
print("\n")


# get contexts shared by two or more words
text2.common_contexts(["monstrous", "very"])
print("\n")

# get words with similar context to "happy" in "Mobdy Dick"
text1.similar("happy")
print("\n")
text1.common_contexts("happy", "interesting")
print("\n")


# get dispersion plot to see distribution of occurrences
text4.dispersion_plot(["citizens", "democracy", "freedom", "duties", "America", "liberty", "constitution"])
plt.savefig("figures/dispersion_plot_inaugural_address.png")
plt.close()

# distribution of "whale" and "kill" in "Moby Dick"
text1.dispersion_plot(["whale", "kill"])
plt.savefig("figures/dispersion_plot_moby_dick.png")
plt.close()


# generate random text in specific style
text3.generate()
print("\n")



# 1.4 Counting Vocabulary

# get number of words and punctuation symbols in text
print(len(text3), "\n")

# get the vocabulary (unique words) of a text
print(set(text3), "\n")

# sort the vocabulary of a text
print(sorted(set(text3)), "\n")

# get amount of vocabulary in text
print(len(set(text3)), "\n")

# get lexical richness of text by looking at proportion of distinct words to total words
print(f"lexical richness of text3: {len(set(text3))/len(text3)}\n")

# get count of specific word
print(f"number of occurrences of 'smote' in text3: {text3.count('smote')}\n")  # number of occurrences of smote
print(f"percentage of times 'a' occurs in text4: {100 * text4.count('a')/len(text4)}\n")

# number of occurences and percentage of 'lol' in text 5
lol_count = text5.count('lol')
print(f"number of occurrences of 'lol' in text5: {lol_count}\n")
print(f"percentage of times 'lol' occurs in text5: {100 * lol_count/len(text5)}\n")


# function test
print(f"lexical diversity of text3: {lexical_diversity(text3)}")
print(f"lexical diversity of text5: {lexical_diversity(text5)}")
print(f"word proportion 'a' in text4: {word_proportion(text4, 'a', True)}\n")



# 2 A Closer Look at Python: Texts as Lists of Words

# 2.1 Lists

sent1 = ["Call", "me", "Ishmael", "."]
print(sent1)
print(f"length of sent1: {len(sent1)}")
print(f"lexical diversity of sent1: {lexical_diversity(sent1)}\n")

print(sent2, "\n", sent3, "\n")

ex1 = ["Monty", "Python", "and", "the", "Holy", "Grail"]
print(ex1)
print(sorted(ex1), "\n", len(set(ex1)), "\n", ex1.count('the'))

# concatenate list
print(['Monthy', 'Python'] + ['and', 'the', 'Holy', 'Grail'], '\n')

print(sent4 + sent1)

# append to list
sent1.append("Some")
print(sent1)


# 2.2 Indexing Lists

# get specific entry
print(text4[173], "\n")

# get index of first entry of word in text
print(text4.index("awaken"), "\n")

# get slice of text
print(text5[16715:16735], "\n")
print(text6[1600:1625], "\n")



# 3 Computing with Language: Simple Statistics

# 3.1 Frequency Distributions

# get frequency distribution of text
fdist1 = FreqDist(text1)
print(fdist1, "\n")  # gives information about number of words

print(f"get most common 50 words:\n{fdist1.most_common(50)}\n")

# get occurences of 'whale' in Moby Dick
print(f"occurrences of 'whale' in Moby Dick: {fdist1['whale']}\n")


fdist2 = FreqDist(text2)
print(f"get most common 50 words in text 2:\n{fdist2.most_common(50)}\n")



# generate cumulative frequency plot to analyze 50 most common words in first text
fdist1.plot(50, cumulative=True)

# get words that only occur once (i.e. unique words or hapaxes) from frequency distribution
# print(f"hapaxes of Mobdy Dick:\n{fdist1.hapaxes()}")  # large list, only uncomment if you wish to view it


# 3.2 Fine-grained Selection of Words

# select long words
V = set(text1)
long_words = [w for w in V if len(w) > 15]
print(f"sorted list of long words in Moby Dick:\n{sorted(long_words)}\n")


# select long words in text4 and text5
long_words_text4 = [w for w in set(text4) if len(w) > 15]
print(f"sorted list of long words in text4:\n{sorted(long_words_text4)}\n\n")
#long_words_text5 = [w for w in set(text5) if len(w) > 15]
#print(f"sorted list of long words in text5:\n{sorted(long_words_text5)}\n")


# select frequently occuring moderately long words from text5
fdist5 = FreqDist(text5)
print(f"frequently occuring words of moderate length:\n{sorted(w for w in set(text5) if len(w) > 7 and fdist5[w] > 7)}\n")




# 3.3 Collocations and Bigrams

# get bigrams (word pairs) from a list of words
print(f"collection of bigrams:\n{list(bigrams(['more', 'is', 'said', 'than', 'done']))}\n")

# get collocations in text4 and text8
print("collocations in text 4:\n")
print(f"{text4.collocations()}\n")
print("collocations in text 8:\n")
print(f"{text8.collocations()}\n")




# 3.4 Counting Other Things

# create frequency distribution of word lengths in text
fdist = FreqDist([len(w) for w in text1])
print(fdist)

# get most common word lengths
print(fdist.most_common())

# get word length with most common length
print(fdist.max())

# get number of words of length 3
print(fdist[3])

# get word frequency as a proportion
print(fdist.freq(3))



# 5. Automatic Natural Language Understanding
print(nltk.chat.chatbots())



























