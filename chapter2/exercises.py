import nltk
from nltk.corpus import gutenberg
from nltk.corpus import brown
from nltk.corpus import webtext
from nltk.corpus import state_union
from nltk.corpus import wordnet as wn
from nltk.corpus import names
from nltk.probability import FreqDist
from nltk.corpus import cmudict
from nltk.corpus import stopwords
from nltk.util import bigrams
from nltk.corpus import udhr

import pandas as pd
import matplotlib.pyplot as plt
import random

# Q1: Create a variable 'phrase' containing a list of words. Review the operations described in the previous
#     chapter, including addition, multiplication, indexing, slicing, and sorting.

phrase = ["This", "is", "a", "list", "of", "words"]
print(f"Addition: {phrase + ['.']}")  # apply list addition
print(f"Multiplication: {2*phrase}")  # apply list multiplication
print(f"Indexing: {phrase[3]}")  # apply indexing
print(f"Slicing: {phrase[0][1:3]}")  # applying slicing
print(f"Sorting: {sorted(phrase)}\n")  # apply sorting



# Q2: Use the corpus module to explore 'austen-persuasion.txt'. How many word tokens does this book have?
#     How many word types?

# get number of tokens in text
austen_text = gutenberg.words("austen-persuasion.txt")
print(f"Austen Persuasion has {len(austen_text)} word tokens.\n")

# get number of word types
types = [a for a in set([d for (w, d) in nltk.pos_tag(austen_text)]) if a.isalpha()]
print(f"Number of word types: {len(types)}")
print(types, "\n")

# question probably meant this
print(f" Austen Persuasion has {len(set(austen_text))} word types.\n")



# Q3: Use the Brown corpus reader nltk.corpus.brown.words() or the Web text corpus reader
#     nltk.corpus.webtext.words() to access some sample text in two different genres.

# print categories and text in specific categories
print(brown.categories(), "\n")
print(brown.words(categories="adventure"), "\n")
print(brown.words(categories="mystery"), "\n")



# Q4: Read in the texts of the State of the Union addresses, using the state_union corpus reader. Count
#     occurrences of men, women, and people in each document. What has happened to the usage of these words
#     over time?

# get plot of counts of men, women, and people in state of union address against time
cfd = nltk.ConditionalFreqDist(
    (word_of_interest, file[:4])
    for word in ["men", "women", "people"]
    for file in nltk.corpus.state_union.fileids()
    for word_of_interest in [w for w in nltk.corpus.state_union.words(file) if w == word]
)

cfd.plot()



# Q5: Investigate the holonym-meronym relations for some nouns. Remember that there are three kinds of
#     holonym-meronym relation, so you need to use: member_meronyms(), part_meronyms(),
#     substance_meronyms(), member_holonyms(), part_holonyms(), and substance_holonyms().

# print synsets, definitions, meronyms, and holonyms
noun = "orange"
print(wn.synsets(noun))
print(wn.synset("orange.n.01").definition())
print(wn.synset("orange.n.01").part_meronyms())
print(wn.synset("orange.n.01").substance_meronyms())
print(wn.synset("orange.n.01").member_meronyms())
print(wn.synset("orange.n.01").part_holonyms())
print(wn.synset("orange.n.01").substance_holonyms())
print(wn.synset("orange.n.01").member_holonyms(), "\n")

noun = "television"
print(wn.synsets(noun))
print(wn.synset("television.n.01").definition)
print(wn.synset("television.n.01").part_meronyms())
print(wn.synset("television.n.01").substance_meronyms())
print(wn.synset("television.n.01").member_meronyms())
print(wn.synset("television.n.01").part_holonyms())
print(wn.synset("television.n.01").substance_holonyms())
print(wn.synset("television.n.01").member_holonyms(), "\n")
    
noun = "cloth"
print(wn.synsets(noun))
print(wn.synset("cloth.n.01").definition)
print(wn.synset("cloth.n.01").part_meronyms())
print(wn.synset("cloth.n.01").substance_meronyms())
print(wn.synset("cloth.n.01").member_meronyms())
print(wn.synset("cloth.n.01").part_holonyms())
print(wn.synset("cloth.n.01").substance_holonyms())
print(wn.synset("cloth.n.01").member_holonyms(), "\n")



# Q6: In the discussion of comparative wordlists, we created an object called translate which you could look
#     up using words in both German and Spanish in order to get corresponding words in English. What problem
#     might arise with this approach? Can you suggest a way to avoid this problem?

print(f"A problem could arise if the word has meaning in both German and Spanish (which could be different).\nWe could add a requirement that we specify which language we want it translated from.")



# Q7: According to Strunk and White's 'Elements of Style', the word 'however', used at the start of a sentence,
#     means "in whatever way" or "to whatever extent", and not "nevertheless". They give this example of correct
#     usage: However you advise him, he will probably do as he thinks best.
#     (http://www.bartleby.com/141/strunk3.html) Use the concordance tool to study actual usage of this word in
#     the various texts we have been considering. See also the LanguageLog posting "Fossilized prejudices about
#     'however' at http://itre.cis.upenn.edu/~myl/languagelog/archives/001913.html"

# look at usage of 'However' at the start of a sentence
print(gutenberg.fileids())
nltk.Text(gutenberg.words("austen-emma.txt")).concordance("However")
print("\n")



# Q8: Define a conditional frequency distribution over the Names corpus that allows you to see which initial
#     letteres are more frequent for males vs. females (cf. 4.4).

# plot male/female against first initial
cfd = nltk.ConditionalFreqDist(
    sorted((filename[:-4], name[0])
    for filename in names.fileids()
    for name in names.words(filename))
)

cfd.plot()



# Q9: Pick a pair of texts and study the difference between them, in terms of vocabulary, vocabulary richness,
#     genre, etc. Can you find pairs of words which have quite different meanings across the two texts, such as
#     monstous in Moby Dick and in Sense and Sensibility?

# look at differences between texts in terms of vocabulary and vocabulary richness
print(gutenberg.fileids())
print("\n")
moby_dick_text = nltk.Text(gutenberg.words("melville-moby_dick.txt"))
austen = nltk.Text(gutenberg.words("austen-sense.txt"))
hamlet = nltk.Text(gutenberg.words("shakespeare-hamlet.txt"))

moby_dick_vocabulary = len(set(moby_dick_text))/len(moby_dick_text)
austen_vocabulary = len(set(austen))/len(austen)
print(f"Amount of vocabulary in Moby Dick: {len(set(moby_dick_text))}")
print(f"Amount of vocabulary in Austen: {len(set(austen))}")
print(f"Vocabulary richness precentage of Moby Dick: {moby_dick_vocabulary}")
print(f"Vocabulary richness percentage of Austen: {austen_vocabulary}")
print("\n")



# Q10: Read the BBC News article: UK's Vicky Pollards 'left behind'
#      http://news.bbc.co.uk/1/hi/education/6173441.stm. The article gives the following statistic about teen
#      language: "the top 20 words used, including yeah, no, but and like, account for around a third of all words."
#      How many word types account for a third of all word tokens, for a variety of text sources? What do you
#      conclude about this statistic? Read more about this on LanguageLog, at
#      http://itre.cis.upenn.edu/~myl/languagelog/archives/003993.html.

# get frequency distribution of words in Moby Dick
moby_dick_fd = FreqDist(word.lower() for word in moby_dick_text if word.isalpha())
moby_dick_fd.plot(20, percents=True, cumulative=True)  # plot cumulative percentage for top 20 words


# get frequency distribution of words in Austen
austen_fd = FreqDist(word.lower() for word in austen_text if word.isalpha())
austen_fd.plot(20, percents=True, cumulative=True)  # plot cumulative percentage for top 20 words

hamlet_fd = FreqDist(word.lower() for word in hamlet if word.isalpha())
hamlet_fd.plot(20, percents=True, cumulative=True)  # plot cumulative percentage for top 20 words
print(f"It appears that not many words make up most word tokens.")
print(f"In particular, 20 words seems enough to get a third of words in a text.")



# Q11: Investigate the table of modal distributions and look for other patterns. Try to explain them in terms of
#      your own impressionistic understanding of the different genres. Can you find other closed classes of words
#      that exhibit significant differences across different genres? 

# get modals
modals = ["can", "could", "may", "might", "must", "will"]

# get conditional frequency distribution of genre against modal
modal_cfd = nltk.ConditionalFreqDist(
    (genre, modal)
    for genre in brown.categories()
    for modal in brown.words(categories=genre)
)

# create table
modal_cfd.tabulate(conditions=brown.categories(), samples=modals)
print(f"Most common modal in fiction is 'could'. This could be because fiction appeals to imagination.")
print(f"Most common modal in news is 'will'. This coud be because news intends to predict what will happen.")

# get questions
w_questions = ["what", "where", "when", "why", "how"]

# get conditional frequency distribution of genre against questions
w_cfd = nltk.ConditionalFreqDist(
    (genre, w_ques)
    for genre in brown.categories()
    for w_ques in brown.words(categories=genre)
)

# create table
w_cfd.tabulate(conditions=brown.categories(), samples=w_questions)
print(f"Distribution of 'w' questions seems to distinguish genres.", "\n")



# Q12: The CMU Pronouncing Dictionary contains multiple pronounciations for certain words. How many distinct
#      words does it contain? What fraction of words in this dictionary have more than one possible pronounciation?

# find fraction of words in cmu pronouncing dictionary with more than one possible pronounciation
print(f"The cmu dictionary has {len(cmudict.entries())} distinct words.")
print(f"{100*len([word for word, pron in cmudict.entries() if len(pron)>1])/len(cmudict.entries())}% of the words in the cmu dictionary have more than one possible pronounciation.\n")



# Q13: What percentage of noun synsets have no hyponyms? You can get all noun synsets using
#      wn.all_synsets('n').

# get all nouns
all_nouns = [noun for noun in wn.all_synsets("n")]
nouns_no_hyponyms = [noun for noun in all_nouns if len(noun.hyponyms())==0]  # get nouns with no hyponyms
print(f"Percentage of noun synsets with no hyponyms: {100*len(nouns_no_hyponyms)/len(all_nouns)}%.\n")  # get percentage of nouns with no hyponyms



# Q14: Define a function supergloss(s) that takes a synset s as its argument and returns a string consisting of the
#      concatentation of the definition of s, and the definitions of all the hypernyms and hyponyms of s.

# takes in synset and provides concatenation of its defiition and the definition of its hypernyms and hyponyms
def supergloss(s):
    s_defs = "Definitions:\n"
    for syn in s:
        s_defs += "-" + syn.definition() + "\n\n"
    s_defs += "\nHypernym Definitions:\n"
    for syn in s:
        for syn_hyper in syn.hypernyms():
            s_defs += "-" + syn_hyper.definition() + "\n\n"
    s_defs += "\nHyponym Definitions:\n"
    for syn in s:
        for syn_hypo in syn.hyponyms():
            s_defs += "-" + syn_hypo.definition() + "\n\n"
    return s_defs


print(supergloss(wn.synsets("car")))



# Q15: Write a program to find all words that occur at least three times in the Brown Corpus.

# count all words that occur at least there times in the Brown corpus
def brown_corpus_three():
    brown_corpus = nltk.corpus.brown.words()
    brown_fd = FreqDist(brown_corpus)
    return set(word for word in brown_corpus if brown_fd[word] >= 3)

print(brown_corpus_three(), "\n")



# Q16: Write a program to generate a table of lexical diversity scores (i.e. token/type ratios), as we saw in 1.1.
#      Include the full set of Brown Corpus genres (nltk.corpus.brown.categories()). Which genre has the lowest
#      diversity (greatest number of tokens per type)? Is this what you would have expected?

genres = nltk.corpus.brown.categories()  # get genres from brown corpus
tokens = [len(brown.words(categories=genre)) for genre in genres]  # get lengths of text from each genre
types = [len(set(brown.words(categories=genre))) for genre in genres]  # get vocabulary of text from each genre
lex_diversity = [len(brown.words(categories=genre))/len(set(brown.words(categories=genre))) for genre in genres]  # lexical diversity for each genre

# create list containing each genre and numerical information
paired_list = [[genres[i], tokens[i], types[i] , lex_diversity[i]] for i in range(len(genres))]

# make dataframe
genres_df = pd.DataFrame(paired_list,
                         columns=["genres", "tokens", "types", "lex_diversity"])
print(genres_df, "\n")

# get genre with minimum lexical diversity
print(f"genre with minimum lexical diversity: {genres_df.iloc[genres_df['lex_diversity'].idxmin(axis=0, skipna=True), 0]}\n")


# 17.

# get 50 most frequent words in text that are not stop words
def most_freq_nonstop(text):
    text_fd = FreqDist(text)  # get frequency distribution over text
    stopwords = nltk.corpus.stopwords.words("english")  # get english stop words
    filtered_words = [word for word, count in text_fd.most_common(200) if word not in stopwords and word.isalpha()]  # filter most common words for ones that are not stop words and must contain only letters
    return filtered_words[:50]  # return only top 50


print(f"Top 50 word in Moby Dick that are not stopwords in english:\n{most_freq_nonstop(nltk.Text(gutenberg.words('melville-moby_dick.txt')))}\n")



# 18.

# get 50 most frequent bigrams in some text and which have no stop words in either component
def bigrams_nonstop(text):
    text_bigrams = bigrams(text)  # form bigrams over text
    stopwords = nltk.corpus.stopwords.words("english")  # get english stop words
    bigrams_fd = FreqDist(text_bigrams)  # get frequency distribution over bigrams
    filtered_bigrams = [bigram for bigram, count in bigrams_fd.most_common(2000)   # filter bigrams for stopwords and must contain only letters
                        if bigram[0] not in stopwords 
                        and bigram[1] not in stopwords
                        and bigram[0].isalpha()
                        and bigram[1].isalpha()]
    return filtered_bigrams[:50]

print(f"Top 50 bigrams in Mobdy Dick where neither word is a stop word:\n{bigrams_nonstop(nltk.Text(gutenberg.words('melville-moby_dick.txt')))}\n")



# 19.

# make conditional frequency over genre, word pairs
cfd = nltk.ConditionalFreqDist(
    (genre, word)
    for genre in brown.categories()
    for word in brown.words(categories=genre)
)

genres = brown.categories()
words = ["world", "laugh", "love", "help", "grim", "learn"]
cfd.tabulate(conditions=genres, samples=words)  # tabulate count of each sample word for each genre
print("\n")


# 20.

def word_freq(brown_section, word):  # input section of Brown corpus and desired word
    brown_section_text = brown.words(categories=brown_section)  # get text for desired section
    brown_section_fd = FreqDist(brown_section_text)  # put frequency distribution over words
    return brown_section_fd[word]  # get word frequency for desired word

print(f"Word frequency of 'world' in 'adventure' text: {word_freq('adventure', 'world')}\n")



# 21.

# predict number of syllables by counting parts of cmu dictionary entries which have a vowel (syllables are typically formed around a vowel)
def syllable_guesser(text):
    return len([phone for phone in cmudict.entries()  # check all cmu dictionary entries
                    if phone[0]==text  # only look at those where the input text is the first entry
                    for phn in phone[1]  # check for a vowel in all second components
                    if set(["a", "e", "i", "o", "u"]).intersection(set(phn.lower()))])


print(syllable_guesser("adventure"), "\n")



# 22.

def hedge(text):
    text_words = text.split()  # split up text
    new_text = ""  # define new text
    for i in range(len(text_words)):
        if not (i%3) and i>0:  # if index is divisible by 3 and not zero add 'like' before it
            new_text += " like "
        new_text += " " + text_words[i] + " "
    return new_text


print(hedge("This is a test sentence for the function."))



# 23.

# a.
# create function to illustrate zipf law by plotting frequency against rank
def zipf_law_plot(text, plot_name):
    text_words = text.split()  # split up input text
    text_fd = FreqDist(text_words)  # get frequency distribution over text words

    # get sorted array of tuples ordered by most frequently ocurring (i.e. order in reverse by second component)
    sorted_text_tuple = sorted([(text, text_fd[text]) for text in text_words], key=lambda x: x[1], reverse=True)
    ranks = [i+1 for i in range(len(text_words))]  # make array of ranks
    freqs = [sorted_text_tuple[i][1] for i in range(len(text_words))]  # make array of frequencies

    fig = plt.figure(figsize=(6.5, 5.5))
    plt.plot(ranks, freqs, color="b")  # plot rank against frequency
    plt.xlabel("words")
    plt.ylabel("frequency")
    plt.xscale("log")  # use log scale
    plt.savefig(f"figures/{plot_name}.png")
    plt.close()

# test string
zipf_law_plot("This is just test text to test if the function is working as ecpected.\nLater we will use a corpus.", "zipf_law")

# Moby Dick test
zipf_law_plot(" ".join(gutenberg.words("melville-moby_dick.txt")), "zipf_law_moby_dick")

# Austen Persuation test
zipf_law_plot(" ".join(gutenberg.words("austen-persuasion.txt")), "zipf_law_austen")


# b.
random_text = ""
random_start = random.randint(1, 5000)
random_end = random.randint(10000, 15000)
for i in range(random_start, random_end):
    random_text += random.choice("abcdefg ")

# zipf plot for random text
zipf_law_plot(random_text, "zipf_law_random_text")

print(f"The plots seem consistent with Zipf's law.\n")



# 24.

# a.
# generate text with given starter word
def generate_model(cfdist, word, num=15, n=1):
    for i in range(num):  # iterate for desired length of text
        print(word, end=" ")  # print word
        m = min(len(cfdist[word]), n)  # get minimum between number of choices of next word and desired amount of choices (i.e. n)
        words = list(word_entry for word_entry, count in cfdist[word].items())[:m] if m>0 else ["the"]  # make a list of 'm' most likely words that should follow, if m=0 use 'the'
        word = random.choice(words)  # update word randomly from list

text = nltk.corpus.genesis.words("english-kjv.txt")  # get text
bigrams = nltk.bigrams(text)  # get bigrams based on text
cfd =nltk.ConditionalFreqDist(bigrams)  # get conditional frequency distribution to predict next word
generate_model(cfdist=cfd, word="living", num=15, n=3)  # generate new text
print("\n")

# b., c.
start_words = ["After", "We", "In", "Beginning"]  # get starter words

# get texts
texts = [nltk.corpus.genesis.words("english-kjv.txt"),
         brown.words(categories="humor"),
         brown.words(categories="science_fiction"),
         brown.words(categories=["humor", "science_fiction"])]

# generate new text from input text using a starter word
for starter in start_words:
    bigrams = nltk.bigrams(texts[3])
    cfd = nltk.ConditionalFreqDist(bigrams)
    generate_model(cfdist=cfd, word=starter, n=3)
    print("\n")



# 25.

# get Latin1 languages
latin1_languages = [language for language in udhr.fileids() if "Latin1" in language]

# get Latin1 languages that have desired word
def find_language(sample):
    return [language_with_sample for language_with_sample in latin1_languages if sample in udhr.words(language_with_sample)]

print(f"List of Latin1 encoded languages containing word: {find_language('umat')}\n")



# 26.

# get all noun synsets
all_nouns= wn.all_synsets("n")

# make list for hypernym counts
hypernym_counts = []

# get hypernym counts for all nouns
for noun in all_nouns:
    if len(noun.hypernyms())>0:
        hypernym_counts += [len(noun.hypernyms())]

# get average branching factor
branching_factor = sum(hypernym_counts)/len(hypernym_counts)
print(f"Branching factor of noun hypernym hierarchy is: {branching_factor}\n")



# 27.

# dog has 7 senses as a noun
print(f"Dog has 7 senses as noun {len(wn.synsets('dog', 'n'))}\n")
def avg_polysemy(part_of_speech):

    # get dictionary mapping part of speech to letter used by Wordnet
    wordnet_dict = {"noun":"n", "verb":"v", "adjective":"a", "adverb":"r"}

    # get words that have a sense matching desired part of speech
    # we used set to avoid repitition (some words may have many senses as part of speech)
    words_of_interest = set(lem.name()  # get name from each lemma
            for syn in wn.all_synsets(wordnet_dict[part_of_speech])  # get all synsets
            for lem in syn.lemmas())  # get all lemmas of synset
    polysemy = [len(wn.synsets(word, wordnet_dict[part_of_speech]))
                for word in words_of_interest]  # get number of senses fo each word that has a definition matching desired sesne
    return sum(polysemy)/len(polysemy)  # get average polysemy

for part_of_speech in ["noun", "verb", "adjective", "adverb"]:
    print(f"Average polysem of {part_of_speech + 's'}: {avg_polysemy(part_of_speech)}\n")
print("\n")



# 28.

# define path similarity function
def similarity(word1, word2, similarity_function):
    synset1 = [syn for syn in wn.synsets(word1)]  # get synsets of word1
    synset2 = [syn for syn in wn.synsets(word2)]  # get synsets of word2
    distances = [similarity_function(syn1, syn2) for syn1 in synset1 for syn2 in synset2]  # calculate path similarity between words
    return max(distances)

# test path similarity function
print(similarity("orca", "tortoise", wn.path_similarity))

# get word pairs
word_pairs = [
    ["car", "automobile"],
    ["gem", "jewel"],
    ["journey", "voyage"],
    ["boy", "lad"],
    ["coast", "shore"],
    ["asylum", "madhouse"],
    ["magician", "wizard"],
    ["midday", "noon"],
    ["furnace", "stove"],
    ["food", "fruit"],
    ["bird", "cock"],
    ["bird", "crane"],
    ["tool", "implement"],
    ["brother", "monk"],
    ["lad", "brother"],
    ["crane", "implement"],
    ["journey", "car"],
    ["monk", "oracle"],
    ["cemetery", "woodland"],
    ["food", "rooster"],
    ["coast", "hill"],
    ["forest", "graveyard"],
    ["shore", "woodland"],
    ["monk", "slave"],
    ["coast", "forest"],
    ["lad", "wizard"],
    ["chord", "smile"],
    ["glass", "magician"],
    ["rooster", "voyage"],
    ["noon", "string"]
]

similarity_functions = [["Path Similarity", wn.path_similarity],  # path similarity
                        #["Leacock-Chodorow", wn.lch_similarity],  # Leacock-Chodorow similarity (can't use this distance as it requires same sense of speech)
                        ["Wu-Palmer", wn.wup_similarity]  # Wu-Palmer similarity
]

# print word pair distances for each type of similarity function
for entry in word_pairs:
    for function in similarity_functions:
        print(f"{function[0]} distance between {entry[0]} and {entry[1]} is: {similarity(entry[0], entry[1], function[1])}")
    print("\n")


