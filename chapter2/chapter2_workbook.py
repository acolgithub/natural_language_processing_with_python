import nltk
from nltk.corpus import gutenberg  # import text from Project Gutenberg
from nltk.corpus import webtext  # import web text
from nltk.corpus import nps_chat  # import instant messaging text
from nltk.corpus import brown  # import corpus from Brown University
from nltk.corpus import reuters  # import reuters corpus
from nltk.corpus import inaugural  # import inaugural addresses
from nltk.corpus import stopwords  # import stop words
from nltk.corpus import names  # get common first names
from nltk.corpus import cmudict  # get cmu pronouncing dictionary for US English
from nltk.corpus import swadesh  # get swadesh wordlists
from nltk.corpus import toolbox  # get toolbox
from nltk.corpus import wordnet as wn  # get wordnet

# matplotlib
import matplotlib.pyplot as plt

# data
emma = nltk.corpus.gutenberg.words("austen-emma.txt")

# 1 Accessing Text Corpora

# 1.1 Gutenberg Corpus

# get file identifiers for gutenberg corpus
print(nltk.corpus.gutenberg.fileids())

# get length of Emma by Jane Austen
print(len(emma), "\n")

# to do concordancing we must first convert the words to text
emma_text = nltk.Text(nltk.corpus.gutenberg.words("austen-emma.txt"))
emma_text.concordance("surprize")
print("\n")

# loop over file ids and print statistics for each text (average word length, average sentence length, average number of times each vocabulary item appears in the text)
for fileid in gutenberg.fileids():
    num_chars = len(gutenberg.raw(fileid))
    num_words = len(gutenberg.words(fileid))
    num_sents = len(gutenberg.sents(fileid))
    num_vocab = len(set(w.lower() for w in gutenberg.words(fileid)))
    print(round(num_chars/num_words),
          round(num_words/num_sents),
          round(num_words/num_vocab),
          fileid)
print("\n")


# get length of raw text (raw gives no linguistic processing)
print(len(gutenberg.raw("blake-poems.txt")), "\n")

# divide text up into sentences
macbeth_sentences = gutenberg.sents("shakespeare-macbeth.txt")
print(macbeth_sentences, "\n")

# print specific sentence
print(macbeth_sentences[1116], "\n")

# get longest length
longest_len = max(len(s) for s in macbeth_sentences)

# get longest sentences
print([s for s in macbeth_sentences if len(s) == longest_len], "\n")


# 1.2 Web and Chat Text
for fileid in webtext.fileids():
    print(fileid, webtext.raw(fileid)[:65], "...")


chatroom = nps_chat.posts("10-19-20s_706posts.xml")
print(chatroom[123], "\n")


# 1.3 Brown Corpus

# get categories
print(brown.categories(), "\n")

# get list of words from specific category
print(brown.words(categories="news"), "\n")

# get list of words from specific id
print(brown.words(fileids=["cg22"]), "\n")

# get sentences from specific categories
print(brown.sents(categories=["news", "editorial", "reviews"]), "\n")


# analyze news text

# get words corresponding to news categories
news_text = brown.words(categories="news")

# get frequency distribution of words in news text
fdist = nltk.FreqDist(w.lower() for w in news_text)

# define modal verbs
modals = ["can", "could", "may", "might", "must", "will"]

# get frequency of each modal verb in news text
print(f"frequency of modal verbs in news text:")
for m in modals:
    print(m + ":", fdist[m], end="\n")
print("\n")


# get words corresponding to mystery
mystery_text = brown.words(categories="mystery")

# get frequency distribution of words in mystery text
mystery_dist = nltk.FreqDist(w.lower() for w in mystery_text)

# define 'wh' words
wh_words = ["what", "when", "where", "who", "why"]

# get frequency of 'wh'
for w in wh_words:
    print(f"{w}: {mystery_dist[w]}")
print("\n")


# get conditional frequency distribution
cfd = nltk.ConditionalFreqDist(
    (genre, word)
    for genre in brown.categories()
    for word in brown.words(categories=genre))

# define genres and modal verbs
genres = ["news", "religion", "hobbies", "science_fiction", "romance", "humor"]
modals = ["can", "could", "may", "might", "must", "will"]

# get counts of modal verbs conditions by genre
cfd.tabulate(conditions=genres, samples=modals)
print("\n")


# 1.4 Reuters Corpus

# get reuters file ids
print(reuters.fileids(), "\n")

# get reuters categories
print(reuters.categories(), "\n")

# get categories in single training set
print(reuters.categories("training/9865"), "\n")

# get categories in list of training sets
print(reuters.categories(["training/9865", "training/9880"]), "\n")

# get sets related to single file id
print(reuters.fileids("barley"), "\n")

# get sets related to list of file ids
print(reuters.fileids(["barley", "corn"]), "\n")


# get subset of words from training set
print(reuters.words("training/9865")[:14], "\n")

# get words from multiple training sets
print(reuters.words(["training/9865", "training/9880"]), "\n")

# get words corresponding to category
print(reuters.words(categories="barley"), "\n")

# get words corresponding to many categories
print(reuters.words(categories=["barley", "corn"]), "\n")




# 1.5 Inaugural Addres Corpus

# inaugural address file ids
print(inaugural.fileids(), "\n")

# print first four characters of file id (i.e. the year of the address)
print([fileid[:4] for fileid in inaugural.fileids()], "\n")

# get conditional frequency distribution of 'america' and 'citizen' vs. year of address
cfd = nltk.ConditionalFreqDist(
    (target, fileid[:4])
    for fileid in inaugural.fileids()  # check each file id
    for w in inaugural.words(fileid)  # check each word in text
    for target in ["america", "citizen"]  # look for targets
    if w.lower().startswith(target))  # convert word to lowercase and check if it starts with target

cfd.plot()



# 1.7 Corpora in Other Languages
from nltk.corpus import udhr

# get languages
languages = ["Chickasaw", "English", "German_Deutsch",
            "Greenlandic_Inuktikut", "Hungarian_Magyar", "Ibibio_Efik"]

# get conditional frequency distribution of lanaguage against length of words in Universal Declaration of Human Rights
cfd = nltk.ConditionalFreqDist(
    (lang, len(word))
    for lang in languages
    for word in udhr.words(lang + "-Latin1"))

# make cumulative plot of word lengths
cfd.plot(cumulative=True)



# get udhr file ids
print(udhr.fileids(), "\n")

# get raw text
raw_text = udhr.raw("Filipino_Tagalog-Latin1")

# plot frequency distribution of letters of the text
nltk.FreqDist(raw_text).plot()
plt.show()




# 2 Conditional Frequency Distributions

# 2.2 Counting Words by Genre

# get conditional distribution which pairs words with genre in brown corpus
cfd = nltk.ConditionalFreqDist(
    (genre, word)
    for genre in brown.categories()
    for word in brown.words(categories=genre))


# loop of all words in genre for each genre
genre_word = [(genre, word)
              for genre in ["news", "romance"]
              for word in brown.words(categories=genre)]


# gets total number of words over all genres considered
print(len(genre_word), "\n")

# news category is at the beginning
print(genre_word[:4], "\n")

# romance is at the end
print(genre_word[-4:], "\n")

# make conditional frequency distribution
cfd = nltk.ConditionalFreqDist(genre_word)

# get information about conditional frequency distribution
print(cfd, "\n")

# get the conditions of the distribution
print(cfd.conditions(), "\n")

# print conditional distribution information on category
print(cfd["news"], "\n")
print(cfd["romance"], "\n")

# get 2o most common words and corresponding frequencies
print(cfd["romance"].most_common(20), "\n")

# get frequency of specific word
print(cfd["romance"]["could"])



# 2.3 Plotting and Tabulating Distributions

# get back conditional frequency distribution for language
cfd = nltk.ConditionalFreqDist(
    (lang, len(word))
    for lang in languages
    for word in udhr.words(lang + "-Latin1")
)

# tabulate frequency of words of length less than 10 for each language given
cfd.tabulate(conditions=["English", "German_Deutsch"],
             samples=range(10),
             cumulative=True)



# define list of genres
genres = ["news", "romance"]

# get conditional distribution
cfd = nltk.ConditionalFreqDist(
    (genre, word)
    for genre in brown.categories()
    for word in brown.words(categories=genre)
)

# define list of days of the week
days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

# tabulate frequency of words for desired genres
cfd.tabulate(conditions=genres, samples=days)

# plot frequencies of words for desired genres
cfd.plot(conditions=genres, samples=days)



# 2.4 Generating Random Text with Bigrams

sent = ["In", "the", "beginning", "God", "created", "the", "heaven",
        "and", "the", "earth", "."]
print(list(nltk.bigrams(sent)), "\n")


# 3 More Python: Reusing Code

# 3.2 Functions

def plural(word):
    if word.endswith("y"):
        return word[:-1] + "ies"
    elif word[-1] in "sx" or word[-2:] in ["sh", "ch"]:
        return word + "es"
    elif word.endswith("an"):
        return word[:-2] + "en"
    else:
        return word + "s"
    
print(plural("fairy"))
print(plural("woman"), "\n")


# 4 Lexical Resources

# 4.1 Wordlist Corpora

# define function to identify unusual words
def unusual_words(text):
    text_vocab = set(w.lower() for w in text if w.isalpha())  # make word from text lowercase if alphabetical word
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())  # make word from corpus lowercase if alphabetical word
    unusual = text_vocab - english_vocab  # find words in text not included in corpus
    return sorted(unusual)  # return sorted collection of unusual words

print(unusual_words(nltk.corpus.gutenberg.words("austen-sense.txt")))
print(unusual_words(nltk.corpus.nps_chat.words()), "\n")

# print stop words
print(stopwords.words("english"), "\n")

# function to determine fraction of words in text that are not stopwords
def content_fraction(text):
    stopwords = nltk.corpus.stopwords.words("english")
    content = [w for w in text if w.lower() not in stopwords]
    return len(content)/len(text)

print(content_fraction(nltk.corpus.reuters.words()), "\n")

# try to solve word puzzle
puzzle_letters = nltk.FreqDist("egivrvonl")
obligatory = "r"
wordlist = nltk.corpus.words.words()
print([w for w in wordlist if len(w) >= 6
                     and obligatory in w
                     and nltk.FreqDist(w) <= puzzle_letters], "\n")

# build function to try and solve similar word puzzles (included default length as 4)
def word_puzzle_solver(letters, center, length=4):
    letter_freq_dist = nltk.Freq.Dist(letters)
    return [w for w in nltk.corpus.words.words()
                if len(w) >= length
                and center in w
                and nltk.FreqDist(w) <= letter_freq_dist]


# get file ids contained in names
print(names.fileids(), "\n")

# get male and female names
male_names, female_names = names.words("male.txt"), names.words("female.txt")

# get gender neutral names
print([w for w in male_names if w in female_names], "\n")


# plot counts of last letter in name
cfd = nltk.ConditionalFreqDist(
    (fileid, name[-1])  # plot male names and female names against last letter
    for fileid in names.fileids()  # for each of male and female file
    for name in names.words(fileid))  # for each name in file

cfd.plot()  # plot frequencies


# 4.2 A Pronouncing Dictionary

# get entries of cmu dictionary
entries = cmudict.entries()

# get number of entries
print(len(entries), "\n")

# print out specific entries
for entry in entries[42371:42379]:
    print(entry)
print("\n")

# scan lexicon for entries whose pronounciation consists of three phones
for word, pron in entries:
    if len(pron) == 3:
        ph1, ph2, ph3 = pron
        if ph1 == "P" and ph3 == "T":
            print(word, ph2, end=" ")
print("\n")

# find all words whose pronounciation ends with a syllable sounding like "nicks"
syllable = ["N", "IH0", "K", "S"]
print([word for word, pron in entries if pron[-4:] == syllable], "\n")

# find all words ending in 'n' but are pronounced as 'm'
print([w for w, pron in entries if pron[-1] == "M" and w[-1] == "n"], "\n")

# print first two letters of words whose beginning pronounciation is like 'N' but the word does not start with 'n'
print(sorted(set(w[:2] for w, pron in entries if pron[0] == "N" and w[0] != "n")), "\n")


# define function to extract the stress digits and scan lexicon to find words having a particular stress pattern
def stress(pron):
    return [char for phone in pron for char in phone if char.isdigit()]


# searches
print([w for w, pron in entries if stress(pron) == ["0", "1", "0", "2", "0"]], "\n")
print([w for w, pron in entries if stress(pron) == ["0", "2", "0", "1", "0"]], "\n")



# find all "p"-words consisting of three sounds and group them according to their first and last sounds
p3 = [(pron[0] + "-" + pron[2], word)
      for (word, pron) in entries
      if pron[0] == "P" and len(pron) == 3]

# get conditional frequency distribution of "p"-words with 3 sounds
cfd = nltk.ConditionalFreqDist(p3)


for template in sorted(cfd.conditions()):
    if len(cfd[template]) > 10:
        words = sorted(cfd[template])
        wordstring = " ".join(words)
        print(template, wordstring[:70] + "...")


# get cmu dictionary
prondict = cmudict.dict()

# look up fire
print(prondict["fire"], "\n")

# look up each word in the pronounciation dictionary
text = ["natural", "language", "processing"]
print([ph for w in text for ph in prondict[w][0]])


# 4.3 Comparative Wordlists

# print swadesh file ids
print(swadesh.fileids())

# print swadesh words
print(swadesh.words("en"))

# access cognate words from multiple languages

# french to english
fr2en = swadesh.entries(["fr", "en"])
print(fr2en)

# get translator from french to english
translate = dict(fr2en)

# translate chien
print(translate["chien"])

# translate jeter
print(translate["jeter"])



# add more languages to translator

# german to english
de2en = swadesh.entries(["de", "en"])

# spanish to english
es2en = swadesh.entries(["es", "en"])

# update translator
translate.update(dict(de2en))
translate.update(dict(es2en))

# translate Hund
print(translate["Hund"])

# translate perro
print(translate["perro"])

# compare words in various Germanic Romance languages
languages = ["en", "de", "nl", "es", "fr", "pt", "la"]
for i in [139, 140, 141, 142]:
    print(swadesh.entries(languages)[i])
print("\n")


# print toolbox entries for rotokas language
# print(toolbox.entries("rotokas.dic"))  # do not uncomment, quite long



# 5 WordNet

# 5.1 Senses and Synonyms
print(wn.synsets("motorcar"))  # produces car.n.01 which is one possible meaning (the first noun sense of car)
# this produces a synset (synonym set)

# synonym names of car.n.01
print(wn.synset("car.n.01").lemma_names())

# get prose definition and example sentence
print(wn.synset("car.n.01").definition())
print(wn.synset("car.n.01").examples(), "\n")

# take a given synset
print(wn.synset("car.n.01").lemmas())

# get lemma of a given synset
print(wn.lemma("car.n.01.automobile"))

# get synset corresponding to a lemma
print(wn.lemma("car.n.01.automobile").synset())

# get the name of a lemma
print(wn.lemma("car.n.01.automobile").name())

# car has give synsets
print(wn.synsets("car"))

# get lemma names of car
for synset in wn.synsets("car"):
    print(synset.lemma_names())
print("\n")

# can also access all lemmas involving "car" as follows:
print(wn.lemmas("car"))

# get all synsets of "dish"
print(wn.synsets("dish"))

# get definition and example of each sense of dish
for syn in wn.synsets("dish"):
    print(f"Definition: {syn.definition()}\nExample: {syn.examples()}")

print("\n")



# 5.2 The WordnNet Hierarchy

# get hyponyms of motorcar (more specific)
motorcar = wn.synset("car.n.01")
types_of_motorcar = motorcar.hyponyms()
print(types_of_motorcar[0], "\n")

print(sorted(lemma.name() for synset in types_of_motorcar for lemma in synset.lemmas()))
print("\n")

# get hypernyms
print(motorcar.hypernyms())

paths = motorcar.hypernym_paths()
print(len(paths))

print([synset.name() for synset in paths[0]], "\n")
print([synset.name() for synset in paths[1]], "\n")

# get root hypernyms (most general)
print(motorcar.root_hypernyms())


# 5.3 More Lexical Relations

# get part meronyms of tree (parts of it)
print(wn.synset("tree.n.01").part_meronyms(), "\n")

# get substance meronyms of tree (substance of it)
print(wn.synset("tree.n.01").substance_meronyms(), "\n")

# get member holonyms (things they are contained in)
print(wn.synset("tree.n.01").member_holonyms(), "\n")


# get definitions of mint
for synset in wn.synsets("mint", wn.NOUN):  # specify noun
    print(synset.name() + ":", synset.definition())  # get definition
print("\n")

# get part holonyms of mint
print(wn.synset("mint.n.04").part_holonyms())
print("\n")

# get substance holonyms of mint
print(wn.synset("mint.n.04").substance_holonyms())
print("\n")


# get entailments of walk as a verb
print(wn.synset("walk.v.01").entailments())

# get entailments of eat as a verb
print(wn.synset("eat.v.01").entailments())

# get entailments of tease
print(wn.synset("tease.v.03").entailments())
print("\n")


# get lexical relationships between lemmas (antonymy)
print(wn.lemma("supply.n.02.supply").antonyms())
print(wn.lemma("rush.v.01.rush").antonyms())
print(wn.lemma("horizontal.a.01.horizontal").antonyms())
print(wn.lemma("staccato.r.01.staccato").antonyms())




# 5.4 Semantic Similarity

right = wn.synset("right_whale.n.01")
orca = wn.synset("orca.n.01")
minke = wn.synset("minke_whale.n.01")
tortoise = wn.synset('tortoise.n.01')
novel = wn.synset('novel.n.01')

# get least common hypernym between two words
right.lowest_common_hypernyms(minke)
right.lowest_common_hypernyms(orca)
right.lowest_common_hypernyms(tortoise)
right.lowest_common_hypernyms(novel)
print("\n")

# calculate how general a word is by depth in tree
print(wn.synset('baleen_whale.n.01').min_depth())
print(wn.synset('whale.n.02').min_depth())
print(wn.synset('vertebrate.n.01').min_depth())
print(wn.synset('entity.n.01').min_depth())
print("\n")

# calculate path similarity between two words
print(right.path_similarity(minke))
print(right.path_similarity(orca))
print(right.path_similarity(tortoise))
print(right.path_similarity(novel))



