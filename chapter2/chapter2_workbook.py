import nltk
from nltk.corpus import gutenberg  # import text from Project Gutenberg
from nltk.corpus import webtext  # import web text
from nltk.corpus import nps_chat  # import instant messaging text
from nltk.corpus import brown  # import corpus from Brown University
from nltk.corpus import reuters  # import reuters corpus
from nltk.corpus import inaugural  # import inaugural addresses

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
















