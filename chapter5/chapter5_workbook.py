import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import brown
from nltk.tbl import demo as brill_demo
from collections import defaultdict
from operator import itemgetter
import regex
import numpy as np
from pickle import dump
from pickle import load



# 1 Using a Tagger

# get text
text = word_tokenize("And now for something completely different")

# get part of speech tags
print(nltk.pos_tag(text))
print("\n")

# get help for tags
nltk.help.upenn_tagset("CC")
nltk.help.upenn_tagset("RB")
nltk.help.upenn_tagset("IN")
nltk.help.upenn_tagset("NN")
nltk.help.upenn_tagset("JJ")
print("\n")

# another example
text = word_tokenize("They refuse to permit us to obtain the refuse permit")
print(nltk.pos_tag(text), "\n")

# find similar contexts
text = nltk.Text(word.lower() for word in nltk.corpus.brown.words())
print(text.similar("woman"), "\n")
print(text.similar("brought"), "\n")
print(text.similar("over"), "\n")
print(text.similar("the"), "\n")



# 2 Tagged Corpora

# 2.1 Representing Tagged Tokens
tagged_token = nltk.tag.str2tuple("fly/NN")
print(tagged_token)
print(tagged_token[0])
print(tagged_token[1], "\n")

# get tagged tuple from string
sent = """
The/AT grand/JJ jury/NN commented/VBD on/IN a/AT number/NN of/IN
other/AP topics/NNS ,/, AMONG/IN them/PPO the/AT Atlanta/NP and/CC
Fulton/NP-tl County/NN-tl purchasing/VBG departments/NNS which/WDT it/PPS
said/VBD ``/`` ARE/BER well/QL operated/VBN and/CC follow/VB generally/RB
accepted/VBN practices/NNS which/WDT inure/VB to/IN the/AT best/JJT
interest/NN of/IN both/ABX governments/NNS ''/'' ./."""

print([nltk.tag.str2tuple(t) for t in sent.split()], "\n")



# 2.2 Reading Tagged Corpora
print(nltk.corpus.brown.tagged_words(), "\n")
print(nltk.corpus.brown.tagged_words(tagset="universal"), "\n")


# more examples
print(nltk.corpus.nps_chat.tagged_words(), "\n")
print(nltk.corpus.conll2000.tagged_words(), "\n")
print(nltk.corpus.treebank.tagged_words(), "\n")


# using universal tag
print(nltk.corpus.brown.tagged_words(tagset="universal"))
print(nltk.corpus.treebank.tagged_words(tagset="universal"), "\n")


# other languages
nltk.download('sinica_treebank')
nltk.download('indian')
nltk.download('mac_morpho')
nltk.download('cess_cat')

print(nltk.corpus.sinica_treebank.tagged_words())
print(nltk.corpus.indian.tagged_words())
print(nltk.corpus.mac_morpho.tagged_words())
print(nltk.corpus.conll2002.tagged_words())
print(nltk.corpus.cess_cat.tagged_words(), "\n")



# 2.3 A Universal Part-of-Speech Target

brown_news_tagged = brown.tagged_words(categories="news", tagset="universal")

# get distribution of tagged words
tag_fd = nltk.FreqDist(tag for (word, tag) in brown_news_tagged)

# get most common
print(tag_fd.most_common(), "\n")

# plot
tag_fd.plot(cumulative=True)



# 2.4 Nouns

# get word tag pairs
word_tag_pairs = nltk.bigrams(brown_news_tagged)

# get tags of words preceding nouns
noun_preceders = [a[1] for (a, b) in word_tag_pairs if b[1] == "NOUN"]

# make frequency distribution over noun preceders
fdist = nltk.FreqDist(noun_preceders)

print([tag for (tag, _) in fdist.most_common()])
print("\n")


# 2.5 Verbs

# get tagged words
wsj = nltk.corpus.treebank.tagged_words(tagset="universal")

# get frequency distribution
word_tag_fd = nltk.FreqDist(wsj)

# get verbs
print([wt[0] for (wt, _) in word_tag_fd.most_common() if wt[1] == "VERB"])
print("\n")


# use conditional frequency distribution and treat word as conditiona and tag as event
cfd1 = nltk.ConditionalFreqDist(wsj)
print(cfd1["yield"].most_common(), "\n")
print(cfd1["cut"].most_common(), "\n")


# reverse word order pairs to get likey word given tag
wsj = nltk.corpus.treebank.tagged_words()
cfd2 = nltk.ConditionalFreqDist((tag, word) for (word, tag) in wsj)
print(list(cfd2["VBN"]), "\n")


# find examples of words that are both past tense and past participle
print(sorted([w for w in cfd1.conditions() if w in cfd2["VBD"] and w in cfd2["VBN"]]), "\n")

# get index of kicked as VBD
idx1 = wsj.index(("kicked", "VBD"))

# print pairs with nearby index
print(wsj[idx1-4:idx1+1], "\n")

# get index of kicked as VBN
idx2 = wsj.index(("kicked", "VBN"))
print(wsj[idx2-4:idx2+1], "\n")


# get list of past participles nad part of speech
past_part = list(cfd2["VBN"])
tagged_pairs = nltk.bigrams([(word, cfd1[word]) for word in cfd1])

print([(w[0][0], *w[0][1].keys()) for w in tagged_pairs if w[1][0] in past_part], "\n")



# 2.7 Unsimplified Tags

def findtags(tag_prefix, tagged_text):
    # get conditional frequency distribution over tagged text if tag has prefix
    cfd = nltk.ConditionalFreqDist((tag, word) for (word, tag) in tagged_text
                                   if tag.startswith(tag_prefix))
    
    return dict((tag, cfd[tag].most_common(5)) for tag in cfd.conditions())


tagdict = findtags("NN", nltk.corpus.brown.tagged_words(categories="news"))
for tag in sorted(tagdict):
    print(tag, tagdict[tag])
print("\n")



# 2.8 Exploring Tagged Corpora

brown_learned_text = brown.words(categories="learned")

# get words that follow "often"
print(sorted(set(b for (a, b) in nltk.bigrams(brown_learned_text) if a == "often")), "\n")


# tabulate part of speech that follows "often"
brown_lrnd_tagged = brown.tagged_words(categories="learned", tagset="universal")
tags = [b[1] for (a, b) in nltk.bigrams(brown_lrnd_tagged) if a[0] == "often"]
fd = nltk.FreqDist(tags)
fd.tabulate()
print("\n")


# print matching three word sequence "verb to verb"
def process(sentence):
    for (w1, t1), (w2, t2), (w3, t3) in nltk.trigrams(sentence):
        if (t1.startswith("V") and t2 == "TO" and t3.startswith("V")):
            print(w1, w2, w3)

# process each tagged sentence
for tagged_sent in brown.tagged_sents():
    process(tagged_sent)
print("\n")


# look for words that have ambiguous part of speech
brown_news_tagged = brown.tagged_words(categories="news", tagset="universal")
data = nltk.ConditionalFreqDist((word.lower(), tag)
                                for (word, tag) in brown_news_tagged)

for word in sorted(data.conditions()):
    if len(data[word]) > 3:
        tags = [tag for (tag, _) in data[word].most_common()]
        print(word, " ".join(tags))
print("\n")



text = brown.tagged_words(categories="news", tagset="universal")
temp_list = []
for (word, tag) in text:
    temp_list.append(word + "/" + tag)
text = nltk.text.Text(" ".join(temp_list).split())

def tag_context(word, context="", text=text):
    if context:
        text.concordance(f"{word}/{context}")
    else:
        text.concordance(word)

tag_context(word="The", context="DET")
print("\n")
        



# 3 Mapping Words to Properties Using Python Dictionaries

# 3.4 Default Dictionaries
frequency = defaultdict(int)
frequency["colourless"] = 4
print(frequency["ideas"])  # default initialized to zero

pos = defaultdict(list)
pos["sleep"] = ["NOUN", "VERB"]
print(pos["ideas"], "\n")


# create default value
pos = defaultdict(lambda: "NOUN")
pos["colourless"] = "ADJ"
print(pos["blog"])
print(list(pos.items()), "\n")


# lambda expression called by no parameters
f = lambda: "NOUN"
print(f(), "\n")



alice = nltk.corpus.gutenberg.words("carroll-alice.txt")
vocab = nltk.FreqDist(alice)
v1000 = [word for (word, _) in vocab.most_common(1000)]
mapping = defaultdict(lambda: "UNK")
for v in v1000:
    mapping[v] = v

alice2 = [mapping[v] for v in alice]
print(alice2[:100])
print(len(set(alice2)), "\n")



# 3.5 Incrementally Updating a Dictionary

counts = defaultdict(int)
from nltk.corpus import brown
for (word, tag) in brown.tagged_words(categories="news", tagset="universal"):
    counts[tag] += 1

print(counts["NOUN"])
print(sorted(counts), "\n")


# sort dictionary entries by ordering dictionary values in reverse
print(sorted(counts.items(), key=itemgetter(1), reverse=True))
print([t for t, c in sorted(counts.items(), key=itemgetter(1), reverse=True)], "\n")


pair = ("NP", 8336)
print(pair[1])
print(itemgetter(1)(pair), "\n")



last_letters = defaultdict(list)
words = nltk.corpus.words.words("en")
for word in words:
    key = word[-2:]
    last_letters[key].append(word)

print(last_letters["ly"])
print(last_letters["zy"], "\n")



# create anagram dictionary
anagrams = defaultdict(list)
for word in words:  # loop over words defined earlier
    key = "".join(sorted(word))  # sort word in alphabetical order to get key
    anagrams[key].append(word)  # assign to this key the original word

print(anagrams["aeilnrt"], "\n")  # anagrams is non-empty only for letters which can be rearranged to form the alphabetical order of entrail


# alternative way to create default dictionary
anagrams = nltk.Index(("".join(sorted(w)), w) for w in words)
print(anagrams["aeilnrt"])



# 3.6 Complex Keys and Values

# find how many values a tag can take given the word and previous tag
pos = defaultdict(lambda: defaultdict(int))
brown_news_tagged = brown.tagged_words(categories="news", tagset="universal")
for ((w1, t1), (w2, t2)) in nltk.bigrams(brown_news_tagged):
    pos[(t1,w2)][t2] += 1

print(pos[("DET", "right")], "\n")



# 3.7 Inverting a Dictionary

# find key given value
counts = defaultdict(int)
for word in nltk.corpus.gutenberg.words("milton-paradise.txt"):
    counts[word] += 1

print([key for (key, value) in counts.items() if value==32], "\n")


# another way to initialize dictionary with value, key pairs
pos = {"colourless": "ADJ", "ideas": "N", "sleep": "V", "furiously": "ADV"}
pos2 = dict((value, key) for (key, value) in pos.items())
print(pos2["N"], "\n")



# update dictionary
pos.update({"cats": "N", "scratch": "V", "peacefully": "ADV", "old": "ADJ"})
pos2 = defaultdict(list)
for key, value in pos.items():
    pos2[value].append(key)

print(pos2["ADV"], "\n")


# simpler way to invert keys and values
pos2 = nltk.Index((value, key) for (key, value) in pos.items())
print(pos2["ADV"], "\n")



# 4 Automatic Tagging

# sentences and tagged sentences
brown_tagged_sents = brown.tagged_sents(categories="news")
brown_sents = brown.sents(categories="news")


# 4.1 The Default Tagger

# get tags
tags = [tag for (word, tag) in brown.tagged_words(categories="news")]
print(nltk.FreqDist(tags).max(), "\n")

# create default tagger
raw = "I do not like green eggs and ham, I do not like them Sam I am !"
tokens = nltk.word_tokenize(raw)
default_tagger = nltk.DefaultTagger("NN")
print(default_tagger.tag(tokens), "\n")


# score default tagger
print(default_tagger.evaluate(brown_tagged_sents), "\n")



# 4.2 The Regular Expression Tagger

# list of regular expression patterns
patterns = [
    (r".*ing$", "VBG"),                # gerunds
    (r".*ed$", "VBD"),                 # simple past
    (r".*es$", "VBZ"),                 # 3rd singular present
    (r".*ould$", "MD"),                # modals
    (r".*\'s$", "NN$"),                # possessive nouns
    (r".*s$", "NNS"),                  # plural nouns
    (r"^-?[0-9]+(\.[0-9]+)?$", "CD"),  # cardinal numbers
    (r".*", "NN")                      # nouns (default)
]

regexp_tagger = nltk.RegexpTagger(patterns)
print(regexp_tagger.tag(brown_sents[3]), "\n")

# score
print(regexp_tagger.evaluate(brown_tagged_sents), "\n")



# 4.3 The Lookup Tagger

# make lookup tagger
fd = nltk.FreqDist(brown.words(categories="news"))
cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories="news"))
most_freq_words = fd.most_common(100)
likely_tags = dict((word, cfd[word].max()) for (word, _) in most_freq_words)
baseline_tagger = nltk.UnigramTagger(model=likely_tags)
print(baseline_tagger.evaluate(brown_tagged_sents), "\n")


# check on untagged input text
sent = brown.sents(categories="news")[3]
print(baseline_tagger.tag(sent), "\n")


# combine lookup tagger with default tagger
baseline_tagger = nltk.UnigramTagger(model=likely_tags,
                                     backoff=nltk.DefaultTagger("NN"))



def performance(cfd, wordlist):
    lt = dict((word, cfd[word].max()) for word in wordlist)
    baseline_tagger = nltk.UnigramTagger(model=lt, backoff=nltk.DefaultTagger("NN"))
    return baseline_tagger.evaluate(brown.tagged_sents(categories="news"))


def display():
    import matplotlib.pyplot as plt

    word_freqs = nltk.FreqDist(brown.words(categories="news")).most_common()
    words_by_freq = [w for (w, _) in word_freqs]
    cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories="news"))
    sizes = 2 ** np.arange(15)
    perfs = [performance(cfd, words_by_freq[:size]) for size in sizes]

    # plot tagger performance
    plt.figure(figsize=(6.5,5.5))
    plt.plot(sizes, perfs, color="b", linestyle="-", marker="o", markersize=3.9, linewidth=0.5)
    plt.title("Lookup Tagger Performance with Varying Model Size")
    plt.xlabel("Model Size")
    plt.ylabel("Performance")
    plt.xticks(np.arange(0, 20000, 2000))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlim(left=0, right=18000)
    plt.ylim(bottom=0, top=1)
    plt.grid()
    plt.savefig("figures/tagger_plot.png")
    plt.close()


display()




# 5 N-Gram Tagging

# 5.1 Unigram Tagging

brown_tagged_sents = brown.tagged_sents(categories="news")
brown_sents = brown.sents(categories="news")
unigram_tagger = nltk.UnigramTagger(brown_tagged_sents)
print(unigram_tagger.tag(brown_sents[2007]), "\n")

# get score
print(unigram_tagger.evaluate(brown_tagged_sents), "\n")



# 5.2 Sepsrating the Training and Testing Data

size = int(len(brown_tagged_sents) * 0.9)
print(size)

train_sents = brown_tagged_sents[:size]
test_sents = brown_tagged_sents[size:]
unigram_tagger = nltk.UnigramTagger(train_sents)
print(unigram_tagger.evaluate(test_sents))



# 5.3 General N-Gram Tagging

bigram_tagger = nltk.BigramTagger(train_sents)
print(bigram_tagger.tag(brown_sents[2007]), "\n")

unseen_sent = brown_sents[4203]
print(bigram_tagger.tag(unseen_sent), "\n")

# get accuracy
print(bigram_tagger.evaluate(test_sents), "\n")



# 5.4 Combining Taggers

t0 = nltk.DefaultTagger("NN")
t1 = nltk.UnigramTagger(train_sents, backoff=t0)
t2 = nltk.BigramTagger(test_sents, backoff=t1)
print(t2.evaluate(test_sents), "\n")

t3 = nltk.TrigramTagger(train_sents, backoff=t2)



# 5.6 Storing Taggers

# store tagger in pickle file
output = open("t2.pkl", "wb")
dump(t2, output, -1)
output.close()

# open pickle file
input = open("t2.pkl", "rb")
tagger = load(input)
input.close()

# use for tagging
text = """The board's action shows what free enterprise
is up against in our complex maze of regulatory laws."""
tokens = text.split()
print(tagger.tag(tokens), "\n")



# 5.7 Performance Limitations

cfd = nltk.ConditionalFreqDist(
    ((x[1], y[1], z[0]), z[1])
    for sent in brown_tagged_sents
    for x, y, z in nltk.trigrams(sent)
)

# ambiguous trigrams percentage
ambiguous_contexts = [c for c in cfd.conditions() if len(cfd[c]) > 1]
print(sum(cfd[c].N() for c in ambiguous_contexts) / cfd.N(), "\n")


# get confusion matrix
test_tags = [tag for sent in brown.sents(categories="editorial")
                 for (word, tag) in t2.tag(sent)]
gold_tags = [tag for (word, tag) in brown.tagged_words(categories="editorial")]
print(nltk.ConfusionMatrix(gold_tags, test_tags), "\n")



# 6 Transformation-Based Tagging

brill_demo.demo()


