import re
import nltk
from urllib import request
from bs4 import BeautifulSoup
from nltk.corpus import brown, words, abc
import pandas as pd
import random
from nltk.corpus import wordnet as wn
from nltk.corpus import udhr
import numpy as np



# 1.
s = "colorless"
s = s[0:4] + "u" + s[4:]
print(s, "\n")



# 2.
word1 = "dish-es"
word2 = "run-ning"
word3 = "nation-ality"
word4 = "un-do"
word5 = "pre-heat"
print(word1[:-3])
print(word2[:-5])
print(word3[:-6])
print(word4[:-3])
print(word5[:-5], "\n")



# 3.
test_string = "test"
# print(test_string[-5])  # goes to far left



# 4.
test_sentence = "This is a test sentence to see subscripting with different step size."

# print with step size 2
print(test_sentence[6:11:2])

# print with step size -2
print(test_sentence[10:5:-2], "\n")



# 5.
print(test_sentence[::-1], "\n")  # prints backwards



# 6.
# a.
print(f"The first regex matches strings consisting of alphabetical letters (both lower and upper case). There must be at least one character.")
print(nltk.re_show(r"[a-zA-Z]+", "happy"), "\n")

# b.
print(f"The second regex matches strings consisting of a single capital letter followed by zero or more lowercase letters.")
print(nltk.re_show(r"[A-Z][a-z]*", "Happy"), "\n")

# c.
print(f"The third regex matches strings consisting of the letter 'p' followed by at most 2 vowels and ending in t.")
print(nltk.re_show(r"p[aeiou]{,2}t", "pat"), "\n")

# d.
print(f"The fourth regex matches strings consisting of a digit one or more times followed by a period and more digits (period and extra digits can appear at most one time in matched string).")
print(nltk.re_show(r"\d+(\.\d+)?", "3.14.15"), "\n")

# e.
print(f"The fifth regex matches the empty string as well as strings consisting of concatentations of 3 character strings each of which starts and ends with a non-vowel character with a vowel in the middle.")
print(nltk.re_show(r"([^aeiou][aeiou][^aeiou])*", "patpataatcat"), "\n")

# f.
print(f"The sixth regex matches postive number of alphabetical letteres or a positive number of characters which are not alphabetical or space.")
print(nltk.re_show(r"\w+|[^\w\s]+", "%\nAb,&\n\r"), "\n")



# 7.
# a.
print(f"a[n]?|the", "\n")

# b.
print(f"\d+(\+\d+|\*\d+)+", "\n")



# 8.
def url_contents(url):
    
    # get url contents
    contents = request.urlopen(url).read().decode("utf8")

    # strip off html
    raw = BeautifulSoup(contents, "html.parser").get_text()
    
    return raw


print(url_contents("http://nltk.org/"), "\n")



# 9.
# a.
def load(f):

    # open file
    with open(f) as file:
        read_file = file.read()

    return read_file

text = load("data/corpus.txt")
print(text, "\n")

pattern = r"""(?x)  # set flag to allow verbose regexps
(\w+)[,!]?  # extract words followed by space, comma, or exclamation mark
"""


tokenizer = nltk.regexp_tokenize
print(tokenizer(text, pattern), "\n")


# b.
text = "$3.14 $5 Alex Barry C DE words 11/12/2023 11-12-2023"
pattern = r"""(?x)  # set flag to allow verbose regexps
\$\d+(?:\.\d+)?
| [A-Z][a-z]+
| \d{1,2}/\d{1,2}/\d{4}
| \d{1,2}-\d{1,2}-\d{4}
"""
print(tokenizer(text, pattern), "\n")



# 10.
sent = ["The", "dog", "gave", "John", "the", "newspaper"]
result = []
print([(word, len(word)) for word in sent], "\n")



# 11.
raw = "This is a test sentence which I will tokenize using a non-space character."
print(raw.split(sep="e"), "\n")



# 12.
for c in "Test string":
    print(c)
print("\n")



# 13.
print("Test string".split())
print("Test string".split(" "))
print("Test\tstring".split())
print("Test\t    \t\t\t\t string".split())
print("\n")



# 14.
words_list = ["This", "is", "a", "list", "of", "various", "words"]
print(words_list)  # print list
print(sorted(words_list))  # returns sorted list but keeps original list as is
print(words_list)  # same as earlier
print(words_list.sort())  # modified sequence in place and returns None
print(words_list)  # this is now sorted
print("\n")



# 15.
print("3"*7)  # prints seven 3's
print(3*7)  # prints 21
print(int("3")*7)  # prints 21
print(str(3)*7)  # prints seven 3's
print("\n")



# 16.
from data.prog import monty
print(monty, "\n")



# 17.
short_word = "short"
word = "much_longer"
print("Test string, %6s" % short_word)  # right aligned
print("Test string, %-6s" % short_word)  # left aligned
print("Test string, %6s" % word)  # neither left nor right aligned since exceeds space
print("Test string, %-6s" % word)  # neither left nor right aligned since exceeds space
print("\n")



# 18.
learned_words = " ".join(brown.words(categories="learned"))
tokenized_words = " ".join(tokenizer(learned_words, r"\w+"))
wh_words = re.findall(r"wh(?:o|at|ich|ere|en|om|ose|y)", tokenized_words)
print(sorted(wh_words))

print("\n")
fd = nltk.FreqDist(wh_words)
fd.tabulate()
print("\n")



# 19.
with open("data/word_frequencies.txt", "r+") as f:
    word_frequencies_list = []
    for line in f:
        word_frequencies_list.append(line.split())

for pair in word_frequencies_list:
    pair[1] = int(pair[1])
print(word_frequencies_list, "\n")



# 20.
url = "https://weather.gc.ca/forecast/hourly/on-165_metric_e.html"
html = request.urlopen(url).read().decode("utf8")
raw = BeautifulSoup(html, "lxml")
table_rows = raw.find_all("tr")

time = []
temp = []
weather = []
pop = []

for row in table_rows:
    times = row.find("td", {"headers": "header1"})
    if times != None:
        time.append(times.text)

    temps = row.find("td", {"headers": "header2"})
    if temps != None:
        temp.append(int(temps.text))

    weather_conditions = row.find("td", {"headers": "header3"})
    if weather_conditions != None:
        weather.append(weather_conditions.text)

    pop_likelihood = row.find("td", {"headers": "header4"})
    if pop_likelihood != None:
        pop.append(pop_likelihood.text + "%")

weather_df = pd.DataFrame(columns=["Time", "Temp.", "Conditions", "POP"])

weather_df["Time"] = time
weather_df["Temp."] = temp
weather_df["Conditions"] = weather
weather_df["POP"] = pop

print(weather_df)
print("\n")



# 21.
def unknown(url):
    html = request.urlopen(url).read().decode("utf8")
    # soup = BeautifulSoup(html, "lxml")
    # text = " ".join(soup.text.split())
    lowercase_words = set(re.findall(r"\b[a-z]+\b", html))
    unknown_words = set(word for word in lowercase_words if word not in words.words())

    js_words = ["href", "png", "rval", "xlink", "jquery", "noscript",
                "onclick", "gs", "pv", "css", "cmd", "haspopup",
                "lang", "reir", "rgba", "vkontakte", "hf", "ux",
                "ul", "mv", "navp", "mb", "progid", "flexbox", "subnav",
                "svg", "odnoklassniki", "src", "ol", "rms", "vd", "qa",
                "btn", "cta", "mt", "gnl", "img", "len", "edr", "nw",
                "smp", "dir", "ls", "jssignals", "jpg", "gt", "gn", "av",
                "js", "uid", "lx", "uri", "config", "nojs", "hreflang",
                "gb", "waf", "renren", "fff", "concat", "ptrt", "ws",
                "eol", "mjs", "eles", "cx", "ngas", "pw", "emp", "pb"]
    non_js_unknown_words = set(word for word in unknown_words if word not in js_words)
    return non_js_unknown_words

# print(unknown("https://weather.gc.ca/forecast/hourly/on-165_metric_e.html"))
# print("\n")



# 22.
# bbc_unknown_words = unknown("http://news.bbc.co.uk/")
# print(bbc_unknown_words)



# 23.
test_word = "don't"
test_regex = r"(\w+)(n't)"
split_word = re.findall(test_regex, test_word)
print(split_word)
print(f"The given regular expression will not work since it will include 'n' with 'do'.\n")



# 24.
hAck3r_dict = {"ate":"8", "e":"3", "i":"1", "o":"0", "l":"|", "^s":"$",
               "(?=.+)s": "5",  # used lookahead to check that there is a chracter in front of s
               "\.":"5w33t!"}

def hAck3r_converter(text):
    text = text.lower()
    for entry in hAck3r_dict:
        text = re.sub(entry, hAck3r_dict[entry], text)
    return text

print(hAck3r_converter("slaepios ate ."), "\n")



# 25.
# a.
def pig_latin_converter(word):
    if word[0:2].lower() == "qu":
        if word[0] == "Q":
            word = "q" + word[1:]
            return word[2].upper() + word[3:] + word[0:2] + "ay"
        else:
            return word[2:] + word[0:2] + "ay"
    elif word[0].islower():
        return re.sub(r"^([^aeiou]*)([a-z]*)(\.){,1}", r"\2\1ay\3", word)
    elif word[0] in ["A", "E", "I", "O", "U"]:
        return word + "ay"
    else:
        word = word[0].lower() + word[1:]
        captures = re.findall(r"^[^aeiou]*([aeiou])", word)
        intermediate = re.sub(r"^([^aeiou]*)[aeiou](.*)", r"\2\1ay", word)
        return captures[0].upper() + intermediate

print(pig_latin_converter("string"))

sentence = "This is a test sentence for pig latin."
pig_latin_sentence =  [pig_latin_converter(word) for word in sentence.split()]
print(pig_latin_sentence, "\n")

# b.
def pig_latin_sentence(sentence):
    return " ".join([pig_latin_converter(word) for word in sentence.split()])
print(pig_latin_sentence(sentence), "\n")

print(pig_latin_converter("Stray"))

# c.
print(pig_latin_sentence("Quiet quiet"), "\n")



# 26.
url = "https://hu.wikipedia.org/wiki/Magyar_nyelv"
html = request.urlopen(url).read().decode("utf8")
soup = BeautifulSoup(html, "lxml").text
tokens = re.findall(r"[a-zA-Z]{2,}", soup)
vowel_pairs = [vp for word in tokens for vp in re.findall("[aeiou][aeiou]", word)]
vowel_pairs_fd = nltk.ConditionalFreqDist(vowel_pairs)
vowel_pairs_fd.tabulate()
print("\n")



# 27.
def random_string(n):
    random_list = []
    for i in range(n):
        random_list.append(random.choice("aehh "))
    return "".join(random_list)

print(" ".join(random_string(500).split()), "\n")



# 29.
lore_text = " ".join(brown.words(categories="lore"))
learned_text = " ".join(brown.words(categories="learned"))
lore_text = re.findall(r"[a-zA-Z]+", lore_text)
learned_text = re.findall(r"[a-zA-Z]+", learned_text)

lore_sentences = brown.sents(categories="lore")
learned_sentences = brown.sents(categories="learned")

def average_number_letters(text_words_list):
    return sum([len(word) for word in text_words_list])/len(text_words_list)

def average_number_words(text_words_list, text_sentences_list):
    return len(text_words_list)/len(text_sentences_list)

def automated_readability_index(category):
    raw_words = " ".join(brown.words(categories=category))
    processed_words = re.findall(r"[a-zA-Z]+", raw_words)

    sentences = brown.sents(categories=category)

    return 4.71*average_number_letters(processed_words) + 0.5*average_number_words(processed_words, sentences) - 12.43

print(f"lore ARI scoe: {automated_readability_index('lore')}")
print(f"learned ARI scoe: {automated_readability_index('learned')}", "\n")



# 30.
porter_stem = nltk.PorterStemmer()
lancaster_stem = nltk.LancasterStemmer()

text = brown.words(categories="humor")
text = text[:100]

print([porter_stem.stem(t) for t in text], "\n")
print([lancaster_stem.stem(t) for t in text], "\n")



# 31.

saying = ["After", "all", "is", "said", "and", "done", ",", "more",
          "is", "said", "than", "done", "."]

lengths = []
for word in saying:
    lengths.append(len(word))

lengths2 = [len(word) for word in saying]
print(lengths == lengths2, "\n")



# 32.
silly = "newly formed bland ideas are inexpressible in an infuriating way"

# a.
bland = silly.split()
print(bland, "\n")

# b.
second_letters = "".join([word[1] for word in bland])
print(second_letters, "\n")

# c.
joined_word = " ".join(bland)
print(joined_word, "\n")

# d.
alphabetical_words = sorted(bland)
print(alphabetical_words, "\n")



# 33.

# a.
print("inexpressible".index("re"))
print(f"we get the first index of the first occurrence of the substring\n")

# b.
words = ["this", "is", "a", "list", "of", "some", "words"]
print(f"index position of 'list' is: {words.index('list')}\n")

# c.
print(silly[:silly.index("in") + len("inexpressible")], "\n")



# 34.
url = "https://en.wikipedia.org/wiki/List_of_adjectival_and_demonymic_forms_of_place_names"
html = request.urlopen(url).read().decode("utf8")
soup = BeautifulSoup(html, "lxml")
table = soup.find("span", {"id": "Canadian_provinces_and_territories"}).find_next("table")
table_rows = table.find_all("tr")[2:]

province_or_territory = []
demonym = []

for row in table_rows:
    row_data = row.find_all("td")
    province_or_territory.append(row_data[0].text)
    demonym.append(row_data[2].text)

province_or_territory = [re.sub(r"\s+\(\w+\)", "", name) for name in province_or_territory]  # remove bracketed word
demonym = [re.sub(r"\n", "", word) for word in demonym]  # remove new line characters
demonym = [re.findall(r"[A-Z][a-z]+(?:\s[A-Z][a-z]+){,2}", word)[0] for word in demonym]  # filter by spaced words starting with capitals, take first one

demonym_df = pd.DataFrame(columns=["Province_Territory", "Demonym"])
demonym_df["Province_Territory"] = province_or_territory
demonym_df["Demonym"] = demonym

print(demonym_df)

def get_demonym(name):
    names_list = name.split()
    last_name = names_list[-1]  # get last part of name
    if last_name[-1] in r"[aeiou]":  # if ends in vowel drop last letter and add 'an'
        return name[:-1] + "an"
    elif last_name[-2:] == "es":  # if ends in 'es' drop last two letters and add 'an'
        return name[:-2] + "an"
    elif last_name[-2:] == "an":  # if ends in 'an' drop last letter and add 'ian'
        return name + "ian"
    elif last_name[-1] == "t":  # if ends in 't' drop last letter and add 'mmiut'
        return name[:-1] + "mmiut"
    else:  # if ends in consonant add 'er'
        return name + "er"


print([get_demonym(name) for name in demonym_df.Province_Territory])
print("\n")



# 35.
pronoun = ["i", "you", "he", "she", "it", "we", "they"]
phrase1 = "as best as "
phrase2 = "as best "
pronoun_phrase1 = [phrase1 + p for p in pronoun]
pronoun_phrase2 = [phrase2 + p + " can" for p in pronoun]


url = "http://itre.cis.upenn.edu/~myl/languagelog/archives/002733.html"
html = request.urlopen(url).read().decode("utf8")
soup = BeautifulSoup(html, "lxml")
red_text = soup.find_all("blockquote")  # look for blockquotes
text = ""
for p in red_text:
    intermediate_text = ""
    intermediate_text = re.sub(r"\s+|\.{2,}", " ", p.text)  # remove ellipsis and extra space
    intermediate_text = re.sub(r"^\s+", "", intermediate_text)  # remove leading space
    text += intermediate_text + "\n"  # add each piece of text separated by new line

# make all lower case
text = text.lower()

# search for first type of pronoun phrase
print(f"First type of phrase")
for entry in pronoun_phrase1:
    print(re.findall(entry, text))
    print("\n")

print(f"Second type of phrase")
# search for secnod type of pronoun phrase
for entry in pronoun_phrase2:
    print(re.findall(entry, text))
    print("\n")



# 36.
lolcat_genesis = nltk.corpus.genesis.words("lolcat.txt")
print([word for word in lolcat_genesis])



# 37.
# help(re.sub)



# 38.

# a.
def find_line_broken(word):
    return re.findall(r"[a-zA-Z]+-\n[a-zA-Z]+", word)

print(find_line_broken("test-\nword"))
print(find_line_broken("another test-\nword"))

# b.
def remove_newline_character(word_list):
    return_list = []
    for word in word_list:
        return_list.append(re.sub("\n", "", word))
    return return_list

print(remove_newline_character(find_line_broken("here is a-\nlist of test-\nwords")))

# c.
print(f"You can check if the portion of the word before the hyphen is a member of common words (e.g. word corpus). If not, then the hyphen may be just due to not fitting on the line.")
print("\n")



# 39.
def american_soundex(word, sql=False):

    # store first letter
    first_letter = word[0]
    return_word = word

    # replace vowels with @
    return_word = re.sub(r"[aeiouAEIOU]", "@", return_word)

    # replace 'h', 'w', and 'y' with 0
    return_word = re.sub(r"[hwyHWY]", r"0", return_word)

    # use soundex pattern for replacing remaining constants
    return_word = re.sub(r"[bfpvBFPV]", "1", return_word)
    return_word = re.sub(r"[cgjkqsxzCGJKQSXZ]", r"2", return_word)
    return_word = re.sub(r"[dtDT]", r"3", return_word)
    return_word = re.sub(r"[lL]", r"4", return_word)
    return_word = re.sub(r"[mnMN]", r"5", return_word)
    return_word = re.sub(r"[rR]", r"6", return_word)

    # store replacement for first letter for later comparison
    first_letter_digit = return_word[0]

    # replace repeated numbers with single instance
    return_word = re.sub(r"1{2,}", "1", return_word)
    return_word = re.sub(r"2{2,}", "2", return_word)
    return_word = re.sub(r"3{2,}", "3", return_word)
    return_word = re.sub(r"4{2,}", "4", return_word)
    return_word = re.sub(r"5{2,}", "5", return_word)
    return_word = re.sub(r"6{2,}", "6", return_word)

    # split based on sql algorithm or standard
    if not sql:
        # replace instancs of the same number being separated by a zero with single instance of digit
        return_word = re.sub(r"1[01]+", "1", return_word)
        return_word = re.sub(r"2[02]+", "2", return_word)
        return_word = re.sub(r"3[03]+", "3", return_word)
        return_word = re.sub(r"4[04]+", "4", return_word)
        return_word = re.sub(r"5[05]+", "5", return_word)
        return_word = re.sub(r"6[06]+", "6", return_word)


    return_word = re.sub(r"[0@]", "", return_word)

    # if current first digit matches conversion of original first letter then omit first digit and add capital version of original starting letter to remaining digits
    if first_letter_digit == return_word[0]:
        return_word = first_letter.upper() + return_word[1:]
    else:  # otherwise append capital version of original first letter to remaining digits
        return_word = first_letter.upper() + return_word

    # if length is smaller than 4 append zeroes until string is of length 4
    if len(return_word) < 4:
        return return_word + "0"*(4 - len(return_word))
    else:  # otherwise return first 4 characters
        return return_word[0:4]
    
        
    
print(american_soundex("Robert", False))
print(american_soundex("Rupert", False))
print(american_soundex("Tymczak", False))
print(american_soundex("Honeyman", False), "\n")



# 40.
# abc.words()



# 41.
words = ["attribution", "confabulation", "elocution",
         "sequoia", "tenacious", "unidirectional"]
print(sorted(set("".join(re.findall(r"[aeiou]", word)) for word in words)), "\n")



# 42.
class IndexedText(object):

    def __init__(self, stemmer, text):
        self._text = text
        self._stemmer = stemmer
        self._index = nltk.Index((wn.synsets(self._stem(word))[0].offset(), i)
                                 for (i, word) in enumerate(text)
                                 if wn.synsets(self._stem(word)) ) 

    def concordance(self, word, width=40):
        key = self._stem(word)
        wc = int(width/4)                # words of context
        for i in self._index[key]:
            lcontext = ' '.join(self._text[i-wc:i])
            rcontext = ' '.join(self._text[i:i+wc])
            ldisplay = '{:>{width}}'.format(lcontext[-width:], width=width)
            rdisplay = '{:{width}}'.format(rcontext[:width], width=width)
            print(ldisplay, rdisplay)

    def _stem(self, word):
        return self._stemmer.stem(word).lower()

porter = nltk.PorterStemmer()
grail = nltk.corpus.webtext.words("grail.txt")
text = IndexedText(porter, grail)
text.concordance("lie")



# 43.
# get latin1 languages
latin1_languages = [language for language in udhr.fileids() if "Latin1" in language]

# set random seed for reproducibility
random.seed(42)
indices = []  # empty indices to store languages
number_of_languages = 100 # 5

# get 5 random latin1 languages
for i in range(number_of_languages):
    indices.append(random.choice(range(len(latin1_languages))))
target_languages = [latin1_languages[i] for i in indices]

# get frequency distribution for each language
target_languages_fds = [nltk.FreqDist(udhr.words(language)) for language in target_languages]

# normalize frequency distributions for each language
for i in range(len(target_languages_fds)):
    total = target_languages_fds[i].N()
    for word in target_languages_fds[i]:
        target_languages_fds[i][word] /= float(total) 

# get frequency distribution on text
random_language = target_languages[np.random.choice(number_of_languages)]
random_text = udhr.words(random_language)
random_text_fd = nltk.FreqDist(random_text)

# normalize frequency distribution on text
random_total = random_text_fd.N()
for word in random_text_fd:
    random_text_fd[word] /= random_total

# array to store Pearson scores
scores = []

# get Pearson scores
for i in range(len(target_languages_fds)):
    scores.append(nltk.spearman_correlation(random_text_fd, target_languages_fds[i]))

print(scores)
print(random_language)
print(target_languages)

def guess_language(number_of_languages, random_sample_size=None):
    # get random language and its text
    random_language = target_languages[np.random.choice(number_of_languages)]
    random_text = udhr.words(random_language)

    # get random sample of words
    if random_sample_size != None:
        random_sample_size = min(random_sample_size, len(random_text))  # ensure sample size does not exceed length of text
        random_sample_indices = np.random.choice(len(random_text), random_sample_size)  # get random sample of indices
        random_text = [random_text[i] for i in random_sample_indices]  # get associated words at random indices

    # get frequency distribution
    random_text_fd = nltk.FreqDist(random_text)

    # normalize frequency distribution on text
    random_total = random_text_fd.N()
    for word in random_text_fd:
        random_text_fd[word] /= random_total

    # array to store Pearson scores
    scores = []

    # get Pearson scores
    for i in range(len(target_languages_fds)):
        scores.append(nltk.spearman_correlation(random_text_fd, target_languages_fds[i]))

    # get index of most correlated language
    best_approximation_index = np.argmax(np.array(scores))

    # get best approximation
    best_approximation = target_languages[best_approximation_index]

    return best_approximation, random_language


def get_accuracy(number_of_languages, number_of_tests=1, random_sample_size=None):
    
    # empty array for scores
    accuracy_scores = []

    for i in range(number_of_tests):
        # get language guess and actual language
        approximation, actual = guess_language(number_of_languages=number_of_languages, random_sample_size=random_sample_size)

        # check if prediction matches actual language of text
        accuracy_scores.append(approximation == actual)
    
    # get accuracy
    return sum(accuracy_scores)/len(accuracy_scores)


print(f"accuracy: {get_accuracy(number_of_languages=number_of_languages, number_of_tests=100, random_sample_size=100)}")
print("\n")



# 44.
sentence = "An unusual test sentence."
split_sentence = sentence.split()

for word in split_sentence:

    # get a synset of word corresponding to context using lesk
    context_synset = nltk.wsd.lesk(split_sentence, word)

    # get all synsets of word
    synsets_list = wn.synsets(word)

    # if the context synset and synset list are not empty compute path similarities
    if context_synset and synsets_list:
        print(f"path similarity of synsets of '{word}' and {context_synset}:")
        for syn in synsets_list:
            print(context_synset.path_similarity(syn))
        print("\n")



# 45.


























