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



# Q1: Define a string s = 'colorless'. Write a Python statement that changes this to "colourless" using only
#     the slice and concatenation operations.

s = "colorless"
s = s[0:4] + "u" + s[4:]
print(s, "\n")



# Q2: We can use the slice notation to remove the morphological endings on words. For example, 'dogs'[:-1]
#     removes the last character of dogs, leaving dog. Use slice notation to remove the affixes from these words
#     (we've inserted a hyphen to indicate the affix boundary, but omit this from your strings): dish-es, run-ning,
#     nation-ality, un-do, pre-heat.

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



# Q3: We saw how we can generate an IndexError by indexing beyond the end of a string. Is it possible to
#     construct an index that goes too far to the left, before the start of the string?

test_string = "test"
# print(test_string[-5])  # goes to far left



# Q4: We can specify a "step" size for the slice. The following returns every second character within the slice:
#     monty[6:11:2]. It also works in the reverse direction: monty[10:5:-2]. Try these for yourself, then
#     experiment with different step values.

test_sentence = "This is a test sentence to see subscripting with different step size."

# print with step size 2
print(test_sentence[6:11:2])

# print with step size -2
print(test_sentence[10:5:-2], "\n")



# Q5: What happens if you ask the interpreter to evaluate monty[::-1]? Explain why this is a reasonable result

print(test_sentence[::-1], "\n")  # prints backwards, reasonable since it considers full slice with -1 step size



# Q6: Describe the class of strings matched by the following regular expressions.

# a: [a-zA-Z]+

print(f"The first regex matches strings consisting of alphabetical letters (both lower and upper case). There must be at least one character.")
print(nltk.re_show(r"[a-zA-Z]+", "happy"), "\n")

# b: [A-Z][a-Z]*

print(f"The second regex matches strings consisting of a single capital letter followed by zero or more lowercase letters.")
print(nltk.re_show(r"[A-Z][a-z]*", "Happy"), "\n")

# c: p[aeiou]{,2}t

print(f"The third regex matches strings consisting of the letter 'p' followed by at most 2 vowels and ending in t.")
print(nltk.re_show(r"p[aeiou]{,2}t", "pat"), "\n")

# d: \d+(\.\d+)?

print(f"The fourth regex matches strings consisting of a digit one or more times followed by a period and more digits (period and extra digits can appear at most one time in matched string).")
print(nltk.re_show(r"\d+(\.\d+)?", "3.14.15"), "\n")

# e: ([^aeiou][aeiou][^aeiou])*

print(f"The fifth regex matches the empty string as well as strings consisting of concatentations of 3 character strings each of which starts and ends with a non-vowel character with a vowel in the middle.")
print(nltk.re_show(r"([^aeiou][aeiou][^aeiou])*", "patpataatcat"), "\n")

# f: \w+|[^\w\s]+
print(f"The sixth regex matches postive number of alphabetical letteres or a positive number of characters which are not alphabetical or space.")
print(nltk.re_show(r"\w+|[^\w\s]+", "%\nAb,&\n\r"), "\n")



# Q7: Write regular expression to match the following classes of strings:

# a: A single determiner (assume that a, an, and the are the only determiners).

print(f"a[n]?|the", "\n")

# b: An arithmetic expression using integers, addition, and multiplication, such as 2*3+8.

print(f"\d+(\+\d+|\*\d+)+", "\n")



# Q8: Write a utility function that takes a URL as its argument, and returns the contents of the URL, with all
#     HTML markup removed. Use from urllib import request and then
#     request.urlopen('http://nltk.org/').read().decode('utf8') to access the contents of the URL.

def url_contents(url):
    
    # get url contents
    contents = request.urlopen(url).read().decode("utf8")

    # strip off html
    raw = BeautifulSoup(contents, "html.parser").get_text()
    
    return raw


print(url_contents("http://nltk.org/"), "\n")



# Q9: Save some text into a file corpus.txt. Define a function load(f) that reads from the file named in its sole
#     argument, and returns a string containing the text of the file.

# a: Use nltk.regexp_tokenize() to create a tokenizer that tokenizes the various kinds of punctuation in
#    this text. Use one multi-line regular expression, with inline comments, using the verbose flag (?x).

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


# b: Use nltk.regexp_tokenize() to create a tokenizer that tokenizes the following kinds of expression:
#    monetary amounts; dates; names of people and organizations.

text = "$3.14 $5 Alex Barry C DE words 11/12/2023 11-12-2023"
pattern = r"""(?x)  # set flag to allow verbose regexps
\$\d+(?:\.\d+)?
| [A-Z][a-z]+
| \d{1,2}/\d{1,2}/\d{4}
| \d{1,2}-\d{1,2}-\d{4}
"""
print(tokenizer(text, pattern), "\n")



# Q10: Rewrite the following loop as a list comprehension
#
#      >>> sent = ['The', 'dog', 'gave', 'John', 'the', 'newspaper']
#      >>> result = []
#      >>> for word in sent:
#      ...     word_len = (word, len(word))
#      ...     result.append(word_len)
#      >>> result
#      [('The', 3), ('dog', 3), ('gave', 4), ('John', 4), ('the', 3), ('newspaper', 9)]


sent = ["The", "dog", "gave", "John", "the", "newspaper"]
result = []
print([(word, len(word)) for word in sent], "\n")



# Q11: Define a string raw containing a sentence of your own choosing. Now, split raw on some character other
#      than space, such as 's'.

raw = "This is a test sentence which I will tokenize using a non-space character."
print(raw.split(sep="e"), "\n")



# Q12: Write a for loop to print out the characters of a string, one per line.

for c in "Test string":
    print(c)
print("\n")



# Q13: What is the difference between calling split on a string with no argument or with ' ' as the argument,
#      e.g. sent.split() versus sent.split(' ')? What happens when the string being split contains tab
#      characters, consecutive space characters, or a sequence of tabs and spaces? (In IDLE you will need to use
#      '\t' to enter a tab character.)

print("Test string".split())
print("Test string".split(" "))
print("Test\tstring".split())
print("Test\t    \t\t\t\t string".split())
print("\n")



# Q14: Create a variable words containing a list of words. Experiment with words.sort() and sorted(words).
#      What is the difference?

words_list = ["This", "is", "a", "list", "of", "various", "words"]
print(words_list)  # print list
print(sorted(words_list))  # returns sorted list but keeps original list as is
print(words_list)  # same as earlier
print(words_list.sort())  # modified sequence in place and returns None
print(words_list)  # this is now sorted
print("\n")



# Q15: Explore the difference between strings and integers by typing the following at a Python prompt: "3" * 7
#      and 3 * 7. Try converting between strings and integers using int("3") and str(3).

print("3"*7)  # prints seven 3's
print(3*7)  # prints 21
print(int("3")*7)  # prints 21
print(str(3)*7)  # prints seven 3's
print("\n")



# Q16: Use a text editor to create a file called prog.py containing the single line monty = 'Monty Python'. Next,
#      start up a new session with the Python interpreter, and enter the expression monty at the prompt. You will get
#      an error from the interpreter. Now, try the following (note that you have to leave off the .py part of the
#      filename):
#
#      >>> from prog import monty
#      >>> monty
#
#      This time, Python should return a value. You could also try import prog, in which case Python should be
#      able to evaluate the expression prog.monty at the prompt.

from data.prog import monty
print(monty, "\n")



# Q17: What happens when the formatting strings %6s and %-6s are used to display strings that are longer than
#      six characters?

short_word = "short"
word = "much_longer"
print("Test string, %6s" % short_word)  # right aligned
print("Test string, %-6s" % short_word)  # left aligned
print("Test string, %6s" % word)  # neither left nor right aligned since exceeds space
print("Test string, %-6s" % word)  # neither left nor right aligned since exceeds space
print("\n")



# Q18: Read in some text from a corpus, tokenize it, and print the list of all wh-word types that occur. (wh-words
#      in English are used in questions, relative clauses and exclamations: who, which, what, and so on.) Print
#      them in order. Are any words dupllicated in this list, because of the presence of case distinctions or
#      punctuation?

learned_words = " ".join(brown.words(categories="learned"))
tokenized_words = " ".join(tokenizer(learned_words, r"\w+"))
wh_words = re.findall(r"wh(?:o|at|ich|ere|en|om|ose|y)", tokenized_words)
print(sorted(wh_words))

print("\n")
fd = nltk.FreqDist(wh_words)
fd.tabulate()
print("\n")



# Q19: Create a file consisting of words and (made up) frequencies, where each line consists of a word, the
#     space character, and a positive integer, e.g. fuzzy 53. Read the file into a Python list using
#     open(filename).readlines(). Next, break each line into its two fields using split(), and convert the
#     number into an integer using int(). The result should be a list of the form: [['fuzzy', 53], ...].

with open("data/word_frequencies.txt", "r+") as f:
    word_frequencies_list = []
    for line in f:
        word_frequencies_list.append(line.split())

for pair in word_frequencies_list:
    pair[1] = int(pair[1])
print(word_frequencies_list, "\n")



# Q20: Write code to access a favorite webpage and extract some text from it. For example, access a weather site
#      and extract the forecast top temperature for your own town or city today.

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



# Q21: Write a function unknown() that takes a URL as its argument, and returns a list of unknown words that
#      occur on that webpage. In order to do this, extract all substrings consisting of lowercase letters (using
#      re.findall()) and remove any items from this set that occur in the Words Corpus (nltk.corpus.words). Try
#      to categorize these words manually and discuss your findings.

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



# Q23: Are you able to write a regular expression to tokenize text in such a way that the word don't is tokenized
#      into do and n't? Explain why this regular expression won't work: <<n't|\w+>>

test_word = "don't"
test_regex = r"(\w+)(n't)"
split_word = re.findall(test_regex, test_word)
print(split_word)
print(f"The given regular expression will not work since it will include 'n' with 'do'.\n")



# Q24: Try to write code to convert text into hAck3r, using regular expressions and substitution, where e->3, i
#      ->1, o->0, l->|, s->5, .->5w33t!, ate->8. Normalize the text to lowercase before converting it. Add
#      more substitutions of your own. Now try to map s to two different values: $ for word-initial s, and 5 for
#      word-internal s.

hAck3r_dict = {"ate":"8", "e":"3", "i":"1", "o":"0", "l":"|", "^s":"$",
               "(?=.+)s": "5",  # used lookahead to check that there is a chracter in front of s
               "\.":"5w33t!"}

def hAck3r_converter(text):
    text = text.lower()
    for entry in hAck3r_dict:
        text = re.sub(entry, hAck3r_dict[entry], text)
    return text

print(hAck3r_converter("slaepios ate ."), "\n")



# Q25: Pig Latin is a simple transformation of English text. Each word of the text is converted as follows: move
#      any consonant (or consonant cluster) that appears at the start of the word to the end, then append ay, e.g.
#      string->ingstray, idle->idleay. http://en.wikipedia.org/wiki/Pig_Latin

# a: Write a function to convert a word to Pig Latin.

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

# b: Write code that converts text, instead of individual words.

def pig_latin_sentence(sentence):
    return " ".join([pig_latin_converter(word) for word in sentence.split()])
print(pig_latin_sentence(sentence), "\n")

print(pig_latin_converter("Stray"))

# c: Extend it further to preserve capitalization, to keep qu together (i.e. so that quiet becomes ietquay),
#    and to detect when y is used as a consonant (e.g. yellow) vs a vowl (e.g. style)

print(pig_latin_sentence("Quiet quiet"), "\n")



# Q26: Download some text from a language that has vowel harmony (e.g. Hungarian), extract the vowel
#      sequence of words, and create a vowel bigram table

url = "https://hu.wikipedia.org/wiki/Magyar_nyelv"
html = request.urlopen(url).read().decode("utf8")
soup = BeautifulSoup(html, "lxml").text
tokens = re.findall(r"[a-zA-Z]{2,}", soup)
vowel_pairs = [vp for word in tokens for vp in re.findall("[aeiou][aeiou]", word)]
vowel_pairs_fd = nltk.ConditionalFreqDist(vowel_pairs)
vowel_pairs_fd.tabulate()
print("\n")



# Q27: Python's random module includes a function choice() which randomly chooses an item from a sequence,
#      e.g. choice("aehh ") will produce one of four possible characters, with the letter h being twice as frequent
#      as the others. Write a generator expression that produces a sequence of 500 randomly chosen letters drawn
#      from the string "aehh ", and put this expression inside a call to the ''.join() function, to concatenate them
#      into one long string. You should get a result that looks like uncontrolled sneezing or maniacal laughter:
#      he  haha ee  heheeh eha. Use split() and join() again to normalize the whitespace in this string. 

def random_string(n):
    random_list = []
    for i in range(n):
        random_list.append(random.choice("aehh "))
    return "".join(random_list)

print(" ".join(random_string(500).split()), "\n")



# Q29: Readability measures are used to score the reading difficulty of a text, for the purposes of selecting texts
#      of appropriate difficulty for language learners. Let us define μw to be the average number of letters per
#      word, and μs to be the average number of words per sentence, in a guven text. The automated Readability
#      Index (ARI) of the text is defined to be: 4.71 μw + 0.5 μs - 21.43. Compute the ARI score for various
#      sections of the Brown Corpus, including section f (lore) and j (learned). Make use of the fact that
#      nltk.corpus.brown.words() produces a sequence of words, while nltk.corpus.brown.sents() produces a
#      sequence of sentences.

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



# Q30: Use the Porter Stemmer to normalize some tokenized text, calling the stemmer on each word. Do the
#      same thing with the Lancaster Stemmer and see if you observe any differences.

porter_stem = nltk.PorterStemmer()
lancaster_stem = nltk.LancasterStemmer()

text = brown.words(categories="humor")
text = text[:100]

print([porter_stem.stem(t) for t in text], "\n")
print([lancaster_stem.stem(t) for t in text], "\n")



# Q31: Define the variable saying to contain the list
#      ['After', 'all', 'is', 'said', 'and', 'done', ',', 'more',
#      'is', 'said', 'than', 'done', '.']. Process this list using a for loop, and store the length of each word
#      in a new list lengths. Hint: begin by assigning the empty list to lengths, using lengths = []. Then each
#      time through the loop, use append() to add another length value to the list. Now do the same thing using a
#      list comprehension

saying = ["After", "all", "is", "said", "and", "done", ",", "more",
          "is", "said", "than", "done", "."]

lengths = []
for word in saying:
    lengths.append(len(word))

lengths2 = [len(word) for word in saying]
print(lengths == lengths2, "\n")



# Q32: Define a variable silly to contain the string:
#      'newly formed bland ideas are inexpressible in an infuriating
#      way'. (This happens to be the legitimate interpretation that bilingual English-Spanish speakers can assign to
#      Chomsky's famous nonsense phrase, colorless green ideas sleep furiously according to Wikipedia). Now
#      write code to perform the following tasks

silly = "newly formed bland ideas are inexpressible in an infuriating way"

# a: Split silly into a list of strings, one per word, using Python's split() operation, and save this to a
#    variable called bland.

bland = silly.split()
print(bland, "\n")

# b: Extract the second letter of each word in silly and join them into a string, to get 'eoldrnnnna'.

second_letters = "".join([word[1] for word in bland])
print(second_letters, "\n")

# c: Combine the words in bland back into a single string, using join(). Make sure the words in the
#    resulting string are separated with whitespace.

joined_word = " ".join(bland)
print(joined_word, "\n")

# d: Print the words of silly in alphabetical order, one per line.

alphabetical_words = sorted(bland)
print(alphabetical_words, "\n")



# 33: The index() function can be used to look up items in sequences. For example,
#     'inexpressible'.index('e') tells us the index of the first position of the letter e.

# a: What happens when you look up a substring e.g. 'inexpressible'.index('re')?

print("inexpressible".index("re"))
print(f"we get the first index of the first occurrence of the substring\n")

# b: Define a variable words containing a list of words. Now use words.index() to look up a position of
#    an individual word.

words = ["this", "is", "a", "list", "of", "some", "words"]
print(f"index position of 'list' is: {words.index('list')}\n")

# c: Define a variable silly as in the exercise above. Use the index() function in combination with list
#    slicing to build a list phrase consisting of all the words up to (but not including) in in silly

print(silly[:silly.index("in") + len("inexpressible")], "\n")



# Q34: Write code to convert nationality adjectives like Canadian and Australian to their corresponding nouns
#      Canada and Australia (see http://en.wikipedia.org/wiki/List_of_adjectival_forms_of_place_names).

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



# Q35: Read the LanguageLog post on phrases of the form as best as p can and as best p can, where p is a
#      pronoun. Investigate this phenomenon with the help of a corpus and the findall() method for searching
#      tokenized text described in 3.5. http://itre.cis.upenn.edu/~myl/languagelog/archives/002733.html

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



# Q38: An interesting challenge for tokenization is words that have been split across a line-break. E.g. if long-
#      term is split, then we have the string long-\nterm.

# a: Write a regular expression that identifies words that are hyphenated at a line-break. The expression
#    will need to include the \n character.

def find_line_broken(word):
    return re.findall(r"[a-zA-Z]+-\n[a-zA-Z]+", word)

print(find_line_broken("test-\nword"))
print(find_line_broken("another test-\nword"))

# b: Use re.sub() to remove the \n character from these words.

def remove_newline_character(word_list):
    return_list = []
    for word in word_list:
        return_list.append(re.sub("\n", "", word))
    return return_list

print(remove_newline_character(find_line_broken("here is a-\nlist of test-\nwords")))

# c: How might you identify words that should not remain hyphenated once the newline is removed, e.g.
#    'encyclo-\npedia'?

print(f"You can check if the portion of the word before the hyphen is a member of common words (e.g. word corpus). If not, then the hyphen may be just due to not fitting on the line.")
print("\n")



# Q39: Read the Wikipedia entry on Soundex. Implement this algorithm in Python

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



# Q41: Rewrite the following nested loop as a nested list comprehension:
#
#      >>> words = ['attribution', 'confabulation', 'elocution',
#                   'sequoia', 'tenacious', 'unidirectional']
#      >>> vsequences = set()
#      >>> for word in words:
#      ...     vowels = []
#      ...     for char in word:
#      ...         if char in 'aeiou':
#      ...             vowels.append(char)
#      ...     vsequences.add(''.join(vowels))
#      >>> sorted(vsequences)
#      ['aiuio', 'eaiou', 'eouio', 'euoia', 'oauaio', 'uiieioa']

words = ["attribution", "confabulation", "elocution",
         "sequoia", "tenacious", "unidirectional"]
print(sorted(set("".join(re.findall(r"[aeiou]", word)) for word in words)), "\n")



# Q42: Use WordNet to create a semantic index for a text collection. Extend the concordance search program in
#      3.6, indexing each word using the offset of its first synset, e.g. wn.synsets('dog')[0].offset (and
#      optionally the offset of some of its ancestors in the hypernym hierarchy).

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



# Q43: With the help of a multilingual corpus such as the Universal Declaration of Human Rights Corpu
#      (nltk.corpus.udhr), and NLTK's frequency distribution and rank correlation functionaity (nltk.FreqDist,
#      nltk.spearman_correlation), develop a system that guesses the language of a previously unseen text. For
#      simplicity, work with a single character encoding and just a few languages.

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



# Q44: Write a program that processes a text and discovers cases where a word has been used with a novel
#      sense. For each word, compute the WordNet similarity between all synsets of the word and all synsets of the
#      words in its context. (Note that this is a crude approach; doing it will is a difficult, open research problem.)

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


























