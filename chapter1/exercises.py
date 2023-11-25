from curses.ascii import isupper
from re import split
import nltk
from nltk.book import *
from nltk.corpus import brown

# matplotlib
import matplotlib.pyplot as plt

def lexical_diversity(text, as_percentage=False):
    return 100 * len(set(text))/len(text) if as_percentage else len(set(text))/len(text)


# 2.
print(f"There are {26**100} hundred letter strings with a 26 letter alphabet.\n")

# 3.
print(f"You should obtain a list where the entries are repeated multiple times.")
print(f"We obtain {20 * ['Monty', 'Python']}\n")

# 4.
print(f"There are {len(set(text2))} distinct words in text 2.\n")

# 5.
rom = brown.words(categories="romance")
hum = brown.words(categories="humor")
print([f"Romance is more lexically diverse." if lexical_diversity(rom) >= lexical_diversity(hum)
       else f"Humor is more lexically diverse."], "\n")

# 6.
protagonists = ["Elinor", "Marianne", "Edward", "Willoughby"]
text2.dispersion_plot(protagonists)
# could also use plt.show() here
plt.savefig("figures/exercise6.png")
plt.close()
print(f"The couples appear to be Willoughby/Elinor and Edward/Marianne based on occurrences.\n")

# 7.
print(f"Collocation of text 5:\n")
print(text5.collocations(), "\n")

# 8.
print(f"The expression len(set(text4)) is used to get the number of distinct words in text4 (i.e. the vocabulary).\n")

# 9.
# a.
my_string = "A very interesting string."
print(my_string, "\n")

# b.
print(f"Add string with +: {my_string + my_string}\n")
print(f"Add string by multiplying: {3*my_string}\n")

# 10.
# a.
my_sent = ["These", "are", "my", "own", "words"]
joined_string = ' '.join(my_sent)
print(f"Create new string using join with space between: {joined_string}\n")

# b.
split_string = joined_string.split()
print(f"Split string: {split_string}\n")

# 11.
phrase1, phrase2 = "Let's go for", "a walk."
sentence = ' '.join([phrase1, phrase2])
print(f"This is a joined sentence: {sentence}\n")
print(f"The lengths should be the sum plus one due to the space added between them: {len(sentence) - (len(phrase1) + len(phrase2))}\n")

# 12.
print(f"The first slices characters 6-12 which correspond to Python. The second selects the second list item.\n")

# 13.
print(f"sent1[2][2] will choose the second word of sent1 and then choose the third letter of sent1[2].\n")

# 14.
print(f"The indices of 'the' in sent3 are: {[i for i in range(len(sent3)) if sent3[i] == 'the']}\n")

# 15.
print(f"All words from text5 starting with 'b' and sorted.\n")
print(sorted([w for w in text5 if w.startswith('b')]), "\n")

# 17.
print(f"The index of 'sunset' in text9 is: {text9.index('sunset')}")
print(text9[621:644], "\n")  # complete sentence

# 18.
vocabulary = sorted(set(sent1 + sent2 + sent3 + sent4 + sent5 + sent6 + sent7 + sent8))
print(vocabulary, "\n")

# 19.
print(f"The first line converts all words to lower case then forms a set and sorts.")
print(f"The second line forms a set from the words then converts all elements of the set to lower case and sorts.")
print(f"The second one gives a larger value since the resulting set may have duplicates of a word. This happens since making a set first distinguishes case and the resulting set after lowering case will have repeats.")
print(f"This can happen for any text where two strings with same letters occur with different case like 'Will' and 'will'.\n")

# 20.
print(f"The first tests if all entries of w is uppercase while the second only tests that the entries are not all lowercase (i.e. some could be lowercase).\n")

# 21.
print(f"Last two words of text2 is: {text2[-2:]}\n")

# 22.
print(f"All four letter words:\n")
len_4 = set(w for w in text5 if len(w) == 4)
print(len_4, "\n")
fdist5 = FreqDist(text5)
print(f"Print words in decreasing order of frequency:\n")
print(sorted(set(len_4), key = lambda x: fdist5[x], reverse = True), "\n")

# 23.
print(*[w for w in text6 if w.isupper()], sep="\n")  # asterisk unpacks list passing each element separately to print
print("\n")

# 24.
# a.
print([w for w in text6 if w.endswith('ise')], "\n")

# b.
print([w for w in text6 if 'z' in w], "\n")

# c.
print([w for w in text6 if 'pt' in w], "\n")

# d.
print([w for w in text6 if w[1:].islower()], "\n")

# 25.
sent = ["she", "sells", "sea", "shells", "by", "the", "sea", "shore"]

# a.
print([w for w in sent if w.startswith("sh")], "\n")

# b.
print([w for w in sent if len(w) > 4], "\n")

# 26.
print(f"This finds the sum of the lengths of all words in text1.")
total_words_length = sum(len(w) for w in text1)
num_words = len(text1)
print(f"The average word length in text1 is: {total_words_length/num_words} letters\n")

# 27.
def vocab_size(text):
    return len(set(text))

# 28.
def percent(word, text):
    return 100 * text.count(word)/len(text)

# 29.
print(set(sent3) < set(text1))
print(set("ridiculous") < set(text1))
print(set("happy") < set(text1))
print(f"Appears to look if left-hand side is subset to right-hand side.")











