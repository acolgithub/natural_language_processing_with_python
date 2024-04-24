from curses.ascii import isupper
from re import split
import nltk
from nltk.book import *
from nltk.corpus import brown

# matplotlib
import matplotlib.pyplot as plt

def lexical_diversity(text, as_percentage=False):
    return 100 * len(set(text))/len(text) if as_percentage else len(set(text))/len(text)



# Q2: Given an alphabet of 26 letters, there are 26 to the power 10, or 26 ** 10, ten-letter strings we can form.
#     That works out to 141167095653376. How many hundred-letter strings are possible

print(f"There are {26**100} hundred letter strings with a 26 letter alphabet.\n")



# Q3: The Python multiplication operation can be applied to lists. What happens when you type
#     ['Monty', 'Python'] * 20, ro 3 * sent1?

print(f"You should obtain a list where the entries are repeated multiple times.")
print(f"We obtain {20 * ['Monty', 'Python']}\n")



# Q4: Review 1 on computing with language. How many words are there in text2? How many distinct words
#     are there?

print(f"There are {len(set(text2))} distinct words in text 2.\n")



# Q5: Compare the lexical diversity scores for humor and romance fiction in 1.1. Which genre is more lexically
#     diverse?

rom = brown.words(categories="romance")
hum = brown.words(categories="humor")
print([f"Romance is more lexically diverse." if lexical_diversity(rom) >= lexical_diversity(hum)
       else f"Humor is more lexically diverse."], "\n")



# Q6: Produce a dispersion plot of the four main protagonists in Sense and Sensibility: Elinor, Marianne,
#     Edward, and Willoughby. What can you observe about the different roles played by the males and females in
#     this novel? Can you identify the couples?

protagonists = ["Elinor", "Marianne", "Edward", "Willoughby"]
text2.dispersion_plot(protagonists)
# could also use plt.show() here
plt.savefig("figures/exercise6.png")
plt.close()
print(f"The couples appear to be Willoughby/Elinor and Edward/Marianne based on occurrences.\n")



# Q7: Find the collocations in text5.

print(f"Collocation of text 5:\n")
print(text5.collocations(), "\n")



# Q8: Consider the following Pythong expression: len(set(text4)). State the purpose of this expression.
#     Describe the two steps involved in performing this computation.

print(f"The expression len(set(text4)) is used to get the number of distinct words in text4 (i.e. the vocabulary).\n")



# Q9: Review 2 on lists and strings.

# a: Define a string and assign it to a variable, e.g., my_string = 'My String' (but put something more
#    interesting in the string). Print the contents of this variable in two ways, first by simply typing the
#    variable name and pressing enter, then by using the print statement.

my_string = "A very interesting string."
print(my_string, "\n")

# b: Try adding the string to itself using my_string + my_string, or multiplying it by a number, e.g.,
#    my_string * 3. Notice that the strings are joined together without any spaces. How could you fix this?

print(f"Add string with +: {my_string + my_string}\n")
print(f"Add string by multiplying: {3*my_string}\n")
print(f"Adding string with + and space: {my_string + ' ' + my_string}\n")



# Q10: Define a variable my_sent to be a list of words, using the syntax my_sent = ["My", sent""] (but with your
#     own words, or a favourite saying).

# a: Use ' '.join(my_sent) to convert this into a string.
my_sent = ["These", "are", "my", "own", "words"]
joined_string = ' '.join(my_sent)
print(f"Create new string using join with space between: {joined_string}\n")

# b: Use split() to split the string back into the list form you had to start with.
split_string = joined_string.split()
print(f"Split string: {split_string}\n")



# Q11: Define several variables containing lists of words, e.g., phrase1, phrase2, and so on. Join them together in
#     various combinations (using the plus operator) to form whole sentences. What is the relationship between
#     len(phrase1 + phrase2) and len(phrase1) + len(phrase2)?
phrase1, phrase2 = "Let's go for", "a walk."
sentence = ' '.join([phrase1, phrase2])
print(f"This is a joined sentence: {sentence}\n")
print(f"The lengths should be the sum plus one due to the space added between them: {len(sentence) - (len(phrase1) + len(phrase2))}\n")



# Q12: Consider the following two expressions, which have the same value. Which one will typically be more
#     relevant in NLP? Why>
#
#     a. "Monty Python"[6:12]
#     b. ["Monty", "Python"][1]

print(f"The first slices characters 6-12 which correspond to Python. The second selects the second list item.\n")



# Q13: We have seen how to represent a sentence as a list of words, where each word is a sequence of characters.
#      What does sent1[2][2] do? Why? Experiment with other index values.

print(f"sent1[2][2] will choose the second word of sent1 and then choose the third letter of sent1[2].\n")



# Q14: The first sentence of text3 is provided to you in the variable sent3. The index of 'the' in sent3 is 1,
#      because sent3[1] gives us 'the'. What are the indexes of the two other occurrences of this word in sent3?

print(f"The indices of 'the' in sent3 are: {[i for i in range(len(sent3)) if sent3[i] == 'the']}\n")



# Q15: Review the discussion of conditionals in 4. Find all words in the Chat Corpus (text5) starting with the
#      letter b. Show them in alphabetical order.

print(f"All words from text5 starting with 'b' and sorted.\n")
print(sorted([w for w in text5 if w.startswith('b')]), "\n")



# Q17: Use text9.index() to find the index of the word sunset. You'll need to insert this word as an argument
#      between the parentheses. By a process of trial and error, find the slice for the complete sentence that contains
#      this word.
print(f"The index of 'sunset' in text9 is: {text9.index('sunset')}")
print(text9[621:644], "\n")  # complete sentence



# Q18: Using list addition, and the set and sorted operations, compute the vocabulary of the sentences sent1 ...
#      sent8.

vocabulary = sorted(set(sent1 + sent2 + sent3 + sent4 + sent5 + sent6 + sent7 + sent8))
print(vocabulary, "\n")



# Q19: What is the difference between the following two lines? Which one will give a larger value? Will this be
#      the case for other texts?
#
# >>>  sorted(set(w.lower() for w in text1))
# >>>  sorted(w.lower() for w in set(text1))

print(f"The first line converts all words to lower case then forms a set and sorts.")
print(f"The second line forms a set from the words then converts all elements of the set to lower case and sorts.")
print(f"The second one gives a larger value since the resulting set may have duplicates of a word. This happens since making a set first distinguishes case and the resulting set after lowering case will have repeats.")
print(f"This can happen for any text where two strings with same letters occur with different case like 'Will' and 'will'.\n")



# Q20: What is the difference between the following two tests: w.isupper() and not w.islower()?

print(f"The first tests if all entries of w is uppercase while the second only tests that the entries are not all lowercase (i.e. some could be lowercase).\n")



# Q21: Write the slice expression that extracts the last two words of text2.

print(f"Last two words of text2 is: {text2[-2:]}\n")



# Q22: Find all the four-letter words in the Chat Corpus(text5). With the help of a frequency distribution
#      (FreqDist), show these words in decreasing order of frequency.

print(f"All four letter words:\n")
len_4 = set(w for w in text5 if len(w) == 4)
print(len_4, "\n")
fdist5 = FreqDist(text5)
print(f"Print words in decreasing order of frequency:\n")
print(sorted(set(len_4), key = lambda x: fdist5[x], reverse = True), "\n")



# Q23: Review the discussion of looping with conditions in 4. Use a combination of for and if statements to
#      loop over the words of the movie script for 'Monty Python and the Holy Grail' (text6) and 'print' all the
#      uppercase words, one per line.

print(*[w for w in text6 if w.isupper()], sep="\n")  # asterisk unpacks list passing each element separately to print
print("\n")



# Q24: Write expressions for finding all words in text6 that meet the conditions listed below. The result should
#      be in the form of a list of words: ['word1', 'word2', ...].
#
# a: Ending in 'ise'

print([w for w in text6 if w.endswith('ise')], "\n")

# b: Containing the letter 'z'
print([w for w in text6 if 'z' in w], "\n")

# c: 'Containing the sequence of letters 'pt'
print([w for w in text6 if 'pt' in w], "\n")

# d: Having all lowercase letters except for an initial capital (i.e., titlecase)
print([w for w in text6 if w[1:].islower()], "\n")



# Q25: Define 'sent' to be the list of words ['she', 'sells', 'sea', 'shells', 'by', 'the', 'sea', 'shore'].
#      Now write code to perform the following tasks: 

sent = ["she", "sells", "sea", "shells", "by", "the", "sea", "shore"]

# a: Print all words beginning with 'sh'
print([w for w in sent if w.startswith("sh")], "\n")

# b: Print all words longer than four characters
print([w for w in sent if len(w) > 4], "\n")



# Q26: What does the following Python code do? sum(len(w) for w in text1)? Can you use it to work out the
#      acerage word length of a text?

print(f"This finds the sum of the lengths of all words in text1.")
total_words_length = sum(len(w) for w in text1)
num_words = len(text1)
print(f"The average word length in text1 is: {total_words_length/num_words} letters\n")



# Q27: Define a function called vocab_size(text) that has a single parameter for the text, and which returns the
#      vocabulary size of the text.

def vocab_size(text):
    return len(set(text))



# Q28: Define a function percent(word, text) that calculates how often a given word occurs in a text, and
#      expresses the result as a percentage.
 
def percent(word, text):
    return 100 * text.count(word)/len(text)



# Q29: We have been using sets to store vocabularies. Try the following Python expression:
#      set(sent3) < set(text1). Experiment with this using different arguments to set(). What does it do? Can
#      you think of a practical application for this?

print(set(sent3) < set(text1))
print(set("ridiculous") < set(text1))
print(set("happy") < set(text1))
print(f"Appears to look if left-hand side is subset to right-hand side.")

