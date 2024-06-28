import random
from util.utility_methods import conllu_to_pos
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import time
from nltk.tokenize import word_tokenize

input_file = 'train_data\\train_data.conllu'
test_file = 'test_data\\test_data2.txt'

# reading a few sentences i.e. is a list of (word, pos) tuples
sentences = conllu_to_pos(input_file)
print("--------Printing a few sentences from the corpus--------")
print(sentences[:3])
print("--------------------------------------------------------")


# Splitting into train and test
random.seed(123)
train_set, test_set = train_test_split(sentences, test_size=0.3)
print("No. of sentences in training set", len(train_set))
print("No. of sentences in testing set", len(test_set))

# Getting list of tagged words
train_tagged_words = [tup for sent in train_set for tup in sent]

print("---------------Exploratory Data Analysis---------------")
# tokens
tokens = [pair[0] for pair in train_tagged_words]
# vocabulary
V = set(tokens)
print("No. of unique tokens in the corpus: ", len(V))
# number of tags
T = set([pair[1] for pair in train_tagged_words])
print("No. of unique tags in the corpus: ", len(T))
print("--------------------------------------------------------")


# computing P(w/t) and storing in T x V matrix
t = len(T)
v = len(V)
w_given_t = np.zeros((t, v))


# compute word given tag: Emission Probability
def word_given_tag(word, tag, train_bag=train_tagged_words):
    tag_list = [pair for pair in train_bag if pair[1] == tag]
    count_tag = len(tag_list)
    w_given_tag_list = [pair[0] for pair in tag_list if pair[0] == word]
    count_w_given_tag = len(w_given_tag_list)

    return (count_w_given_tag, count_tag)

# compute tag given tag: tag2(t2) given tag1 (t1), i.e. Transition Probability
def t2_given_t1(t2, t1, train_bag=train_tagged_words):
    tags = [pair[1] for pair in train_bag]
    count_t1 = len([t for t in tags if t == t1])
    count_t2_t1 = 0
    for index in range(len(tags) - 1):
        if tags[index] == t1 and tags[index + 1] == t2:
            count_t2_t1 += 1
    return (count_t2_t1, count_t1)

# creating t x t transition matrix of tags
# each column is t2, each row is t1
# thus M(i, j) represents P(tj given ti)
tags_matrix = np.zeros((len(T), len(T)), dtype='float32')
for i, t1 in enumerate(list(T)):
    for j, t2 in enumerate(list(T)):
        tags_matrix[i, j] = t2_given_t1(t2, t1)[0] / t2_given_t1(t2, t1)[1]

print(tags_matrix)

# convert the matrix to a df for better readability
tags_df = pd.DataFrame(tags_matrix, columns=list(T), index=list(T))

# Viterbi Heuristic
def Viterbi(words, train_bag=train_tagged_words):
    state = []
    T = list(set([pair[1] for pair in train_bag]))

    for key, word in enumerate(words):
        # initialise list of probability column for a given observation
        p = []
        for tag in T:
            if key == 0:
                transition_p = tags_df.loc['SYM', tag]
            else:
                transition_p = tags_df.loc[state[-1], tag]

            # compute emission and state probabilities
            emission_p = word_given_tag(words[key], tag)[0] / word_given_tag(words[key], tag)[1]
            state_probability = emission_p * transition_p
            p.append(state_probability)

        pmax = max(p)
        # getting state for which probability is maximum
        state_max = T[p.index(pmax)]
        state.append(state_max)
    return list(zip(words, state))


# Running on entire test dataset would take more than 3-4hrs.
# Let's test our Viterbi algorithm on a few sample sentences of test dataset

random.seed(1234)

# choose random 5 sents
rndom = [random.randint(1, len(test_set)) for x in range(5)]
# list of sents
test_run = [test_set[i] for i in rndom]
# list of tagged words
test_run_base = [tup for sent in test_run for tup in sent]
# list of untagged words
test_tagged_words = [tup[0] for sent in test_run for tup in sent]

print("-----Displaying test sentences with Actual Tagging-----")
print(test_run)
print("-------------------------------------------------------")

# tagging the test sentences
start = time.time()
tagged_seq = Viterbi(test_tagged_words)
end = time.time()
difference = end - start
print("Time taken in seconds: ", difference)
print("-----Displaying test sentences with Actual Tagging------")
print(tagged_seq)
print("--------------------------------------------------------")

# accuracy
check = [i for i, j in zip(tagged_seq, test_run_base) if i == j]
accuracy = len(check) / len(tagged_seq)
print("--------------Accuracy of the PoS-tagger----------------")
print(accuracy)
print("--------------------------------------------------------")

## Testing
print("-----------Testing with another test sentence----------")
with open(test_file, 'r', encoding='utf-8') as infile:
    for line in infile:
        words = word_tokenize(line)
        start = time.time()
        tagged_seq = Viterbi(words)
        print(tagged_seq)
print("--------------------------------------------------------")