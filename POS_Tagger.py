import random
import time
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from util.utility_methods import conllu_to_pos

# Constants
INPUT_FILE = 'train_data\\train_data.conllu'
TEST_FILE = 'test_data\\test_data.txt'
OUTPUT_FILE = 'test_data\\output.txt'
SEED = 123


def load_sentences(input_file):
    """Load sentences from a CoNLL-U formatted file."""
    return conllu_to_pos(input_file)


def split_data(sentences, test_size=0.3, seed=123):
    """Split sentences into training and testing sets."""
    random.seed(seed)
    return train_test_split(sentences, test_size=test_size)


def exploratory_data_analysis(train_tagged_words):
    """Perform exploratory data analysis."""
    tokens = [pair[0] for pair in train_tagged_words]
    vocabulary = set(tokens)
    tags = set(pair[1] for pair in train_tagged_words)

    print("---------------Exploratory Data Analysis---------------")
    print("No. of unique tokens in the corpus:", len(vocabulary))
    print("No. of unique tags in the corpus:", len(tags))
    print("--------------------------------------------------------")

    return vocabulary, tags


def compute_emission_probabilities(train_tagged_words):
    """Compute emission probabilities P(w|t)."""
    emission_counts = {}
    tag_counts = {}

    for word, tag in train_tagged_words:
        if tag not in emission_counts:
            emission_counts[tag] = {}
        if word not in emission_counts[tag]:
            emission_counts[tag][word] = 0
        emission_counts[tag][word] += 1

        if tag not in tag_counts:
            tag_counts[tag] = 0
        tag_counts[tag] += 1

    return emission_counts, tag_counts


def word_given_tag(word, tag, emission_counts, tag_counts):
    """Return the count of word given tag and the count of the tag."""
    return emission_counts.get(tag, {}).get(word, 0), tag_counts.get(tag, 0)


def compute_transition_probabilities(train_tagged_words):
    """Compute transition probabilities P(t2|t1)."""
    transition_counts = {}
    tag_counts = {}
    tags = [pair[1] for pair in train_tagged_words]

    for i in range(len(tags) - 1):
        t1 = tags[i]
        t2 = tags[i + 1]

        if t1 not in transition_counts:
            transition_counts[t1] = {}
        if t2 not in transition_counts[t1]:
            transition_counts[t1][t2] = 0
        transition_counts[t1][t2] += 1

        if t1 not in tag_counts:
            tag_counts[t1] = 0
        tag_counts[t1] += 1

    return transition_counts, tag_counts


def create_transition_matrix(tags, transition_counts, tag_counts):
    """Create a transition matrix for tags."""
    tags_matrix = np.zeros((len(tags), len(tags)), dtype='float32')
    tag_list = list(tags)

    for i, t1 in enumerate(tag_list):
        for j, t2 in enumerate(tag_list):
            count_t2_t1, count_t1 = transition_counts.get(t1, {}).get(t2, 0), tag_counts.get(t1, 0)
            tags_matrix[i, j] = count_t2_t1 / count_t1 if count_t1 != 0 else 0

    return pd.DataFrame(tags_matrix, columns=tag_list, index=tag_list)


def viterbi_algorithm(words, tags_df, emission_counts, tag_counts, start_tag='SYM'):
    """Implement the Viterbi algorithm for POS tagging."""
    state = []
    tag_list = list(tags_df.columns)

    for key, word in enumerate(words):
        p = []
        for tag in tag_list:
            if key == 0:
                transition_p = tags_df.loc[start_tag, tag]
            else:
                transition_p = tags_df.loc[state[-1], tag]

            emission_p, count_tag = word_given_tag(word, tag, emission_counts, tag_counts)
            emission_p = emission_p / count_tag if count_tag != 0 else 0
            state_probability = emission_p * transition_p
            p.append(state_probability)

        pmax = max(p)
        state_max = tag_list[p.index(pmax)]
        state.append(state_max)

    return list(zip(words, state))


def main():
    # Load sentences
    sentences = load_sentences(INPUT_FILE)
    print("--------Printing a few sentences from the corpus--------")
    print(sentences[:3])
    print("--------------------------------------------------------")

    # Split data into training and testing sets
    train_set, test_set = split_data(sentences, test_size=0.3, seed=SEED)
    print("No. of sentences in training set:", len(train_set))
    print("No. of sentences in testing set:", len(test_set))

    # Get list of tagged words from training set
    train_tagged_words = [tup for sent in train_set for tup in sent]

    # Exploratory data analysis
    vocabulary, tags = exploratory_data_analysis(train_tagged_words)

    # Compute emission and transition probabilities
    emission_counts, tag_counts = compute_emission_probabilities(train_tagged_words)
    transition_counts, tag_counts = compute_transition_probabilities(train_tagged_words)

    # Create transition matrix
    tags_df = create_transition_matrix(tags, transition_counts, tag_counts)

    # Test Viterbi algorithm on sample sentences
    random.seed(1234)
    test_sample_indices = [random.randint(1, len(test_set)) for _ in range(5)]
    test_samples = [test_set[i] for i in test_sample_indices]
    test_sample_words = [tup[0] for sent in test_samples for tup in sent]
    test_sample_tags = [tup for sent in test_samples for tup in sent]

    print("-----Displaying test sentences with Actual Tagging-----")
    print(test_samples)
    print("-------------------------------------------------------")

    start = time.time()
    tagged_sequence = viterbi_algorithm(test_sample_words, tags_df, emission_counts, tag_counts)
    end = time.time()

    print("Time taken in seconds:", end - start)
    print("-----Displaying test sentences with Predicted Tagging------")
    print(tagged_sequence)
    print("--------------------------------------------------------")

    # Calculate accuracy
    correct_predictions = [i for i, j in zip(tagged_sequence, test_sample_tags) if i == j]
    accuracy = len(correct_predictions) / len(tagged_sequence)
    print("--------------Accuracy of the PoS-tagger----------------")
    print(accuracy)
    print("--------------------------------------------------------")

    # Test on new sentences from the test file
    print("-----------Testing with another test sentence----------")
    with open(TEST_FILE, 'r', encoding='utf-8') as infile, open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        for line in infile:
            words = word_tokenize(line)
            tagged_sequence = viterbi_algorithm(words, tags_df, emission_counts, tag_counts)
            print(tagged_sequence)
            tagged_sequence_str = ' '.join([f"{word}/{tag}" for word, tag in tagged_sequence])
            outfile.write(tagged_sequence_str + '\n')
    print("--------------------------------------------------------")


if __name__ == "__main__":
    main()
