**Nandakrishnan, Nair, 12402313**

**POS-Tagger for Hindi Language**

# Project Description

* A POS tagger was created for the language **Hindi**, which is the national language of India.

* For this task a prebuilt POS tagged corpus for Hindi was used to train the Hidden Markov Model and  identify **Transmission** and **Emission Probabilities**. Then applying **Viterbi Algorithm** on Testing Data Set.

* Then finally returning the tagging sequence with the highest probability.

* The POS tagged corpus is taken from http://universaldependencies.org .

# Prerequisites

Python Version >= 3.10.

Libraries mentioned in the requirements.txt

# Installation

Make sure all the libraries mentioned in requirements.txt is installed with command:-
pip install -r requirements.txt


# Basic Usage

* The driver code exists in **POS_Tagger.py** file, so it can be used to initiate the POS tagger.

* The Tagged corpus is present in **train_data** directory in .conllu format, which is preprocessed in the **util/utility_methods.py** file to get the training set in the desired format.

* A Test sentence is present in the **test_data/test_data.txt** file to check the working of the POS tagger on a new sentence.
* This file can be updated to test the POS tagging of some other sentences in Hindi.

* The output is shown in the console itself under various sections like Displaying test sentences with Actual Tagging, Displaying test sentences with Predicted Tagging, Testing with another test sentence, etc.

* The predicted tag set for the test sentence from **test_data/test_data.txt** is also written to file, **test_data/output.txt** .

* A part of the corpus text is stored into **train_data/corpus_text.txt** just for readability

* A translated version of the corpus text is stored into **train_data/corpus_text_translated.txt** .


# Alternative Language

The alternative language used in this project is Hindi (https://en.wikipedia.org/wiki/Hindi).

**What is a Sentence?**


A sentence in Hindi is a grammatical unit that expresses a complete thought. It typically contains a subject and a predicate and ends with a punctuation mark such as a period (।), question mark (?), or exclamation mark (!). Sentences in Hindi may vary in length and complexity, from simple single-clause structures to complex multi-clause constructs.

Examples:

Simple Sentence:

राम घर जाता है। (Ram goes home.)

Complex Sentence:

जब वह स्कूल गया, तब उसने पढ़ाई की। (When he went to school, he studied.)

**What is a Word Token?**

A word token in Hindi is a sequence of characters that form a word, typically separated by whitespace or punctuation in written text. Word tokens are the basic units used for linguistic analysis in tasks such as POS tagging.

Examples:

राम (Ram)
घर (home)
जाता (goes)
है (is)

**Parts of Speech in Hindi**

**1. JJ (Adjective)**

Descriptive words that modify nouns.
- **Examples**:
  - खूबसूरत (Beautiful)
  - बड़ा (Big)

**2. RB (Adverb)**

Adverbs that modify verbs, adjectives, or other adverbs.
- **Examples**:
  - अकेले (Alone)
  - तत्काल (immediately)

**3. NN (Noun)**

General tag for nouns.
- **Examples**:
  - देश (Country)
  - मंत्री (Minister)

**4. VM (Main Verb)**

Main action verbs.
- **Examples**:
  - बना (to change)
  - मिलने (to meet)

**5. NNPC (Proper Noun Case)**

Proper nouns with specific case marking.
- **Examples**:
  - केंद्र (Center)
  - मानव (the human)

**6. INJ (Interjection)**

Interjections are words that express strong emotion or sudden feelings.
- **Examples**:
  - अरे! (Oh!)
  - वाह! (Wow!)

**7. PSP (Postposition)**

Postpositions are placed after nouns or pronouns to show their relationship with other words in the sentence.
- **Examples**:
  - में (in)
  - से (from)

**8. VAUX (Auxiliary Verb)**

Auxiliary verbs help the main verb and express tense, aspect, mood, etc.
- **Examples**:
  - हुआ (Happened)
  - जाएगी (will go)

**9. CC (Coordinating Conjunction)**

Joins two phases/clauses
- **Examples**:
  - और (and)
  - लेकिन (but)

**10. DEM (Demonstrative)**

Demonstrative pronouns or adjectives.
- **Examples**:
  - यह (this)
  - वह (that)

**11. QO (Ordinal)**

Ordinal numbers.
- **Examples**:
  - पहला (first)
  - दूसरा (second)

**12. RP (Particle)**

A function that must be associated with another word
- **Examples**:
  - ही (only)
  - भी (also)

**13. PRP (Pronoun)**

General pronoun tag.
- **Examples**:
  - वह (he/she/it)
  - हम (we)

**14. CCC (Coordinating Conjunction)**

Conjunctions that join two words, phrases, or clauses of equal rank.
- **Examples**:
  - और (and)
  - या (or)

**15. SYM (Symbol)**

Symbols, punctuation marks, etc.
- **Examples**:
  - । (period)
  - , (comma)


**16. INTF (Intensifier)**

Words that intensify the meaning of the word they modify.

- **Examples**:
  - बहुत (very)
  - काफी (quite)

**17. PRPC (Pronoun Case)**

Pronouns with specific case marking.

- **Examples**:
  - मुझे (to me)
  - तुम्हें (to you)

**18. WQ (Wh-word/Question word)**

Words used to form questions.

- **Examples**:
  - क्या (what)
  - कौन (who)

**19. JJC (Comparative Adjective)**

Adjectives used to compare two things.

- **Examples**:
  - बड़ा (bigger)
  - अधिक (more)

**20. NST (Proper Noun: Spatial/Temporal)**

Proper nouns that indicate spatial or temporal entities.

- **Examples**:
  - पास (Near)
  - साथ (with)

**21. QFC (Cardinal Fraction)**

Fractional numbers.

- **Examples**:
  - आधा (half)
  - तिहाई (third)

**22. QC (Cardinal)**

Cardinal numbers.

- **Examples**:
  - एक (one)
  - दस (ten)

**23. QF (Quantifier)**

Quantifiers express quantity.

- **Examples**:
  - कुछ (some)
  - कई (many)

**24. NEG (Negation)**

Negative particles.

- **Examples**:
  - नहीं (no/not)
  - न (not)

**25. RDP (Reduplicative)**

Words formed by reduplication.

- **Examples**:
  - अलग-अलग (Different)
  - साथ-साथ (together)

**26. RBC (Comparative Adverb)**

Adverbs used in comparative contexts.

- **Examples**:
  - अधिक (more)
  - कम (less)

**27. RBC (Comparative Adverb)**

Adverbs used in comparative contexts.

- **Examples**:
  - अधिक (more)
  - कम (less)

**28. UNK (Unknown)**

Unknown or unrecognized words.

- **Examples**:
  - Any word not recognized by the POS tagger.

# Used AI

ChatGPT was used to get the definition of various language specific tags which are not present in Universal Part-of-Speech Tagset.
 To get the hindi to English translation of the corpus, ChatGPT was used.

**Prompts:**

Show all the POS tags used in the language Hindi.
Translate the following text from hindi to english.
