import gensim
import nltk
nltk.download()
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
""""A1 (1).pptx"brown_path = nltk.data.find('corpora/brown')brown_path"""
sentences = nltk.corpus.brown.sents()
len(sentences)
type(sentences)
sentences_list = list(sentences)
type(sentences_list)
import gensim.downloader as api
wv = api.load("word2vec-google-news-300")
from gensim.models import Word2Vec

## Preprocessing
# Convert all sentences to lowercase
lowercased_sentences = [[token.lower() for token in sentence] for sentence in sentences_list]
#Print the first sentences before and after conversion

print("Original Sentences:")
print(sentences_list[0])   # Print the first sentences
print("\nLowercase Sentences:")
print(lowercased_sentences[0])  # Print the first sentences after conversion
# Lemmatize the lowercase_sentences
lemmatized_sentences = [[lemmatizer.lemmatize(token) for token in sentence] for sentence in lowercased_sentences]
#Print the first sentences before and after lemmatization
print("Original Sentences:")
print(lemmatized_sentences[0])  # Print the first sentences

print("\nLemmatized Sentences:")
print(lemmatized_sentences[0])  # Print the first sentences after lemmatization
brown_corpus = lemmatized_sentences

## Task 1
# Train Word2Vec on selected corpus (first_model)
first_model = Word2Vec(sentences=brown_corpus,sg =1,epochs =5,vector_size =100,min_count =5)
"""embedding_size = first_model.vector_size
print(f"The size of word vectors is: {embedding_size}")"""
"""# Get the keyed vectors
keyed_vectors = first_model.wv

# Count the number of words with vectors
num_words_with_vectors = len(keyed_vectors.index_to_key)

print(f"The number of words with vectors is: {num_words_with_vectors}")"""
first_model.wv.save_word2vec_format("first_model.txt")
# Train Word2Vec on selected corpus (second_model)
second_model = Word2Vec(sentences=brown_corpus,sg =0,epochs =5,vector_size =100,min_count =5)
second_model.wv.save_word2vec_format("second_model.txt")
#first_model.wv["county"] 
#second_model.wv["county"] 

## Task 2
import A1_helper
x_vals, y_vals, labels = A1_helper.reduce_dimensions(first_model.wv)
import random

# Get the keyed vectors
keyed_vectors = first_model.wv

# Get the list of words in the vocabulary
vocab_words = keyed_vectors.index_to_key

# Set a seed for reproducibility
random.seed(42)

# Randomly select 20 words from the vocabulary
selected_words = random.sample(vocab_words, 20)

print("Randomly selected 20 words:")
print(selected_words)
A1_helper.plot_with_matplotlib(x_vals, y_vals, labels, selected_words)
x_vals_sec, y_vals_sec, labels_sec = A1_helper.reduce_dimensions(second_model.wv)
A1_helper.plot_with_matplotlib(x_vals_sec, y_vals_sec, labels_sec, selected_words)

## Task 3
import numpy as np

# Set a seed for reproducibility
np.random.seed(42)

# Get vocabulary from the trained model
vocabulary = list(first_model.wv.index_to_key)

# Create word pairs
word_pairs = []

# Select 10 random words from the vocabulary
for i in range(10):
    word1, word2 = np.random.choice(vocabulary, size=2, replace=False)
    word_pairs.append((word1, word2))

# Print the list of word pairs
print(word_pairs)
# Corresponding suggested similarity scores
similarity_scores = [0.3, 0.7, 0.6, 0.2, 0.4, 0.1, 0.1, 0.3, 0.5, 0.6]

# Write to a tab-delimited text file
with open('input.txt', 'w') as file:
    for pair, score in zip(word_pairs, similarity_scores):
        file.write(f'{pair[0]}\t{pair[1]}\t{score}\n')

print("File 'input.txt' has been created.")
wv.evaluate_word_pairs('input.txt')
similarity_scores_first_model = first_model.wv.evaluate_word_pairs('input.txt')
similarity_scores_first_model
similarity_scores_second_model = second_model.wv.evaluate_word_pairs('input.txt')
similarity_scores_second_model

## Task 4
new_selected_words = selected_words[0:5]
print(new_selected_words)
wv.most_similar("bewildered")
# Find most similar words for each model
similar_words_first_model = {word: [similar[0] for similar in first_model.wv.most_similar(word, topn=6)] for word in new_selected_words}
similar_words_second_model = {word: [similar[0] for similar in second_model.wv.most_similar(word, topn=5)] for word in new_selected_words}
similar_words_wv = {word: [similar[0] for similar in wv.most_similar(word, topn=5)] for word in new_selected_words}

# Print the results
print("Most similar words using first_model:")
print(similar_words_first_model)

print("\nMost similar words using second_model:")
print(similar_words_second_model)

print("\nMost similar words using wv:")
print(similar_words_wv)
