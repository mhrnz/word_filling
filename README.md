# word_filling

word filling using bidirectional lstm , any word that needs to be filled is considered as a blank with random embedding

main functions in preprocessing file:
get_words() : returns words of a book using nltk.corpus.gutenberg, different books for train and test
words_to_int() : returns a list of size equal to the number of all words in the corpus, containing indices in the glove_vector for each         word. by encountering a word which is not in the glove dataset, we add it to stoi and assign a random embedding to it

biLstm_word_filling class:
a class for building the bidirectional lstm model

train file:
get_batch() : batch generator, returns a tensor of size [batch_size,window_size] containing indices of words in those windows , and random embeddings for blanks
train() : for training the model
