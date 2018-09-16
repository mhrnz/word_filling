
# coding: utf-8

# In[12]:

# do not use extra information
# import tensorflow as tf
import collections
# import torchtext.vocab as vocab
import numpy as np
import nltk
import pickle


# In[13]:
# be aware of extra comments caused by ipynb

# do not use a code in main body unless in if __name__ == "__main__ part"
# nltk.download('gutenberg')

def get_words(istraining):
    if istraining==True:
        # train data
        words=nltk.corpus.gutenberg.words('austen-emma.txt')
    else:
        # test data
        words=nltk.corpus.gutenberg.words('austen-persuasion.txt')
    return words


# In[14]:


# biuling glove_vectors and stoi , stoi is a dictionary from words, to indices in the glove_vectors, glove_vectors is a 2d np array from indices to embeddings
# it is better to input file instead of it's path
def get_glove_vectors(path):
"""
Decription:
building a glove vectors and dictionary from words

input:
the path of file of glove vectors in txt format,
each line starts with the word name and then vectors elements;

output:
glove_vector: a numpy array of word i in index i
stoi: word dictionary
"""
    with open(path,encoding='UTF-8') as f:
        # do not use pre-knowledge about size of vocabulary specially extra memory
        # glove_vectors=np.zeros((400000,50),dtype=np.float32)
        glove_vectors=[]
        stoi={}
        i=0
        for line in f.readlines():
            parts=line.split()
            stoi[parts[0]]=i
            glove_vectors.append([float(num) for num in parts[1:]])
            i+=1
        glove_vectors = np.array(glove_vectors)
    return glove_vectors,stoi


# In[15]:


def save(glove_vectors,stoi,int_words):
    with open('glove_vectors.pickle','wb') as handle:
        pickle.dump(glove_vectors,handle,protocol=pickle.HIGHEST_PROTOCOL)
    with open('stoi.pickle','wb') as handle:
        pickle.dump(stoi,handle,protocol=pickle.HIGHEST_PROTOCOL)
    with open('int_words.pickle','wb') as handle:
        pickle.dump(int_words,handle,protocol=pickle.HIGHEST_PROTOCOL)


# In[16]:


#input: istraining is a boolean
#output: a list of size equal to the number of all words in the corpus, containing indices in the glove_vector for each word
# by encountering a word which is not in the glove dataset, we add it to stoi and assign a random embedding to it

# it is a principal modules but it uses local paths and variables
# it is better to do this:
def words_to_int(words, glove_vectors, stoi):
"""
Description:
...

input:
...

output:
...
"""
    # words=get_words(istraining)
    # glove_vectors,stoi=get_glove_vectors(glove_path)
    int_words=[]
    for word in words:
        try:
            # int_words.append() is faster
            int_words=int_words+[stoi[word]]
        except:
            stoi[word]=glove_vectors.shape[0]+1
            glove_vectors=np.vstack([glove_vectors,np.random.normal(size=glove_vectors.shape[1])])
            int_words+=[stoi[word]]
    return glove_vectors, stoi, int_words

if __name__=='__main__':
    nltk.download('gutenberg')
    words = get_words(True)
    glove_vector, stoi = get_glove_vectors(r'D:\uni\shenakht pajuh\work\glove_data\glove.6B.50d.txt')
    glove_vectors, stoi, int_words = words_to_int(words, glove_vectors, stoi)
    save(glove_vectors, stoi, int_words)
