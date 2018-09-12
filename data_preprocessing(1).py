
# coding: utf-8

# In[12]:


import tensorflow as tf
import collections
import torchtext.vocab as vocab
import numpy as np
import nltk
import pickle


# In[13]:


nltk.download('gutenberg')

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
def get_glove_vectors(path):
    with open(path,encoding='UTF-8') as f:
        glove_vectors=np.zeros((400000,50),dtype=np.float32)
        stoi={}
        i=0
        for line in f.readlines():
            parts=line.split()
            stoi[parts[0]]=i
            glove_vectors[i]=[float(num) for num in parts[1:]]
            i+=1
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
def words_to_int(istraining,glove_path):
    words=get_words(istraining)
    glove_vectors,stoi=get_glove_vectors(glove_path)
    int_words=[]
    for word in words:
        try:
            int_words=int_words+[stoi[word]]
        except:
            stoi[word]=glove_vectors.shape[0]+1
            glove_vectors=np.vstack([glove_vectors,np.random.normal(size=glove_vectors.shape[1])])
            int_words+=[stoi[word]]
    save(glove_vectors,stoi,int_words)

if __name__=='__main__':
    words_to_int(True,r'D:\uni\shenakht pajuh\work\glove_data\glove.6B.50d.txt')

