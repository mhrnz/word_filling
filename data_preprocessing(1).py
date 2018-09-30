
# coding: utf-8

# In[2]:


import numpy as np
import nltk
import pickle

import tensorflow as rf


# In[7]:


def get_words(istraining):
    if istraining==True:
        # train data
        words=nltk.corpus.gutenberg.words('austen-emma.txt')
    else:
        # test data
        words=nltk.corpus.gutenberg.words('austen-persuasion.txt')
    return words  


# In[8]:


'''
biuling glove_vectors and stoi , stoi is a dictionary from words, to indices in the glove_vectors, glove_vectors is a 2d np array from indices to embeddings
'''
def get_glove_vectors(path,words):
    with open(path,encoding='UTF-8') as f:
        glove_vectors=np.zeros((400000,50),dtype=np.float32)
        stoi={}
        i=0
        for line in f.readlines():
            parts=line.split()
            stoi[parts[0]]=i
            glove_vectors[i]=[float(num) for num in parts[1:]]
            i+=1
        j=0
        for word in words:
            word=word.lower()
            j+=1
            if j%100==0:
                print(j)
            try:
                k=stoi[word]
            except:
                stoi[word]=glove_vectors.shape[0]
                glove_vectors=np.vstack([glove_vectors,np.random.normal(size=glove_vectors.shape[1])]) 
        return glove_vectors,stoi
def int_to_vocab(stoi):
        int_to_vocab={i:word for i,word in enumerate(stoi)}
        return int_to_vocab


# In[9]:


def save(istraining,glove_vectors,stoi,int_words,int_to_vocab):
    if(istraining==True):
        print('True')
        with open('glove_vectors.pickle','wb') as handle:
            pickle.dump(glove_vectors,handle,protocol=pickle.HIGHEST_PROTOCOL)
        with open('stoi.pickle','wb') as handle:
            pickle.dump(stoi,handle,protocol=pickle.HIGHEST_PROTOCOL)
        with open('int_words_train.pickle','wb') as handle:
            pickle.dump(int_words,handle,protocol=pickle.HIGHEST_PROTOCOL)
    else :
        print(istraining)
        with open('int_words_test.pickle','wb') as handle:
            pickle.dump(int_words,handle,protocol=pickle.HIGHEST_PROTOCOL)
        with open('int_to_vocab_test.pickle','wb') as handle:
            pickle.dump(int_to_vocab,handle,protocol=pickle.HIGHEST_PROTOCOL)


# In[10]:


'''
input: istraining is a boolean
output: a list of size equal to the number of all words in the corpus, containing indices in the glove_vector for each word

by encountering a word which is not in the glove dataset, we add it to stoi and assign a random embedding to it
'''
def words_to_int(words,stoi):
    int_words=[]
    i=0
    for word in words:
        if i%100==0:
            print(i)
        i+=1
        word=word.lower()
        int_words.append(stoi[word])
    return int_words

if __name__=='__main__':
    nltk.download('gutenberg')
    train_words=get_words(True)
    print(len(train_words))
    test_words=get_words(False)
    print(len(test_words))
    all_words=train_words+test_words
#     glove_vectors,stoi=get_glove_vectors(r'D:\uni\shenakht pajuh\work\glove_data\glove.6B.50d.txt',all_words)
#     print(glove_vectors.shape)
#     # for train 
#     int_words_train=words_to_int(train_words,stoi)
#     int_to_vocab_dict={}
#     save(True,glove_vectors,stoi,int_words_train,int_to_vocab_dict)

    # for test
    stoi={}
    glove_vectors=np.ones(1)
    with open('stoi.pickle','rb') as handle:
        stoi=pickle.load(handle)
    int_words_test=words_to_int(test_words,stoi)
    int_to_vocab_dict=int_to_vocab(stoi)
    save(False,glove_vectors,stoi,int_words_test,int_to_vocab_dict)

