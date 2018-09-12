
# coding: utf-8

# In[1]:


import tensorflow as tf
import collections
import tensorflow.contrib.rnn as rnn
import torchtext.vocab as vocab
import numpy as np
import nltk
import pickle
import import_ipynb
import data_preprocessing
import biLstm_word_filling


# In[2]:


# batch generator
# output : x is a tensor of size [batch_size,window_size] containing indices of words in those windows , 
# random_blanks : random embeddings for blanks
def get_batch(batch_size,epochs,window_size,int_words,embedding_dim,glove_vectors_size):
    batch_len=len(int_words)//batch_size
    int_words=np.array(int_words[:batch_len*batch_size])
    int_words=np.reshape(int_words,[batch_size,batch_len])
    # we take middle word of the window (that is going to be predicted) as a blank word with a random embedding ,
    # we add these random embeddings to the end of the glove_vectors
    blank_indices=[glove_vectors_size+i for i in range(batch_size)]
    for epoch in range(epochs):
        for step_start in range(batch_len-window_size+1):
            x=np.copy(int_words[:,step_start:step_start+window_size])
            y=np.copy(x[:,int(window_size//2)])
#             stoi['blank{}'.format(i)]=glove_vectors.shape[0]+1
#             blank_embedding=np.mean(x[:,window_size//2-5:window_size//2],axis=1)+np.mean(x[:,window_size//2+1:window_size//2+5],axis=1)
            
            # random embeddings for blanks
            random_blanks=np.random.normal(size=(batch_size,embedding_dim))
            x[:,int(window_size//2)]=blank_indices
            yield x,y,random_blanks


# In[3]:


def train():
    
    #hyperparameters
    window_size=100
    num_layers=2
    batch_size=128
    hidden_size=200
    keep_prob=0.7
    epochs=100
    
    with open(r'C:\Users\mehrnaz\Anaconda3\envs\environment1\Codes\work\word filling\int_words.pickle','rb') as handle:
        int_words=pickle.load(handle)
    with open(r'C:\Users\mehrnaz\Anaconda3\envs\environment1\Codes\work\word filling\glove_vectors.pickle','rb') as handle:
        glove_vectors=pickle.load(handle)
    vocab_size=glove_vectors.shape[0]
    embedding_dim=glove_vectors.shape[1]
    # building bidirectional lstm model with appropriate parameters
    model=biLstm_word_filling.biLstm_word_filling(window_size=window_size,embedding_dim=embedding_dim,hidden_size=hidden_size,keep_prob=keep_prob,num_layers=num_layers,vocab_size=vocab_size)

    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run([model.embedding_init],feed_dict={model.embedding_placeholder:glove_vectors})
        
        batch_generator=get_batch(batch_size=batch_size,epochs=epochs,window_size=window_size,int_words=int_words,embedding_dim=embedding_dim,glove_vectors_size=vocab_size)
        i=0
        for x,y,random_blanks in batch_generator:
            
            feed_dict={model.inputs:x,model.output:y,model.random_blanks:random_blanks}
            _,loss=sess.run([model.optimizer,model.cost],feed_dict=feed_dict)
#             print(model.logits.eval())
#             print('-------------')
#             print(model.one_hot_outputs.eval())
            if i%50==0:
                print('i:',i,' loss:',loss)
            i+=1
            
if __name__=='__main__':
    train()

