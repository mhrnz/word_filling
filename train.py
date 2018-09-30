
# coding: utf-8

# In[1]:


import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import numpy as np
import pickle
import import_ipynb
import biLstm_word_filling


# In[2]:


'''
description: batch generator
output : x is a tensor of size [batch_size,window_size] containing indices of words in those windows , 
 y is a tensor of size [batch_size] containing indices of labels(middle words in windows)
 random_blanks : random embeddings for blanks (different random embeddings for different blanks)
''' 

def get_batch(batch_size,window_size,int_words,embedding_dim,glove_vectors_size):
    def coef(x):
            '''assign a coeficient p to frequent words
            # 0:the -> p=0.7 , 1:, -> p=0.5 , 2:. -> p=0.5 , 3:of -> p=0.7 , 4:to -> p=0.7 , 5:and -> p=0.7
            '''
        return {0:0.7 , 1:0.5 , 2:0.5 , 3:0.7 , 4:0.7 , 5:0.7}[x]
    batch_len=(len(int_words)//batch_size)
    print(batch_len)
    int_words=np.array(int_words[:batch_len*batch_size])
    int_words=np.reshape(int_words,[batch_size,batch_len])
    '''
    we take middle word of the window (that is going to be predicted) as a blank word with a random embedding ,
    we add these random embeddings to the end of the glove_vectors
    '''
    blank_indices=[glove_vectors_size+i for i in range(batch_size)]
    for step_start in range(batch_len-window_size+1):
        x=np.copy(int_words[:,step_start:step_start+window_size])
        y=np.copy(x[:,int(window_size//2)])
        frequent_words_coefs=np.ones(y.shape)
        for i in range(batch_size):
            if y[i] in range(6):
                frequent_words_coefs[i]=coef(y[i])
        #    y=np.expand_dims(y,axis=1)
    #             stoi['blank{}'.format(i)]=glove_vectors.shape[0]+1
    #             blank_embedding=np.mean(x[:,window_size//2-5:window_size//2],axis=1)+np.mean(x[:,window_size//2+1:window_size//2+5],axis=1)

        '''random embeddings for blanks'''
        random_blanks=np.random.normal(size=(batch_size,embedding_dim))
        x[:,int(window_size//2)]=blank_indices
        yield x,y,random_blanks,frequent_words_coefs


# In[3]:


def train(int_words, glove_vectors, window_size=101, num_layers=2,
          batch_size=64, hidden_size=200, keep_prob=0.8, epochs=10):
    
    vocab_size=glove_vectors.shape[0]
    embedding_dim=glove_vectors.shape[1]
    '''building bidirectional lstm model with appropriate parameters'''
    model=biLstm_word_filling.biLstm_word_filling(window_size=window_size,embedding_dim=embedding_dim,hidden_size=hidden_size,keep_prob=keep_prob,num_layers=num_layers,vocab_size=vocab_size)

    with tf.name_scope('run_session'):
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        sess.run(tf.global_variables_initializer())
        sess.run([model.embedding_init],feed_dict={model.embedding_placeholder:glove_vectors})

        batch_generator=get_batch(batch_size=batch_size,epochs=epochs,window_size=window_size,int_words=int_words,embedding_dim=embedding_dim,glove_vectors_size=vocab_size)
        i=0
        for x,y,random_blanks in batch_generator:

            feed_dict={model.inputs:x,model.output:y,model.random_blanks:random_blanks}
            _,loss=sess.run([model.optimizer,model.cost],feed_dict=feed_dict)
            if i%50==0:
                print('i:',i,' loss:',loss)
            i+=1
        saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)    
        saver.save(sess, './trained_word_filling.ckpt')
            
if __name__=='__main__':
    with open('glove_vectors.pickle','rb') as handle:
        glove_vectors=pickle.load(handle)
    with open(r'int_words_train.pickle','rb') as handle:
        int_words=pickle.load(handle)
        
    train(int_words,glove_vectors)

