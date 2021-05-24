import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text 
from sklearn import metrics
from sklearn.feature_extraction import text
import pandas as pd
import numpy as np
import regex as re

def get_pre_encoders(preprocessor_link, encoder_link):

  """
  Arguements:
    preprocessor_link : link for embedding to preprocess text
    encoder_link : weights for embedding to encode text into tensors
    
  Returns:
    weights of preprocessor and encoder
  """

  preprocessor = hub.KerasLayer(preprocessor_link)
  encoder = hub.KerasLayer(encoder_link, trainable=True)

  return preprocessor, encoder


def get_bert_embeddings(text, preprocessor, encoder):
  
  """
  Gets the embedding of the text
  Arguements:
   text : input text
   preprocessor: link for embedding to preprocess text
   encoder: weights for embedding to encode text into tensors
   
  Return:
    Encoded text 
  """

  text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
  encoder_inputs = preprocessor(text_input)
  outputs = encoder(encoder_inputs)
  embedding_model = tf.keras.Model(text_input, outputs['pooled_output'])
  sentences = tf.constant([text])
  return embedding_model(sentences)

def preprocess_text():

  """
  Gets the cleaned input text to be queried for
  """

  text = input()
  text = text.lower()
  text = re.sub('[^A-Za-z0-9]+', ' ', text)
  return text

def embed_df(list_words, pre, enc):
  
  """
  Arguements:
    list_words: list of sentences
    pre: preprocessor
    end: encoder
  
  Returns the dataframe with text and corresponding encodings
  """
  
  list_df = pd.DataFrame(list_words)
  list_df.columns = ['text']
  list_df['embeds'] = list_df['text'].apply(lambda x : get_bert_embeddings(x, pre, enc))

  return list_df

def get_result(query_text, df, pre, enc):

  """
  Arguements:
    query_text: text to be queried
    df: base dataframe with text and encodings 
    pre: preprocessor
    end: encoder
  
  Returns the dataframe with final semantic scores.
  """

  query_encoding = get_bert_embeddings(query_text, pre, enc)
  df['similarity_score'] = df['embeds'].apply(lambda x: metrics.pairwise.cosine_similarity(x, query_encoding)[0][0])
  df_results = df.sort_values(by=['similarity_score'], ascending=False)

  return df_results
