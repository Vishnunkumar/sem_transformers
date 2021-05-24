# sem_transformers
A simple semantic search library using transformers and tensorflow hub

## sample code 
```
pre, enc =  get_pre_encoders("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3", "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1")
embeds =  get_bert_embeddings("Hi how are you", pre, enc)
list_words = ["great to have you back", "get the hell out", "welcome back", "Work like a mad man", "we are on fire"]
list_df =  embed_df(list_words, pre, enc)
quer_text =  preprocess_text()
df_f =  get_result(quer_text, list_df, pre, enc)
```
