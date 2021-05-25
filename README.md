# sem_transformers
A simple semantic search library using transformers and tensorflow hub

## Installation

```
pip install sem-transformers
```

## Sample code 
```
from sem_transformers import semantic

# please visit tf hub for information regarding preprocessors and encoders
pre, enc =  semantic.get_pre_encoders("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3", "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1")
embeds =  semantic.get_bert_embeddings("Hi how are you", pre, enc)

# Encoding for a list of sentences
list_words = ["great to have you back", "get the hell out", "welcome back", "Work like a mad man", "we are on fire"]
list_df =  semantic.embed_df(list_words, pre, enc)

# Phrase to be queried for
quer_text =  semantic.preprocess_text()

# Results ordered from most similar to least. 
df_f =  semantic.get_result(quer_text, list_df, pre, enc)
```
- Please note: This is under development, please raise an issue regarding the same. 
