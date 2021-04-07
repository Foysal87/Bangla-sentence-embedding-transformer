# Bangla-sentence-embedding-transformer

This is a Transformer base bangla sentence embedding. I trained 2,50,000 Bangla sentences(wiki) by sentence transformer.
Embedding dimension is 300d.

## What do you get from here?

* A pretrained Sentence embedding using transformer
* How to add new data and train it?
* How to create Your own pretrained sentence embedding model?

## Python package
```buildoutcfg
sentence_transformers

#install it by below command

pip3 install sentence_transformers
```

## Model download
As my model size is 1.1gb, I can't upload it here. So i upload it in google drive.
[drive link](https://drive.google.com/file/d/1qvljgnus6L4vYR5XxxyhJjolM9v6Ws-Q/view?usp=sharing)

**Or You can use our python module [sbnltk](https://github.com/Foysal87/sbnltk) . Check it!**

Clone this project, then download my model. 
After download, unzip the folder in 'Bangla-sentence-embedding-transformer' directory.

## How to use it?

```python
from Bangla-sentence-embedding-transformer.Bangla_transformer import Bangla_sentence_transformer_small

transformer=Bangla_sentence_transformer_small()

sentences=['আপনার বয়স কত','আমি তোমার বয়স জানতে চাই','আমার ফোন ভাল আছে','আপনার সেলফোনটি দুর্দান্ত দেখাচ্ছে']

sentences_embeddings=transformer.encode(sentences)

for i in range(len(sentences)):
    j=i+1
    while j<len(sentences):
        s1=sentences[i]
        s2=sentences[j]
        print(s1,' --- ',s2,transformer.similarity(sentences_embeddings[s1],sentences_embeddings[s2]))
        j+=1
```

**Output:**

```
আপনার বয়স কত  ---  আপনার বয়স কত tensor([[1.0000]])
আপনার বয়স কত  ---  আমি তোমার বয়স জানতে চাই tensor([[0.8607]])
আপনার বয়স কত  ---  আমার ফোন ভাল আছে tensor([[0.1994]])
আপনার বয়স কত  ---  আপনার সেলফোনটি দুর্দান্ত দেখাচ্ছে tensor([[0.2581]])
আমি তোমার বয়স জানতে চাই  ---  আমি তোমার বয়স জানতে চাই tensor([[1.0000]])
আমি তোমার বয়স জানতে চাই  ---  আমার ফোন ভাল আছে tensor([[0.1960]])
আমি তোমার বয়স জানতে চাই  ---  আপনার সেলফোনটি দুর্দান্ত দেখাচ্ছে tensor([[0.2495]])
আমার ফোন ভাল আছে  ---  আমার ফোন ভাল আছে tensor([[1.0000]])
আমার ফোন ভাল আছে  ---  আপনার সেলফোনটি দুর্দান্ত দেখাচ্ছে tensor([[0.9281]])
আপনার সেলফোনটি দুর্দান্ত দেখাচ্ছে  ---  আপনার সেলফোনটি দুর্দান্ত দেখাচ্ছে tensor([[1.0000]])
```

## How to add and train new data?

If you want to train more data or add data, you should install 'Cuda' GPU.\
If you haven't any nvidia graphics card, You should use Google Colab GPU. \
CPU is very much slow for transformers model.

Suppose you have a dataset, now you want to train and with my model.
```python
from Bangla-sentence-embedding-transformer.Bangla_transformer import Bangla_sentence_transformer_small

transformer=Bangla_sentence_transformer_small()

path='/dataset.txt'
transformer.train(path)
```

Run this in google colab. **You must download my model from drive**

## How to create my own model?

It was pretty much same way.
```python
from Bangla-sentence-embedding-transformer.Bangla_transformer import Bangla_sentence_transformer_small

transformer=Bangla_sentence_transformer_small()

path='/dataset.txt'
transformer.train_new(path)
```

**You don't need to download my model, if you want to create your own model**

## How to prepare a dataset?

This model needs parallel dataset of english-bangla. First line of your text file must be a english and bangla sentence separated by
a tab. **Sentence length should be less than 128**

```
English sentence1 \tab Bangla sentence1
English sentence2 \tab Bangla sentence2
English sentence3 \tab Bangla sentence3
- - - 
- - -
```

Suppose you have only bangla Sentence, You can use Google translator and manually check it.
Or you can directly use it. Google translator accuracy(85%).

## About my model

I prepared 2,50,000 parallel dataset for training using google translator. Then i roughly check it.

Epochs=5 \
Every Epochs iteration=7000 \
Device=google colab gpu \
MSE=11.39 \
Evaluation_size=500 \
training_time=3 hours 44 minutes
