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