from string import punctuation
from nltk import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups
import gensim.downloader
import pickle
import numpy as np
import nltk



categories = ['sci.med','sci.space']
nltk.download('stopwords')

newsgroups = fetch_20newsgroups(categories=categories, remove=('headers', 'footers', 'quotes'))
eng_stopwords = set(stopwords.words('english'))

tokenizer = RegexpTokenizer(r'\s+', gaps=True)
stemmer = PorterStemmer()
translate_tab = {ord(p): u" " for p in punctuation}

def text2tokens(raw_text):
    """Split the raw_text string into a list of stemmed tokens."""
    clean_text = raw_text.lower().translate(translate_tab)
    tokens = [token.strip() for token in tokenizer.tokenize(clean_text)]
    tokens = [token for token in tokens if token not in eng_stopwords]
    #stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return [token for token in tokens if len(token) > 2]  # skip short tokens

dataset = [text2tokens(txt) for txt in newsgroups['data'] if len(text2tokens(txt))>0]  # convert a documents to list of tokens
targets = [newsgroups['target'][i] for i,txt in enumerate(newsgroups['data']) if len(text2tokens(txt))>0]

from gensim.corpora import Dictionary
dictionary = Dictionary(documents=dataset, prune_at=None)
dictionary.filter_extremes(no_below=5, no_above=0.6, keep_n=None)  # use Dictionary to remove un-relevant tokens
dictionary.compactify()

vocab = dictionary.token2id

print("Newsgroup loaded")

print("Downloading fasttext")
fasttext_vectors = gensim.downloader.load('fasttext-wiki-news-subwords-300')
print("Fasttext downloaded")

embeddings = np.zeros((len(dictionary), 300))
for w,i in dictionary.token2id.items():
    try:
        embeddings[i] = fasttext_vectors.wv[w]
    except KeyError:
        pass
    
sentences = []

for sentence in dataset:
    l = []
    for word in sentence:
        try:
            index = dictionary.token2id[word]
            if np.linalg.norm(embeddings[index])>0:
                l.append(index)
        except KeyError:
            pass
    sentences.append(l)
    

N_sentences = 100000
size_sentences = 12
np.random.seed(79)

lenghts = np.array([len(sentence) for sentence in sentences])
probas = lenghts/np.sum(lenghts)

all_sentences = np.zeros((N_sentences, size_sentences))

index_sentences = np.random.choice(np.arange(len(sentences)), size = N_sentences, p=probas)
labels = np.zeros(N_sentences)

for i,index in enumerate(index_sentences):
    if len(sentences[index])<=size_sentences:
        all_sentences[i, 0:len(sentences[index])] = np.array(sentences[index])
    else:
        start = int(np.random.randint(0, len(sentences[index])-size_sentences))
        all_sentences[i] = np.array(sentences[index][start:start+size_sentences])
    labels[i] = targets[index]
    
print("Sentences generated")
    
np.save("newsgroup/sentences.npy", all_sentences)
np.save("newsgroup/embeddings.npy", embeddings)
np.save("newsgroup/labels.npy", labels)
with open('newsgroup/vocab.p', 'wb') as outfile:
    pickle.dump(vocab, outfile)
    