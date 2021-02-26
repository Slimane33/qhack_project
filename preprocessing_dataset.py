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
    
print("Sentences newgroup generated")
    
np.save("newsgroup/sentences.npy", all_sentences)
np.save("newsgroup/embeddings.npy", embeddings)
np.save("newsgroup/labels.npy", labels)
with open('newsgroup/vocab.p', 'wb') as outfile:
    pickle.dump(vocab, outfile)
    
print('Generating dummy dataset')
    
names = ['woman', 'man','chef','policeman','dog','cat','apple','fish','teacher','toy',
         'kid','vegetable','doctor','car','boat','bird','meat','professor','president','student', 'chair', 'table']

adjectives = ['big','old','young','tiny','long','heavy','blue','strong','red','discret',
              'tender','rotten','gentle','funny','sad','light','complex','green','cheap','expensive']

verbs = ['eat','cut','cook','burn','fix','repair','build','hit','take','make', 'bake','paint','throw','push','create','look','pick', 'chop']

min_len = min(len(names),len(adjectives),len(verbs))

dictionary = names+adjectives+verbs

vocab = { word:i for i,word in enumerate(dictionary)}

names1 = [['man','chef', 'teacher', 'doctor', 'kid', 'president'], ['woman', 'policeman', 'doctor', 'kid', 'professor','president','student' ]]
adjectives1 = [['big','old','young','gentle', 'sad', 'tiny', 'funny'], ['big','old','young','gentle', 'sad', 'tiny', 'funny']]

verbs = [['eat','cut','cook','burn', 'bake', 'make','throw', 'pick', 'chop'], ['fix','repair','build','hit','take','throw', 'paint', 'pick', 'create']]

names2 = [['apple', 'meat', 'vegetable', 'dog', 'fish', 'bird'], ['chair', 'table', 'car', 'boat', 'toy']]
adjectives2 = [['expensive', 'tender', 'red', 'cheap', 'rotten'], ['old', 'light', 'blue', 'heavy', 'green']]

def generate_sentence(num_sentences, word_per_sentence):
    assert word_per_sentence in [3,4,5]
    all_sentences = []
    for i in range(num_sentences//2):
        sentence = []
        if word_per_sentence==3:
            R = np.random.randint(len(names1[0]))
            sentence.append(names1[0][R])
            R = np.random.randint(len(verbs[0]))
            sentence.append(verbs[0][R])
            R = np.random.randint(len(names2[0]))
            sentence.append(names2[0][R])
        elif word_per_sentence==4:
            c = np.random.randint(2)
            if c==0:
                R = np.random.randint(len(adjectives1[0]))
                sentence.append(adjectives1[0][R])
                R = np.random.randint(len(names1[0]))
                sentence.append(names1[0][R])
                R = np.random.randint(len(verbs[0]))
                sentence.append(verbs[0][R])
                R = np.random.randint(len(names2[0]))
                sentence.append(names2[0][R])
            else : 
                R = np.random.randint(len(names1[0]))
                sentence.append(names1[0][R])
                R = np.random.randint(len(verbs[0]))
                sentence.append(verbs[0][R])
                R = np.random.randint(len(adjectives2[0]))
                sentence.append(adjectives2[0][R])
                R = np.random.randint(len(names2[0]))
                sentence.append(names2[0][R])

        elif word_per_sentence==5:
            R = np.random.randint(len(adjectives1[0]))
            sentence.append(adjectives1[0][R])
            R = np.random.randint(len(names1[0]))
            sentence.append(names1[0][R])
            R = np.random.randint(len(verbs[0]))
            sentence.append(verbs[0][R])
            R = np.random.randint(len(adjectives2[0]))
            sentence.append(adjectives2[0][R])
            R = np.random.randint(len(names2[0]))
            sentence.append(names2[0][R])
        all_sentences.append(sentence)
            
    for i in range(num_sentences//2, num_sentences):
        sentence = []
        if word_per_sentence==3:
            R = np.random.randint(len(names1[1]))
            sentence.append(names1[1][R])
            R = np.random.randint(len(verbs[1]))
            sentence.append(verbs[1][R])
            R = np.random.randint(len(names2[1]))
            sentence.append(names2[1][R])
        elif word_per_sentence==4:
            c = np.random.randint(2)
            if c==0:
                R = np.random.randint(len(adjectives1[1]))
                sentence.append(adjectives1[1][R])
                R = np.random.randint(len(names1[1]))
                sentence.append(names1[1][R])
                R = np.random.randint(len(verbs[1]))
                sentence.append(verbs[1][R])
                R = np.random.randint(len(names2[1]))
                sentence.append(names2[1][R])
            else : 
                R = np.random.randint(len(names1[1]))
                sentence.append(names1[1][R])
                R = np.random.randint(len(verbs[1]))
                sentence.append(verbs[1][R])
                R = np.random.randint(len(adjectives2[1]))
                sentence.append(adjectives2[1][R])
                R = np.random.randint(len(names2[1]))
                sentence.append(names2[1][R])

        elif word_per_sentence==5:
            R = np.random.randint(len(adjectives1[1]))
            sentence.append(adjectives1[1][R])
            R = np.random.randint(len(names1[1]))
            sentence.append(names1[1][R])
            R = np.random.randint(len(verbs[1]))
            sentence.append(verbs[1][R])
            R = np.random.randint(len(adjectives2[1]))
            sentence.append(adjectives2[1][R])
            R = np.random.randint(len(names2[1]))
            sentence.append(names2[1][R])
        all_sentences.append(sentence)
            
    return all_sentences

np.random.seed(48)
dummy_sentences_3 = generate_sentence(num_sentences=3000, word_per_sentence=3)
dummy_sentences_4 = generate_sentence(num_sentences=6000, word_per_sentence=4)
dummy_sentences_5 = generate_sentence(num_sentences=10000, word_per_sentence=5)

def create_sentence_index(sentences):
    all_sentences = []
    for sentence in sentences:
        s = []
        for word in sentence:
            s.append(vocab[word])
        all_sentences.append(s)
    return all_sentences

embeddings = np.zeros((len(vocab.items()), 300))

for word,i in vocab.items():
    embeddings[i] = fasttext_vectors.wv[word]
    
dummy_sentences_index_3 = create_sentence_index(dummy_sentences_3)
dummy_sentences_index_4 = create_sentence_index(dummy_sentences_4)
dummy_sentences_index_5 = create_sentence_index(dummy_sentences_5)
    
np.save('dummy_dataset_example/dummy_sentences_3', np.array(dummy_sentences_index_3))
np.save('dummy_dataset_example/dummy_sentences_4', np.array(dummy_sentences_index_4))
np.save('dummy_dataset_example/dummy_sentences_5', np.array(dummy_sentences_index_5))

np.save('dummy_dataset_example/embeddings', embeddings)
import pickle

with open('dummy_dataset_example/vocab.p', 'wb') as outfile:
    pickle.dump(vocab, outfile)
    