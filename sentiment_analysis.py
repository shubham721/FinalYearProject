import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import pickle
import random
from nltk.stem import WordNetLemmatizer
from collections import Counter
from googletrans import Translator
from dbn_outside.dbn.tensorflow import SupervisedDBNClassification


lemmatizer = WordNetLemmatizer()
hm_lines =5000000
translator = Translator()
def create_lexicon(pos,neg):
    lexicon = []
    for file_name in [pos,neg]:
        with open(file_name,'r') as f:
            contents = f.readlines()
            for line in contents[:hm_lines]:
                all_words = word_tokenize(line.lower())
                lexicon += list(all_words)
    #print (lexicon[0:15])
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    #print (lexicon[0:15])
    word_counts = Counter(lexicon) # it will return kind of dictionary
    #print ((word_counts))
    l2 = []
    for word in word_counts:
        if 1500>word_counts[word]>5 :
            l2.append(word)
    
    print(l2)
    #print (len(l2))
    return l2

def sample_handling(sample, lexicon, classification):
    featureset= []
    with open(sample,'r') as f:
        contents = f.readlines()
        for line in contents[:hm_lines]:
            all_words = word_tokenize(line.lower())
            all_words = [lemmatizer.lemmatize(i) for i in all_words]
            features = np.zeros(len(lexicon))
            for word in all_words:
                if word.lower() in lexicon:
                    idx = lexicon.index(word)
                    features[idx] += 1
            features = list(features)
            featureset.append([features, classification])
    return featureset

def create_feature_set_and_labels(pos, neg, test_size = 0.2):
    lexicon = create_lexicon(pos,neg)
    features = []
    features += sample_handling(pos, lexicon, 1)
    features += sample_handling(neg, lexicon, 0)
    random.shuffle(features)
    features = np.array(features)
    print (len(features))
    testing_size = int(test_size*len(features))

    x_train = list(features[:,0][:-testing_size]) # taking features array upto testing_size
    y_train = list(features[:,1][:-testing_size]) # taking labels upto testing_size

    x_test = list(features[:,0][testing_size:])
    y_test = list(features[:,1][testing_size:])

    return x_train,y_train,x_test,y_test

def check_class(text,lexicon):
    line = translator.translate(text,dest='hi').text
    classifier = SupervisedDBNClassification.load('dbn.pkl')
    predict_set=[]
    all_words = word_tokenize(line.lower())
    all_words = [lemmatizer.lemmatize(i) for i in all_words]
    features = np.zeros(len(lexicon))
    for word in all_words:
        if word.lower() in lexicon:
            idx = lexicon.index(word)
            features[idx] += 1
    features = list(features)
    predict_set.append(features)
    predict_set = np.array(predict_set,dtype=np.float32) 
    predict_set = classifier.predict(predict_set)
    print(predict_set)


#if __name__ == '__main__':
    
    #x_train,y_train,x_test,y_test = create_feature_set_and_labels('pos.txt','neg.txt')
    #with open('sentiment_data.pickle','wb') as f:
     #   pickle.dump([x_train,y_train,x_test,y_test],f)
     #lexicon = create_lexicon('pos_hindi.txt','neg_hindi.txt')
    # check_class('while the performances are often engaging , this loose collection of largely improvised numbers would probably have worked better as a one-hour tv documentary . \
    #interesting , but not compelling . ',lexicon)
