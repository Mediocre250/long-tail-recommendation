import re
import os
import codecs
import gensim
import numpy as np
from gensim.models import word2vec
import logging


# Manner1: load GoogleNews vectors
class Manner1():
    def __init__(self):
        super(Manner1, self).__init__()

    def extract_feats(self, word):
        model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
        return model[word]


# Manner2: train new vectors
class Manner2():
    def __init__(self):
        super(Manner2, self).__init__()

    # Step 1: download your corpus

    # Step 2: Process the data
    def process_symbol(self, input_file):
        p1 = re.compile(r'-\{.*?(zh-en):([^;]*?)(;.*?)?\}-')
        p2 = re.compile(r'[（\(][,;.?!\s]*[）\)]')
        p3 = re.compile(r'[「『《]')
        p4 = re.compile(r'[」』》]')
        p5 = re.compile('<doc (.*)>')
        p6 = re.compile('</doc>')
        outfile = codecs.open('std_' + input_file, 'w', 'utf-8')
        with codecs.open(input_file, 'r', 'utf-8') as myfile:
            for line in myfile:
                line = p1.sub(r'\2', line)
                line = p2.sub(r'', line)
                line = p3.sub(r'“', line)
                line = p4.sub(r'”', line)
                line = p5.sub('', line)
                line = p6.sub('', line)
                outfile.write(line)
        outfile.close()

    # Step 3：word segmentaion

    # Step 4: train word2vec model
    def train(self):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        sentences = word2vec.LineSentence('./cut_std_en_wiki')
        model = word2vec.Word2Vec(sentences, size=200, window=5, min_count=5, workers=4)
        model.save('./word2vecModel/WikiENModel')

    def extract_feats(self, word):
        model = word2vec.Word2Vec.load('./word2vecModel/WikiCHModel')
        return model[word]

''
if __name__ == '__main__':

    path = "/home/..."
    feature_path = "/home/..."
    manner1 = Manner1()
    manner2 = Manner2()
    for text in os.listdir(path):
        with open(os.path.join(path,text),"r") as f:
            words=f.readlines()
        wordlist=[]
        for word in words:
            word_feature=manner1.extract_feats(word)
            #feature=manner2.extract_feats(word)
            wordlist.append(word_feature)
        feature=np.array(wordlist)
        np.save(feature_path + text, feature)