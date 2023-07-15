import fasttext
from scipy import spatial
from numpy.linalg import norm
import yake
import numpy as np
from numpy.linalg import norm
import json

def find_keywords(text):
    kw_extractor = yake.KeywordExtractor()
    keywords = kw_extractor.extract_keywords(text)
    return keywords

def spatialDistance(vector1, vector2):
    return spatial.distance.euclidean(vector1, vector2)

def cosineSimulatory(vector1, vector2):
    return np.dot(vector1,vector2)/(norm(vector1)*norm(vector2))

def findSimulatory(PTKB_keyword, Question_keyword):
    model = fasttext.load_model('./data/fastText/cc.en.300.bin')
    PTKB_cosine = model.get_sentence_vector(PTKB_keyword)
    Question_cosine = model.get_sentence_vector(Question_keyword)
    return cosineSimulatory(PTKB_cosine,Question_cosine)

with open('./data/2023_train_topics.json', 'r') as f: 
    data = json.load(f)
    PTKBs = [None] * len(data[0]['ptkb'])
    index = 0
    while index < len(data[0]['ptkb']) :
        PTKBs[index] = data[0]['ptkb'][str(index+1)]
        index += 1
    for ptkb in PTKBs:
        a = findSimulatory(ptkb,'Do you want to continue your bachelors studies and obtain a degree in computer science?') # Rank PTKBs from response (reverse-order)
        print(ptkb + ": " + str(a))

