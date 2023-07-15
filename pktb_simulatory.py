import fasttext
from scipy import spatial
from numpy.linalg import norm
import yake
import numpy as np
from numpy.linalg import norm
import json
import random

model = fasttext.load_model('./data/fastText/cc.en.300.bin')

def find_keywords(text):
    kw_extractor = yake.KeywordExtractor()
    keywords = kw_extractor.extract_keywords(text)
    return keywords

def spatialDistance(vector1, vector2):
    return spatial.distance.euclidean(vector1, vector2)

def cosine_similarity(text1, text2):
    embedding1 = np.mean([model.get_word_vector(word) for word in text1.split()], axis=0)
    embedding2 = np.mean([model.get_word_vector(word) for word in text2.split()], axis=0)
    
    similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    
    return similarity

def rankPTKBS():
    file_name = random.randint(0,673786)
    with open('./data/2023_train_topics.json', 'r') as f: 
        data = json.load(f)
        index = 0;
        for item in data:
            PTKBs = [None] * len(item['ptkb'])
            for turn in item['turns']:
                print(f"Sample Response: {turn['response']}")

                index = 0
                while index < len(item['ptkb']):
                    PTKBs[index] = item['ptkb'][str(index+1)]
                    index += 1
                scoredPTKB = {}
                for ptkb in PTKBs:
                    a = cosine_similarity(ptkb,turn["response"]) # Rank PTKBs from response (reverse-order)
                    scoredPTKB[ptkb.replace(',','🔥')] = a
                    
                rankedPTKBS = sorted(scoredPTKB.items(), key=lambda x:x[1],reverse=True)

                with open(f"./output/{file_name}.csv", 'a') as f2:
    
                    f2.write(f"{turn['response'].replace(',','🔥')},{rankedPTKBS}  \n")
                    f2.close()

                
rankPTKBS()
