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

def cosineSimulatory(vector1, vector2):
    return np.dot(vector1,vector2)/(norm(vector1)*norm(vector2))

def findSimulatory(PTKB_keyword, Question_keyword):
    model = fasttext.load_model('./data/fastText/cc.en.300.bin')
    PTKB_cosine = model.get_sentence_vector(PTKB_keyword)
    Question_cosine = model.get_sentence_vector(Question_keyword)
    return cosineSimulatory(PTKB_cosine,Question_cosine)

def cosine_similarity(text1, text2):
    embedding1 = np.mean([model.get_word_vector(word) for word in text1.split()], axis=0)
    embedding2 = np.mean([model.get_word_vector(word) for word in text2.split()], axis=0)
    
    similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    
    return similarity

def keywordSim(PTKB, Question):
    MATCHING_THRESHOLD = 0.5
    PTKB_keywords = find_keywords(PTKB)
    Question_keywords = find_keywords(Question)
    total = {}
    for PTKB_keyword in PTKB_keywords:
        for question_keyword in Question_keywords:
            PTKB_cosine = model.get_word_vector(PTKB_keyword[0])
            Question_cosine = model.get_word_vector(question_keyword[0])
            cs = cosineSimulatory(PTKB_cosine,Question_cosine)
            if cs >= MATCHING_THRESHOLD:
                total[PTKB_keyword[0] + "-" + question_keyword[0]] = cs
    return len(total)
        
def rankPTKBS():
    file_name = random.randint(0,673786)
    with open('./data/2023_train_topics.json', 'r') as f: 
        data = json.load(f)
        index = 0;
        for item in data:
            PTKBs = [None] * len(item['ptkb'])
            for turn in item['turns']:
               # print(f"Sample Response: {turn['response']}")

                index = 0
                while index < len(item['ptkb']):
                    PTKBs[index] = item['ptkb'][str(index+1)]
                    index += 1
                scoredPTKB = {}
                for ptkb in PTKBs:
                    a = keywordSim(ptkb,turn["response"]) # Rank PTKBs from response (reverse-order)
                    scoredPTKB[ptkb.replace(',','🔥')] = a
                    
                rankedPTKBS = sorted(scoredPTKB.items(), key=lambda x:x[1],reverse=True)
                hitOnOneProv = False
                for pktpRel in turn["ptkb_provenance"]:
                    if (item['ptkb'][str(pktpRel)] == rankedPTKBS[0][0]):
                        hitOnOneProv = print()
                
                    
                with open(f"./output/{file_name}.csv", 'a') as f2:
    
                    f2.write(f"{turn['response'].replace(',','🔥')},{rankedPTKBS}  \n")
                    f2.close()

                
rankPTKBS()
