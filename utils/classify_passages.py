from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import re
from pyserini.search.lucene import LuceneSearcher
import json
import time
import os

searcher = LuceneSearcher('data/passage-index')
def getPassagesFromSearchQuery(query, maxNumberPassages=10):
    hits = searcher.search(q=query,k=maxNumberPassages)
    return hits

features = []
labels = []

# GET UNRELIABLE PASSAGES FROM QUORA & ADVERTISEMENTS USING GET_PASSAGES
EXPLICIT_UNTRUSTWORTHY_SITES = ["quora","reddit","twitter","forum","u/","r/"]
EXPLICIT_UNTRUSTWORTHY_SITES_PASSAGES = []
qp = getPassagesFromSearchQuery("quora",500)
for keyword in EXPLICIT_UNTRUSTWORTHY_SITES:
    ap = getPassagesFromSearchQuery(keyword,10)
    for p in ap:
        EXPLICIT_UNTRUSTWORTHY_SITES_PASSAGES.append(json.loads(p.raw)["contents"])

ADVERTISEMENT_PASSAGES = []
spam_keywords = [
    "Free", "Guaranteed", "Discount", "Limited Time", "Exclusive", "Win",
    "Earn Money", "Cash", "Congratulations", "Click Here", "Act Now",
    "Unsubscribe", "Viagra", "Cialis", "Pharmacy", "Online Casino",
    "Work from Home", "Dating", "Enlarge", "Enhance",
    "Credit Repair", "Investment", "Multi-level Marketing", "Pornography", "Supplements", "Sex"
]
for keyword in spam_keywords:
    ap = getPassagesFromSearchQuery(keyword,10)
    for p in ap:
        ADVERTISEMENT_PASSAGES.append(json.loads(p.raw)["contents"])

# GET WIKIPEDIA (RELIABLE) PASSAGES FROM WIKITEXT
WIKIPEDIA_PASSAGES = []
with open('./data/text-classification/wikitext-103/wiki.train.tokens', 'r') as f: 
    lines = f.readlines()
    tmp = ""
    for line in lines:
        processable_line = re.sub(r'\s+', '', line)
        if len(processable_line) > 3:
            isSubjectLine = processable_line[0] == '=' and processable_line[1] != '='
            if not isSubjectLine:
                tmp += line
            else:
                WIKIPEDIA_PASSAGES.append(tmp)
                tmp = ""
    WIKIPEDIA_PASSAGES.pop(0)
    WIKIPEDIA_PASSAGES = WIKIPEDIA_PASSAGES[:(len(EXPLICIT_UNTRUSTWORTHY_SITES_PASSAGES)+len(ADVERTISEMENT_PASSAGES))]

# GET RELIABLE NEWS ARTICLES
NEWS_ARTICLES_FOLDER_PATH = './data/news-articles'
NEWS_ARTICLES = []
files = os.listdir(NEWS_ARTICLES_FOLDER_PATH)
for file in files:
    file_path = os.path.join(NEWS_ARTICLES_FOLDER_PATH, file)
    if os.path.isfile(file_path):
        with open(file_path, 'r') as f:
            NEWS_ARTICLES.append(f.read())


for wp in WIKIPEDIA_PASSAGES:
    features.append(f"This is a reliable passage from Wikipedia: {wp}")
    labels.append("reliable")

for qp in EXPLICIT_UNTRUSTWORTHY_SITES_PASSAGES:
    features.append(f"This is an unreliable passage from an untrustworthy website: {qp}")
    labels.append("unreliable")

for ap in ADVERTISEMENT_PASSAGES:
    features.append(f"This is a spam or junk passage: {ap}")
    labels.append("junk")

for na in NEWS_ARTICLES:
    features.append(f"This is a reliable passage from a news source: {na}")
    labels.append("reliable")

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

tfidf_vectorizer = TfidfVectorizer(max_features=len(WIKIPEDIA_PASSAGES)*2)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

predictions = model.predict(X_test_tfidf)
print("INITIALIZED TFID Vectorizer")

def determinePassageReliability(passage):
    START_TIME = time.time()
    new_passage_tfidf = tfidf_vectorizer.transform([passage])

    predicted_label = model.predict(new_passage_tfidf)
    #print(f"FINISHED determinePassageReliability in {time.time()-START_TIME} SECONDS")
    return predicted_label[0]
