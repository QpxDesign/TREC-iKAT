from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import re
from pyserini.search.lucene import LuceneSearcher
import json

searcher = LuceneSearcher('data/clueweb/indexes/ikat_collection_2023_02') #indexes/ikat_collection_2023_01
def getPassagesFromSearchQuery(query, maxNumberPassages=10):
    hits = searcher.search(q=query,k=maxNumberPassages)
    return hits

features = []
labels = []

# GET UNRELIABLE PASSAGES FROM QUORA & ADVERTISEMENTS USING GET_PASSAGES
QUORA_PASSAGES = []
qp = getPassagesFromSearchQuery("quora",500)
for p in qp:
    QUORA_PASSAGES.append(json.loads(p.raw)["contents"])

ADVERTISEMENT_PASSAGES = []
spam_keywords = [
    "Free", "Guaranteed", "Discount", "Limited Time", "Exclusive", "Win",
    "Earn Money", "Cash", "Congratulations", "Click Here", "Act Now",
    "Unsubscribe", "Viagra", "Cialis", "Pharmacy", "Online Casino",
    "Work from Home", "Dating", "Enlarge", "Enhance",
    "Credit Repair", "Investment", "Multi-level Marketing", "Pornography", "Supplements"
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
    WIKIPEDIA_PASSAGES = WIKIPEDIA_PASSAGES[:(len(QUORA_PASSAGES)+len(ADVERTISEMENT_PASSAGES))]

for wp in WIKIPEDIA_PASSAGES:
    features.append(f"This is a reliable passage from Wikipedia: {wp}")
    labels.append("reliable")

for qp in QUORA_PASSAGES:
    features.append(f"This is an unreliable passage from Quora: {qp}")
    labels.append("unreliable")

for ap in ADVERTISEMENT_PASSAGES:
    features.append(f"This is a spam or junk passage: {ap}")
    labels.append("junk")

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

tfidf_vectorizer = TfidfVectorizer(max_features=len(WIKIPEDIA_PASSAGES)*2)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

predictions = model.predict(X_test_tfidf)

def determinePassageReliability(passage):
    new_passage_tfidf = tfidf_vectorizer.transform([passage])

    predicted_label = model.predict(new_passage_tfidf)

    return predicted_label[0]
