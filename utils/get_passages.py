from pyserini.search.lucene import LuceneSearcher
import json

searcher = LuceneSearcher('data/clueweb/indexes/ikat_collection_2023_02') #indexes/ikat_collection_2023_01

def getPassagesFromSearchQuery(query):
    hits = searcher.search(query)
    return hits


"""a = getPassagesFromSearchQuery("Low-carb diet Ketogenic diet High-fat diet Low-glycemic index diet Ketosis Carbohydrate restriction Fat metabolism Weight loss Diabetes management Cardiovascular health Nutritional ketosis Atkins 40 Atkins 20 Induction phase Ongoing weight loss phase Pre-maintenance phase Maintenance phase Atkins diet food list Atkins diet meal plan Atkins diet recipes")[1]
b = json.loads(a.raw)["contents"]
print(b)
"""