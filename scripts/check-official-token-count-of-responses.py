from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
import json
nlp = English()

tokenizer = Tokenizer(nlp.vocab)
tokenizer = Tokenizer(nlp.vocab)

with open('./output/BEST_RUN_250_TOKEN.json', 'r') as f: 
    data = json.load(f)
    for response in data[0]:
        a = data[0]["responses"][0]['text']
        if [*a][-1] != ".":
            print("=========================")
            print(a)
            print("=========================")
            """
            if (len(a) > 250):
            print("🚩")
            """

        
