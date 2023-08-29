import re
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
nlp = English()

tokenizer = Tokenizer(nlp.vocab)


def prevent_trail_off(response):
    tokens = tokenizer(response)
    new_response = ""
    for token_index in range(len(tokens)-1):
        if token_index < 250:
            new_response += str(tokens[token_index]) + " "
        else:
            break

    list_matches = re.findall("[1-9]\.\d*|\d+\.", new_response)
    if len(list_matches) > 1:  # checks if it's listing off things
        r = new_response.split(list_matches[-1])
        return r[0]
    else:
        return re.sub('\.[^.]*$', '.', new_response)
