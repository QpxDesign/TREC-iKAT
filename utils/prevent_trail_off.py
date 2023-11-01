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


# a = prevent_trail_off("The BMW 4 Series Gran Coupe is a practical and sporty car that offers extra practicality. The Hyundai Coupe is a sporty option for those looking for a do-it-all car at a low price. The Ferrari California T is a sleek and practical supercar that is recommended as a first foray into the Ferrari world. It has 540 horsepower and is a convertible.")
# print(a)
