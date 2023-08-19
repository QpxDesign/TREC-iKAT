import re
def prevent_trail_off(response):
    list_matches = re.findall("[1-9]\.\d*|\d+\.",response)
    if len(list_matches) > 1: #checks if it's listing off things 
        r = response.split(list_matches[-1])
        return r[0]
    else:
        return re.sub('\.[^.]*$','.',response)
        