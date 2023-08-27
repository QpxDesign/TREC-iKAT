def sort_passage_provenances(passages): # PUT USED PASSAGES ON TOP
    sorted_passages = []
    for passage in passages:
        if passage['used']:
            sorted_passages.append(passage)
    for passage in passages:
        if not passage['used']:
            sorted_passages.append(passage)
    return sorted_passages
