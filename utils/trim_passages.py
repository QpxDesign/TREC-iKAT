def trim_passages(passages):
    return passages[:3]
    """cutoffValue = 10 # TO-DO: FIND GOOD CUT OFF VALUE
    for i in range(len(passages)):
        if passages[i].score < cutoffValue:
            return passages[0:i]
    return passages
"""