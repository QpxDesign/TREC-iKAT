def trim_passages(passages):
    cutoffValue = 30 
    for i in range(len(passages)):
        if passages[i].score < cutoffValue:
            return passages[0:i]
    return passages
