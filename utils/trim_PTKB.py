def trim(PTKBs):
    cutoffValue = .25 
    for i in range(len(PTKBs)):
        if PTKBs[i][1] < cutoffValue:
            return PTKBs[0:i]
    return PTKBs
