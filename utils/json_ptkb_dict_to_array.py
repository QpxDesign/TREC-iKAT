def format(PTKB_DICT):
    PTKBs = [None] * len(PTKB_DICT)
    index = 0
    while index < len(PTKB_DICT):
        PTKBs[index] = PTKB_DICT['ptkb'][str(index+1)]
        index += 1
    return PTKBs