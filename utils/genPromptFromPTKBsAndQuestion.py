
def genPromptFromPTKBAndQuestion(PTKBs, question):
    prompt_pktbs = []
    for d in PTKBs:
        prompt_pktbs.append(d[0])
    prmpt = ""
    if len(prompt_pktbs) == 1: 
        return f"{prompt_pktbs[0]} {question}" 
    if len(prompt_pktbs) == 0:
        return question
    
    index = 0
    for ptkb in prompt_pktbs:
        if index != len(prompt_pktbs)-1: 
            prmpt += f"{ptkb.replace('.','')} and "
        else:
            prmpt += f"{ptkb} {question}"
        index += 1 
    return prmpt

