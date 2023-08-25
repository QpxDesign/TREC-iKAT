from sentence_transformers import SentenceTransformer, util
from utils.extract_keywords import extract_keywords

st_model = SentenceTransformer('paraphrase-MiniLM-L6-v2',device='cuda')

def transformerSim(statements, question):
    question_embedding = st_model.encode(question, convert_to_tensor=True)

    statement_embeddings = st_model.encode(statements, convert_to_tensor=True)

    cosine_scores = util.pytorch_cos_sim(question_embedding, statement_embeddings)

    ranked_statements = sorted(
        zip(statements, cosine_scores.tolist()[0]), key=lambda x: x[1], reverse=True
    )

    return ranked_statements 

def rankPTKBS(PTKBs, Text):
    # print(f"Sample Response: {turn['response']}")
    text_keywords = extract_keywords(Text)
    if len(text_keywords) == 0:
        return []
    keyword = extract_keywords(Text)[0][0]
    a = transformerSim(PTKBs,keyword) # Rank PTKBs from response (reverse-order)
    return a
        
"""
a = rankPTKBS(["I'm lactose intolerant.", "I'm vegetarian", "I'm allergic to soybeans","Because of my kidney problem, I have to drink water frequently to stay hydrated.", "I have an expensive car."], "Can you held me find a diet?")
print(a)
"""