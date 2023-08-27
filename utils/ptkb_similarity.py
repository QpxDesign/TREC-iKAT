from sentence_transformers import SentenceTransformer, util
# from utils.extract_keywords import extract_keywords

st_model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device='cuda')


def transformerSim(statements, question):
    question_embedding = st_model.encode(question, convert_to_tensor=True)

    statement_embeddings = st_model.encode(statements, convert_to_tensor=True)

    cosine_scores = util.pytorch_cos_sim(
        question_embedding, statement_embeddings)

    ranked_statements = sorted(
        zip(statements, cosine_scores.tolist()[0]), key=lambda x: x[1], reverse=True
    )

    return ranked_statements


def rankPTKBS(PTKBs, Text):
    # print(f"Sample Response: {turn['response']}")
    a = transformerSim(PTKBs, Text)
    return a


"""
a = rankPTKBS([
    "I don't like the new spin-off; because I keep comparing the two and it has lower quality.",
    "Because of my kidney problem, I have to drink water frequently to stay hydrated.",
    "I'm going to change my phone.",
    "I can't exercise too much because of the heart problem that I have.",
    "I'm vegetarian.",
    "I'm lactose intolerant.",
    "I'm allergic to soybeans.",
    "I just finished watching the Game of Thrones.",
    "I didn't like how the series ended, especially the war scenes.",
    "I'm an Android user."
], "I prefer a natural diet, not a pill-based diet. Which of the aforementioned ones is natural?")
print(a)
"""
