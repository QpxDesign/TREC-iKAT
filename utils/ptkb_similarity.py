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
"I drink milk every day",
"I own a Christy Dawn leather shoe",
"I hold a bachelor's degree in sociology",
"I love formula one races",
"I love Rally races.",
"I enjoy riding bikes.",
"I like to shop.",
"I like sightseeing.",
"I own a black Harley-Davidson motorcycle.",
"My children live in Germany.",
"I live in the UK.",

], "I'm looking for a car, can you help me?")
print(a)
"""
