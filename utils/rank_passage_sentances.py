from sentence_transformers import SentenceTransformer, util
st_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def rank(passage, question):
    MAX_SENTANCES = 50
    def merge_sentances_into_paragraph(sentances):
        ans = ""
        for s in sentances:
            ans += f"{s[0]}. "
        return ans
        
    sentances = passage.split('.')
    question_embedding = st_model.encode(question, convert_to_tensor=True)

    statement_embeddings = st_model.encode(sentances, convert_to_tensor=True)

    cosine_scores = util.pytorch_cos_sim(question_embedding, statement_embeddings)

    ranked_statements = sorted(
        zip(sentances, cosine_scores.tolist()[0]), key=lambda x: x[1], reverse=True
    )
    return merge_sentances_into_paragraph(ranked_statements[:MAX_SENTANCES])