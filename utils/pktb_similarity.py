from sentence_transformers import SentenceTransformer, util

st_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
"""
def find_keywords(text):
    kw_extractor = yake.KeywordExtractor()
    keywords = kw_extractor.extract_keywords(text)
    return keywords

def spatialDistance(vector1, vector2):
    return spatial.distance.euclidean(vector1, vector2)

def cosineSimulatory(vector1, vector2):
    return np.dot(vector1,vector2)/(norm(vector1)*norm(vector2))

def findSimulatory(PTKB_keyword, Question_keyword):
    model = fasttext.load_model('./data/fastText/cc.en.300.bin')
    PTKB_cosine = model.get_sentence_vector(PTKB_keyword)
    Question_cosine = model.get_sentence_vector(Question_keyword)
    return cosineSimulatory(PTKB_cosine,Question_cosine)

def cosine_similarity(text1, text2):
    embedding1 = np.mean([model.get_word_vector(word) for word in text1.split()], axis=0)
    embedding2 = np.mean([model.get_word_vector(word) for word in text2.split()], axis=0)
    
    similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    
    return similarity

def keywordSim(PTKB, Question):
    MATCHING_THRESHOLD = 0.5
    PTKB_keywords = find_keywords(PTKB)
    Question_keywords = find_keywords(Question)
    total = {}
    for PTKB_keyword in PTKB_keywords:
        for question_keyword in Question_keywords:
            PTKB_cosine = model.get_word_vector(PTKB_keyword[0])
            Question_cosine = model.get_word_vector(question_keyword[0])
            cs = cosineSimulatory(PTKB_cosine,Question_cosine)
            if cs >= MATCHING_THRESHOLD:
                total[PTKB_keyword[0] + "-" + question_keyword[0]] = cs
    return len(total)
        """
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
    a = transformerSim(PTKBs,Text) # Rank PTKBs from response (reverse-order)
    return a


        
        
        

                