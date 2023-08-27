import yake
kw_extractor = yake.KeywordExtractor()


def extract_keywords(text):
    keywords = kw_extractor.extract_keywords(text)
    return keywords


"""a = extract_keywords("1) Vegan Mediterranean: No, it excludes animal products including fish and other seafood. 2) Vegan Keto: No, it also excludes fish and seafood.3) Eco-Atkins: It allows for up to one serving per day of low-fat dairy or fish. However, if you choose not to eat these foods, the diet can still be followed without them. 4) The Ornish Diet (with limited exercise): No, it excludes all animal products including fish and seafood.")
print(a)"""
