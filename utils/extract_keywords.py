import yake
kw_extractor = yake.KeywordExtractor()

def extract_keywords(text):
    keywords = kw_extractor.extract_keywords(text)
    return keywords

"""
a = extract_keywords("Black garlic is a versatile ingredient that can be used to add depth and complexity to various dishes. It can be roasted to create a sweet and savory glaze for meats, or blended into a paste for added richness. Here are some ideas on how you can use black garlic in your gourmet dishes:  1. Roasted Black Garlic Glaze: Roast black garlic cloves in the oven until they become caramelized and sweet, then mash them into a glaze to brush over meats or vegetables during cooking.  2. Black Garlic Paste: Blend roasted black garlic with olive oil and lemon juice to create a rich and creamy paste that can be used as a topping for pasta dishes, risottos, or mashed potatoes.  3. Black Garlic-Stuffed Meats: Stuff black garlic cloves into chicken breasts, pork loin, or lamb racks before roasting them to create a flavorful and savory dish.")
print(a)
"""