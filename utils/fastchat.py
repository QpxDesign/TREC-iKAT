from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers import (   
    AutoModelForSeq2SeqLM,   
    T5Tokenizer,
)
tokenizer = AutoTokenizer.from_pretrained("lmsys/fastchat-t5-3b-v1.0")
model = AutoModelForSeq2SeqLM.from_pretrained("lmsys/fastchat-t5-3b-v1.0")

def summerize_with_fastchat(passage):
    passage = f"Human: summerize this in 1 senatance: {passage} Assistant: "  # Ensure the task is properly formatted
    encoded_input = tokenizer.encode(passage, return_tensors='pt')
    output = model.generate(encoded_input, max_length=512, temperature=0.7, top_p=1)
    decoded_output = tokenizer.decode(output[0, :], skip_special_tokens=True)
    decoded_output = decoded_output.replace('<pad> ','')
    decoded_output = decoded_output.replace("  "," ")
    return decoded_output

res = summerize_with_fastchat(
"""
Lipids - an overview | ScienceDirect Topics\nLipids\nJ. Abraham Dom\u00ednguez-Avila, Gustavo A. Gonz\u00e1lez-Aguilar, in Postharvest Physiology and Biochemistry of Fruits and Vegetables, 2019\nAbstract\nLipids are structurally and functionally diverse. The most abundant lipids are triacylglycerols (TAGs), which are used by plants as for dense energy storage. Lipids of vegetable origin are regularly consumed by humans as part of our everyday diet, for example, as cooking oil (>99% lipids) or from nuts and seeds (up to 75%). Some specific plant lipids have to be consumed by humans, while others are highly recommended to prevent disease. This chapter defines the chemical structure of the most relevant lipids (fatty acids, TAGs, and phospholipids), describes their synthesis, lists the percentage of lipids in various fruits and vegetables, and how they relate to humans.\nView chapter Purchase book\nLipids\nAnet Re\u017eek Jambrak, Dubravka \u0160kevin, in Nutraceutical and Functional Food Components  (Second Edition), 2017\n4.1 Introduction\nLipids are usually referred to as fats and oils. Fats are materials that are solid at ambient temperature and oils are those liquid at ambient temperature. Lipids [characterized as oils, greases, fats, and fatty acids (FAs)] are one of the most important components of natural foods and many synthetic compounds and emulsions. The contribution of bioactive lipids to health is determined by their compositional factors. FA composition (especially levels of omega-3, omega-6, and omega-9 FAs) and other high-value minor lipid compounds (e.g., glycolipids, phospholipids, tocols, phytosterols, aroma compounds, and phenolics) have been shown to exhibit health-promoting properties and positively affect the physiological functions of our body.
""")
print(res)