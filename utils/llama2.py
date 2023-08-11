from llama_cpp import Llama
llm = Llama(model_path="./models/llama-2-13b-chat.ggmlv3.q4_1.bin", n_ctx=10240)
#llm = Llama(model_path="./models/llama-2-70b-chat.ggmlv3.q4_1.bin", n_ctx=10240)

def gen_response(prompt,previous_chats):
    full_prompt = ""
    for pc in previous_chats:
        full_prompt += f"Q: {pc['resolved_utterance']} A: {pc['response']}"

    full_prompt += f" Q: {prompt} A: "
    output = llm(full_prompt,max_tokens=250, stop=["Q:"], echo=True) # 250 tokens is TREC iKAT limit
    ans = output["choices"][-1]["text"]
    ans = ans.split(" A: ")[-1]
    return ans

def gen_summary(passage):
    prompt = f"Q: summerize this passage in 1 sentance: {passage} A:"
    output = llm(prompt,max_tokens=250, stop=["Q:"], echo=True)
    ans = output["choices"][-1]["text"]
    ans = ans.split(" A: ")[-1]
    ans = ans.replace('Here is a summary of the passage in one sentence:','')
    return ans


r = gen_summary("""
Lipids - an overview | ScienceDirect Topics\nLipids\nJ. Abraham Dom\u00ednguez-Avila, Gustavo A. Gonz\u00e1lez-Aguilar, in Postharvest Physiology and Biochemistry of Fruits and Vegetables, 2019\nAbstract\nLipids are structurally and functionally diverse. The most abundant lipids are triacylglycerols (TAGs), which are used by plants as for dense energy storage. Lipids of vegetable origin are regularly consumed by humans as part of our everyday diet, for example, as cooking oil (>99% lipids) or from nuts and seeds (up to 75%). Some specific plant lipids have to be consumed by humans, while others are highly recommended to prevent disease. This chapter defines the chemical structure of the most relevant lipids (fatty acids, TAGs, and phospholipids), describes their synthesis, lists the percentage of lipids in various fruits and vegetables, and how they relate to humans.\nView chapter Purchase book\nLipids\nAnet Re\u017eek Jambrak, Dubravka \u0160kevin, in Nutraceutical and Functional Food Components  (Second Edition), 2017\n4.1 Introduction\nLipids are usually referred to as fats and oils. Fats are materials that are solid at ambient temperature and oils are those liquid at ambient temperature. Lipids [characterized as oils, greases, fats, and fatty acids (FAs)] are one of the most important components of natural foods and many synthetic compounds and emulsions. The contribution of bioactive lipids to health is determined by their compositional factors. FA composition (especially levels of omega-3, omega-6, and omega-9 FAs) and other high-value minor lipid compounds (e.g., glycolipids, phospholipids, tocols, phytosterols, aroma compounds, and phenolics) have been shown to exhibit health-promoting properties and positively affect the physiological functions of our body.
""")
print(r)