from utils.remove_extra_spaces import remove_extra_spaces
import torch
import json
import time
# from utils.prevent_trail_off import prevent_trail_off
from utils.rank_passage_sentances import rank
from llama_cpp import Llama
# llm = Llama(model_path="./models/llama-2-13b-chat.ggmlv3.q4_1.bin",
#  n_ctx=10240, n_threads=16)  # - USE IF NO GPU/NOT ENOUGH VRAM
# llm = Llama(model_path="./models/llama-2-70b-chat.ggmlv3.q2_K.bin", n_gqa=8, n_ctx=10240, n_threads=16) - USE WITH NO GPU (SEE README) OR IF YOU HAVE LOADS OF VRAM
llm = Llama(model_path="./models/llama-2-13b-chat.ggmlv3.q3_K_M.bin",
            n_ctx=10240, n_gpu_layers=50)  # NEEDS ~8GB+ OF VRAM, MUCH FASTER
# llm = Llama(model_path='./models/llama-2-7b-chat.ggmlv3.q2_K.bin',
# n_ctx=10240, n_gpu_layers=50)

torch.cuda.empty_cache()


def gen_response(prompt, previous_chats):
    full_prompt = ""
    for pc in previous_chats:
        full_prompt += f"Q: {pc['utterance']} A: {pc['response']}"

    # When answering, always include all necessary information, repeating anything from the previous question(s) when necessary.
    full_prompt += f" Q: {prompt} A: "
    output = llm(full_prompt, max_tokens=250, stop=["Q:"], echo=True)
    ans = output["choices"][-1]["text"]
    ans = ans.split(" A: ")[-1]
    return ans


def gen_summary(passage, question):
    MAX_PASSAGE_LENGTH = 512  # num characters
    prompt = f"Q: Summarize this passage in 1 sentence: {rank(passage, question)[:MAX_PASSAGE_LENGTH]} A: "
    output = llm(prompt, max_tokens=250, stop=["Q:"], echo=True)
    ans = output["choices"][-1]["text"]
    ans = ans.split(" A: ")[-1]
    ans = ans.replace(
        "Sure! Here is a summary of the passage in one sentence:", '')
    ans = ans.replace('Here is a summary of the passage in one sentence:', '')
    ans = ans.replace(
        'Here is the summary in one sentence based on the information from the passages:', '')
    return ans


def answer_question_from_passage(passage, question, previous_chats):
    prompt = f"Answer this question - {question} - using this information - {remove_extra_spaces(passage)} "
    full_prompt = ""
    for pc in previous_chats:
        full_prompt += f"Q: {pc['resolved_utterance']} A: {pc['response']}"

    full_prompt += f" Q: {prompt} A: "
    output = llm(full_prompt, max_tokens=250, stop=["Q:"], echo=True)
    ans = output["choices"][-1]["text"]
    ans = ans.split(" A: ")[-1]
    ans = ans.replace(
        "Sure! Here is a summary of the passage in one sentence:", '')
    ans = ans.replace('Here is a summary of the passage in one sentence:', '')
    ans = ans.replace(
        'Here is the summary in one sentence based on the information from the passages:', '')
    ans = ans.replace(
        'Sure! Based on the information provided, here is a summary of the passage in one sentence:', '')
    ans = ans.replace("\n", "")
    return ans


def determine_passage_relevance(passage, statement):
    prompt = f"Q: Is this passage - {json.loads(passage.raw())['contents']} relevant to this response - {statement}? Answer with either 'yes' or 'no'. A: "
    output = llm(prompt, max_tokens=256, stop=["Q:"], echo=True)
    ans = output["choices"][-1]["text"]
    ans = ans.split(" A: ")[-1]
    ans = remove_extra_spaces(ans)
    print(ans)
    if ans[0:2].lower() == "no":
        return False
    if ans[0:3].lower() == 'yes':
        return True
    return False


"""
start_time = time.time()
full_prompt = "Q: List the names of 5 good diets to use to loose weight. A: 1. DASH Diet, 2. Keto, 3. Mediterranean. Q: Can i eat fish in any of those diets? A:"
output = llm(full_prompt, max_tokens=1024, stop=["Q:"], echo=True)
print(output)
ans = output["choices"][-1]["text"]
ans = ans.split(" A: ")[-1]
print(ans)
print(f"GENERATED OUTPUT in {time.time()-start_time} seconds")
"""
"""
turn_obj = [
    {
        "turn_id": 1,
        "utterance": "I'm looking for a car, can you help me?",
        "resolved_utterance": "I'm looking for a car, can you help me?",
        "response": "Sure, do you have a particular brand in mind?",
        "ptkb_provenance": [],
        "response_provenance": []
    },
    {
        "turn_id": 2,
        "utterance": "No, help me to buy one that best suits me.",
        "resolved_utterance": "No, help me to buy a car that best suits me.",
        "response": "Sure, since you like American-made goods, Ford, Dodge, Chevrolet, Pontiac, GMC, Buick, Cadillac, and Chrysler are some car brands from the Us. Surprisingly, Toyota, which is a Japanese brand, also makes cars in America for the US market. Also, Buick is still one of the most reliable American car manufacturers operating for 116 years now.",
        "ptkb_provenance": [2, 9],
        "response_provenance": [
            "clueweb22-en0036-68-06144:8",
            "clueweb22-en0033-81-08944:2",
            "clueweb22-en0040-24-07046:0",
            "clueweb22-en0040-24-07046:1",
            "clueweb22-en0040-24-07046:2",
            "clueweb22-en0040-24-07046:3",
            "clueweb22-en0040-24-07046:4",
            "clueweb22-en0040-24-07046:5",
            "clueweb22-en0040-24-07046:6",
            "clueweb22-en0040-24-07046:7",
            "clueweb22-en0040-24-07046:8",
            "clueweb22-en0040-24-07046:9",
            "clueweb22-en0040-24-07046:10",
            "clueweb22-en0040-24-07046:11",
            "clueweb22-en0040-24-07046:12",
            "clueweb22-en0007-02-13589:10",
            "clueweb22-en0007-02-13589:5"
        ]
    },
    {
        "turn_id": 3,
        "utterance": "Nice, I love American goods and I might like American cars, but for now let's talk about manufacturers of race cars.",
        "resolved_utterance": "Nice, I love American goods and I might like American cars, but for now let's talk about manufacturers of race cars.",
        "response": "Here are some manufacturers that race in F1 racing: McLaren, Aston Martin, Mercedes, Alpine, Ferrari, Alfa Romeo, Caterham, Renault, and Honda. Mercedes-Benz even went further and used F1 high-tech in creating the project-one model. Moreover, in Rally races Toyota, Hyundai, Volkswagen, Ford, Subaru, Citroen, Peugeot, and Mitsubishi are the manufacturers that won the recent 20 years of Rally races.",
        "ptkb_provenance": [5, 3],
        "response_provenance": [
            "clueweb22-en0002-85-15489:1",
            "clueweb22-en0002-85-15489:2",
            "clueweb22-en0002-85-15489:3",
            "clueweb22-en0002-85-15489:4",
            "clueweb22-en0030-85-06976:0",
            "clueweb22-en0009-19-02707:5"
        ]
    },
    {
        "turn_id": 4,
        "utterance": "Wow the project-one is an F1 tech car! Ok, but let's be realistic, I don't want a million-dollar one. Which ones are made here?",
        "resolved_utterance": "Wow the project-one is an F1 tech car! Ok, but let's be realistic, I don't want a million-dollar car. Which ones are made in the UK?",
        "response": "Do you mean the manufacturers that won Rally or the manufacturers that take part in the F1?",
        "ptkb_provenance": [11],
        "response_provenance": []
    },
    {
        "turn_id": 5,
        "utterance": "Both of them.",
        "resolved_utterance": "The manufacturers that won both Rally and F1.",
        "response": "The 2021 Formula 1 teams based in the UK are McLaren, Williams, Aston Martin, Mercedes, and Alpine. But, Mercedes-Benz is made in Germany. None of the Rally winner cars were made in the UK.",
        "ptkb_provenance": [],
        "response_provenance": [
            "clueweb22-en0025-83-03426:0",
            "clueweb22-en0042-11-15285:0",
            "clueweb22-en0042-11-15285:1",
            "clueweb22-en0042-11-15285:2",
            "clueweb22-en0042-11-15285:3"
        ]
    },
    {
        "turn_id": 6,
        "utterance": "Hmm, ok, what about Germany?",
        "resolved_utterance": "Hmm, ok, what about Germany?",
        "response": "Mercedes-Benz and Volkswagen are from Germany. There are other great car manufacturers in Germany, such as Audi, BMW, and Porsche. Audi R8 is for example a very good car for racing. The BMW M lineup is also great for racing. The Porsche 917 is considered one of the most iconic racing cars of all time and gave Porsche their first 24 Hours of Le Mans win.",
        "ptkb_provenance": [10, 5, 3],
        "response_provenance": [
            "clueweb22-en0042-11-15285:0",
            "clueweb22-en0042-11-15285:1",
            "clueweb22-en0042-11-15285:2",
            "clueweb22-en0042-11-15285:3",
            "clueweb22-en0011-05-06348:0",
            "clueweb22-en0021-51-04302:1",
            "clueweb22-en0015-73-15716:1"
        ]
    },
    {
        "turn_id": 7,
        "utterance": "Hmm, the Porsche 917 is an awesome car. But, I don't want to go that far. Can you recommend some good ones from the one you mentioned before?",
        "resolved_utterance": "Hmm, the Porsche 917 is an awesome car. But, I don't want to go that far. Can you recommend some good ones from the BMW M lineup?",
        "response": "There are some models such as BMW 1 series, X1, and X2.",
        "ptkb_provenance": [],
        "response_provenance": [
            "clueweb22-en0036-61-03055:3",
            "clueweb22-en0036-61-03055:4",
            "clueweb22-en0019-01-01618:0",
            "clueweb22-en0019-01-01618:1",
            "clueweb22-en0019-01-01618:2",
            "clueweb22-en0019-49-14412:0"
        ]
    },
    {
        "turn_id": 8,
        "utterance": "What's the difference between the first and second one?",
        "resolved_utterance": "What's the difference between the BMW 1 series and BMW X1?",
        "response": "If you are looking at the smallest BMW - the 1 Series - might like that it offers BMW quality and performance in a compact body but, if it's likely to be a squeeze next time you're packing for your holiday, you might be better off in the marque's compact SUV - the X1. So, if you're looking to acquire the BMW image for the least amount of money, we've compared the 1 Series and X1 to see which is the best to buy. BMW 1 Series on the left, X1 on the right Engines The 1 Series offers nine different engines while the X1 only offers six. However, the X1 offers more flexibility by offering four-wheel drive on a wider engine choice than the 1 Series. All the engines available for the X1 (20i, 16d, 18d, 20d, and 25d) are all available on the 1 Series but due to the X1 being a heavier, taller, wider, and overall less aerodynamic machine than the 1 Series, it's noticeably less economical.",
        "ptkb_provenance": [],
        "response_provenance": ["clueweb22-en0041-09-09771:0"]
    },
    {
        "turn_id": 9,
        "utterance": "Nice. How does X1 compare to Porsche?",
        "resolved_utterance": "Nice. How does BMW X1 compare to Porsche 917?",
        "response": "System: Well, it is not fair to compare a regular SUV production car of 21 century to a legendary racing car of the 20th century. Despite the age difference, the comparison is in the following. The 1969 Porsche 917 had a 12-cylinder engine with a V 180@ Displacement, 4,494 ccm capacity, and an output of 426 kW (620 hp). This iconic car weighed 800 kg with an astonishing top speed of 320 km/h in the short-tail version. The 2022 BMW X1 has a smaller 2.0-liter, 4-cylinder (Gasoline Fuel) engine with an output of 228 hp. This SUV weighs 1565 Kg with a top speed of 222 km/h. Hence, BMW X1 is heavier and slower than the Porsche 917.",
        "ptkb_provenance": [],
        "response_provenance": [
            "clueweb22-en0000-72-03607:0",
            "clueweb22-en0038-04-08696:9",
            "clueweb22-en0013-84-13757:0",
            "clueweb22-en0005-01-06045:0",
            "clueweb22-en0018-93-17608:0",
            "clueweb22-en0046-53-03571:6",
            "clueweb22-en0009-73-11404:0"
        ]
    },
    {
        "turn_id": 10,
        "utterance": "Nice. Ok then tell me more cars that won the Rally races.",
        "resolved_utterance": "Nice. Ok then tell me more cars from the German manufacturer that won the Rally races.",
        "response": "In Rally races Toyota, Hyundai, Volkswagen, Ford, Subaru, Citroen, Peugeot, and Mitsubishi are the manufacturer that won the recent 20 years of Rally races.",
        "ptkb_provenance": [],
        "response_provenance": ["clueweb22-en0009-19-02707:5"]
    },
    {
        "turn_id": 11,
        "utterance": "But Toyota is Japanese, I'm looking for a German one!",
        "resolved_utterance": "But Toyota is Japanese, I'm looking for a German car!",
        "response": "Oh, sorry. Here are some models from Volkswagen: Golf, Jetta, Touareg, Passat, and Polo.",
        "ptkb_provenance": [],
        "response_provenance": ["clueweb22-en0006-10-17022:0"]
    },
    {
        "turn_id": 12,
        "utterance": "I like the last option. Please compare it with the Audi A1.",
        "resolved_utterance": "I like Polo. Please compare the Polo with the Audi A1.",
        "response": "The two are also identical when it comes to performance: VW and Audi claim both can go from 0-62mph in 6.9 seconds, whilst only one mile-per-hour separates the A1's top speed from the Polo's superior 142mph. VW Polo GTI Audi is allegedly, though, putting the finishing touches, which should comfortably put the Polo GTI in its place in the Volkswagen Group hot hatch pecking order. Though, understandably, the Audi will most likely retail for an awful lot more than the Volkswagen. The A1 is also quite a pricey car when compared with the Polo, though the high-quality cabin, the smart styling, and the Audi badge go some way to justifying the premium. Moreover, VW Polo R WRC was a car that won a Rally race which I think you might wanna look into.",
        "ptkb_provenance": [],
        "response_provenance": [
            "clueweb22-en0025-06-15661:6",
            "clueweb22-en0019-86-13099:6"
        ]
    }
]
a = gen_response(
    "I own a black Harley-Davidson motorcycle. I'm looking for a car, can you help me?", turn_obj[0:5])
print(a)
"""
