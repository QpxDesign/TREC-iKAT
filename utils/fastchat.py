from transformers import AutoTokenizer, T5ForConditionalGeneration
from utils.rank_passage_sentances import rank
from transformers import (
    AutoModelForSeq2SeqLM,
    T5Tokenizer,
)
import torch
import time

# SET THIS (NEED A LOT OF VRAM IF USING LLAMA WITH GPU AND FASTCHAT WITH GPU)
USE_GPU = False

device = 'cuda' if torch.cuda.is_available() and USE_GPU else 'cpu'


tokenizer = AutoTokenizer.from_pretrained(
    "lmsys/fastchat-t5-3b-v1.0", legacy=False)
# ONLY HAVE load_in_8bit True if using GPU with limited VRAM
if USE_GPU:
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "lmsys/fastchat-t5-3b-v1.0", load_in_8bit=True)
else:
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "lmsys/fastchat-t5-3b-v1.0")


def summarize_with_fastchat(passage, question, rankSentances=True):
    MAX_PASSAGE_LENGTH = 512  # char
    if rankSentances == False:
        passage = f"Human: Summarize this passage in 1-2 sentences {passage} Assistant: "
    else:
        passage = f"Human: Summarize this passage in 1-2 sentences {rank(passage, question)[:MAX_PASSAGE_LENGTH]} Assistant: "
    encoded_input = tokenizer.encode(passage, return_tensors='pt').to(device)
    output = model.generate(
        encoded_input, max_length=1024, temperature=0.7, top_p=1)
    decoded_output = tokenizer.decode(output[0, :], skip_special_tokens=True)
    decoded_output = decoded_output.replace('<pad> ', '')
    decoded_output = decoded_output.replace("  ", " ")
    print("GENERATED FASTCHAT SUMMARY")
    return decoded_output


def awnser_question(question):
    prompt = f"Human: {question} Assistant: "
    encoded_input = tokenizer.encode(prompt, return_tensors='pt').to(device)
    output = model.generate(
        encoded_input, max_length=1024, temperature=0.7, top_p=1)
    decoded_output = tokenizer.decode(output[0, :], skip_special_tokens=True)
    decoded_output = decoded_output.replace('<pad> ', '')
    decoded_output = decoded_output.replace("  ", " ")
    print("GENERATED FASTCHAT SUMMARY")
    return decoded_output


# start_time = time.time()
# a = summarize_with_fastchat(
#    "The four terrestrial or inner planets have dense, rocky compositions, few or no moons, and no ring systems. They are in hydrostatic equilibrium, forming a rounded shape, and have undergone planetary differentiation, causing chemical elements to accumulate at different radii. They are composed largely of refractory minerals such as silicates—which form their crusts and mantles—and metals such as iron and nickel which form their cores. Three of the four inner planets (Venus, Earth and Mars) have atmospheres substantial enough to generate weather; all have impact craters and tectonic surface features, such as rift valleys and volcanoes. The term inner planet should not be confused with inferior planet, which designates those planets that are closer to the Sun than Earth (i.e. Mercury and Venus).[89] The inner Solar System is the region comprising the terrestrial planets and the asteroid belt.[87] Composed mainly of silicates and metals,[88] the objects of the inner Solar System are relatively close to the Sun; the radius of this entire region is less than the distance between the orbits of Jupiter and Saturn. This region is also within the frost line, which is a little less than 5 AU (750 million km; 460 million mi) from the Sun.[27] The four outer planets, also called giant planets or Jovian planets, collectively make up 99% of the mass known to orbit the Sun.[f] Jupiter and Saturn are together more than 400 times the mass of Earth and consist overwhelmingly of the gases hydrogen and helium, hence their designation as gas giants.[129] Uranus and Neptune are far less massive—less than 20 Earth masses (M🜨) each—and are composed primarily of ice. For these reasons, some astronomers suggest they belong in their own category, ice giants.[130] All four giant planets have rings, although only Saturn's ring system is easily observed from Earth. The term superior planet designates planets outside Earth's orbit and thus includes both the outer planets and Mars.[89] The outer region of the Solar System is home to the giant planets and their large moons. The centaurs and many short-period comets also orbit in this region. Due to their greater distance from the Sun, the solid objects in the outer Solar System contain a higher proportion of volatiles, such as water, ammonia, and methane than those of the inner Solar System because the lower temperatures allow these compounds to remain solid, without significant rates of sublimation.[15]", "Which planets have weather?", rankSentances=True)
# print(a)
