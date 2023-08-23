import openai
from utils.remove_extra_spaces import remove_extra_spaces
from utils.prevent_trail_off import prevent_trail_off
import os
from dotenv import load_dotenv
import json

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def answer_question_from_passage(passage,question,previous_chats):
    RECEIVED_RESPONSE = False
    response = []
    while not RECEIVED_RESPONSE:
        try:
            openai.api_key = os.getenv("OPENAI_API_KEY")
            messages = []
            for chat in previous_chats:
                messages.append({"role": 'user',"content":chat['resolved_utterance']})
                messages.append({"role": 'assistant',"content":chat['response']})
            prompt = f"Answer this question - {question} - using this information - {remove_extra_spaces(passage)} and your own knowledge."
            messages.append({"role":"user","content":prompt})
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=250
            )
            RECEIVED_RESPONSE = True
        except:
            print("FAILED TO CONNECT TO OPENAI SERVERS - RETRYING")
    return prevent_trail_off(response['choices'][0]['message']['content'])

def determine_passage_relevance(passage, statement):
    RECEIVED_RESPONSE = False
    response = []
    while not RECEIVED_RESPONSE:
        try:
            prompt = f"Is this passage - {json.loads(passage.raw)['contents']} relevant to this response - {statement}? Answer with either 'yes' or 'no'."
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {'role':'user','content':prompt}
                ],
                max_tokens=10
            )
            RECEIVED_RESPONSE = True
        except:
            print("FAILED TO CONNECT TO OPENAI SERVERS - RETRYING")
    ans = remove_extra_spaces(response['choices'][0]['message']['content'])
    if ans[0:2].lower() == "no":
        return False
    if ans[0:3].lower() == 'yes':
        return True
    return False
"""
a = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        {"role": "user", "content": "Write an essay about it."}
    ],
   max_tokens=250
)

print(a)
"""