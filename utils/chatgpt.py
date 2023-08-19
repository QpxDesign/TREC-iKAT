import openai
from utils.remove_extra_spaces import remove_extra_spaces
from utils.prevent_trail_off import prevent_trail_off

def answer_question_from_passage(passage,question,previous_chats):
    messages = []
    for chat in previous_chats:
        messages.append({"role": 'user',"content":chat['resolved_utterance']})
        messages.append({"role": 'assistant',"content":chat['response']})
    prompt = f"Answer this question - {question} - using this information - {remove_extra_spaces(passage)} and your own knowledge."
    response = messages.append({"role":"user","content":prompt})
    return prevent_trail_off(response['choices'][0]['message']['content'])

a = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        {"role": "user", "content": "Where was it played?"}
    ]
)
print(a)