# Load model directly
# Use a pipeline as a high-level helper
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoConfig
import torch
from timer import Timer
import logging
logging.basicConfig(filename='outputs.log',level=logging.INFO)
max_new_tokens = 1024

template = """
A virtual assistant answers questions from a user based on the provided text.
USER: Text: {}
ASSISTANT: I’ve read this text.
USER: What describes Products in the text?
ASSISTANT: (model's predictions in JSON format)
"""       
prompt = 'Given a paragraph, your task is to extract Product entity and concept, and define their type using a short sentence. The output should be in the following format: [("entity", "definition of entity type in a short sentence"), ... ]'
text="Fresh Kiwi Fruit"
name = "Universal-NER/UniNER-7B-definition"
# path = "./uniner_model/"
print("loading model ..  \n")


def get_raw_answer(answer):
  answer_list = answer[0]['generated_text'].split('\n')
  return answer[-1]
model_8bit = AutoModelForCausalLM.from_pretrained(name, device_map="auto", load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained(name)

pipe = pipeline("text-generation", model="Universal-NER/UniNER-7B-type",do_sample=False)
recognizer = pipeline("text-generation", model=model_8bit, tokenizer=tokenizer,do_sample=False,max_new_tokens=max_new_tokens)


text = '1'
while text != '0':
    text = str(input("enter your input : \n"))
    if text == '' : continue  

    model_input = template.format(text)
    logging.info("input: " + str(text))
    with Timer("Type model",logging):
      result = pipe(model_input,max_new_tokens=max_new_tokens)
    print("the pipe result is : \n" + result[0]['generated_text'].split('\n')[-1])
    logging.info("the pipe result is : \n" + result[0]['generated_text'].split('\n')[-1])

    with Timer("Defenition model",logging):
      res2 = recognizer(model_input,max_new_tokens=max_new_tokens)

    print("the pipe ner result is : \n" + res2[0]['generated_text'].split('\n')[-1])
    logging.info("the pipe ner result is : \n" + res2[0]['generated_text'].split('\n')[-1])
    print()
