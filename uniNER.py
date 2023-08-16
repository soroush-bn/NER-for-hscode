# Load model directly
# Use a pipeline as a high-level helper
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch



max_new_tokens = 512
def generate_from_model(text,model, tokenizer):
  encoded_input = tokenizer(text, return_tensors='pt')
  output_sequences = model.generate(input_ids=encoded_input['input_ids'].cuda(),max_new_tokens=max_new_tokens )
  print("out sequence is : " +  str(output_sequences))
  return tokenizer.decode(output_sequences[0], skip_special_tokens=True,clean_up_tokenization_spaces=False)


examples = [{"conversations": [{"from": "human", "value": f"Text: I want some Fresh Kiwi Fruits"}, {"from": "gpt", "value": "I've read this text."}, {"from": "human", "value": f"What describes Product in the text?"}, {"from": "gpt", "value": "[]"}]}]
        
prompt = 'Given the paragraph: {}, your task is to extract all entities and concepts, and define their type using a short sentence. The output should be in the following format: [("entity", "definition of entity type in a short sentence"), ... ]'
text="Fresh Kiwi Fruit"
name = "Universal-NER/UniNER-7B-definition"
path = "./uniner_model/"
print("loading model ..  \n")
model_8bit = AutoModelForCausalLM.from_pretrained(name, device_map="auto", load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained(name)

pipe = pipeline("text-generation", model="Universal-NER/UniNER-7B-type")
recognizer = pipeline("text-generation", model=model_8bit, tokenizer=tokenizer)
text = ''
while text != '0':
    if text = '' : continue  
    text = str(input("enter your input : \n"))
    input_pipe =  prompt.format(text)
    # print("ur paragraph is :  \n" + str(input_pipe))
    # result  =generate_from_model(input_pipe,model_8bit,tokenizer)
    result = pipe(text)
    print("the pipe result is : \n" + str(result))
    res2 = recognizer(text)
    print("the pipe ner result is : \n" + str(res2))

    print()
    # prompt = 'Given a paragraph, your task is to extract all entities and concepts, and define their type using a short sentence. The output should be in the following format: [("entity", "definition of entity type in a short sentence"), ... ] the paragraph is : {}'
# result  =generate_from_model(prompt.format("Where is my Fresh Kiwi Fruit ?"),model_8bit,tokenizer)

# print(result)