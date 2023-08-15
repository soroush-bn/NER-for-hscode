# Load model directly
# Use a pipeline as a high-level helper
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch




def generate_from_model(text,model, tokenizer):
  encoded_input = tokenizer(text, return_tensors='pt')
  output_sequences = model.generate(input_ids=encoded_input['input_ids'].cuda())
  return tokenizer.decode(output_sequences[0], skip_special_tokens=True)



max_new_tokens = 512
text="Fresh Kiwi Fruit"
name = "Universal-NER/UniNER-7B-definition"
path = "./uniner_model/"
print("loading model ..  \n")
# model_8bit = AutoModelForCausalLM.from_pretrained(name, device_map="auto", load_in_8bit=True)
# tokenizer = AutoTokenizer.from_pretrained(name)

prompt = '{} What describes Product in the text?'
pipe = pipeline("text-generation", model="Universal-NER/UniNER-7B-type")

text = '0'
while text != '0':
    text = str(input("enter your input : \n"))
    input =  prompt.format(text)

    result = pipe(input)
    print(result)
    # prompt = 'Given a paragraph, your task is to extract all entities and concepts, and define their type using a short sentence. The output should be in the following format: [("entity", "definition of entity type in a short sentence"), ... ] the paragraph is : {}'
# result  =generate_from_model(prompt.format("Where is my Fresh Kiwi Fruit ?"),model_8bit,tokenizer)

print(result)