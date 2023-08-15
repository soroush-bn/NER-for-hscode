# Load model directly
# Use a pipeline as a high-level helper
# from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch




def generate_from_model(text,model, tokenizer):
  encoded_input = tokenizer(text, return_tensors='pt')
  output_sequences = model.generate(input_ids=encoded_input['input_ids'].cuda())
  return tokenizer.decode(output_sequences[0], skip_special_tokens=True)



max_new_tokens = 20
text="Fresh Kiwi Fruit"
name = "Universal-NER/UniNER-7B-definition"
path = "./uniner_model/"
print("loading model ..  \n")
model_8bit = AutoModelForCausalLM.from_pretrained(path, device_map="auto", load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained(path)


# pipe = pipeline("text-generation", model="Universal-NER/UniNER-7B-type")

# text = str(input("enter your input : \n"))
 

# result = pipe(text)
# print(result)

result  =generate_from_model("Where is my Fresh Kiwi Fruit ?",model_8bit,tokenizer)
