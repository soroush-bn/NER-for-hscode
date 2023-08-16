# Load model directly
# Use a pipeline as a high-level helper
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoConfig
import torch

from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
# generation_config = GenerationConfig(
#     temperature=0.2,
#     top_p=0.75,
#     top_k=40,
#     num_beams=4,
#     max_new_tokens=32,
# )

max_new_tokens = 512
def generate_from_model(text,model, tokenizer):
  encoded_input = tokenizer(text, return_tensors='pt')
  output_sequences = model.generate(input_ids=encoded_input['input_ids'].cuda(),max_new_tokens=max_new_tokens,return_dict_in_generate=True )
  print("out sequence is : " +  str(output_sequences))
  return tokenizer.decode(output_sequences[0], skip_special_tokens=True,clean_up_tokenization_spaces=False)


def generate_prompt(instruction: str, paragraph: str = None) -> str:
    if paragraph:
          return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

  ### Instruction:
  {instruction}

  ### Input:
  {paragraph}

  ### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""




examples = [{"conversations": [{"from": "human", "value": f"Text: I want some Fresh Kiwi Fruits"}, {"from": "gpt", "value": "I've read this text."}, {"from": "human", "value": f"What describes Product in the text?"}, {"from": "gpt", "value": "[]"}]}]
        
prompt = 'Given a paragraph, your task is to extract Product entity and concept, and define their type using a short sentence. The output should be in the following format: [("entity", "definition of entity type in a short sentence"), ... ]'
text="Fresh Kiwi Fruit"
name = "Universal-NER/UniNER-7B-definition"
path = "./uniner_model/"
print("loading model ..  \n")
config = AutoConfig.from_pretrained(
  name,
  trust_remote_code=True,
  max_new_tokens=max_new_tokens
)
llm = HuggingFacePipeline.from_model_id(
    model_id=name,
    task="text-generation",
    model_kwargs= {'max_length' : max_new_tokens}
    
)
template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
 What describes Product in the text?
### Input:
{paragraph}

### Response:"""
prompt = PromptTemplate(template=template, input_variables=["paragraph"])

llm_chain = LLMChain(prompt=prompt, llm=llm)

def ask_question(paragraph):
    result = llm_chain(paragraph)
    # print(result['question'])
    print("result is  ======> : \n")
    print(result)


# model_8bit = AutoModelForCausalLM.from_pretrained(name, device_map="auto", load_in_8bit=True)
# tokenizer = AutoTokenizer.from_pretrained(name)

# pipe = pipeline("text-generation", model="Universal-NER/UniNER-7B-type")
# recognizer = pipeline("text-generation", model=model_8bit, tokenizer=tokenizer)

# model_id = "lmsys/fastchat-t5-3b-v1.0"

text = '1'
while text != '0':
    if text == '' : continue  
    text = str(input("enter your input : \n"))
    # input_pipe =  prompt.format(text)
    # print("ur paragraph is :  \n" + str(input_pipe))
    # result  =generate_from_model(input_pipe,model_8bit,tokenizer)
    # model_input = generate_prompt(prompt,text)
    # result = pipe(model_input)
    # print("the pipe result is : \n" + str(result))
    # res2 = recognizer(model_input)
    # print("the pipe ner result is : \n" + str(res2))
    ask_question(text)
    print()
    # prompt = 'Given a paragraph, your task is to extract all entities and concepts, and define their type using a short sentence. The output should be in the following format: [("entity", "definition of entity type in a short sentence"), ... ] the paragraph is : {}'
# result  =generate_from_model(prompt.format("Where is my Fresh Kiwi Fruit ?"),model_8bit,tokenizer)

# print(result)