# Load model directly
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="Universal-NER/UniNER-7B-type")

text = str(input("enter your input : \n"))
 

result = pipe(text)
print(result)