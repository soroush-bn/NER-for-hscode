from simpletransformers.ner import NERModel, NERArgs
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import wandb


df = pd.read_csv("NER.csv",error_bad_lines=False)
df = df.dropna()
df = df.drop(["Unnamed: 0"],axis = 1 )
df["labels"]=df.labels.map(str)
df["sentence_id"]=df.sentence_id.map(str)
train, test = train_test_split(df, test_size=0.2)
test,validation = train_test_split(test, test_size=0.2)

cuda_available = torch.cuda.is_available()

print(cuda_available)

wandb.login()
wandb.init(
    # set the wandb project where this run will be logged
    project="NER",

)


model_args = NERArgs()
model_args.labels_list = df.labels.unique().tolist()
model_args.classification_report=True
model_args.wandb_project = "NER"
model_args.num_train_epochs=1
model_args.overwrite_output_dir = True
model_args.reprocess_input_data = True



model = NERModel(
    model_type= "roberta",
    model_name= "roberta-base",
    labels =df.labels.unique().tolist(),
    args=model_args,
    use_cuda=True,


)


model.train_model(train,output_dir='./out_train/',show_running_loss=True)

result, model_outputs, wrong_preds = model.eval_model(validation,output_dir = "./out_eval/",verbose= True)
print("result : " + str(result))
print("wrong preds" + str(wrong_preds))
wandb.join()