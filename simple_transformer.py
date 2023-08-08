from simpletransformers.ner import NERModel, NERArgs
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split



df = pd.read_csv("NER.csv")

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


model = NERModel(
    model_type= "roberta",
    model_name= "roberta-base",
    labels =df.labels.unique().tolist(),
    args=model_args,
    use_cuda=True,


)


model.train_model(train,output_dir='./out_train/',show_running_loss=True)

result, model_outputs, wrong_preds = model.eval_model(eval_data,output_dir = "./out_eval/",verbose= True)
wandb.join()