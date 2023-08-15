
from simpletransformers.ner import NERModel, NERArgs
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import wandb




df = pd.read_csv("bioNER_final.csv")
df = df.dropna()
df = df.drop(["Unnamed: 0"],axis = 1 )
df.rename(columns={"tokens": "words", "tags": "labels"},inplace=True)
df["labels"]=df.labels.map(str)
df["sentence_id"]=df.sentence_id.map(str)
train, test = train_test_split(df, test_size=0.2)
test,validation = train_test_split(test, test_size=0.2)

cuda_available = torch.cuda.is_available()

print(cuda_available)

# wandb.login()
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="NER_IOB",

# )


model_args = NERArgs()
model_args.labels_list = df.labels.unique().tolist()
model_args.classification_report=True
# model_args.wandb_project = "NER_IOB"
model_args.num_train_epochs=1000
model_args.overwrite_output_dir = True
model_args.reprocess_input_data = True
model_args.use_early_stopping = True
model_args.early_stopping_delta = 0.01
model_args.early_stopping_metric = "mcc"
model_args.early_stopping_metric_minimize = False
model_args.early_stopping_patience = 5
model_args.evaluate_during_training_steps = 1000


saved_model  = "./out_train_ner_bio/checkpoint-302604-epoch-906"
model = NERModel(
    model_type= "roberta",
    model_name= saved_model,
    labels =df.labels.unique().tolist(),
    args=model_args,
    use_cuda=True,
)

result, model_outputs, wrong_preds = model.eval_model(validation,output_dir = "./out_eval_ner_bio/",verbose= True)
print("result : " + str(result))

p = "1"
while p!='0':
    p = str(input("enter sth ... \n"))
    predictions, raw_outputs = model.predict([p])

    print("prediction :  " + str(predictions))
    print("---------------------")
    # print("raw outputs :" + str(raw_outputs))
