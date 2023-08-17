# NER-for-hscode
NER with simple transformers

## Quick start

clone the repository : 
`git clone https://github.com/soroush-bn/NER-for-hscode.git`

install the requirements : 

`pip install -r requirements.txt`

you can also exclude those libs that you can install via `conda` channels and install the rest with `pip` manually.

train the model :

1- sign in to your [wandb account](https://wandb.ai/) and get your private key.

2- run `python simple_transformer.py` (you can add you config within the code, arg parser not implemented yet)

3- you can go to your wandb account and see the training process.

testing the model:

1- find the output dir of your saved model, the default dir is `./out_train_ner_bio/`, replace it with the otput dir in the `test_NER.py`

2- run `python test_NER.py`


inference with UniNER: 

1- `python uniNER.py`


