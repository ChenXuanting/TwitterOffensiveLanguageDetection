import sys
import torch
import numpy as np
from transformers import (BertConfig, BertForSequenceClassification, BertTokenizer,
                                  RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
                                  AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer,
                                  XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer,
                                  )
from DataLoader import load_train_data, load_test_data
from trainer import trainer
from predict import predict_sentences
from utils import save_model, save_predictions, eval_stats

if len(sys.argv) > 2:
    raise ValueError("Only two arguments are accepted.")
elif len(sys.argv) < 2:
    raise ValueError("You have to provide the model name.")

torch.cuda.empty_cache()

"""!!!Modify this part to change models!!!"""

Model_name = sys.argv[1]
    #Available models: 'albert-base-v1', 'albert-large-v1', 'albert-xlarge-v1', 'albert-xxlarge-v1'
                #'albert-base-v2', 'albert-large-v2', 'albert-xlarge-v2', 'albert-xxlarge-v2'
                #'roberta-base', 'roberta-large','xlm-roberta-base', 'xlm-roberta-large', 'bert-base-uncased', 'bert-large-uncased'
                #Browse https://huggingface.co/ for more pretrained models
Model_type = Model_name.split('-')[0]
if Model_type == 'xlm':
    Model_type = 'xlmroberta'
Output_path = 'cached-results/' + Model_type + '/'

seq_len = 128
train_batch_size = 18
eval_batch_size = 18
epochs = 6
weight_decay = 0
LR = 5e-6
adam_eps = 1e-9
max_norm = 1.0

Output_path = Output_path+Model_name+"/"
print("Output Dir:",Output_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_mapping = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'albert': (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    'xlmroberta': (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer)
}

config_class, model_class, tokenizer_class = model_mapping[Model_type]

config = config_class.from_pretrained(Model_name, num_labels=2)
tokenizer = tokenizer_class.from_pretrained((Model_name))
X_train = load_train_data(tokenizer, seq_len)
X_test = load_test_data(tokenizer, seq_len)

"""Do train"""
print(f"Start training {Model_name}:")
model = model_class.from_pretrained(Model_name, num_labels=2)
step, tr_loss = trainer(model, X_train, Model_type, train_batch_size, epochs, LR, weight_decay, adam_eps, max_norm, device)
print(" step =", step, ", average loss =", tr_loss)

save_model(model, tokenizer, Output_path)

"""Do test"""
predict_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_test['input_ids'].values.tolist(), dtype=torch.long),
            torch.tensor(X_test['input_mask'].values.tolist(), dtype=torch.long),
            torch.tensor(X_test['segment_ids'].values.tolist(), dtype=torch.long),
            torch.tensor([1] * X_test.shape[0], dtype=torch.long)
)

prob_scores = predict_sentences(model, predict_dataset, Model_type, eval_batch_size, device)
predicted_labels = [a.argmax() for a in prob_scores]

save_predictions(predicted_labels, Output_path)

result = eval_stats(np.array(X_test['label'].values.tolist()), predicted_labels)

print("Model performance:")
print("NOT:" + "\t" +  "P: %s" %(str(round(result["p_not"]*100, 3))) + "\t" +  "R: %s" %(str(round(result["r_not"]*100, 2))) + "\t" +  "F1: %s" %(str(round(result["f1_not"]*100, 2))))
print("OFF:" + "\t" +  "P: %s" %(str(round(result["p_off"]*100, 2))) + "\t" +  "R: %s" %(str(round(result["r_off"]*100, 2))) + "\t" +  "F1: %s" %(str(round(result["f1_off"]*100, 2))))
print("F1: %s" %(str(round(result["f1"]*100, 2))) + "\t" + "ACC: %s" %(str(round(result["acc"]*100, 2))))