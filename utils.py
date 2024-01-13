import os
import pickle
from sklearn.metrics import (matthews_corrcoef, confusion_matrix,
                              accuracy_score, f1_score, precision_score, recall_score)

def save_model(model, tokenizer, Output_path):
    """Save trained model"""

    os.makedirs(Output_path, exist_ok=True)
    print("Saving model checkpoint to " + Output_path)

    model.save_pretrained(Output_path)
    tokenizer.save_pretrained(Output_path)

def save_predictions(predicted_labels, Output_path):
    """Save predictions"""

    os.makedirs(Output_path, exist_ok=True)

    pickle.dump(predicted_labels, file=open(os.path.join(Output_path, "testset_predictions.p"), "wb"))

    print(f"Prediction saved to {Output_path}testset_predictions.p")

def eval_stats(labels, preds):
    """Generate stats"""

    mcc = matthews_corrcoef(labels, preds)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    acc = accuracy_score(labels, preds)

    # macro
    f1 = f1_score(labels, preds, average='macro')
    p = precision_score(labels, preds, average='macro')
    r = recall_score(labels, preds, average='macro')

    # not
    f1_0 = f1_score(labels, preds, average='binary', pos_label=0)
    p_0 = precision_score(labels, preds, average='binary', pos_label=0)
    r_0 = recall_score(labels, preds, average='binary', pos_label=0)

    # off
    f1_1 = f1_score(labels, preds, average='binary', pos_label=1)
    p_1 = precision_score(labels, preds, average='binary', pos_label=1)
    r_1 = recall_score(labels, preds, average='binary', pos_label=1)

    return {
        "mcc": mcc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "acc" : acc,
        "f1" : f1,
        "precision" : p,
        "recall" : r,
        "p_not" : p_0,
        "r_not" : r_0,
        "f1_not" : f1_0,
        "p_off" : p_1,
        "r_off" : r_1,
        "f1_off" : f1_1
    }