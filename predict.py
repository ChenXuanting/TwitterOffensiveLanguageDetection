
import numpy as np
import torch
from tqdm import tqdm

def predict_sentences(model, predict_dataset, Model_type, eval_batch_size, device):
    """Testing process"""

    eval_sampler = torch.utils.data.SequentialSampler(predict_dataset)
    eval_dataloader = torch.utils.data.DataLoader(predict_dataset, sampler=eval_sampler, batch_size=eval_batch_size)
    prefix = ""

    print("***** Running evaluation {} *****".format(prefix))
    print("  Num examples =", len(predict_dataset))
    print("  Batch size =", eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if Model_type in ['bert'] else None,
                      'labels':         batch[3]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps

    sm = torch.nn.Softmax(dim=1)
    probabilities = sm(torch.from_numpy(preds)).numpy()

    return probabilities