import torch
from torch.optim import AdamW
from tqdm import tqdm, trange

def trainer(model, train_data, Model_type, train_batch_size, epochs, LR, weight_decay, adam_eps, max_norm, device):
    """Training process"""
    model.to(device)

    train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(train_data['input_ids'].values.tolist(), dtype=torch.long),
            torch.tensor(train_data['input_mask'].values.tolist(), dtype=torch.long),
            torch.tensor(train_data['segment_ids'].values.tolist(), dtype=torch.long),
            torch.tensor(train_data['label'].values.tolist(), dtype=torch.long)
    )

    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=LR, eps=adam_eps)

    print("===== Training =====")
    print("  Num examples =", len(train_dataset))
    print("  Num Epochs =", epochs)
    print("  Total train batch size =", train_batch_size)

    step = 0
    tr_loss = 0.0
    model.zero_grad()
    train_iterator = trange(epochs, desc="Epoch")

    epoch_i = 0
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        epoch_i += 1
        print("\nTraining Epoch", epoch_i)
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = [t.to(device) for t in batch]
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if Model_type in ['bert'] else None,
                      'labels':         batch[3]}
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            print("\rReal-time loss: %f" % loss, end='')

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            tr_loss += loss.item()

            optimizer.step()
            model.zero_grad()

            step += 1

        # Save model checkpoint
        # output_dir = os.path.join(Output_path, 'checkpoint-{}'.format(epoch_i))
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)
        # model.save_pretrained(output_dir)
        # print(" ")
        # print("Saving model checkpoint to", output_dir)


    return step, tr_loss / step