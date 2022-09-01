from modeling_rmt import RMTEncoderForSequenceClassification
import torch
import json
import logging
import os
import re
import string
from pathlib import Path
from torch.utils.data import DataLoader, DistributedSampler, Dataset
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import transformers  # noqa: E402
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, AutoModel
from matplotlib import pyplot as plt
from utils import grad_layer_attention_map, plot_attention_weights, grad_heads_attention_map
from attention import print_by_layer_grad, print_by_layer_head

model_name = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = {
    'num_mem_tokens': 10,
    'input_size': 512,
    # 'input_seg_size': args.input_seg_size,
    'model_attr': 'deberta',
    # 'backbone_cls': backbone_cls,
    'bptt_depth': -1, 
    'pad_token_id': 0,
    'cls_token_id': tokenizer.cls_token_id, 
    'sep_token_id': tokenizer.sep_token_id,
    'eos_token_id': 102,
    "data_path": "data/test.jsonl",
    "batch_size": 4,
    "gradient_accumulation_steps": 2,
    "data_n_workers": 1,
}
labels_map = {'false': 0, 'true': 1}
encode_plus_kwargs = {
    'max_length': 998,
    'truncation': True,
    'padding': 'longest',
    'pad_to_multiple_of': 1
    }

class HyperpartisanDataset(Dataset):
    def __init__(self, datafile, x_field='text', label_field='label'):
        if isinstance(datafile, str):
            # convert str path to folder to Path
            datafile = Path(datafile)
        self.data = []
        for line in datafile.open('r'):
            self.data += [json.loads(line)]
        self.x_field = x_field
        self.label_field = label_field

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx][self.x_field]
        label = self.data[idx][self.label_field]
        return x, label

def collate_fn(batch):
    inputs, labels = zip(*batch)
    features = tokenizer.batch_encode_plus(list(inputs), return_tensors='pt', **encode_plus_kwargs)
    labels = np.array([labels_map[t] for t in labels])
    labels = {'labels': torch.from_numpy(labels)}
    return {**features, **labels}

def main():
    load_path = "/home/admin/t5-experiments/Julia/deberta/run_1/model_best.pth"
    checkpoint = torch.load(load_path, map_location='cpu')
    model = RMTEncoderForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.set_params(num_mem_tokens=10, 
                    input_size=512,
                    input_seg_size=998,
                    model_attr=config["model_attr"],
                    # backbone_cls=transformers.BartForConditionalGeneration,
                    bptt_depth=-1, 
                    pad_token_id=tokenizer.pad_token_id,
                    cls_token_id=tokenizer.cls_token_id, 
                    sep_token_id=tokenizer.sep_token_id,
                    eos_token_id=config["eos_token_id"],)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.cuda()
    model.zero_grad()

    test_dataset = HyperpartisanDataset('/home/admin/t5-experiments/Julia/data/test.jsonl')
    test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)

    i = 0 
    for batch in test_dataloader:
        i += 1
        if i == 5:
            batch["input_ids"] = batch["input_ids"].to("cuda")
            batch["token_type_ids"] = batch["token_type_ids"].to("cuda")
            batch["attention_mask"] = batch["attention_mask"].to("cuda")
            batch["labels"] = batch["labels"].to("cuda")
            pred = model(
                **batch,
                output_attentions=True,
                return_dict=True,
                # output_hidden_states=True
                )
            break
    
    # attentions = np.array([layer_atts[0].cpu().detach().squeeze().numpy() if layer_atts[0] is not None else None for layer_atts in pred[1]["attentions"]])
    attentions = np.array([attn[1][0].detach().squeeze().cpu().numpy() for attn in pred[1]['attentions']])
    avg_per_layer_att = attentions.mean(axis=1)
    print(avg_per_layer_att.shape)


    attention_maps = grad_layer_attention_map(model, batch, pred[1], tqdm_enable=True)
    tokens = [tokenizer.convert_ids_to_tokens(t_id).replace('▁', '') for t_id in batch["input_ids"][0].tolist()[-500:]]
    tokens = ["[CLS]"] + ["[MEM]"] * 10 + ["[SEP]"] + tokens
    grad_based_attentions = attention_maps[0]
    
    # plot_attention_weights([grad_based_attentions], tokens, layer=0, layout=(2,6), figsize=(20,7), caption='Layer', save=True)
    # print_by_layer_grad(grad_based_attentions)

    # print('layer-avg attention maps:')
    plot_attention_weights([avg_per_layer_att], tokens, layer=0, layout=(6,2), figsize=(30, 90), caption='Layer', save=True, filename="att_layer.png")
    # print('layer-avg grad-based attention maps:')
    plot_attention_weights([grad_based_attentions], tokens, layer=0, layout=(6,2), figsize=(30, 90), caption='Layer', save=True, filename="att_grad.png")

    # shape = pred[1]['attentions'][1][0].shape
    # hiddens_per_head = torch.stack([attn[0].view(*shape[:-1], model.config.num_attention_heads, -1).permute(0, 2, 1, 3) for attn in pred[1]['attentions']])
    # bs x n_layers x n_heads x seq_len x per_head_hidden
    # hiddens_per_head = hiddens_per_head.permute(1, 0, 2, 3, 4)
    # grad_based_attentions = grad_heads_attention_map(model, batch, hiddens_per_head, pred[1])
    # grad_based_attentions = grad_based_attentions[0]
    # for i in range(12):
    #     print_by_layer_head(grad_based_attentions, i)
    
    # for i in range(12):
    #     plot_attention_weights(grad_based_attentions, tokens, layer=i, layout=(6,2), figsize=(30, 90), caption='Head', save=True, filename=f"layer_{i}.png")



if __name__=="__main__":
    main()
