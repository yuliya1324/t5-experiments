import numpy as np
from statistics import mean
from matplotlib import pyplot as plt

def get_atts_mean(attentions, i, example, segment, layer, head, read=True, mem=False):
    if mem:
        if read:
            return [np.mean(attentions[example][segment]["attentions"][layer][0][head][1:11].detach().numpy(), axis=0)[0],
            np.mean(np.mean(attentions[example][segment]["attentions"][layer][0][head][1:11].detach().numpy(), axis=0)[1:11]),
            np.mean(attentions[example][segment]["attentions"][layer][0][head][1:11].detach().numpy(), axis=0)[11],
            np.mean(np.mean(attentions[example][segment]["attentions"][layer][0][head][1:11].detach().numpy(), axis=0)[12:-1]),
            np.mean(attentions[example][segment]["attentions"][layer][0][head][1:11].detach().numpy(), axis=0)[-1],]
        else:
            return [np.mean(attentions[example][segment]["attentions"][layer][0][head][0][1:11].detach().numpy(), axis=0),
            np.mean(np.mean(attentions[example][segment]["attentions"][layer][0][head][1:11].detach().numpy(), axis=0)[1:11]),
            np.mean(attentions[example][segment]["attentions"][layer][0][head][11][1:11].detach().numpy(), axis=0),
            np.mean(np.mean(attentions[example][segment]["attentions"][layer][0][head][12:-1].detach().numpy(), axis=0)[1:11]),
            np.mean(attentions[example][segment]["attentions"][layer][0][head][-1][1:11].detach().numpy(), axis=0),]
    else:
        if read:
            return [attentions[example][segment]["attentions"][layer][0][head][i][0].detach().numpy().item(),
            np.mean(attentions[example][segment]["attentions"][layer][0][head][i][1:11].detach().numpy()),
            attentions[example][segment]["attentions"][layer][0][head][i][11].detach().numpy().item(),
            np.mean(attentions[example][segment]["attentions"][layer][0][head][i][12:-1].detach().numpy()),
            attentions[example][segment]["attentions"][layer][0][head][i][-1].detach().numpy().item(),]
        else:
            return [attentions[example][segment]["attentions"][layer][0][head][0][i].detach().numpy().item(),
            np.mean(attentions[example][segment]["attentions"][layer][0][head][1:11][i].detach().numpy()),
            attentions[example][segment]["attentions"][layer][0][head][11][i].detach().numpy().item(),
            np.mean(attentions[example][segment]["attentions"][layer][0][head][12:-1][i].detach().numpy()),
            attentions[example][segment]["attentions"][layer][0][head][-1][i].detach().numpy().item(),]

def get_atts_max(attentions, i, example, segment, layer, head, read=True, mem=False):
    if mem:
        if read:
            return [np.max(attentions[example][segment]["attentions"][layer][0][head][1:11].detach().numpy(), axis=0)[0],
            np.max(np.max(attentions[example][segment]["attentions"][layer][0][head][1:11].detach().numpy(), axis=0)[1:11]),
            np.max(attentions[example][segment]["attentions"][layer][0][head][1:11].detach().numpy(), axis=0)[11],
            np.max(np.max(attentions[example][segment]["attentions"][layer][0][head][1:11].detach().numpy(), axis=0)[12:-1]),
            np.max(attentions[example][segment]["attentions"][layer][0][head][1:11].detach().numpy(), axis=0)[-1],]
        else:
            return [np.max(attentions[example][segment]["attentions"][layer][0][head][0][1:11].detach().numpy(), axis=0),
            np.max(np.max(attentions[example][segment]["attentions"][layer][0][head][1:11].detach().numpy(), axis=0)[1:11]),
            np.max(attentions[example][segment]["attentions"][layer][0][head][11][1:11].detach().numpy(), axis=0),
            np.max(np.max(attentions[example][segment]["attentions"][layer][0][head][12:-1].detach().numpy(), axis=0)[1:11]),
            np.max(attentions[example][segment]["attentions"][layer][0][head][-1][1:11].detach().numpy(), axis=0),]
    else:
        if read:
            return [attentions[example][segment]["attentions"][layer][0][head][i][0].detach().numpy().item(),
            np.max(attentions[example][segment]["attentions"][layer][0][head][i][1:11].detach().numpy()),
            attentions[example][segment]["attentions"][layer][0][head][i][11].detach().numpy().item(),
            np.max(attentions[example][segment]["attentions"][layer][0][head][i][12:-1].detach().numpy()),
            attentions[example][segment]["attentions"][layer][0][head][i][-1].detach().numpy().item(),]
        else:
            return [attentions[example][segment]["attentions"][layer][0][head][0][i].detach().numpy().item(),
            np.max(attentions[example][segment]["attentions"][layer][0][head][1:11][i].detach().numpy()),
            attentions[example][segment]["attentions"][layer][0][head][11][i].detach().numpy().item(),
            np.max(attentions[example][segment]["attentions"][layer][0][head][12:-1][i].detach().numpy()),
            attentions[example][segment]["attentions"][layer][0][head][-1][i].detach().numpy().item(),]

def print_layers(example=0, segment=1, multiple=False):
    """
    Mean by layer for each head
    """
    di_layer = {
        "CLS1_read": [],
        "CLS1_write": [],
        "SEP1_read": [],
        "SEP1_write": [],
        "MEM_read": [],
        "MEM_write": [],
        "SEP2_read": [],
        "SEP2_write": [],
    }
    ind = {
        "CLS1": 0,
        "SEP1": 1,
        "SEP2": -1
    }

    for key in di_layer:
        for layer in range(12):
            di_layer[key].append([])
            for head in range(12):
                if key[:3] == "MEM":
                    di_layer[key][layer].append(get_atts_mean(0, example, segment, layer, head, mem=True, read=key.endswith("read")))
                else:
                    di_layer[key][layer].append(get_atts_mean(ind[key[:4]], example, segment, layer, head, read=key.endswith("read")))
        di_layer[key] = np.mean(np.array(di_layer[key]), axis=0)

    t = ["CLS", "MEM", "SEP1", "tokens", "SEP2"]
    fig = plt.figure(figsize=(30, 30))
    i = 1
    for key in di_layer:
        ax = fig.add_subplot(4, 4, i)
        ax.matshow(di_layer[key], cmap='Reds')
        ax.set_title(key)
        ax.set_xticks(range(len(t)))
        ax.set_yticks(range(12))
        ax.set_yticklabels(range(12), fontdict={'fontsize': 10})
        ax.set_xticklabels(t,fontdict={'fontsize': 10}, rotation=90)
        i += 1

def print_heads(example=0, segment=1, multiple=False):
    """
    Max by head for each layer
    """
    di_head = {
        "CLS1_read": [],
        "CLS1_write": [],
        "SEP1_read": [],
        "SEP1_write": [],
        "MEM_read": [],
        "MEM_write": [],
        "SEP2_read": [],
        "SEP2_write": [],
    }
    ind = {
        "CLS1": 0,
        "SEP1": 1,
        "SEP2": -1
    }

    for key in di_head:
        for layer in range(12):
            di_head[key].append([])
            for head in range(12):
                if key[:3] == "MEM":
                    di_head[key][layer].append(get_atts_max(0, example, segment, layer, head, mem=True, read=key.endswith("read")))
                else:
                    di_head[key][layer].append(get_atts_max(ind[key[:4]], example, segment, layer, head, read=key.endswith("read")))
        di_head[key] = np.max(np.array(di_head[key]), axis=1)

    t = ["CLS", "MEM", "SEP1", "tokens", "SEP2"]
    fig = plt.figure(figsize=(30, 30))
    i = 1
    for key in di_head:
        ax = fig.add_subplot(4, 4, i)
        ax.matshow(di_head[key], cmap='Reds')
        ax.set_title(key)
        ax.set_xticks(range(len(t)))
        ax.set_yticks(range(12))
        ax.set_yticklabels(range(12), fontdict={'fontsize': 10})
        ax.set_xticklabels(t,fontdict={'fontsize': 10}, rotation=90)
        i += 1

def print_by_layer(example=0, segment=1, layer=0, multiple=False):
    """
    For each head in each layer
    """
    di = {
        "CLS1_read": [],
        "CLS1_write": [],
        "SEP1_read": [],
        "SEP1_write": [],
        "MEM_read": [],
        "MEM_write": [],
        "SEP2_read": [],
        "SEP2_write": [],
    }
    ind = {
        "CLS1": 0,
        "SEP1": 1,
        "SEP2": -1
    }

    for key in di:
        for head in range(12):
            if key[:3] == "MEM":
                di[key].append(get_atts_mean(0, example, segment, layer, head, mem=True, read=key.endswith("read")))
            else:
                di[key].append(get_atts_mean(ind[key[:4]], example, segment, layer, head, read=key.endswith("read")))

    t = ["CLS", "MEM", "SEP1", "tokens", "SEP2"]
    fig = plt.figure(figsize=(30, 30))
    i = 1
    for key in di:
        ax = fig.add_subplot(4, 4, i)
        ax.matshow(di[key], cmap='Reds')
        ax.set_title(key)
        ax.set_xticks(range(len(t)))
        ax.set_yticks(range(12))
        ax.set_yticklabels(range(12), fontdict={'fontsize': 10})
        ax.set_xticklabels(t,fontdict={'fontsize': 10}, rotation=90)
        i += 1