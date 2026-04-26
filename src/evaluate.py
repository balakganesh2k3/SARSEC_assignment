import numpy as np
import torch
from interface import TOKEN_PADDING, MAX_LENGTH

def prepare_seq(seq_list):
    seq = seq_list[-MAX_LENGTH:] # keep only the tail for older interactions that are less useful          
    pad_len = MAX_LENGTH - len(seq)
    seq = [TOKEN_PADDING] * pad_len + seq # left-pad so the real tokens land at the end   
    return torch.LongTensor(seq).unsqueeze(0)

def rank_target(scores, target_idx):
    target_score = scores[target_idx]
    rank = int((scores > target_score).sum().item()) # 0 based meands rank 0 is the target second highest
    return rank

def evaluate(model, data, num_items, K=[10, 20], device=None):
    if device is None:
        try:
            device = next(model.parameters()).device # pull device from the model itself if not given
        except StopIteration:
            device = torch.device("cpu")  #no parameters then fall back to CPU
    all_items = torch.arange(1, num_items + 1, device=device)  # item IDs are 1-indexed
    ndcg_acc = {k: 0.0 for k in K}
    recall_acc = {k: 0.0 for k in K}
    num_evaluated = 0
    model.eval()
    with torch.no_grad():
        for u, (seq_list, target) in data.items():
            if len(seq_list) == 0:
                continue  # nothing to work with for this user
            seq_tensor = prepare_seq(seq_list).to(device)
            scores = model.predict(seq_tensor, all_items)  # score every item in the catalogue  
            scores = scores.squeeze(0)   # drop the batch dim - shape (num_items,)                   
            target_idx = target - 1 # convert 1 indexed ID to 0 indexed position
            rank = rank_target(scores, target_idx)  # how many items beat the target   
            for k in K:
                if rank < k:   # target landed in the top k
                    recall_acc[k] += 1.0
                    ndcg_acc[k] += 1.0 / np.log2(rank + 2) # +2 because rank is 0 based and log starts at 1
            num_evaluated += 1
    ndcg10 = ndcg_acc[10] / num_evaluated
    ndcg20 = ndcg_acc[20] / num_evaluated
    recall10 = recall_acc[10] / num_evaluated
    recall20 = recall_acc[20] / num_evaluated
    return ndcg10, ndcg20, recall10, recall20