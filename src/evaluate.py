import numpy as np
import torch
from interface import TOKEN_PADDING, MAX_LENGTH


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _prepare_sequence(seq_list):
    """
    Convert a raw Python list of item IDs into a left-padded LongTensor.

    Steps
    -----
    1. Truncate to the last MAX_LENGTH items (keep most-recent context).
    2. Left-pad with TOKEN_PADDING (0) so the tensor is exactly MAX_LENGTH long.
    3. Add a batch dimension → shape [1, MAX_LENGTH], ready for model.predict().

    Args:
        seq_list : list[int]  — chronological item IDs (oldest first)

    Returns:
        torch.LongTensor of shape [1, MAX_LENGTH]
    """
    seq = seq_list[-MAX_LENGTH:]            # truncate: keep last MAX_LENGTH items
    pad_len = MAX_LENGTH - len(seq)
    seq = [TOKEN_PADDING] * pad_len + seq   # left-pad with 0
    return torch.LongTensor(seq).unsqueeze(0)   # [1, MAX_LENGTH]


def _rank_of_target(scores, target_idx):
    """
    Compute the 0-indexed rank of the target item in the full item score list.

    In full ranking, scores cover ALL items (indices 0 to num_items-1,
    corresponding to item IDs 1 to num_items). The target item sits at
    target_idx = target_id - 1.

    Rank = number of items that scored strictly higher than the target.
    Rank 0 means the target is the top-1 prediction.

    Args:
        scores     : 1-D FloatTensor of length num_items
        target_idx : int — index of the target item in the scores tensor

    Returns:
        int — 0-indexed rank of the target
    """
    target_score = scores[target_idx]
    rank = int((scores > target_score).sum().item())
    return rank


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate(model, data, num_items, K=[10, 20], device=None):
    """
    Evaluate SASRec using FULL RANKING — scores all items, not just 101
    candidates. This is the protocol specified in the assignment instructions.

    Protocol
    --------
    - For each user, use the sequence prefix to produce a score for every
      item in the vocabulary (IDs 1 … num_items).
    - Rank the held-out target item among all num_items items.
    - Items in the user's training history are NOT excluded from ranking
      (standard full-ranking protocol for sequential recommendation).
    - Report Recall@K and NDCG@K for K ∈ {10, 20}.

    NDCG formula (matches reference repo kang205/SASRec):
        NDCG@K = 1 / log2(rank + 2)  if rank < K,  else 0
    where rank is 0-indexed (rank 0 → target is the top-1 prediction).

    Data format expected
    --------------------
    data : dict  { user_id : (seq_list, target_item) }
        seq_list    — list[int], the sequence prefix used to predict target_item.
                      For val  : s[:-2]  (training sequence)
                      For test : s[:-1]  (training + val item)
        target_item — int, the held-out next item (1-indexed item ID).

    Args:
        model     : SASRec — must already be in eval mode before calling.
        data      : dict as described above.
        num_items : int — total vocabulary size (highest valid item ID).
        K         : list[int] — cut-off values, default [10, 20].
        device    : torch.device or None — inferred from model parameters if None.

    Returns:
        (ndcg10, ndcg20, recall10, recall20) — floats, averaged over all users.
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    # All item IDs: 1 … num_items, mapped to tensor indices 0 … num_items-1
    all_items = torch.arange(1, num_items + 1, device=device)   # [num_items]

    # ------------------------------------------------------------------ #
    # Metric accumulators                                                  #
    # ------------------------------------------------------------------ #
    ndcg_acc   = {k: 0.0 for k in K}
    recall_acc = {k: 0.0 for k in K}
    num_evaluated = 0

    model.eval()
    with torch.no_grad():
        for u, (seq_list, target) in data.items():

            # Skip users with empty training sequences
            if len(seq_list) == 0:
                continue

            # ----------------------------------------------------------
            # Prepare input sequence tensor: [1, MAX_LENGTH]
            # ----------------------------------------------------------
            seq_tensor = _prepare_sequence(seq_list).to(device)

            # ----------------------------------------------------------
            # Score ALL items in one forward pass.
            # model.predict() returns shape [1, num_items].
            # Squeeze to [num_items].
            # ----------------------------------------------------------
            scores = model.predict(seq_tensor, all_items)   # [1, num_items]
            scores = scores.squeeze(0)                       # [num_items]

            # ----------------------------------------------------------
            # Target item ID is 1-indexed; tensor index is (target - 1).
            # ----------------------------------------------------------
            target_idx = target - 1
            rank = _rank_of_target(scores, target_idx)      # 0-indexed

            # ----------------------------------------------------------
            # Accumulate Recall@K and NDCG@K
            # ----------------------------------------------------------
            for k in K:
                if rank < k:
                    recall_acc[k] += 1.0
                    # NDCG formula: 1/log2(rank+2); rank=0 → 1/log2(2) = 1.0
                    ndcg_acc[k]   += 1.0 / np.log2(rank + 2)

            num_evaluated += 1

    # ------------------------------------------------------------------ #
    # Average over all evaluated users                                     #
    # ------------------------------------------------------------------ #
    ndcg10   = ndcg_acc[10]   / num_evaluated
    ndcg20   = ndcg_acc[20]   / num_evaluated
    recall10 = recall_acc[10] / num_evaluated
    recall20 = recall_acc[20] / num_evaluated

    return ndcg10, ndcg20, recall10, recall20