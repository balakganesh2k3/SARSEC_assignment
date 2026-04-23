import random
import numpy as np
import torch
from interface import TOKEN_PADDING, MAX_LENGTH, NEG_SAMPLE_SIZE
from sampler import negative_sample


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
    seq = seq_list[-MAX_LENGTH:]                    # truncate: keep last MAX_LENGTH items
    pad_len = MAX_LENGTH - len(seq)
    seq = [TOKEN_PADDING] * pad_len + seq           # left-pad with 0
    return torch.LongTensor(seq).unsqueeze(0)       # [1, MAX_LENGTH]


def _rank_of_target(scores):
    """
    Compute the 0-indexed rank of the target item among all candidates.

    Convention: the target item is always placed at index 0 of the candidate
    list (and therefore at scores[0]).  Rank = number of candidates that
    received a strictly higher score than the target.

    Args:
        scores : 1-D FloatTensor of length (1 + NEG_SAMPLE_SIZE)

    Returns:
        int — 0-indexed rank of the target (0 = best possible)
    """
    target_score = scores[0]
    rank = int((scores > target_score).sum().item())
    return rank


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate(model, data, num_items, K=[10, 20], max_users=10_000, device=None):
    """
    Evaluate SASRec with the standard 1-positive + 100-negative protocol.

    Protocol (matches reference repo kang205/SASRec and PRD §4):
    - For each user, score 1 held-out target item together with NEG_SAMPLE_SIZE
      (= 100) randomly sampled negative items that are NOT in the user's observed
      history for this split.
    - Rank the target among all 101 candidates.
    - Report Recall@K and NDCG@K for K ∈ {10, 20}.

    NDCG formula (PRD-specified, matches reference repo):
        NDCG@K = 1 / log2(rank + 2)  if rank < K,  else 0
    where rank is 0-indexed (rank 0 → target is the top-1 prediction).

    Data format expected
    --------------------
    data : dict  { user_id : (seq_list, target_item) }
        seq_list    — list[int], the sequence prefix used to predict target_item.
                      For val  : s[:-2]  (train sequence)
                      For test : s[:-1]  (train + val item)
        target_item — int, the held-out next item.

    Args:
        model     : SASRec — must already be in eval mode before calling.
        data      : dict as described above.
        num_items : int — total vocabulary size (highest valid item ID).
        K         : list[int] — cut-off values, default [10, 20].
        max_users : int — cap on evaluated users for speed; if len(data) > max_users,
                    a random subset of max_users is drawn (matches reference repo).
        device    : torch.device or None — inferred from model parameters if None.

    Returns:
        (ndcg10, ndcg20, recall10, recall20) — floats, averaged over evaluated users.
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            # Model has no registered parameters (e.g. a test stub) — fall back to CPU
            device = torch.device("cpu")

    # ------------------------------------------------------------------ #
    # 1.  User sampling                                                    #
    # ------------------------------------------------------------------ #
    users = list(data.keys())
    if len(users) > max_users:
        users = random.sample(users, max_users)

    # ------------------------------------------------------------------ #
    # 2.  Metric accumulators                                              #
    # ------------------------------------------------------------------ #
    ndcg_acc   = {k: 0.0 for k in K}
    recall_acc = {k: 0.0 for k in K}
    num_evaluated = 0

    model.eval()
    with torch.no_grad():
        for u in users:
            seq_list, target = data[u]

            # ----------------------------------------------------------
            # 2a.  Build exclude set for negative sampling.
            #      Excludes every item the user has already seen in this
            #      split's sequence prefix, plus the target itself.
            #      This prevents any known positive from appearing as a
            #      negative candidate.
            # ----------------------------------------------------------
            exclude = set(seq_list) | {target}

            # ----------------------------------------------------------
            # 2b.  Sample NEG_SAMPLE_SIZE distinct negatives.
            #      Each sampled item is immediately added to exclude so
            #      the same item cannot be drawn twice.
            # ----------------------------------------------------------
            negatives = []
            for _ in range(NEG_SAMPLE_SIZE):
                neg = negative_sample(exclude, num_items)
                negatives.append(neg)
                exclude.add(neg)    # deduplicate within this user's sample

            # ----------------------------------------------------------
            # 2c.  Build candidate list: target at index 0, negatives after.
            #      Keeping target at a fixed index makes rank computation
            #      straightforward and deterministic.
            # ----------------------------------------------------------
            candidates = [target] + negatives      # length = 101

            # ----------------------------------------------------------
            # 2d.  Prepare tensors and run model.predict().
            # ----------------------------------------------------------
            seq_tensor   = _prepare_sequence(seq_list).to(device)      # [1, MAX_LENGTH]
            item_tensor  = torch.LongTensor(candidates).to(device)     # [101]

            scores = model.predict(seq_tensor, item_tensor)            # [1, 101]
            scores = scores.squeeze(0)                                  # [101]

            # ----------------------------------------------------------
            # 2e.  Rank target and accumulate metrics.
            # ----------------------------------------------------------
            rank = _rank_of_target(scores)          # 0-indexed

            for k in K:
                if rank < k:
                    recall_acc[k] += 1.0
                    # PRD formula: 1/log2(rank+2)   (rank=0 → 1/log2(2) = 1.0)
                    ndcg_acc[k]   += 1.0 / np.log2(rank + 2)

            num_evaluated += 1

    # ------------------------------------------------------------------ #
    # 3.  Average and return                                               #
    # ------------------------------------------------------------------ #
    ndcg10   = ndcg_acc[10]   / num_evaluated
    ndcg20   = ndcg_acc[20]   / num_evaluated
    recall10 = recall_acc[10] / num_evaluated
    recall20 = recall_acc[20] / num_evaluated

    return ndcg10, ndcg20, recall10, recall20