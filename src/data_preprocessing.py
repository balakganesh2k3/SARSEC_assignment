# We are performing Data preprocessing for SASRec on MovieLens-1M
import pandas as pd
from collections import defaultdict
import pickle
import os

def loadRatings(path):
    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["user", "item", "rating", "timestamp"]
    )
    return df

def preprocess(
    path="../data/ratings.dat",
    interactions=5,
    path1="../data/processed/data.pkl"):
    print("Loading ratings...")
    df = loadRatings(path)
    print("Filtering rating >= 4")
    # ratings >= 4 are positive interactions
    df = df[df["rating"] >= 4]
    print("Sorting by user and timestamp...")
    df = df.sort_values(["user", "timestamp"])

    # Constructing chronological sequences of items per user
    seq = defaultdict(list)
    for row in df.itertuples():
        seq[row.user].append(row.item)

    # Filtering users
    seq = {
        u: s for u, s in seq.items()
        if len(s) >= interactions
    }
    # sorting items 
    Items = sorted(set(item for s in seq.values() for item in s))
    item2id = {item: idx + 1 for idx, item in enumerate(Items)}

    # Re-mapping of raw movie IDs to  integers 
    for u in seq:
        seq[u] = [item2id[i] for i in seq[u]]
    train, val, test = {}, {}, {}
    for u, s in seq.items():
        # Leave-one-out split
        train[u] = s[:-2]
        val[u] =(s[:-2], s[-2])   
        test[u] = (s[:-1], s[-1]) 
         # Storing processed data  
        processed = {
        "train":    train,
        "val":      val,
        "test":     test,
        "num_items": len(item2id),
        "item2id":  item2id,
    }

    os.makedirs(os.path.dirname(path1), exist_ok=True)

    with open(path1, "wb") as f:
        pickle.dump(processed, f)

    print("Preprocessing is completed")
    print(f" Users: {len(seq)}")
    print(f" Unique items : {len(item2id)}")
    print(f" Saved to: {path1}")


if __name__ == "__main__":
    preprocess()
