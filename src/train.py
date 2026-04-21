from dataset import get_loader
from dataset import SASRecDataset
from torch.utils.data import DataLoader
import pickle
import torch
import os
from torch.optim import Adam
from model import SASRec
from evaluate import evaluate #to be updated according to evaluate implementation

def train_model(model, train_loader, val_data, num_items, epochs=5, 
                    lr=0.001, patience=5, checkpoint_path = "best_model.pt", device = None)-> dict:
    if device is None:
        device = (
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )
    print(f"training on: {device}")
    model = model.to(device)  # move params before building optimizer
    optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history = {"train_loss": [], "val_ndcg10": []}  # returned for plotting
    best_ndcg10  = 0.0 
    epochs_no_improve = 0
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0 
        for seq, pos, neg in train_loader:
            seq = seq.to(device)
            pos = pos.to(device)
            neg = neg.to(device)
            optimizer.zero_grad()
            h = model(seq)                     
            pos_emb = model.item_emb(pos)       
            neg_emb = model.item_emb(neg)       

            pos_logits = (h*pos_emb).sum(-1)  
            neg_logits = (h*neg_emb).sum(-1)  
            istarget = (pos != 0).float()
            loss = -torch.log(torch.sigmoid(pos_logits) + 1e-8)* istarget
            loss += -torch.log(1 - torch.sigmoid(neg_logits) + 1e-8)* istarget
            loss = loss.sum() / istarget.sum()
            running_loss += loss.item()
            loss.backward() 
            optimizer.step()
        
        scheduler.step()
        avg_train_loss = running_loss / len(train_loader)
        
        model.eval()
        with torch.no_grad():
            ndcg10, ndcg20, recall10, recall20 = evaluate(model, val_data, num_items)
        
        history["train_loss"].append(avg_train_loss)
        history["val_ndcg10"].append(ndcg10)
        
        print(f"Epoch {epoch:02d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val NDCG@10: {ndcg10:.4f}")
        
        # after implementing evaluate change >= to >
        if ndcg10 >= best_ndcg10:
            best_ndcg10 = ndcg10
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"New best — checkpoint saved")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {patience} epochs")
                break
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        torch.save(model.state_dict(), checkpoint_path)
        print("No checkpoint found — saving current model state")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Training complete. Best val NDCG@10: {best_ndcg10:.4f}")
    return history

if __name__ == "__main__":
    with open("../data/processed/data.pkl", "rb") as f:
        data = pickle.load(f)

    dataset = SASRecDataset(data["train"], data["num_items"])
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = SASRec(
        num_items=data["num_items"],
        hidden_size=50,
        head_nums=1,
        block_nums=1,
        maxlen=50,
        dropout_rate=0.2
    )

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_data=data["val"],
        num_items=data["num_items"]
    )
