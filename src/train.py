import os
import pickle
import torch
import torch.backends.cudnn as cudnn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from dataset import SASRecDataset
from model import SASrec
from evaluate import evaluate
from interface import MAX_LENGTH

def train_model(model, train_loader, val_data, num_items, epochs=200, lr=0.001, patience=5, checkpoint_path="checkpoints/best_model.pt", device=None) -> dict:
    if device is None:
        device = ("cuda" if torch.cuda.is_available() else
                  "mps"  if torch.backends.mps.is_available() else
                  "cpu")
    log_file = open("Training_on_cuda.txt", "a")
    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()
    device = torch.device(device)
    log(f"Training on: {device}")
    
    if device.type == "cuda":
        cudnn.benchmark = True
        log(f"GPU : {torch.cuda.get_device_name(0)}")
        log(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        log(f"Created checkpoint directory: {checkpoint_dir}")

    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))
    warmup_epochs = 10
    warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda ep: min((ep + 1) / warmup_epochs, 1.0))
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    scaler = GradScaler("cuda" if device.type == "cuda" else "cpu")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log("\n" + "=" * 60)
    log("Training configs")
    log("=" * 60)
    log(f"Device : {device}" + (f"({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    if device.type == "cuda":
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        log(f"VRAM : {vram:.1f} GB")
    amp_status = "enabled (float16)" if device.type == "cuda" else "disabled (CPU/MPS)"
    log(f"AMP (mixed prec) : {amp_status}")
    log("-" * 60)
    log("Model")
    log(f"Architecture : SASRec")
    log(f"Num items : {model.num_items:,}")
    log(f"Hidden size : {model.hid_size}")
    log(f"Attention heads : {model.head_nums}")
    log(f"Transformer blks : {model.block_nums}")
    log(f"Max seq length : {model.maxlen}")
    log(f"Dropout rate : {model.dropout_rate}")
    log(f"Total params : {total_params:,}")
    log(f"Trainable params: {trainable_params:,}")
    log("-" * 60)
    log("Training")
    log(f"Epochs : {epochs}")
    log(f"Learning rate : {lr}")
    log(f"Optimizer : Adam (β1=0.9, β2=0.98)")
    log(f"Scheduler : Warmup({warmup_epochs} ep) + CosineAnnealingLR")
    log(f"Batch size : {train_loader.batch_size}")
    log(f"Train batches : {len(train_loader)}")
    log(f"Early stopping : patience={patience}")
    log(f"Checkpoint path : {checkpoint_path}")
    log("=" * 60 + "\n")
    history = {"train_loss": [], "val_ndcg10": []}
    best_ndcg10 = 0.0
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{epochs}", leave=False)
        for seq, pos, neg in progress:
            seq = seq.to(device, non_blocking=True)
            pos = pos.to(device, non_blocking=True)
            neg = neg.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
                h = model(seq)                        
                pos_emb = model.item_emb(pos)         
                neg_emb = model.item_emb(neg)         
                pos_logits = (h * pos_emb).sum(-1)    
                neg_logits = (h * neg_emb).sum(-1)  
                istarget = (pos != 0).float()         
                loss  = -torch.log(torch.sigmoid(pos_logits) + 1e-8) * istarget
                loss += -torch.log(1 - torch.sigmoid(neg_logits) + 1e-8) * istarget
                loss  = loss.sum() / istarget.sum()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()
        avg_train_loss = running_loss / len(train_loader)
        model.eval()
        with torch.no_grad():
            ndcg10, ndcg20, recall10, recall20 = evaluate( model, val_data, num_items, device=device)
        history["train_loss"].append(avg_train_loss)
        history["val_ndcg10"].append(ndcg10)
        log(f"Epoch {epoch:03d}/{epochs} | "f"Loss: {avg_train_loss:.4f} | "f"Val NDCG@10: {ndcg10:.4f} | "f"Val Recall@10: {recall10:.4f}")
        if ndcg10 > best_ndcg10:
            best_ndcg10 = ndcg10
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
            log(f"New best checkpoint saved ({checkpoint_path})")
        else:
            if epoch > warmup_epochs: 
              epochs_no_improve += 1
              if epochs_no_improve >= patience:
                 log(f"Early stopping triggered after {patience} epochs without improvement.")
                 break
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        log(f"Best checkpoint restored from {checkpoint_path}")
    else:
        torch.save(model.state_dict(), checkpoint_path)
        log("No checkpoint found saving current model state.")
    log(f"Training complete best val NDCG@10: {best_ndcg10:.4f}")
    log_file.close()
    return history

if __name__ == "__main__":
    torch.manual_seed(42)
    torch.set_num_threads(os.cpu_count())
    torch.set_num_interop_threads(os.cpu_count())
    with open("../data/processed/data.pkl", "rb") as f:
        data = pickle.load(f)
    num_workers = 0 if os.name == "nt" else min(4, os.cpu_count() or 2)
    dataset = SASRecDataset(data["train"], data["num_items"])
    train_loader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=(num_workers > 0))
    model = SASrec(num_items=data["num_items"], hid_size=50, head_nums=1, block_nums=1, maxlen=MAX_LENGTH, dropout_rate=0.2)
    history = train_model(model=model, train_loader=train_loader, val_data=data["val"], num_items=data["num_items"])
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else
                          "cpu")
    model.eval()
    with torch.no_grad():
        ndcg10, ndcg20, recall10, recall20 = evaluate(model, data["test"], data["num_items"], device=device)
    print(f"\nTest result:")
    print(f"NDCG@10 : {ndcg10:.4f}")
    print(f"NDCG@20 : {ndcg20:.4f}")
    print(f"Recall@10 : {recall10:.4f}")
    print(f"Recall@20 : {recall20:.4f}")