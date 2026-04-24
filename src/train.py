import os
import pickle
import torch
import torch.backends.cudnn as cudnn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast  # automatic mixed precision
from tqdm import tqdm

from dataset import SASRecDataset
from model import SASRec
from evaluate import evaluate
from interface import MAX_LENGTH


def train_model(model, train_loader, val_data, num_items, epochs=200,
                lr=0.001, patience=5, checkpoint_path="checkpoints/best_model.pt",
                device=None) -> dict:

    # ------------------------------------------------------------------ #
    # Device setup                                                         #
    # ------------------------------------------------------------------ #
    if device is None:
        device = (
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else
            "cpu"
        )
    device = torch.device(device)
    print(f"Training on: {device}")

    # cuDNN auto-tuner: finds the fastest convolution algorithm for your
    # exact input sizes. Free speed-up on GPU with no code changes needed.
    if device.type == "cuda":
        cudnn.benchmark = True
        print(f"  GPU : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ------------------------------------------------------------------ #
    # Create checkpoint directory if needed                               #
    # ------------------------------------------------------------------ #
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Created checkpoint directory: {checkpoint_dir}")

    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))
    # Warmup for first 10 epochs, then cosine decay — required by PRD
    warmup_epochs = 10
    warmup   = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda ep: min((ep + 1) / warmup_epochs, 1.0)
    )
    cosine   = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )

    # GradScaler enables AMP (automatic mixed precision).
    # On GPU : runs the forward pass in float16 — roughly 2x faster and
    #          uses half the VRAM compared to float32.
    # On CPU / MPS: scaler is a no-op, so the same code runs everywhere.
    scaler = GradScaler("cuda" if device.type == "cuda" else "cpu")

    # ------------------------------------------------------------------ #
    # Configuration summary                                                #
    # ------------------------------------------------------------------ #
    total_params    = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "=" * 60)
    print("  TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"  Device          : {device}" +
          (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    if device.type == "cuda":
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM            : {vram:.1f} GB")
    amp_status = "Enabled (float16)" if device.type == "cuda" else "Disabled (CPU/MPS)"
    print(f"  AMP (mixed prec): {amp_status}")
    print("-" * 60)
    print("  MODEL")
    print(f"  Architecture    : SASRec")
    print(f"  Num items       : {model.num_items:,}")
    print(f"  Hidden size     : {model.hidden_size}")
    print(f"  Attention heads : {model.head_nums}")
    print(f"  Transformer blks: {model.block_nums}")
    print(f"  Max seq length  : {model.maxlen}")
    print(f"  Dropout rate    : {model.dropout_rate}")
    print(f"  Total params    : {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print("-" * 60)
    print("  TRAINING")
    print(f"  Epochs          : {epochs}")
    print(f"  Learning rate   : {lr}")
    print(f"  Optimizer       : Adam (β1=0.9, β2=0.98)")
    print(f"  Scheduler       : Warmup({warmup_epochs} ep) + CosineAnnealingLR")
    print(f"  Batch size      : {train_loader.batch_size}")
    print(f"  Train batches   : {len(train_loader)}")
    print(f"  Early stopping  : patience={patience}")
    print(f"  Checkpoint path : {checkpoint_path}")
    print("=" * 60 + "\n")

    history = {"train_loss": [], "val_ndcg10": []}
    best_ndcg10 = 0.0
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{epochs}", leave=False)
        for seq, pos, neg in progress:
            # non_blocking=True overlaps the CPU→GPU transfer with ongoing
            # GPU compute. Requires pin_memory=True in the DataLoader.
            seq = seq.to(device, non_blocking=True)
            pos = pos.to(device, non_blocking=True)
            neg = neg.to(device, non_blocking=True)

            # set_to_none=True is faster than filling gradients with zeros
            optimizer.zero_grad(set_to_none=True)

            # autocast runs eligible ops in float16 on GPU, falling back
            # to float32 for numerically sensitive ones automatically.
            with autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
                h = model(seq)                        # [B, L, d]
                pos_emb = model.item_emb(pos)         # [B, L, d]
                neg_emb = model.item_emb(neg)         # [B, L, d]

                pos_logits = (h * pos_emb).sum(-1)    # [B, L]
                neg_logits = (h * neg_emb).sum(-1)    # [B, L]

                istarget = (pos != 0).float()         # mask out padding positions
                loss  = -torch.log(torch.sigmoid(pos_logits) + 1e-8) * istarget
                loss += -torch.log(1 - torch.sigmoid(neg_logits) + 1e-8) * istarget
                loss  = loss.sum() / istarget.sum()

            # scaler.scale() replaces loss.backward() under AMP
            scaler.scale(loss).backward()
            # Unscales gradients before the optimizer step so the learning
            # rate remains consistent regardless of the AMP loss scale factor
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_train_loss = running_loss / len(train_loader)

        # Validation — uses the real evaluate.py (no fake metrics)
        model.eval()
        with torch.no_grad():
            ndcg10, ndcg20, recall10, recall20 = evaluate(
                model, val_data, num_items, device=device
            )

        history["train_loss"].append(avg_train_loss)
        history["val_ndcg10"].append(ndcg10)

        print(f"Epoch {epoch:03d}/{epochs} | "
              f"Loss: {avg_train_loss:.4f} | "
              f"Val NDCG@10: {ndcg10:.4f} | "
              f"Val Recall@10: {recall10:.4f}")

        # Strict > so a plateau correctly counts against patience counter
        if ndcg10 > best_ndcg10:
            best_ndcg10 = ndcg10
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> New best — checkpoint saved ({checkpoint_path})")
        else:
            if epoch > warmup_epochs: 
              epochs_no_improve += 1
              if epochs_no_improve >= patience:
                 print(f"Early stopping triggered after {patience} epochs without improvement.")
                 break

    # Restore best weights before returning
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Best checkpoint restored from {checkpoint_path}")
    else:
        torch.save(model.state_dict(), checkpoint_path)
        print("No checkpoint found — saving current model state.")

    print(f"Training complete. Best val NDCG@10: {best_ndcg10:.4f}")
    return history


if __name__ == "__main__":
    torch.manual_seed(42)

    # Use all CPU cores for faster tensor ops
    torch.set_num_threads(os.cpu_count())
    torch.set_num_interop_threads(os.cpu_count())

    with open("../data/processed/data.pkl", "rb") as f:
        data = pickle.load(f)

    # num_workers=0 on Windows — multiprocessing workers cause massive
    # overhead on Windows and are slower than single-process loading.
    num_workers = 0 if os.name == "nt" else min(4, os.cpu_count() or 2)

    dataset = SASRecDataset(data["train"], data["num_items"])
    train_loader = DataLoader(
        dataset,
        batch_size=512,                              # larger batch = fewer iterations
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),        # page-locked RAM for fast GPU transfers
        persistent_workers=(num_workers > 0),        # keep workers alive between epochs
    )

    model = SASRec(
        num_items=data["num_items"],
        hidden_size=50,
        head_nums=1,
        block_nums=2,       # PRD default for ml-1m
        maxlen=MAX_LENGTH,  # 200, defined in interface.py
        dropout_rate=0.2    # PRD default for ml-1m
    )

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_data=data["val"],
        num_items=data["num_items"],
    )

    # Final test evaluation — copy these four numbers into the report
    model.eval()
    with torch.no_grad():
        ndcg10, ndcg20, recall10, recall20 = evaluate(
            model, data["test"], data["num_items"]
        )

    print(f"\nTest Results:")
    print(f"  NDCG@10:   {ndcg10:.4f}")
    print(f"  NDCG@20:   {ndcg20:.4f}")
    print(f"  Recall@10: {recall10:.4f}")
    print(f"  Recall@20: {recall20:.4f}")