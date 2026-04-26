import matplotlib.pyplot as plt
import os

def parse_curve(log_path, target_hid=100, target_heads=2, target_blocks=2):
    train_losses = []
    val_ndcgs = []
    in_best = False
    current_hidden  = None
    current_heads = None
    current_blocks = None
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if "Transformer blks:" in line:
                current_blocks = int(line.split(":")[1].strip())
            elif "Hidden size" in line:
                current_hidden = int(line.split(":")[1].strip())
            elif "Attention heads" in line:
                current_heads = int(line.split(":")[1].strip())
                if (current_hidden == target_hid and current_heads == target_heads and current_blocks == target_blocks):
                    in_best = True
                    train_losses = []
                    val_ndcgs = []
                else:
                    in_best = False

            elif in_best and "Epoch" in line and "Loss:" in line:
                parts = line.split("|")
                loss = float(parts[1].split(":")[1].strip())
                ndcg = float(parts[2].split(":")[1].strip())
                train_losses.append(loss)
                val_ndcgs.append(ndcg)
            elif in_best and "Test results:" in line:
                in_best = False

    return train_losses, val_ndcgs


def plot_curve(train_loss, val_ndcgs, save_path="../results/sasrec_line.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = list(range(1, len(train_loss) + 1))
    fig, ax1 = plt.subplots(figsize=(10, 5))
    color_loss = "#2166AC"
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Train loss", fontsize=11, color=color_loss)
    ax1.plot(epochs, train_loss, color=color_loss, linewidth=2, marker="o", markersize=3, label="Train Loss")
    ax1.tick_params(axis="y", labelcolor=color_loss)
    ax2 = ax1.twinx()
    color_ndcg = "#D6604D"
    ax2.set_ylabel("Val NDCG@10", fontsize=11, color=color_ndcg)
    ax2.plot(epochs, val_ndcgs, color=color_ndcg, linewidth=2, marker="s", markersize=3, linestyle="--", label="Val NDCG@10")
    ax2.tick_params(axis="y", labelcolor=color_ndcg)
    best_epoch = val_ndcgs.index(max(val_ndcgs)) + 1
    ax2.axvline(x=best_epoch, color="grey", linestyle=":", linewidth=1.5, label=f"Best epoch ({best_epoch})")
    ax2.text(best_epoch + 0.5, max(val_ndcgs)* 0.95, f"best\nepoch {best_epoch}", fontsize=8, color="grey", va="top")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=True)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax1.set_axisbelow(True)
    fig.suptitle("SASRec training curve — best config (blocks=2, hidden=100, heads=2)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"line chart saved to {save_path}")


if __name__ == "__main__":
    log_path = "Training_on_cuda.txt"
    train_loss, val_ndcgs = parse_curve(log_path)
    print(f"parsed {len(train_loss)} epochs for best config")
    if train_loss:
        plot_curve(train_loss, val_ndcgs)
    else:
        print("could not find best config in log file — check target params")