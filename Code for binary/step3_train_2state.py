#!/usr/bin/env python3
"""
MAGE Step 3: Train RST-GCN  (3-class)

Classes:
    0 → human
    1 → ai_original
    2 → ai_paraphrased

Loss   : CrossEntropyLoss  (replaces BCELoss)
Predict: argmax(logits)    (replaces threshold 0.5)
"""

import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from rst_gcn_model import RSTGCN, prepare_batch

print("=" * 80)
print("STEP 3: TRAIN RST-GCN  (3-class)")
print("=" * 80)

# ============================================================
# Configuration
# ============================================================

CONFIG = {
    'input_dim':            768,
    'hidden_dim':           128,
    'num_classes':          3,
    'dropout':              0.5,
    'learning_rate':        0.001,
    'epochs':               30,
    'batch_size':           1,
    'save_every':           5,
    'early_stop_patience':  10,
}

print(f"\n⚙️  Configuration:")
for key, value in CONFIG.items():
    print(f"   {key}: {value}")


# ============================================================
# Load Data
# ============================================================

print("\n[1/4] Loading data...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Device: {device}")

data_dir = Path("data/mage_parsed")

def load_split(split_name):
    d = data_dir / split_name
    graphs     = pickle.load(open(d / "graphs.pkl",          'rb'))
    embeddings = pickle.load(open(d / "node_embeddings.pkl", 'rb'))
    spans      = pickle.load(open(d / "node_spans.pkl",      'rb'))
    labels     = pickle.load(open(d / "labels.pkl",          'rb'))
    return graphs, embeddings, spans, labels

train_graphs, train_embeddings, train_spans, train_labels = load_split("train")
val_graphs,   val_embeddings,   val_spans,   val_labels   = load_split("val")

# Verify label distribution
from collections import Counter
label_names = {0: 'human', 1: 'ai_original', 2: 'ai_paraphrased'}

print(f"\n  Train: {len(train_graphs)} samples")
for cls, cnt in sorted(Counter(train_labels).items()):
    print(f"    {cls} ({label_names[cls]}): {cnt}")

print(f"\n  Val:   {len(val_graphs)} samples")
for cls, cnt in sorted(Counter(val_labels).items()):
    print(f"    {cls} ({label_names[cls]}): {cnt}")


# ============================================================
# Initialize Model
# ============================================================

print("\n[2/4] Initializing model...")

model = RSTGCN(
    input_dim=CONFIG['input_dim'],
    hidden_dim=CONFIG['hidden_dim'],
    num_classes=CONFIG['num_classes'],
    dropout=CONFIG['dropout'],
)
model = model.to(device)

print(f"  ✅ Model created")
print(f"     Parameters: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.CrossEntropyLoss()                           # 3-class loss
optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])


# ============================================================
# Training Loop
# ============================================================

print("\n[3/4] Training...")

if torch.cuda.is_available():
    torch.cuda.empty_cache()

best_val_acc     = 0.0
best_epoch       = 0
patience_counter = 0
train_losses     = []
val_accs         = []
# initialise so final-save never crashes if early-stop fires on epoch 1
train_acc        = 0.0
val_macro_f1     = 0.0

model_dir      = Path("models/mage")
checkpoint_dir = model_dir / "checkpoints"
model_dir.mkdir(parents=True, exist_ok=True)
checkpoint_dir.mkdir(exist_ok=True)

print(f"  Models saved to: {model_dir.absolute()}/")

for epoch in range(CONFIG['epochs']):

    # ── Train ─────────────────────────────────────────────────
    model.train()
    epoch_loss = 0.0
    correct    = 0
    total      = 0

    chunk_size = 2000
    num_chunks = (len(train_graphs) + chunk_size - 1) // chunk_size

    for chunk_idx in range(num_chunks):
        s = chunk_idx * chunk_size
        e = min(s + chunk_size, len(train_graphs))

        train_batch = prepare_batch(
            train_graphs[s:e],
            train_embeddings[s:e],
            train_spans[s:e],
            train_labels[s:e],
            device,
        )

        for node_feat, adj, label in tqdm(
            train_batch,
            desc=f"  Ep {epoch+1}/{CONFIG['epochs']} [Train] chunk {chunk_idx+1}/{num_chunks}",
            leave=False,
        ):
            optimizer.zero_grad()

            logits = model(node_feat, adj)              # [num_classes]
            loss   = criterion(
                logits.unsqueeze(0),                    # [1, num_classes]
                label.unsqueeze(0),                     # [1]  LongTensor
            )
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pred        = torch.argmax(logits).item()   # argmax replaces threshold
            correct    += int(pred == label.item())
            total      += 1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    avg_loss  = epoch_loss / max(total, 1)
    train_acc = correct    / max(total, 1)
    train_losses.append(avg_loss)

    # ── Validate ──────────────────────────────────────────────
    model.eval()
    val_correct = 0
    val_total   = 0
    val_preds   = []
    val_true    = []

    val_chunk_size = 500
    num_val_chunks = (len(val_graphs) + val_chunk_size - 1) // val_chunk_size

    with torch.no_grad():
        for chunk_idx in range(num_val_chunks):
            s = chunk_idx * val_chunk_size
            e = min(s + val_chunk_size, len(val_graphs))

            val_batch = prepare_batch(
                val_graphs[s:e],
                val_embeddings[s:e],
                val_spans[s:e],
                val_labels[s:e],
                device,
            )

            for node_feat, adj, label in tqdm(
                val_batch,
                desc=f"  Ep {epoch+1}/{CONFIG['epochs']} [Val] chunk {chunk_idx+1}/{num_val_chunks}",
                leave=False,
            ):
                logits = model(node_feat, adj)
                pred   = torch.argmax(logits).item()
                val_preds.append(pred)
                val_true.append(label.item())
                val_correct += int(pred == label.item())
                val_total   += 1

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    val_acc      = val_correct / max(val_total, 1)
    val_macro_f1 = f1_score(val_true, val_preds, average='macro', zero_division=0)
    val_accs.append(val_acc)

    print(
        f"  Epoch {epoch+1:>3}/{CONFIG['epochs']}  "
        f"Loss={avg_loss:.4f}  "
        f"Train={train_acc:.4f}  "
        f"Val Acc={val_acc:.4f}  "
        f"Val MacroF1={val_macro_f1:.4f}"
    )

    # ── Save best ─────────────────────────────────────────────
    if val_acc > best_val_acc:
        best_val_acc     = val_acc
        best_epoch       = epoch
        patience_counter = 0

        torch.save({
            'epoch':                epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_acc':            train_acc,
            'val_acc':              val_acc,
            'val_macro_f1':         val_macro_f1,
            'config':               CONFIG,
            'dataset':              'ds',
            'num_classes':          CONFIG['num_classes'],
        }, model_dir / "mage_model_best.pt")

        print(f"    ✅ Best model saved  (val_acc={val_acc:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= CONFIG['early_stop_patience']:
            print(f"\n  Early stopping at epoch {epoch+1}")
            break

    # ── Periodic checkpoint ───────────────────────────────────
    if (epoch + 1) % CONFIG['save_every'] == 0:
        torch.save({
            'epoch':                epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_acc':            train_acc,
            'val_acc':              val_acc,
            'val_macro_f1':         val_macro_f1,
            'config':               CONFIG,
            'dataset':              'ds',
            'num_classes':          CONFIG['num_classes'],
        }, checkpoint_dir / f"mage_model_epoch{epoch+1}.pt")

# ── Final model ───────────────────────────────────────────────
torch.save({
    'epoch':                epoch,
    'model_state_dict':     model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_acc':            train_acc,
    'val_acc':              val_accs[-1],
    'val_macro_f1':         val_macro_f1,
    'config':               CONFIG,
    'dataset':              'ds',
    'num_classes':          CONFIG['num_classes'],
}, model_dir / "mage_model_final.pt")


# ============================================================
# Summary
# ============================================================

print("\n[4/4] Training summary...")
print(f"\n  Best val accuracy  : {best_val_acc:.4f}  (epoch {best_epoch+1})")
print(f"  Final val accuracy : {val_accs[-1]:.4f}")

if val_accs[-1] < best_val_acc:
    print(f"  ⚠️  Overfitting detected (best={best_val_acc:.4f}, final={val_accs[-1]:.4f})")
else:
    print(f"  ✅ No overfitting detected")

print("\n" + "=" * 80)
print("✅ TRAINING COMPLETE  (3-class)")
print("=" * 80)
print(f"\n📁 Models saved in: {model_dir}/")
print(f"   mage_model_best.pt        (best val_acc={best_val_acc:.4f})")
print(f"   mage_model_final.pt       (final epoch)")
print(f"   checkpoints/mage_model_epoch*.pt")
print(f"\n🚀 Next step: python step4_test.py")
print("=" * 80)
