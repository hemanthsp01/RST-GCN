#!/usr/bin/env python3
"""
MAGE Step 4: Test RST-GCN  (3-class)

Classes:
    0 → human
    1 → ai_original
    2 → ai_paraphrased

Metrics: Accuracy, Macro/Micro/Weighted F1,
         3×3 Confusion Matrix, Per-class report, ROC-AUC (OvR)
"""

import pickle
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
import torch.nn.functional as F

from rst_gcn_model import RSTGCN, prepare_batch

print("=" * 80)
print("STEP 4: TEST RST-GCN  (3-class)")
print("=" * 80)

# ============================================================
# Configuration
# ============================================================

MODEL_PATH   = Path("models/mage/mage_model_best.pt")
DATA_DIR     = Path("data/mage_parsed")
RESULTS_DIR  = Path("results/mage")
TEST_SPLITS  = ["test", "test1", "test2", "test3", "test4", "test5"]
CLASS_NAMES  = ['Human', 'AI-Original', 'AI-Paraphrased']

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load Model
# ============================================================

print("\n[1/3] Loading trained model...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Device: {device}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}\n"
        "Run step3_train.py first."
    )

checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

config = checkpoint.get('config', {
    'input_dim':   768,
    'hidden_dim':  128,
    'num_classes': 3,
    'dropout':     0.5,
})

print(f"  Config      : input_dim={config['input_dim']}, "
      f"hidden_dim={config['hidden_dim']}, "
      f"num_classes={config.get('num_classes', 3)}, "
      f"dropout={config['dropout']}")
print(f"  Dataset     : {checkpoint.get('dataset', 'unknown')}")
print(f"  Val acc     : {checkpoint.get('val_acc', 'N/A'):.4f}")
print(f"  Val MacroF1 : {checkpoint.get('val_macro_f1', 'N/A')}")
print(f"  Epoch       : {checkpoint.get('epoch', 'N/A')}")

model = RSTGCN(
    input_dim=config['input_dim'],
    hidden_dim=config['hidden_dim'],
    num_classes=config.get('num_classes', 3),
    dropout=config['dropout'],
)
model = model.to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("  ✅ Model loaded")


# ============================================================
# Inference Helper
# ============================================================

def run_test(split_name):
    split_dir   = DATA_DIR / split_name
    graphs_path = split_dir / "graphs.pkl"

    if not graphs_path.exists():
        print(f"  ❌ {graphs_path} not found — skipping {split_name}")
        return None

    with open(split_dir / "graphs.pkl",          'rb') as f: graphs     = pickle.load(f)
    with open(split_dir / "node_embeddings.pkl",  'rb') as f: embeddings = pickle.load(f)
    with open(split_dir / "node_spans.pkl",       'rb') as f: spans      = pickle.load(f)
    with open(split_dir / "labels.pkl",           'rb') as f: labels     = pickle.load(f)

    print(f"  Loaded {len(graphs)} samples")
    dist = Counter(labels)
    label_names = {0: 'human', 1: 'ai_original', 2: 'ai_paraphrased'}
    for cls, cnt in sorted(dist.items()):
        print(f"    {cls} ({label_names.get(cls, '?')}): {cnt}")

    test_batch = prepare_batch(graphs, embeddings, spans, labels, device)

    all_preds  = []
    all_labels = []
    all_probs  = []   # softmax probabilities [N, 3] for ROC-AUC

    with torch.no_grad():
        for node_feat, adj, label in tqdm(
            test_batch, desc=f"  Inference [{split_name}]", leave=True
        ):
            logits = model(node_feat, adj)               # [3]
            probs  = F.softmax(logits, dim=0)            # [3]
            pred   = torch.argmax(logits).item()

            all_preds.append(pred)
            all_labels.append(label.item())
            all_probs.append(probs.cpu().numpy())

    return (
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_probs),    # [N, 3]
    )


# ============================================================
# Metrics Helper
# ============================================================

def compute_metrics(split_name, all_preds, all_labels, all_probs):
    acc      = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro',    zero_division=0)
    micro_f1 = f1_score(all_labels, all_preds, average='micro',    zero_division=0)
    wt_f1    = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    precision, recall, f1_per_class, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None,
        labels=[0, 1, 2], zero_division=0
    )

    # 3×3 confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])

    report = classification_report(
        all_labels, all_preds,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    # ROC-AUC one-vs-rest
    unique_classes = np.unique(all_labels)
    try:
        if len(unique_classes) == 3:
            roc_auc = roc_auc_score(
                all_labels, all_probs,
                multi_class='ovr', average='macro'
            )
        else:
            roc_auc = None
    except Exception:
        roc_auc = None

    # Per-class error rates
    fp_per_class = {}
    fn_per_class = {}
    for c in range(3):
        fp = int(np.sum((all_labels != c) & (all_preds == c)))
        fn = int(np.sum((all_labels == c) & (all_preds != c)))
        fp_per_class[c] = fp
        fn_per_class[c] = fn

    # ── Print ─────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"RESULTS — {split_name.upper()}")
    print(f"{'='*80}")

    dist = Counter(all_labels.tolist())
    print(f"\n  Samples : {len(all_labels)}")
    label_names = {0: 'human', 1: 'ai_original', 2: 'ai_paraphrased'}
    for cls, cnt in sorted(dist.items()):
        print(f"    {cls} ({label_names.get(cls,'?')}): {cnt}")

    print(f"\n  ✦ Accuracy        : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  ✦ Macro F1        : {macro_f1:.4f}")
    print(f"  ✦ Micro F1        : {micro_f1:.4f}")
    print(f"  ✦ Weighted F1     : {wt_f1:.4f}")
    if roc_auc is not None:
        print(f"  ✦ ROC-AUC (OvR)   : {roc_auc:.4f}")

    print(f"\n  Per-class F1:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"    {name:<18}: P={precision[i]:.4f}  R={recall[i]:.4f}  F1={f1_per_class[i]:.4f}")

    print(f"\n  3×3 Confusion Matrix:")
    print(f"  {'':>18}", end="")
    for name in CLASS_NAMES:
        print(f"  {name:>16}", end="")
    print()
    for i, name in enumerate(CLASS_NAMES):
        print(f"  Actual {name:<12}", end="")
        for j in range(3):
            print(f"  {cm[i][j]:>16}", end="")
        print()

    print(f"\n  Per-Class Report:")
    for line in report.splitlines():
        print(f"    {line}")

    print(f"\n  Error Analysis (FP / FN per class):")
    for c, name in enumerate(CLASS_NAMES):
        n_actual = int(np.sum(all_labels == c))
        fp_rate  = fp_per_class[c] / max(int(np.sum(all_labels != c)), 1) * 100
        fn_rate  = fn_per_class[c] / max(n_actual, 1) * 100
        print(f"    {name:<18}: FP={fp_per_class[c]} ({fp_rate:.1f}%)  "
              f"FN={fn_per_class[c]} ({fn_rate:.1f}%)")

    print(f"\n  Interpretation:")
    if acc >= 0.90:
        print(f"    ✅ EXCELLENT  (acc ≥ 90%)")
    elif acc >= 0.80:
        print(f"    ✅ GOOD       (acc ≥ 80%)")
    elif acc >= 0.70:
        print(f"    ⚠️  MODERATE  (acc ≥ 70%)")
    else:
        print(f"    ❌ LOW        (acc < 70%)")

    return {
        'split':                  split_name,
        'num_samples':            len(all_labels),
        'label_counts':           dict(dist),
        'accuracy':               acc,
        'macro_f1':               macro_f1,
        'micro_f1':               micro_f1,
        'weighted_f1':            wt_f1,
        'f1_per_class':           f1_per_class.tolist(),
        'precision_per_class':    precision.tolist(),
        'recall_per_class':       recall.tolist(),
        'roc_auc':                roc_auc,
        'confusion_matrix':       cm.tolist(),
        'classification_report':  report,
        'fp_per_class':           fp_per_class,
        'fn_per_class':           fn_per_class,
        'predictions':            all_preds.tolist(),
        'labels':                 all_labels.tolist(),
        'probabilities':          all_probs.tolist(),
        'model_path':             str(MODEL_PATH),
        'model_config':           config,
        'class_names':            CLASS_NAMES,
    }


# ============================================================
# Run All Test Splits
# ============================================================

print(f"\n[2/3] Running inference on: {TEST_SPLITS}")

all_results = {}

for split_name in TEST_SPLITS:
    print(f"\n{'─'*60}")
    print(f"  Split: {split_name}")
    print(f"{'─'*60}")

    output = run_test(split_name)
    if output is None:
        continue

    all_preds, all_labels, all_probs = output
    metrics = compute_metrics(split_name, all_preds, all_labels, all_probs)
    all_results[split_name] = metrics


# ============================================================
# Save Results
# ============================================================

print(f"\n[3/3] Saving results...")

for split_name, metrics in all_results.items():
    out_pkl = RESULTS_DIR / f"ds_{split_name}_results.pkl"
    with open(out_pkl, 'wb') as f:
        pickle.dump(metrics, f)
    print(f"  ✅ {out_pkl}")

report_path = RESULTS_DIR / "ds_test_report.txt"
with open(report_path, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("RST-GCN TEST RESULTS — DS  (3-class)\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Model : {MODEL_PATH}\n")
    f.write(f"Config: input_dim={config['input_dim']}, "
            f"hidden_dim={config['hidden_dim']}, "
            f"num_classes={config.get('num_classes',3)}, "
            f"dropout={config['dropout']}\n")
    f.write(f"Val acc at save: {checkpoint.get('val_acc','N/A'):.4f}\n\n")

    # Summary table
    f.write("SUMMARY TABLE\n")
    f.write("-" * 75 + "\n")
    f.write(f"{'Split':<10} {'Samples':>8} {'Accuracy':>10} "
            f"{'Macro F1':>10} {'Weighted F1':>12} {'ROC-AUC':>10}\n")
    f.write("-" * 75 + "\n")
    for sn, m in all_results.items():
        roc_str = f"{m['roc_auc']:.4f}" if m['roc_auc'] is not None else "   N/A"
        f.write(f"{sn:<10} {m['num_samples']:>8} {m['accuracy']:>10.4f} "
                f"{m['macro_f1']:>10.4f} {m['weighted_f1']:>12.4f} {roc_str:>10}\n")
    f.write("-" * 75 + "\n\n")

    # Per-split detail
    for sn, m in all_results.items():
        f.write("=" * 80 + "\n")
        f.write(f"{sn.upper()} DETAIL\n")
        f.write("=" * 80 + "\n")
        f.write(f"  Samples     : {m['num_samples']}\n")
        f.write(f"  Accuracy    : {m['accuracy']:.4f}\n")
        f.write(f"  Macro F1    : {m['macro_f1']:.4f}\n")
        f.write(f"  Micro F1    : {m['micro_f1']:.4f}\n")
        f.write(f"  Weighted F1 : {m['weighted_f1']:.4f}\n")
        if m['roc_auc'] is not None:
            f.write(f"  ROC-AUC     : {m['roc_auc']:.4f}\n")
        f.write(f"\n  Per-class F1:\n")
        for i, name in enumerate(CLASS_NAMES):
            f.write(f"    {name:<18}: P={m['precision_per_class'][i]:.4f}  "
                    f"R={m['recall_per_class'][i]:.4f}  "
                    f"F1={m['f1_per_class'][i]:.4f}\n")
        f.write(f"\n  3x3 Confusion Matrix:\n")
        cm = m['confusion_matrix']
        f.write(f"  {'':>18}")
        for name in CLASS_NAMES:
            f.write(f"  {name:>16}")
        f.write("\n")
        for i, name in enumerate(CLASS_NAMES):
            f.write(f"  Actual {name:<12}")
            for j in range(3):
                f.write(f"  {cm[i][j]:>16}")
            f.write("\n")
        f.write(f"\n  Per-Class Report:\n{m['classification_report']}\n\n")

print(f"  ✅ {report_path}")


# ============================================================
# Final Summary Table
# ============================================================

print(f"\n{'='*80}")
print(f"FINAL SUMMARY — ALL TEST SPLITS  (3-class)")
print(f"{'='*80}")
print(f"{'Split':<10} {'Samples':>8} {'Accuracy':>10} "
      f"{'Macro F1':>10} {'Weighted F1':>12} {'ROC-AUC':>10}")
print(f"{'─'*70}")
for sn, m in all_results.items():
    roc_str = f"{m['roc_auc']:.4f}" if m['roc_auc'] is not None else "   N/A"
    print(f"{sn:<10} {m['num_samples']:>8} {m['accuracy']:>10.4f} "
          f"{m['macro_f1']:>10.4f} {m['weighted_f1']:>12.4f} {roc_str:>10}")
print(f"{'─'*70}")

print(f"\n📁 Results saved in: {RESULTS_DIR.absolute()}/")
print(f"   ds_test_results.pkl")
print(f"   ds_test_report.txt")
print("=" * 80)
