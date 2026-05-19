#!/usr/bin/env python3
"""
MAGE Step 2: Create EDU Embeddings
Generates RoBERTa node features for RST graphs
Processes train / val / test splits

No label-related changes needed here —
embeddings are purely text-based and class-agnostic.
"""

import pickle
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

print("=" * 80)
print("STEP 2: CREATE EMBEDDINGS")
print("=" * 80)

# ============================================================
# Load RoBERTa
# ============================================================

print("\n[1/3] Loading RoBERTa model...")

device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Device: {device}")

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model     = AutoModel.from_pretrained("roberta-base")
model     = model.to(device)
model.eval()

print("  ✅ RoBERTa loaded")


# ============================================================
# Embedding Functions
# ============================================================

def get_edu_embedding(edu_text: str) -> np.ndarray:
    """Get RoBERTa [CLS] embedding for a single EDU."""
    if not edu_text or len(edu_text.strip()) == 0:
        edu_text = "[EMPTY]"

    inputs = tokenizer(
        edu_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs   = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze()

    return embedding.cpu().numpy()


def create_graph_embeddings(edus_dict: dict):
    """
    Create RoBERTa embeddings for all EDUs in one graph.

    Returns:
        embeddings : np.ndarray [num_nodes, 768]
        edu_spans  : list of span keys (same order)
    """
    if len(edus_dict) == 0:
        return np.zeros((1, 768)), ["span_1-1"]

    edu_spans  = sorted(edus_dict.keys())
    embeddings = []
    for span in edu_spans:
        emb = get_edu_embedding(edus_dict[span])
        embeddings.append(emb)

    return np.stack(embeddings), edu_spans


# ============================================================
# Configuration
# ============================================================

PARSED_DIR = Path("data/ds_parsed")
SPLITS     = ["train", "val", "test"]

print(f"\n  Input directory: {PARSED_DIR}")
print(f"  Splits         : {SPLITS}")


# ============================================================
# Process All Splits
# ============================================================

for split_name in SPLITS:
    print(f"\n{'='*80}")
    print(f"Processing: {split_name.upper()}")
    print(f"{'='*80}")

    split_dir   = PARSED_DIR / split_name
    edus_path   = split_dir  / "edus.pkl"
    graphs_path = split_dir  / "graphs.pkl"

    if not split_dir.exists():
        print(f"  ⚠️  {split_dir} not found — skipping")
        continue
    if not edus_path.exists():
        print(f"  ⚠️  edus.pkl not found in {split_dir} — skipping")
        continue

    with open(edus_path,   'rb') as f: all_edus   = pickle.load(f)
    with open(graphs_path, 'rb') as f: all_graphs = pickle.load(f)

    print(f"  ✅ Loaded {len(all_edus)} EDU dicts, {len(all_graphs)} graphs")

    # ── Create embeddings ─────────────────────────────────────
    all_embeddings = []
    all_spans      = []
    failed_count   = 0

    for i, edus_dict in enumerate(tqdm(all_edus, desc=f"  {split_name}", leave=False)):
        try:
            embeddings, spans = create_graph_embeddings(edus_dict)
            all_embeddings.append(embeddings)
            all_spans.append(spans)
        except Exception as e:
            print(f"\n  ⚠️  Failed on sample {i}: {e}")
            all_embeddings.append(np.zeros((1, 768)))
            all_spans.append(["span_1-1"])
            failed_count += 1

    if failed_count > 0:
        print(f"  ⚠️  Failed: {failed_count}/{len(all_edus)}")

    avg_nodes     = np.mean([e.shape[0] for e in all_embeddings])
    embedding_dim = all_embeddings[0].shape[1]

    print(f"  ✅ Embeddings created:")
    print(f"    Samples        : {len(all_embeddings)}")
    print(f"    Avg nodes/graph: {avg_nodes:.1f}")
    print(f"    Embedding dim  : {embedding_dim}")

    # ── Save ──────────────────────────────────────────────────
    with open(split_dir / "node_embeddings.pkl", 'wb') as f:
        pickle.dump(all_embeddings, f)
    with open(split_dir / "node_spans.pkl", 'wb') as f:
        pickle.dump(all_spans, f)

    print(f"  ✅ Saved to: {split_dir}/")
    print(f"    node_embeddings.pkl  (RoBERTa 768-dim)")
    print(f"    node_spans.pkl       (EDU span labels)")


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 80)
print("✅ EMBEDDINGS COMPLETE")
print("=" * 80)
print(f"\nEmbeddings saved in: {PARSED_DIR.absolute()}/")
for split_name in SPLITS:
    emb_path = PARSED_DIR / split_name / "node_embeddings.pkl"
    if emb_path.exists():
        print(f"  ✅ {split_name}/  →  node_embeddings.pkl, node_spans.pkl")

print("\n🚀 Next step: python step3_train.py")
print("=" * 80)