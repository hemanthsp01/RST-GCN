#!/usr/bin/env python3
"""
MAGE Step 1: Parse with DMRST
Creates RST trees for train/val/test splits

3-class labels:
    0 → human
    1 → ai_original
    2 → ai_paraphrased
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import torch
from tqdm import tqdm
import networkx as nx
try:
    import regex as re          # preferred — richer Unicode support
except ImportError:
    import re                   # fallback to stdlib re
from transformers import AutoTokenizer, AutoModel

# Setup paths
DMRST_ROOT = Path("DMRST_Parser")
sys.path.insert(0, str(DMRST_ROOT))

from model_depth import ParsingNet

print("=" * 80)
print("STEP 1: PARSE WITH DMRST  (3-class)")
print("=" * 80)

# ============================================================
# Label Mapping  — edit source_type strings here if needed
# ============================================================

SOURCE_TYPE_TO_LABEL = {
    'human':          0,
    'ai_generated':   1,
    'ai_original':    1,   # alias
    'paraphrased':    2,
    'ai_paraphrased': 2,   # alias
}


def resolve_label(row):
    """
    Resolve integer 3-class label from a DataFrame row.
    Priority: source_type column > label column
    """
    # Try source_type first (most specific)
    if 'source_type' in row.index:
        st = str(row['source_type']).strip().lower()
        if st in SOURCE_TYPE_TO_LABEL:
            return SOURCE_TYPE_TO_LABEL[st]

    # Fall back to label column
    if 'label' in row.index:
        val = row['label']
        if isinstance(val, str):
            val = val.strip().lower()
            if val in SOURCE_TYPE_TO_LABEL:
                return SOURCE_TYPE_TO_LABEL[val]
            # binary string fallback
            if val == 'machine':
                return 1
        else:
            v = int(val)
            if v in (0, 1, 2):
                return v

    raise ValueError(
        f"Cannot resolve 3-class label from row: {dict(row)}\n"
        f"Expected source_type in {list(SOURCE_TYPE_TO_LABEL.keys())}"
    )


# ============================================================
# Graph Construction
# ============================================================

def create_graph_from_const_format(format_string: str) -> nx.DiGraph:
    """Build RST graph from parse string"""
    spans    = format_string.strip().split(" ")
    rgx_span = r"\((\d+):(.+)=(.+):(\d+),(\d+):(.+)=(.+):(\d+)\)"
    edges       = []
    nodes_types = {}

    for i, span in enumerate(spans, start=1):
        m_span = re.match(rgx_span, span)
        assert m_span is not None

        left_most_edu_index  = int(m_span.group(1))
        right_most_edu_index = int(m_span.group(8))
        left_type            = m_span.group(2)
        left_relation        = m_span.group(3)
        left_end_edu_index   = int(m_span.group(4))
        right_start_edu_index = int(m_span.group(5))
        right_type           = m_span.group(6)
        right_relation       = m_span.group(7)

        node_label       = f"span_{left_most_edu_index}-{right_most_edu_index}"
        left_node_label  = f"span_{left_most_edu_index}-{left_end_edu_index}"
        right_node_label = f"span_{right_start_edu_index}-{right_most_edu_index}"

        edges.append((left_node_label,  node_label,       "/"))
        edges.append((right_node_label, node_label,       "/"))

        if left_relation != "span":
            edges.append((left_node_label,  right_node_label, left_relation))
        if right_relation != "span":
            edges.append((right_node_label, left_node_label,  right_relation))

        if i == 1:
            nodes_types[node_label] = "root"

        nodes_types[left_node_label]  = left_type
        nodes_types[right_node_label] = right_type

    G = nx.DiGraph()
    for u, v, label in edges:
        G.add_edge(u, v, label_0=label)
    nx.set_node_attributes(G, nodes_types, "label_0")

    return G


# ============================================================
# Parse Function
# ============================================================

SUBWORD_LIMIT = 490    # hard ceiling — keeps 22 tokens as buffer for XLM-RoBERTa's 512


def safe_truncate(tokens, tokenizer, limit=SUBWORD_LIMIT):
    """
    Truncate token list so its subword ID count stays under `limit`.
    Only long texts get trimmed — short texts are never touched.

    Strategy: iteratively shed 10% of tokens until subword count is safe.
    In practice this converges in 1-2 iterations for borderline texts.
    """
    if len(tokenizer.convert_tokens_to_ids(tokens)) <= limit:
        return tokens   # already safe — no truncation needed

    while len(tokens) > 1:
        tokens = tokens[:int(len(tokens) * 0.9)]
        if len(tokenizer.convert_tokens_to_ids(tokens)) <= limit:
            break

    return tokens


def parse_texts(model, tokenizer, texts, batch_size=16):
    """Parse texts with DMRST using smart per-sample truncation.
    Each text is truncated only if its actual subword token count
    exceeds SUBWORD_LIMIT (490). Short texts are never truncated.
    """
    input_sentences = []
    truncated_count = 0

    for text in texts:
        tokens = tokenizer.tokenize(text, add_special_tokens=False)
        original_len = len(tokens)
        tokens = safe_truncate(tokens, tokenizer)
        if len(tokens) < original_len:
            truncated_count += 1
        input_sentences.append(tokens)

    if truncated_count > 0:
        print(f"  ℹ️  Truncated {truncated_count}/{len(texts)} texts "
              f"(subword count exceeded {SUBWORD_LIMIT})")

    all_tokens   = []
    all_segments = []
    all_parsed   = []

    num_batches = int(np.ceil(len(input_sentences) / batch_size))

    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="  Parsing"):
            start = i * batch_size
            end   = min((i + 1) * batch_size, len(input_sentences))
            batch = input_sentences[start:end]

            try:
                _, _, SPAN_batch, _, predict_EDU_breaks = model.TestingLoss(
                    batch,
                    input_EDU_breaks=None,
                    LabelIndex=None,
                    ParsingIndex=None,
                    GenerateTree=True,
                    use_pred_segmentation=True,
                )
                all_tokens.extend(batch)
                all_segments.extend(predict_EDU_breaks)
                all_parsed.extend(SPAN_batch)

            except Exception as e:
                print(f"\n  ⚠️  Batch {i} failed ({e}) — using fallback for {len(batch)} samples")
                # Fallback: treat each sample as single EDU, no parse
                for tok in batch:
                    all_tokens.append(tok)
                    all_segments.append([len(tok) - 1] if tok else [0])
                    all_parsed.append(["NONE"])

    return all_tokens, all_segments, all_parsed


# ============================================================
# Load DMRST Model
# ============================================================

print("\n[1/4] Loading DMRST model...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Device: {device}")

bert_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=True)
bert_model     = AutoModel.from_pretrained("xlm-roberta-base")
bert_model     = bert_model.to(device)

for param in bert_model.parameters():
    param.requires_grad = False

model      = ParsingNet(bert_model, bert_tokenizer=bert_tokenizer)
model      = model.to(device)
model_path = DMRST_ROOT / "depth_mode/Savings/multi_all_checkpoint.torchsave"
checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)
model.load_state_dict(checkpoint, strict=False)
model.eval()

print("  ✅ DMRST model loaded")


# ============================================================
# Parse All Splits
# ============================================================

data_dir   = Path("data/mage")
output_dir = Path("data/mage_parsed")
output_dir.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "val", "test"]

for split in SPLITS:
    print(f"\n{'='*80}")
    print(f"Processing: {split.upper()}")
    print(f"{'='*80}")

    csv_path = data_dir / f"{split}.csv"
    if not csv_path.exists():
        print(f"  ⚠️  {csv_path} not found — skipping")
        continue

    df = pd.read_csv(csv_path)

    # Ensure text column
    if 'text' not in df.columns and 'generation' in df.columns:
        df = df.rename(columns={'generation': 'text'})

    texts  = df['text'].tolist()

    # ── Resolve 3-class labels ────────────────────────────────
    labels = []
    errors = 0
    for _, row in df.iterrows():
        try:
            labels.append(resolve_label(row))
        except ValueError as e:
            print(f"  ⚠️  {e}")
            labels.append(-1)
            errors += 1

    if errors > 0:
        print(f"  ⚠️  {errors} rows could not be labelled — they will be excluded")
        valid_mask = [l != -1 for l in labels]
        texts  = [t for t, m in zip(texts,  valid_mask) if m]
        labels = [l for l, m in zip(labels, valid_mask) if m]

    # Label distribution
    from collections import Counter
    dist = Counter(labels)
    print(f"  Loaded {len(texts)} samples")
    print(f"  Label distribution:")
    label_names = {0: 'human', 1: 'ai_original', 2: 'ai_paraphrased'}
    for cls, count in sorted(dist.items()):
        print(f"    {cls} ({label_names.get(cls, '?')}): {count}")

    # ── Parse ─────────────────────────────────────────────────
    print(f"  Parsing with DMRST...")
    all_tokens, all_segments, all_parsed = parse_texts(
        model, bert_tokenizer, texts, batch_size=16
    )

    # ── Build graphs ──────────────────────────────────────────
    print(f"  Building RST graphs...")
    graphs   = []
    all_edus = []
    failures = 0

    for i in tqdm(range(len(texts)), desc="  Building"):
        # Extract EDUs
        edus     = {}
        last_end = 0
        for edu_i, edu_end in enumerate(all_segments[i], start=1):
            edu_text = "".join(
                all_tokens[i][last_end : edu_end + 1]
            ).replace("▁", " ")
            edus[f"span_{edu_i}-{edu_i}"] = edu_text
            last_end = edu_end + 1

        parsed_str = (
            all_parsed[i][0]
            if all_parsed[i] and len(all_parsed[i]) > 0
            else "NONE"
        )

        if parsed_str == "NONE" or not parsed_str:
            G = nx.DiGraph()
            G.add_node("span_1-1", label_0="N")
            failures += 1
        else:
            try:
                G = create_graph_from_const_format(parsed_str)
            except Exception:
                G = nx.DiGraph()
                G.add_node("span_1-1", label_0="N")
                failures += 1

        graphs.append(G)
        all_edus.append(edus)

    if failures > 0:
        print(f"  ⚠️  Parse failures: {failures}/{len(texts)} "
              f"({failures/len(texts)*100:.1f}%)")

    # ── Save ──────────────────────────────────────────────────
    split_output = output_dir / split
    split_output.mkdir(exist_ok=True)

    pickle.dump(graphs,   open(split_output / "graphs.pkl",  'wb'))
    pickle.dump(all_edus, open(split_output / "edus.pkl",    'wb'))
    pickle.dump(labels,   open(split_output / "labels.pkl",  'wb'))

    stats = {
        'num_samples':    len(graphs),
        'avg_nodes':      np.mean([G.number_of_nodes() for G in graphs]),
        'avg_edges':      np.mean([G.number_of_edges() for G in graphs]),
        'avg_edus':       np.mean([len(e) for e in all_edus]),
        'label_counts':   dict(dist),
        'parse_failures': failures,
        'num_classes':    3,
    }
    pickle.dump(stats, open(split_output / "stats.pkl", 'wb'))

    print(f"  ✅ Saved to: {split_output}/")
    print(f"    Samples  : {stats['num_samples']}")
    print(f"    Avg nodes: {stats['avg_nodes']:.1f}")
    print(f"    Avg edges: {stats['avg_edges']:.1f}")
    print(f"    Avg EDUs : {stats['avg_edus']:.1f}")
    print(f"    Labels   : {stats['label_counts']}")


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 80)
print("✅ PARSING COMPLETE  (3-class)")
print("=" * 80)
print(f"\nParsed data saved in: {output_dir.absolute()}/")
print(f"\nSplits processed:")
for split in SPLITS:
    split_dir = output_dir / split
    if split_dir.exists():
        print(f"  ✅ {split}/  →  graphs.pkl, edus.pkl, labels.pkl, stats.pkl")

print("\n🚀 Next step: python step2_embeddings.py")
print("=" * 80)