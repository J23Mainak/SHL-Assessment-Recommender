import os
import re
import time
import pickle
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss

PRODUCTS_CSV = "data/products.csv"   # produced by your crawler
OUT_DIR = Path("data/index")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMBS_PATH = OUT_DIR / "embeddings.npy"
FAISS_PATH = OUT_DIR / "shl_index.faiss"
PROD_PATH = OUT_DIR / "products.parquet"
BM25_PATH = OUT_DIR / "bm25.pkl"
IDMAP_PATH = OUT_DIR / "id_map.pkl"

EMBEDDING_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 64))
NORMALIZE = True
DEDUP_SIM_THRESHOLD = float(os.environ.get("DEDUP_SIM_THRESHOLD", 0.985))
DEDUP_NEIGHBORS = int(os.environ.get("DEDUP_NEIGHBORS", 5))
IVF_THRESHOLD = int(os.environ.get("IVF_THRESHOLD", 100000))

print(f"Config: model={EMBEDDING_MODEL}, batch={BATCH_SIZE}, dedup_threshold={DEDUP_SIM_THRESHOLD}")

def safe_str(val: Any) -> str:
    if val is None:
        return ""
    # Use pandas.isna to detect pd.NA, np.nan
    if pd.isna(val):
        return ""
    return str(val).strip()

def _normalize_colname(col: str) -> str:
    c = str(col).strip().lower()
    c = c.replace(" ", "_")
    c = c.replace("-", "_")
    c = c.replace("/", "_")
    c = re.sub(r'__+', '_', c)
    return c

def fix_column_names(df: pd.DataFrame) -> pd.DataFrame:
    # Build mapping from existing columns to canonical names
    mapping = {}
    cols = list(df.columns)
    for col in cols:
        n = _normalize_colname(col)
        # Title/name mapping
        if n in ("name", "title", "product", "product_name", "assessment_name"):
            mapping[col] = "title"
        # URL
        elif n in ("url", "href", "link"):
            mapping[col] = "url"
        # Adaptive support variants
        elif n in ("adaptive_irt_support", "adaptive_irt", "adaptive_support", "adaptive"):
            mapping[col] = "adaptive_support"
        # Remote testing variants
        elif "remote" in n and ("test" in n or "testing" in n or "remote_support" in n):
            mapping[col] = "remote_support"
        elif n in ("remote_test", "remote_testing", "remote_test_support"):
            mapping[col] = "remote_support"
        # Test type
        elif n in ("test_type", "test_types", "type", "test_types_"):
            mapping[col] = "test_type"
        # Description
        elif n in ("description", "desc", "product_description", "product_desc"):
            mapping[col] = "description"
        # Duration
        elif n in ("duration", "assessment_length", "assessment_length_minutes"):
            mapping[col] = "duration"
        # Languages
        elif n in ("languages", "language", "lang"):
            mapping[col] = "languages"
        # Job level
        elif n in ("job_level", "job_levels", "level", "seniority"):
            mapping[col] = "job_level"
        # Job family
        elif n in ("job_family", "job_family_group", "family"):
            mapping[col] = "job_family"
        # Industry
        elif n in ("industry", "industries", "sector"):
            mapping[col] = "industry"
        # source tab (keep as-is)
        elif n in ("source_tab", "source"):
            mapping[col] = "source_tab"
        

    if mapping:
        df = df.rename(columns=mapping)

    
    canonical_cols_defaults = {
        "title": pd.NA,
        "url": pd.NA,
        "description": pd.NA,
        "test_type": pd.NA,
        "job_level": pd.NA,
        "remote_support": pd.NA,
        "adaptive_support": pd.NA,
        "job_family": pd.NA,
        "industry": pd.NA
    }
    for c, default in canonical_cols_defaults.items():
        if c not in df.columns:
            df[c] = default

    if "adaptive/irt_support" in df.columns and "adaptive_support" not in df.columns:
        df["adaptive_support"] = df["adaptive/irt_support"]
    # Similarly handle "remote_test" or "remote_testing"
    if "remote_test" in df.columns and "remote_support" not in df.columns:
        df["remote_support"] = df["remote_test"]
    if "remote_testing" in df.columns and "remote_support" not in df.columns:
        df["remote_support"] = df["remote_testing"]

    # Some CSVs use the column name 'name' instead of 'title' and rename may not have replaced (double-check)
    if "name" in df.columns and "title" not in df.columns:
        df["title"] = df["name"]

    # Normalize boolean-like values in adaptive_support / remote_support columns to simple 'yes'/'no' where possible
    for col in ("adaptive_support", "remote_support"):
        if col in df.columns:
            def _norm_bool(x):
                if pd.isna(x):
                    return pd.NA
                s = str(x).strip().lower()
                if s in ("yes", "y", "true", "supported", "green"):
                    return "Yes"
                if s in ("no", "n", "false", "not supported", "red"):
                    return "No"
                return x  # leave original if ambiguous
            df[col] = df[col].apply(_norm_bool)

    return df

def clean_text_for_bm25(text: str) -> str:
    if text is None:
        return ""
    if pd.isna(text):
        return ""
    s = str(text).lower()
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'[^\w\s\.\+\-]', ' ', s) 
    return s.strip()

def create_rich_text(row: pd.Series) -> str:
    """Construct combined text for embedding. Title boosted to emphasize importance."""
    parts: List[str] = []
    title = safe_str(row.get("title", ""))
    if title:
        # Title repeated to boost its importance in embeddings
        parts.append(title)
        parts.append(title)

    test_type = safe_str(row.get("test_type", ""))
    if test_type:
        parts.append(f"Test type: {test_type}")

    # map letters to readable description (helps semantic matching)
    type_map = {
        'K': 'technical knowledge and skills',
        'P': 'personality and behavioral traits',
        'B': 'situational judgement and biodata',
        'S': 'simulations',
        'A': 'cognitive ability and aptitude',
        'C': 'competencies',
        'D': 'development and 360 feedback',
        'E': 'assessment exercises'
    }
    type_descs = []
    for t in test_type.split():
        if t in type_map:
            type_descs.append(type_map[t])
    if type_descs:
        parts.append("Measures: " + ", ".join(type_descs))

    desc = safe_str(row.get("description", ""))
    if desc:
        parts.append(desc[:1500])  # cap description length

    job_level = safe_str(row.get("job_level", ""))
    if job_level:
        parts.append(f"Job level: {job_level}")

    job_family = safe_str(row.get("job_family", ""))
    if job_family:
        parts.append(f"Job family: {job_family}")

    industry = safe_str(row.get("industry", ""))
    if industry:
        parts.append(f"Industry: {industry}")

    if safe_str(row.get("remote_support", "")).lower() == "yes":
        parts.append("Remote testing supported")

    if safe_str(row.get("adaptive_support", "")).lower() == "yes":
        parts.append("Adaptive testing supported")

    # Join with separators so BM25 and embedding sees boundaries
    return " . ".join([p for p in parts if p])

def deduplicate_by_embedding(embeddings: np.ndarray, threshold: float = 0.985, k: int = 5):
    n, dim = embeddings.shape
    if n == 0:
        return np.array([], dtype=bool)


    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    k_search = min(k, n)
    sims, idxs = index.search(embeddings, k_search)  # each row returns itself as first neighbor

    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        neighbors = idxs[i]
        neighbor_sims = sims[i]
        # iterate neighbors excluding itself (first is itself)
        for j_pos in range(1, len(neighbors)):
            nb_idx = int(neighbors[j_pos])
            sim_score = float(neighbor_sims[j_pos])
            if nb_idx != i and sim_score >= threshold:
                # Keep smaller index (earlier) and drop later to be deterministic
                if nb_idx > i:
                    keep[nb_idx] = False
                else:
                    keep[i] = False
                    break
    return keep

def build_indices(products_csv: str = PRODUCTS_CSV) -> Dict[str, Any]:
    if not Path(products_csv).exists():
        raise FileNotFoundError(f"{products_csv} not found. Run crawler first.")

    print("Loading products CSV...")
    # Read CSV with default pandas behaviour; fix_column_names will normalize header variants.
    df = pd.read_csv(products_csv)
    df = fix_column_names(df)
    print(f"Loaded {len(df)} rows")

    # Ensure columns exist
    for c in ["title", "description", "test_type", "job_level", "remote_support",
              "adaptive_support", "job_family", "industry", "url"]:
        if c not in df.columns:
            df[c] = pd.NA

    # Create rich text and BM25 text
    print("Creating rich text and bm25 text...")
    
    df["text"] = df.apply(create_rich_text, axis=1)
    df["text_bm25"] = df["text"].apply(clean_text_for_bm25)

    # Drop empty texts
    df = df[df["text_bm25"].str.strip().astype(bool)].reset_index(drop=True)
    print(f"Documents after cleaning: {len(df)}")

    # Build BM25
    print("Building BM25 index...")
    tokenized = [doc.split() for doc in df["text_bm25"].tolist()]
    bm25 = BM25Okapi(tokenized)

    # Choose embedding model
    print(f"Loading sentence transformer model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Batch encode
    corpus = df["text"].tolist()
    print("Encoding embeddings in batches...")
    all_embs = []
    for i in tqdm(range(0, len(corpus), BATCH_SIZE), desc="Batches"):
        batch = corpus[i:i + BATCH_SIZE]
        embs = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        all_embs.append(embs.astype("float32"))
        time.sleep(0.01)
    embeddings = np.vstack(all_embs)

    # Normalize if configured
    if NORMALIZE:
        faiss.normalize_L2(embeddings)

    # Deduplicate near-duplicates
    print(f"Deduplicating with threshold {DEDUP_SIM_THRESHOLD} ...")
    keep_mask = deduplicate_by_embedding(embeddings, threshold=DEDUP_SIM_THRESHOLD, k=DEDUP_NEIGHBORS)
    num_dup = (~keep_mask).sum()
    if num_dup > 0:
        print(f"Found {num_dup} duplicates -> removing them")
        df = df[keep_mask].reset_index(drop=True)
        embeddings = embeddings[keep_mask]
        tokenized = [tokenized[i] for i, k in enumerate(keep_mask) if k]
    else:
        print("No duplicates found")

    # Save embeddings (cached)
    np.save(EMBS_PATH, embeddings)
    print("Saved embeddings to", EMBS_PATH)

    # Build FAISS index. Choose strategy based on corpus size.
    dim = embeddings.shape[1]
    nvec = embeddings.shape[0]
    print(f"Building FAISS index (n={nvec}, dim={dim})")

    if nvec > IVF_THRESHOLD:
        # Use IndexIVFFlat (requires training)
        nlist = int(min(65536, max(1024, nvec // 100)))
        print(f"Using IndexIVFFlat with nlist={nlist} (requires train)")
        quantizer = faiss.IndexFlatIP(dim)
        ivf_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        ivf_index.train(embeddings)
        # Create ID map wrapper then add with IDs
        index_ivf = faiss.IndexIDMap(ivf_index)
        ids = np.arange(nvec).astype("int64")
        index_ivf.add_with_ids(embeddings, ids)
        index = index_ivf
    else:
        flat = faiss.IndexFlatIP(dim)
        index = faiss.IndexIDMap(flat)
        ids = np.arange(nvec).astype("int64")
        index.add_with_ids(embeddings, ids)

    # Save faiss index
    faiss.write_index(index, str(FAISS_PATH))
    print("Saved FAISS index to", FAISS_PATH)

    # Save BM25 artifact (tokenized corpus + bm25 object)
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "tokenized": tokenized}, f)
    print("Saved BM25 to", BM25_PATH)

    # Save metadata
    df.to_parquet(PROD_PATH, index=False)
    print("Saved product metadata to", PROD_PATH)

    # Save id mapping
    with open(IDMAP_PATH, "wb") as f:
        pickle.dump({"ids": ids.tolist()}, f)

    # Print some stats
    print("\n=== Stats ===")
    print("Total documents:", len(df))
    if "test_type" in df.columns:
        from collections import Counter
        types = []
        for t in df["test_type"].dropna():
            types.extend(str(t).upper().split())
        if types:
            print("Test types distribution:", Counter(types))
        else:
            print("Test types column present but empty")
    print("Done building indices.")

    return {
        "df_path": str(PROD_PATH),
        "faiss_path": str(FAISS_PATH),
        "embs_path": str(EMBS_PATH),
        "bm25_path": str(BM25_PATH),
        "id_map": str(IDMAP_PATH)
    }

def load_artifacts():
    df = pd.read_parquet(PROD_PATH)
    embs = np.load(EMBS_PATH)
    with open(BM25_PATH, "rb") as f:
        bm25_data = pickle.load(f)
    index = faiss.read_index(str(FAISS_PATH))
    return df, embs, bm25_data["bm25"], bm25_data["tokenized"], index


def hybrid_search(query: str, top_k: int = 10, bm25_k: int = 50, alpha: float = 0.6):
    df, embs, bm25, tokenized, index = load_artifacts()

    # BM25
    q_clean = clean_text_for_bm25(query)
    q_tokens = q_clean.split()
    bm25_scores = bm25.get_scores(q_tokens)
    bm25_top_idx = np.argsort(bm25_scores)[-bm25_k:][::-1]
    bm25_top_scores = bm25_scores[bm25_top_idx]

    # Dense
    model = SentenceTransformer(os.environ.get("EMBED_MODEL", EMBEDDING_MODEL))
    q_emb = model.encode([query], convert_to_numpy=True)
    q_emb = q_emb.astype("float32")
    if NORMALIZE:
        faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, bm25_k)
    dense_idx = I[0]
    dense_scores = D[0]

    # union candidates
    candidate_set = list(dict.fromkeys(list(bm25_top_idx) + list(dense_idx)))
    cand_to_pos = {c: i for i, c in enumerate(candidate_set)}

    bm25_vals = np.array([bm25_scores[c] for c in candidate_set], dtype=float)
    bm25_norm = (bm25_vals / (bm25_vals.max() if bm25_vals.max() > 0 else 1.0))

    dense_map = {int(idx): float(score) for idx, score in zip(dense_idx, dense_scores)}
    dense_vals = np.array([dense_map.get(int(c), 0.0) for c in candidate_set], dtype=float)
    dense_norm = (dense_vals / (dense_vals.max() if dense_vals.max() > 0 else 1.0))

    final_scores = alpha * dense_norm + (1.0 - alpha) * bm25_norm
    top_order = np.argsort(final_scores)[-top_k:][::-1]
    top_indices = [candidate_set[i] for i in top_order]

    results = []
    for idx in top_indices:
        row = df.iloc[int(idx)].to_dict()
        results.append({
            "index": int(idx),
            "title": row.get("title"),
            "url": row.get("url"),
            "score": float(final_scores[candidate_set.index(idx)]),
            "bm25_score": float(bm25_scores[idx]),
            "dense_score": float(dense_map.get(int(idx), 0.0))
        })
    return results

if __name__ == "__main__":
    out = build_indices()
    print("\nBuild complete. Artifacts:", out)
