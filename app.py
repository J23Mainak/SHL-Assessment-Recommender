import os
import re
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict, OrderedDict

import numpy as np
import pandas as pd
import faiss
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


INDEX_DIR = "data/index"
PRODUCT_CANDIDATES = [
    os.path.join(INDEX_DIR, "products.parquet"),
    os.path.join("data", "products.parquet"),
    os.path.join(INDEX_DIR, "products.csv"),
    os.path.join("data", "products.csv"),
    "data/products.csv",
]
BM25_CANDIDATES = [
    os.path.join(INDEX_DIR, "bm25.pkl"),
    os.path.join("data", "bm25.pkl"),
    os.path.join(INDEX_DIR, "bm25.pkl"),
]
EMB_CANDIDATES = [
    os.path.join(INDEX_DIR, "embeddings.npy"),
    os.path.join("data", "embeddings.npy"),
    os.path.join(INDEX_DIR, "embs.npy"),
    os.path.join("data", "embs.npy"),
]
FAISS_CANDIDATES = [
    os.path.join(INDEX_DIR, "shl_index.faiss"),
    os.path.join(INDEX_DIR, "index.faiss"),
    os.path.join("data", "shl_index.faiss"),
    os.path.join(INDEX_DIR, "shl_index.index"),
]
IDMAP_FILE = os.path.join(INDEX_DIR, "id_map.pkl")

EMBEDDING_MODEL_NAME = os.environ.get("EMBED_MODEL", "BAAI/bge-base-en-v1.5")

BM25_TOP = int(os.environ.get("BM25_TOP", 200))
DENSE_TOP = int(os.environ.get("DENSE_TOP", 200))
ENSEMBLE_TOP = int(os.environ.get("ENSEMBLE_TOP", 200))
FINAL_K_MIN = int(os.environ.get("FINAL_K_MIN", 5))
FINAL_K_MAX = int(os.environ.get("FINAL_K_MAX", 10))

WEIGHT_DENSE = float(os.environ.get("WEIGHT_DENSE", 0.55))
WEIGHT_BM25 = float(os.environ.get("WEIGHT_BM25", 0.25))
WEIGHT_KEYWORD = float(os.environ.get("WEIGHT_KEYWORD", 0.10))
WEIGHT_TYPE = float(os.environ.get("WEIGHT_TYPE", 0.10))

class SimpleLRU:
    def __init__(self, maxsize=256):
        self.maxsize = maxsize
        self.od = OrderedDict()

    def get(self, key):
        try:
            val = self.od.pop(key)
            self.od[key] = val  # move to end
            return val
        except KeyError:
            return None

    def set(self, key, value):
        if key in self.od:
            self.od.pop(key)
        self.od[key] = value
        if len(self.od) > self.maxsize:
            self.od.popitem(last=False)

def first_existing(paths: List[str]) -> str:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

def find_product_file() -> str:
    p = first_existing(PRODUCT_CANDIDATES)
    if p:
        return p
    # fallback scan
    for d in ("data", INDEX_DIR):
        if os.path.isdir(d):
            for fn in os.listdir(d):
                if fn.endswith(".parquet") or fn.endswith(".csv"):
                    return os.path.join(d, fn)
    raise FileNotFoundError("No product file found in expected locations.")

def find_bm25_file() -> str:
    return first_existing(BM25_CANDIDATES)

def find_emb_file() -> str:
    return first_existing(EMB_CANDIDATES)

def find_faiss_file() -> str:
    return first_existing(FAISS_CANDIDATES)

print("Loading product metadata...")
PRODUCTS_FILE = find_product_file()
print("Products file:", PRODUCTS_FILE)
ext = Path(PRODUCTS_FILE).suffix.lower()
if ext == ".parquet":
    products = pd.read_parquet(PRODUCTS_FILE)
else:
    products = pd.read_csv(PRODUCTS_FILE, dtype=str, keep_default_na=False).fillna("")

# Normalize some common columns
def ensure_column(df: pd.DataFrame, candidates: List[str], new_name: str):
    for c in candidates:
        if c in df.columns:
            df = df.rename(columns={c: new_name})
            return df, new_name
    df[new_name] = ""
    return df, new_name

products, _ = ensure_column(products, ["title", "name", "assessment_name"], "title")
products, _ = ensure_column(products, ["url", "assessment_url", "link", "href"], "url")
products, _ = ensure_column(products, ["description", "text", "desc"], "description")
products, _ = ensure_column(products, ["test_type", "test_types", "type"], "test_type")
products, _ = ensure_column(products, ["duration", "duration_minutes"], "duration")
products, _ = ensure_column(products, ["remote_support", "remote_testing", "remote"], "remote_support")
products, _ = ensure_column(products, ["adaptive_support", "adaptive", "adaptive/irt_support"], "adaptive_support")

# ensure string types
for col in ["title", "url", "description", "test_type", "duration", "remote_support", "adaptive_support"]:
    if col in products.columns:
        products[col] = products[col].astype(str).fillna("").str.strip()
    else:
        products[col] = ""

# Compose BM25 text
def create_rich_text(row: pd.Series) -> str:
    parts = []
    title = str(row.get("title","") or "").strip()
    if title:
        parts.append(title)
    test_type = str(row.get("test_type","") or "").strip()
    if test_type:
        parts.append(f"Test type: {test_type}")
    desc = str(row.get("description","") or "").strip()
    if desc:
        parts.append(desc)
    job_level = str(row.get("job_level","") or "").strip() if "job_level" in row else ""
    if job_level:
        parts.append(f"Job level: {job_level}")
    return " . ".join([p for p in parts if p])

products["text"] = products.apply(create_rich_text, axis=1)
products["text_clean"] = products["text"].str.lower().str.replace(r"\s+", " ", regex=True)

docs = products["text_clean"].fillna("").tolist()
print(f"Loaded {len(products)} products.")

# BM25
BM25_FILE = find_bm25_file()
bm25 = None
tokenized = None
if BM25_FILE:
    print("Attempting to load BM25 from:", BM25_FILE)
    try:
        with open(BM25_FILE, "rb") as f:
            obj = pickle.load(f)
            if isinstance(obj, dict):
                bm25 = obj.get("bm25") or obj.get("bm25_obj") or obj.get("bm25_okapi")
                tokenized = obj.get("tokenized") or obj.get("tokens")
            else:
                bm25 = obj
            print("BM25 loaded.")
    except Exception as e:
        print("Failed to load BM25 pickle:", e)
        bm25 = None

if bm25 is None:
    print("Building BM25 on the fly from product texts...")
    tokenized = [d.split() for d in products["text_clean"].fillna("").tolist()]
    try:
        bm25 = BM25Okapi(tokenized)
        print("BM25 built in-memory.")
    except Exception as e:
        print("BM25 build failed:", e)
        bm25 = None

# Embeddings
EMB_FILE = find_emb_file()
embs = None
if EMB_FILE:
    try:
        print("Loading embeddings from:", EMB_FILE)
        embs = np.load(EMB_FILE)
        print("Embeddings shape:", embs.shape)
    except Exception as e:
        print("Failed to load embeddings:", e)
        embs = None
else:
    print("No embeddings file found.")

# FAISS index
FAISS_FILE = find_faiss_file()
index = None
if FAISS_FILE:
    try:
        print("Loading FAISS index from:", FAISS_FILE)
        index = faiss.read_index(FAISS_FILE)
        print("FAISS index loaded. ntotal:", index.ntotal)
    except Exception as e:
        print("Failed to load FAISS index:", e)
        index = None

if index is None and embs is not None:
    try:
        print("Building in-memory FAISS IndexFlatIP from loaded embeddings...")
        dim = embs.shape[1]
        flat = faiss.IndexFlatIP(dim)
        # ensure normalized for inner product = cosine
        faiss.normalize_L2(embs)
        index = faiss.IndexIDMap(flat)
        ids = np.arange(embs.shape[0]).astype("int64")
        index.add_with_ids(embs, ids)
        print("Built fallback FAISS with ntotal:", index.ntotal)
    except Exception as e:
        print("Failed to build fallback FAISS:", e)
        index = None

# ID map (faiss id -> product row index)
id_map: Dict[int,int] = {}
if os.path.exists(IDMAP_FILE):
    try:
        with open(IDMAP_FILE, "rb") as f:
            m = pickle.load(f)
            if isinstance(m, dict) and "ids" in m:
                # ids list is in same order as the dataframe used during build
                for pos, fid in enumerate(m["ids"]):
                    id_map[int(fid)] = int(pos)
                print("Loaded id_map entries:", len(id_map))
            elif isinstance(m, list):
                for pos, fid in enumerate(m):
                    id_map[int(fid)] = int(pos)
                print("Loaded id_map (list) entries:", len(id_map))
    except Exception as e:
        print("Failed to load id_map:", e)
else:
    print("No id_map file found at", IDMAP_FILE)

# Load embedding model (used for runtime/dense queries)
print("Loading embedding model:", EMBEDDING_MODEL_NAME)
embed_model = None
try:
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Embedding model loaded. dim:", embed_model.get_sentence_embedding_dimension())
except Exception as e:
    print("Failed to load embedding model (embedding queries will be disabled):", e)
    embed_model = None

# If we have embs and embed_model, sanity-check dimension
if embs is not None and embed_model is not None:
    model_dim = embed_model.get_sentence_embedding_dimension()
    if embs.shape[1] != model_dim:
        print(f"-> Embedding dimension mismatch: saved_emb_dim={embs.shape[1]} vs model_dim={model_dim}")
    else:
        print("Embedding dimension matches model.")

# Prepare embedding cache
_emb_cache = SimpleLRU(maxsize=512)

def get_query_embedding(query: str) -> np.ndarray:
    q = query.strip()
    cached = _emb_cache.get(q)
    if cached is not None:
        return cached
    if embed_model is None:
        raise RuntimeError("Embedding model not available")
    emb = embed_model.encode([q], convert_to_numpy=True, show_progress_bar=False)
    emb = emb.astype("float32")
    try:
        faiss.normalize_L2(emb)
    except Exception:
        pass
    _emb_cache.set(q, emb[0])
    return emb[0]

app = FastAPI(title="SHL Assessment Recommender (optimized)")

class QueryRequest(BaseModel):
    query: str
    k: int = 10

class AssessmentResponse(BaseModel):
    assessment_name: str
    url: str
    remote_support: str
    adaptive_support: str
    duration: str
    test_type: str
    score: float

class RecommendationResponse(BaseModel):
    query: str
    results: List[AssessmentResponse]

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()

def extract_keywords(text: str, max_keywords: int=20) -> List[str]:
    s = clean_text(text)
    words = s.split()
    stopwords = {'the','a','an','and','or','but','in','on','at','to','for','of','with','by','from','is','are','was','were','be','been','assessment','test','solution','need','looking','want','hire','role','job'}
    keywords = [w for w in words if len(w) > 3 and w not in stopwords]
    seen = set(); out=[]
    for w in keywords:
        if w not in seen:
            seen.add(w); out.append(w)
        if len(out) >= max_keywords:
            break
    return out

def bm25_search(query: str, top_k: int = BM25_TOP) -> List[int]:
    if bm25 is None:
        return []
    q = clean_text(query)
    tokens = q.split()
    try:
        scores = bm25.get_scores(tokens)
        ranks = np.argsort(scores)[::-1][:top_k].tolist()
        return ranks
    except Exception:
        return []

def semantic_search(query: str, top_k: int = DENSE_TOP) -> List[Tuple[int, float]]:
    """Return list of (product_row_index, score)."""
    if embed_model is None or index is None:
        return []
    try:
        q_emb = get_query_embedding(query).reshape(1, -1).astype("float32")
    except Exception:
        return []
    try:
        D, I = index.search(q_emb, top_k)
        results = []
        for fid, score in zip(I[0], D[0]):
            if int(fid) == -1:
                continue
            # Map faiss id -> saved df row index using id_map, otherwise assume fid==row idx
            mapped = id_map.get(int(fid), int(fid))
            if mapped < 0 or mapped >= len(products):
                continue
            results.append((int(mapped), float(score)))
        return results
    except Exception:
        return []

def keyword_filter_search(query: str, top_k: int = 50) -> List[int]:
    kws = extract_keywords(query, max_keywords=20)
    if not kws:
        return []
    scores = []
    for i, doc in enumerate(docs):
        cnt = 0
        for kw in kws:
            if kw in doc:
                cnt += 1
        if cnt > 0:
            scores.append((i, cnt))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in scores[:top_k]]

def test_type_match_search(query_intent: Dict, top_k: int = 100) -> List[int]:
    required_types = query_intent.get("required_test_types", [])
    if not required_types:
        return []
    matched = []
    for i, row in products.iterrows():
        types = str(row.get("test_type", "") or "").upper()
        if any(t in types for t in required_types):
            matched.append(i)
    return matched[:top_k]

def ensemble_retrieval(query: str, query_intent: Dict, top_k: int = ENSEMBLE_TOP) -> List[Tuple[int, float]]:
    bm25_idx = bm25_search(query, top_k=BM25_TOP)
    bm25_scores_map = {idx: 1.0/(rank+1) for rank, idx in enumerate(bm25_idx)}
    dense_results = semantic_search(query, top_k=DENSE_TOP)
    dense_scores_map = {idx: score for idx, score in dense_results}
    dense_list = [idx for idx, _ in dense_results]
    keyword_idx = keyword_filter_search(query, top_k=50)
    keyword_scores_map = {idx: 1.0/(rank+1) for rank, idx in enumerate(keyword_idx)}
    type_idx = test_type_match_search(query_intent, top_k=100)
    type_scores_map = {idx: 1.0/(rank+1) for rank, idx in enumerate(type_idx)}

    all_idxs = set(bm25_idx + dense_list + keyword_idx + type_idx)
    combined = []
    for idx in all_idxs:
        s = (
            WEIGHT_BM25 * bm25_scores_map.get(idx, 0.0)
            + WEIGHT_DENSE * dense_scores_map.get(idx, 0.0)
            + WEIGHT_KEYWORD * keyword_scores_map.get(idx, 0.0)
            + WEIGHT_TYPE * type_scores_map.get(idx, 0.0)
        )
        combined.append((idx, s))
    combined.sort(key=lambda x: x[1], reverse=True)
    return combined[:top_k]

def balance_by_test_type(candidates: List[int], query_intent: Dict, target_k: int = 10) -> List[int]:
    by_type = defaultdict(list)
    for idx in candidates:
        t = str(products.iloc[idx].get("test_type", "")).upper()
        for token in t.split():
            if token:
                by_type[token].append(idx)
    req = query_intent.get("required_test_types", [])
    min_k = query_intent.get("min_k_tests", 0)
    min_p = query_intent.get("min_p_tests", 0)
    selected = []
    selset = set()
    requirements = [("K", min_k), ("P", min_p)]
    for typ, cnt in requirements:
        if cnt > 0 and typ in by_type:
            for idx in by_type[typ]:
                if idx not in selset:
                    selected.append(idx)
                    selset.add(idx)
                    if len(selected) >= target_k:
                        return selected[:target_k]
    for idx in candidates:
        if idx not in selset:
            selected.append(idx)
            selset.add(idx)
            if len(selected) >= target_k:
                break
    if len(selected) < target_k:
        for typ in req:
            for idx in by_type.get(typ, []):
                if idx not in selset:
                    selected.append(idx)
                    selset.add(idx)
                    if len(selected) >= target_k:
                        break
    return selected[:target_k]

def format_response_with_scores(pairs: List[Tuple[int, float]], query: str):
    results = []
    for idx, score in pairs:
        if idx < 0 or idx >= len(products):
            continue
        row = products.iloc[idx]
        res = {
            "assessment_name": str(row.get("title", "")),
            "url": str(row.get("url", "")),
            "remote_support": str(row.get("remote_support", "")),
            "adaptive_support": str(row.get("adaptive_support", "")),
            "duration": str(row.get("duration", "")),
            "test_type": str(row.get("test_type", "")),
            "score": float(score)
        }
        results.append(res)
    return {"query": query, "results": results}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "products_loaded": len(products),
        "faiss_index_loaded": index is not None,
        "embeddings_loaded": embs is not None,
        "bm25_loaded": bm25 is not None,
        "embedding_model_loaded": embed_model is not None,
        "embed_model_name": EMBEDDING_MODEL_NAME
    }

@app.post("/recommend")
def recommend(request: QueryRequest):
    query = (request.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    k = min(max(request.k or FINAL_K_MIN, FINAL_K_MIN), FINAL_K_MAX)

    # Simple intent heuristics
    qlow = query.lower()
    query_intent = {"required_test_types": [], "min_k_tests": 0, "min_p_tests": 0}
    tech_keywords = ['java','python','sql','javascript','programming','coding','developer','engineer']
    soft_keywords = ['collaborate','communication','teamwork','leadership','personality','interpersonal']
    if any(kw in qlow for kw in tech_keywords):
        query_intent['required_test_types'].append('K'); query_intent['min_k_tests'] = 3
    if any(kw in qlow for kw in soft_keywords):
        query_intent['required_test_types'].append('P'); query_intent['min_p_tests'] = 2
    if not query_intent['required_test_types']:
        query_intent['required_test_types'] = ['K','P']

    # retrieval
    candidates_with_scores = ensemble_retrieval(query, query_intent, top_k=ENSEMBLE_TOP)
    candidate_indices = [idx for idx, _ in candidates_with_scores]

    # balance & select
    balanced = balance_by_test_type(candidate_indices, query_intent, target_k=k)

    # produce final pairs preserving score when available
    idx_to_score = {idx: score for idx, score in candidates_with_scores}
    final_pairs = [(idx, idx_to_score.get(idx, 0.0)) for idx in balanced]

    # format and return
    resp = format_response_with_scores(final_pairs, query)
    return resp

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=False)
