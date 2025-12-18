import os
import time
import re
import json
from typing import Dict, List, Set, Tuple, Any
from urllib.parse import urlparse, urlunparse, unquote
from difflib import get_close_matches
from math import log2

import pandas as pd
import numpy as np
import requests

API_URL = os.environ.get("RECOMMEND_API", "http://localhost:8000/recommend")
HEALTH_URL = os.environ.get("HEALTH_API", "http://localhost:8000/health")

TRAIN_SET = "data/train_set.csv"
TEST_SET = "data/test_set.csv"
CANDIDATE_PRODUCT_FILES = [
    "data/index/products.parquet",
    "data/index_products.csv",
    "data/index/products.csv",
    "data/index_products.parquet",
    "data/products.csv",
    "data/index/products.parquet",
]
TOP_K = 10
API_TIMEOUT = 30
SLEEP_BETWEEN_CALLS = 0.05
RETRY_ATTEMPTS = 2
RETRY_BACKOFF = 0.5

# Outputs
FAILURES_CSV = "mapping_failures.csv"
TRAIN_PRED_CSV = "train_predictions_detailed.csv"
SUBMISSION_PRED_CSV = "submission_predictions.csv"

def find_product_file() -> str:
    for p in CANDIDATE_PRODUCT_FILES:
        if os.path.exists(p):
            return p
    for d in ["data/index", "data"]:
        if os.path.isdir(d):
            for fn in os.listdir(d):
                if fn.lower().endswith((".parquet", ".csv")) and ("product" in fn.lower() or "index" in fn.lower()):
                    return os.path.join(d, fn)
    raise FileNotFoundError("No product file found in expected locations.")

def normalize_url(raw: str) -> str:
    if not raw or not isinstance(raw, str):
        return ""
    s = raw.strip()
    try:
        if not re.match(r"^https?://", s):
            s = "https://" + s.lstrip("/")
        parsed = urlparse(s)
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()
        if "@" in netloc:
            netloc = netloc.split("@")[-1]
        path = unquote(parsed.path or "").rstrip("/")
        path = re.sub(r"/+", "/", path)
        # remove index-like endings
        path = re.sub(r"/(index\.html?|default\.html?)$", "", path)
        normalized = urlunparse((scheme, netloc, path, "", "", ""))
        return normalized
    except Exception:
        return s.lower().rstrip("/")

def normalize_title(t: str) -> str:
    if not isinstance(t, str):
        return ""
    s = t.lower().strip()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def last_path_segment(norm_url: str) -> str:
    try:
        p = urlparse(norm_url).path
        if not p:
            return ""
        seg = p.strip("/").split("/")[-1]
        return seg.lower()
    except Exception:
        return ""

def domain_key(norm_url: str) -> str:
    try:
        return urlparse(norm_url).netloc.lower()
    except Exception:
        return ""

def build_url_mappings(products_df: pd.DataFrame) -> Dict[str, Any]:
    df = products_df.copy().fillna("")
    url_col = None
    for cand in ["url", "link", "assessment_url", "href"]:
        if cand in df.columns:
            url_col = cand
            break
    if url_col is None:
        for c in df.columns:
            if df[c].astype(str).str.contains("http", na=False).any():
                url_col = c
                break
    if url_col is None:
        raise KeyError("No URL column found in products file.")
    title_col = None
    for cand in ["title", "name", "assessment_name"]:
        if cand in df.columns:
            title_col = cand
            break
    if title_col is None:
        title_col = df.columns[0]

    url_to_canonical = {}
    title_to_url = {}
    last_segment_to_urls = {}
    path_to_urls = {}
    domain_to_urls = {}
    canonical_titles_map = {}

    for _, row in df.iterrows():
        raw_url = str(row.get(url_col, "") or "").strip()
        if not raw_url:
            continue
        canon_url = raw_url
        nurl = normalize_url(raw_url)
        if not nurl:
            continue
        url_to_canonical[nurl] = canon_url
        if nurl.startswith("https://www."):
            url_to_canonical[nurl.replace("https://www.", "https://")] = canon_url
        if nurl.startswith("http://www."):
            url_to_canonical[nurl.replace("http://www.", "http://")] = canon_url

        title = str(row.get(title_col, "") or "")
        ntitle = normalize_title(title)
        if ntitle:
            if ntitle not in title_to_url:
                title_to_url[ntitle] = canon_url
            canonical_titles_map[canon_url] = ntitle

        seg = last_path_segment(nurl)
        if seg:
            last_segment_to_urls.setdefault(seg, set()).add(canon_url)
        try:
            p = urlparse(nurl).path
        except Exception:
            p = ""
        if p:
            path_to_urls.setdefault(p, set()).add(canon_url)
        dom = domain_key(nurl)
        if dom:
            domain_to_urls.setdefault(dom, set()).add(canon_url)

        seg_no_suffix = re.sub(r'-(new|latest|solution|product|view|page)$', '', seg)
        if seg_no_suffix and seg_no_suffix != seg:
            last_segment_to_urls.setdefault(seg_no_suffix, set()).add(canon_url)

    all_canonical = set(url_to_canonical.values())
    return {
        "url_to_canonical": url_to_canonical,
        "title_to_url": title_to_url,
        "last_segment_to_urls": last_segment_to_urls,
        "path_to_urls": path_to_urls,
        "domain_to_urls": domain_to_urls,
        "all_canonical": all_canonical,
        "canonical_titles_map": canonical_titles_map,
        "url_col": url_col,
        "title_col": title_col
    }

def _token_overlap_match(pred_norm_title: str, title_candidates: List[Tuple[str,str]]) -> str:
    if not pred_norm_title:
        return ""
    tokens_pred = set(pred_norm_title.split())
    best = ("", 0, "")  # (canon_url, overlap_count, title)
    for ntitle, canon in title_candidates:
        tset = set(ntitle.split())
        overlap = len(tokens_pred & tset)
        if overlap > best[1]:
            best = (canon, overlap, ntitle)
    
    if best[1] >= 1 and best[1] >= max(1, int(0.4 * len(pred_norm_title.split()))):
        return best[0]
    return ""

def match_url_to_canonical(pred_url: str, mappings: Dict[str, Any]) -> str:
    if not pred_url:
        return ""
    pred = pred_url.strip()

    # 1) If full url-like: normalize and match
    n = normalize_url(pred)
    if n and n in mappings["url_to_canonical"]:
        return mappings["url_to_canonical"][n]
    if n and n.startswith("https://www."):
        n2 = n.replace("https://www.", "https://")
        if n2 in mappings["url_to_canonical"]:
            return mappings["url_to_canonical"][n2]
        
    # 2) Last path segment heuristic
    if n:
        seg = last_path_segment(n)
        if seg and seg in mappings["last_segment_to_urls"]:
            candidates = list(mappings["last_segment_to_urls"][seg])
            dom = domain_key(n)
            for c in candidates:
                if domain_key(normalize_url(c)) == dom:
                    return c
            return candidates[0]
        
    # 3) full path exact
    if n:
        try:
            path = urlparse(n).path
        except:
            path = ""
        if path and path in mappings["path_to_urls"]:
            return next(iter(mappings["path_to_urls"][path]))
        
    # 4) If the pred looks like a title (contains spaces or few words), try title mapping
    pnorm = normalize_title(pred)
    if pnorm and pnorm in mappings["title_to_url"]:
        return mappings["title_to_url"][pnorm]
    
    # 5) Fuzzy URL matching (difflib) against known normalized urls
    if n:
        all_urls = list(mappings["url_to_canonical"].keys())
        close = get_close_matches(n, all_urls, n=1, cutoff=0.75)
        if close:
            return mappings["url_to_canonical"][close[0]]
        close2 = get_close_matches(n, all_urls, n=1, cutoff=0.6)
        if close2:
            return mappings["url_to_canonical"][close2[0]]
        
    # 6) Fuzzy title matching
    if pnorm:
        title_keys = list(mappings["title_to_url"].keys())
        close_t = get_close_matches(pnorm, title_keys, n=1, cutoff=0.7)
        if close_t:
            return mappings["title_to_url"][close_t[0]]
        close_t2 = get_close_matches(pnorm, title_keys, n=1, cutoff=0.6)
        if close_t2:
            return mappings["title_to_url"][close_t2[0]]
        
    # 7) Token overlap title match (helpful for partial/word-order variations)
    if pnorm:
        title_candidates = [(k, v) for k, v in mappings["title_to_url"].items()]
        tok_match = _token_overlap_match(pnorm, title_candidates)
        if tok_match:
            return tok_match
    # No match
    return ""

def call_api(query: str, k: int = TOP_K) -> List[str]:
    urls = []
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            resp = requests.post(API_URL, json={"query": query, "k": k}, timeout=API_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", []) if isinstance(data, dict) else data
            for r in results:
                u = None
                if isinstance(r, dict):
                    # common fields
                    for cand in ("url", "assessment_url", "link", "href", "assessment_url", "assessmentName", "assessment_name", "title", "assessment_name"):
                        if cand in r and r[cand]:
                            u = r[cand]
                            break
                    if not u:
                        for v in r.values():
                            if isinstance(v, str) and v.strip():
                                u = v
                                break
                elif isinstance(r, str):
                    u = r
                if u:
                    urls.append(u)
                else:
                    urls.append("")
            return urls
        except Exception as e:
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_BACKOFF * attempt)
                continue
            else:
                print(f"API Error (attempt {attempt}): {e}")
                return []

def recall_at_k(predicted: List[str], relevant: Set[str], k: int = 10) -> float:
    if not relevant:
        return 0.0
    pred_set = set(predicted[:k])
    return len(pred_set & relevant) / len(relevant)

def average_precision_at_k(predicted: List[str], relevant: Set[str], k: int = 3) -> float:
    if not relevant:
        return 0.0
    pred_k = predicted[:k]
    score = 0.0
    num_hits = 0
    for i, p in enumerate(pred_k, start=1):
        if p in relevant:
            num_hits += 1
            score += num_hits / i
    return score / min(len(relevant), k)

def mrr_at_k(predicted: List[str], relevant: Set[str], k: int = 10) -> float:
    pred_k = predicted[:k]
    for i, p in enumerate(pred_k, start=1):
        if p in relevant:
            return 1.0 / i
    return 0.0

def dcg_at_k(predicted: List[str], relevant: Set[str], k: int = 10) -> float:
    pred_k = predicted[:k]
    dcg = 0.0
    for i, p in enumerate(pred_k, start=1):
        rel = 1.0 if p in relevant else 0.0
        if i == 1:
            dcg += rel
        else:
            dcg += rel / log2(i + 0.0)
    return dcg

def ndcg_at_k(predicted: List[str], relevant: Set[str], k: int = 10) -> float:
    ideal_rel_count = min(len(relevant), k)
    # ideal DCG is sum of 1/log2(i+1) for first ideal_rel_count positions
    ideal = 0.0
    for i in range(1, ideal_rel_count + 1):
        if i == 1:
            ideal += 1.0
        else:
            ideal += 1.0 / log2(i + 0.0)
    if ideal == 0.0:
        return 0.0
    return dcg_at_k(predicted, relevant, k=k) / ideal

def evaluate_on_dataset(dataset_path: str, mappings: Dict[str, Any], output_csv: str = None):
    print("--> Evaluating:", dataset_path)

    df = pd.read_csv(dataset_path, dtype=str).fillna("")

    has_query = "Query" in df.columns
    has_assessment = any(c.lower() in {"assessment_url", "assessment url", "assessmenturl"} for c in df.columns)

    if not has_query:
        possible_q = [c for c in df.columns if "query" in c.lower() or "job" in c.lower()]
        if not possible_q:
            raise KeyError("Dataset does not contain a 'Query' column.")
        df = df.rename(columns={possible_q[0]: "Query"})
        print("Renamed", possible_q[0], "-> Query")

    if "Assessment_url" not in df.columns and "assessment_url" in df.columns:
        df = df.rename(columns={"assessment_url": "Assessment_url"})
        has_assessment = True
    if "Assessment URL" in df.columns:
        df = df.rename(columns={"Assessment URL": "Assessment_url"})
        has_assessment = True

    grouped = {}
    if has_assessment:
        grouped = df.groupby("Query")["Assessment_url"].apply(list).to_dict()
        print(f"Found {len(grouped)} unique queries with ground truth")
    else:
        grouped = {q: [] for q in df["Query"].dropna().unique().tolist()}
        print(f"No ground-truth present. Generating predictions for {len(grouped)} queries.")

    queries = list(grouped.keys())
    all_predictions = []
    all_ground_truths = []
    submission_rows = []
    mapping_failures = []
    detailed_rows = []

    for i, query in enumerate(queries, start=1):
        print(f"[{i}/{len(queries)}] Query: {query[:120]}...")
        predicted_raw = call_api(query, k=TOP_K)
        canonical_preds = []
        unmatched_preds = []
        for p in predicted_raw:
            mapped = match_url_to_canonical(p, mappings)
            if mapped:
                canonical_preds.append(mapped)
            else:
                unmatched_preds.append(p)

        # preserve order and deduplicate
        seen = set()
        canonical_preds = [x for x in canonical_preds if not (x in seen or seen.add(x))]

        # ground truth
        g_urls = grouped.get(query, []) if grouped else []
        canonical_gt = set()
        for gu in g_urls:
            mapped_gt = match_url_to_canonical(gu, mappings)
            if mapped_gt:
                canonical_gt.add(mapped_gt)

        if unmatched_preds:
            for up in list(unmatched_preds):
                upnorm = normalize_title(up)
                if upnorm and upnorm in mappings["title_to_url"]:
                    cand = mappings["title_to_url"][upnorm]
                    if cand not in canonical_preds:
                        canonical_preds.append(cand)
                    unmatched_preds.remove(up)

        # collect mapping failures
        if unmatched_preds or any((gu and not match_url_to_canonical(gu, mappings)) for gu in g_urls):
            mapping_failures.append({
                "query": query,
                "predicted_raw": json.dumps(predicted_raw, ensure_ascii=False),
                "predicted_mapped_count": len(canonical_preds),
                "predicted_mapped": json.dumps(canonical_preds, ensure_ascii=False),
                "groundtruth_raw": json.dumps(g_urls, ensure_ascii=False),
                "groundtruth_mapped_count": len(canonical_gt),
                "groundtruth_mapped": json.dumps(list(canonical_gt), ensure_ascii=False),
                "unmatched_preds": json.dumps(unmatched_preds, ensure_ascii=False),
            })

        # metrics for this query
        recall10 = recall_at_k(canonical_preds, canonical_gt, k=TOP_K) if g_urls else 0.0
        recall5 = recall_at_k(canonical_preds, canonical_gt, k=5) if g_urls else 0.0
        mrr10 = mrr_at_k(canonical_preds, canonical_gt, k=TOP_K) if g_urls else 0.0
        ndcg10 = ndcg_at_k(canonical_preds, canonical_gt, k=TOP_K) if g_urls else 0.0
        ap3 = average_precision_at_k(canonical_preds, canonical_gt, k=3) if g_urls else 0.0

        print(f"  Mapped preds: {len(canonical_preds)} ; Mapped GT: {len(canonical_gt)}")
        print(f"  Recall@5: {recall5:.3f}  Recall@10: {recall10:.3f}  MRR@10: {mrr10:.3f}  nDCG@10: {ndcg10:.3f}\n")

        all_predictions.append(canonical_preds)
        all_ground_truths.append(canonical_gt)

        # build submission rows using canonical_preds (top-K) or fallback to raw preds that look like urls
        used_urls = []
        for url in canonical_preds[:TOP_K]:
            if url not in used_urls:
                submission_rows.append({"Query": query, "Assessment_url": url})
                used_urls.append(url)

        if not canonical_preds and predicted_raw:
            for p in predicted_raw[:TOP_K]:
                submission_rows.append({"Query": query, "Assessment_url": p})

        # detailed rows for diagnostics
        detailed_rows.append({
            "Query": query,
            "predicted_raw": json.dumps(predicted_raw, ensure_ascii=False),
            "predicted_mapped": json.dumps(canonical_preds, ensure_ascii=False),
            "groundtruth_raw": json.dumps(g_urls, ensure_ascii=False),
            "groundtruth_mapped": json.dumps(list(canonical_gt), ensure_ascii=False),
            "recall@5": recall5,
            "recall@10": recall10,
            "mrr@10": mrr10,
            "ndcg@10": ndcg10,
            "ap@3": ap3
        })

        time.sleep(SLEEP_BETWEEN_CALLS)

    # compute aggregate metrics
    valid_indices = [i for i, g in enumerate(all_ground_truths) if g]
    if valid_indices:
        recalls_10 = [recall_at_k(all_predictions[i], all_ground_truths[i], k=10) for i in valid_indices]
        recalls_5 = [recall_at_k(all_predictions[i], all_ground_truths[i], k=5) for i in valid_indices]
        mrrs = [mrr_at_k(all_predictions[i], all_ground_truths[i], k=10) for i in valid_indices]
        ndcgs = [ndcg_at_k(all_predictions[i], all_ground_truths[i], k=10) for i in valid_indices]
        aps = [average_precision_at_k(all_predictions[i], all_ground_truths[i], k=3) for i in valid_indices]
        mean_recall_10 = float(np.mean(recalls_10))
        mean_recall_5 = float(np.mean(recalls_5))
        mean_mrr_10 = float(np.mean(mrrs))
        mean_ndcg_10 = float(np.mean(ndcgs))
        map_3 = float(np.mean(aps))
    else:
        mean_recall_10 = mean_recall_5 = mean_mrr_10 = mean_ndcg_10 = map_3 = 0.0

    print(f"-> Mean Recall@5:  {mean_recall_5:.4f} ({mean_recall_5*100:.2f}%)")
    print(f"-> Mean Recall@10: {mean_recall_10:.4f} ({mean_recall_10*100:.2f}%)")
    print(f"-> Mean MRR@10:    {mean_mrr_10:.4f}")
    print(f"-> Mean nDCG@10:   {mean_ndcg_10:.4f}")
    print(f"-> MAP@3:          {map_3:.4f}\n")

    # Save mapping failures
    if mapping_failures:
        mf_df = pd.DataFrame(mapping_failures)
        mf_df.to_csv(FAILURES_CSV, index=False)
        print(f"Saved mapping failures to: {FAILURES_CSV}")

    if detailed_rows:
        det_df = pd.DataFrame(detailed_rows)
        det_df.to_csv(TRAIN_PRED_CSV, index=False)
        print(f"Saved detailed per-query predictions to: {TRAIN_PRED_CSV}")

    # Save submission CSV (unique pairs)
    if submission_rows:
        out_df = pd.DataFrame(submission_rows).drop_duplicates().reset_index(drop=True)
        out_df.to_csv(output_csv or SUBMISSION_PRED_CSV, index=False)
        print(f"Saved submission predictions to: {output_csv or SUBMISSION_PRED_CSV}")

    return {
        "mean_recall_10": mean_recall_10,
        "mean_recall_5": mean_recall_5,
        "mean_mrr_10": mean_mrr_10,
        "mean_ndcg_10": mean_ndcg_10,
        "map_3": map_3
    }

def main():
    print("Building URL mappings...")
    prod_file = find_product_file()
    print("Using products file:", prod_file)
    if prod_file.lower().endswith(".parquet"):
        products = pd.read_parquet(prod_file)
    else:
        products = pd.read_csv(prod_file, dtype=str).fillna("")

    mappings = build_url_mappings(products)
    print("Loaded", len(mappings["all_canonical"]), "canonical products\n")

    # check API health
    try:
        r = requests.get(HEALTH_URL, timeout=5)
        if r.status_code == 200:
            print("-> API is running\n")
        else:
            print("API health check returned", r.status_code)
    except Exception:
        print("API not reachable; start it and re-run.")

    if os.path.exists(TRAIN_SET):
        evaluate_on_dataset(TRAIN_SET, mappings, output_csv="train_submission.csv")
    else:
        print("Train set not found:", TRAIN_SET)

    if os.path.exists(TEST_SET):
        tmp = pd.read_csv(TEST_SET, dtype=str).fillna("")
        if any(c.lower() in {"assessment_url", "assessment url", "assessmenturl"} for c in tmp.columns):
            evaluate_on_dataset(TEST_SET, mappings, output_csv=SUBMISSION_PRED_CSV)
        else:
            print("\nTest set doesn't have ground-truth. Generating submission predictions only.")
            evaluate_on_dataset(TEST_SET, mappings, output_csv=SUBMISSION_PRED_CSV)
    else:
        print("Test set not found:", TEST_SET)

if __name__ == "__main__":
    main()
