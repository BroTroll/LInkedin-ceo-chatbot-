import math
import re


SYNONYM_GROUPS: list[frozenset] = [
    frozenset({"fintech", "financial", "finance", "banking", "bank"}),
    frozenset({"tech", "technology", "software", "saas", "it"}),
    frozenset({"healthcare", "health", "biotech", "pharma", "medical"}),
    frozenset({"media", "entertainment", "content", "streaming"}),
    frozenset({"retail", "ecommerce", "commerce", "shopping"}),
    frozenset({"energy", "oil", "gas", "utilities", "power"}),
    frozenset({"ai", "artificial", "intelligence", "ml", "machine"}),
    frozenset({"automotive", "auto", "vehicle", "car", "ev"}),
    frozenset({"telecom", "telecommunications", "wireless", "mobile"}),
]

def _synonym_of(token: str) -> str:
    for group in SYNONYM_GROUPS:
        if token in group:
            return next(iter(group))
    return token

def get_tokens(text: str) -> set[str]:
    raw = set(re.sub(r"[^\w\s]", "", str(text).lower()).split())
    return {_synonym_of(t) for t in raw}


SIZE_FLOOR = 0.15   # minimum score on size dimension — no CEO fully eliminated
SIZE_SIGMA = 1.2   # log10-space std dev (was 0.7 — too tight)


def score_geo(target_city: str, row_city: str) -> float:
    if not target_city:
        return 1.0
    user_tok = get_tokens(target_city)
    ceo_tok  = get_tokens(row_city or "")
    union = user_tok | ceo_tok
    return len(user_tok & ceo_tok) / len(union) if union else 0.0


def score_industry(target_industry: str, row_industry: str) -> float:
    if not target_industry:
        return 1.0
    user_tok = get_tokens(target_industry)
    ceo_tok  = get_tokens(row_industry or "")
    return len(user_tok & ceo_tok) / len(user_tok) if user_tok else 0.0


def score_size(ideal_size: int, actual_size) -> float | None:
    """
    Log-normal decay with floor.
    Returns 1.0   when ideal_size is not specified.
    Returns None  when actual_size is unknown (caller skips dimension).
    Returns float in [SIZE_FLOOR, 1.0] otherwise.
    """
    if not ideal_size:
        return 1.0
    if actual_size is None or actual_size <= 0:
        return None

    log_dist = abs(math.log10(float(actual_size)) - math.log10(float(ideal_size)))
    raw = math.exp(-(log_dist ** 2) / (2 * SIZE_SIGMA ** 2))
    return round(max(raw, SIZE_FLOOR), 4)


def score_network(degree: int) -> float:
    return {1: 1.0, 2: 0.70, 3: 0.40}.get(int(degree or 3), 0.20)


DEFAULT_WEIGHTS = {"geo": 0.35, "ind": 0.30, "size": 0.25, "network": 0.10}


def _build_active_weights(base_weights: dict, size_available: bool) -> dict:
    if size_available:
        total = sum(base_weights.values()) or 1.0
        return {k: v / total for k, v in base_weights.items()}
    reduced = {k: v for k, v in base_weights.items() if k != "size"}
    total   = sum(reduced.values()) or 1.0
    return {k: v / total for k, v in reduced.items()}


def compute_expert_match(
    row: dict,
    target_city:     str  = None,
    target_industry: str  = None,
    ideal_size:      int  = None,
    weights:         dict = None,
) -> float:
    base_w      = {**DEFAULT_WEIGHTS, **(weights or {})}
    actual_size = row.get("company_size") or row.get("employee_count") or row.get("size")
    s_size      = score_size(ideal_size, actual_size)
    active_w    = _build_active_weights(base_w, size_available=(s_size is not None))

    s_geo     = score_geo(target_city, row.get("city", ""))
    s_ind     = score_industry(target_industry, row.get("industry", ""))
    s_network = score_network(row.get("degree", 3))
    quality   = float(row.get("base_quality_score") or 0.7)

    raw = (active_w.get("geo",0)*s_geo + active_w.get("ind",0)*s_ind +
           active_w.get("network",0)*s_network)
    if s_size is not None:
        raw += active_w.get("size", 0) * s_size

    return round(raw * quality, 4)


def compute_expert_match_batch(
    rows:            list[dict],
    target_city:     str   = None,
    target_industry: str   = None,
    ideal_size:      int   = None,
    weights:         dict  = None,
    min_score:       float = 0.0,
) -> list[dict]:
    base_w = {**DEFAULT_WEIGHTS, **(weights or {})}
    scored = []

    for row in rows:
        actual_size = row.get("company_size") or row.get("employee_count")
        s_size      = score_size(ideal_size, actual_size)
        active_w    = _build_active_weights(base_w, size_available=(s_size is not None))

        s_geo     = score_geo(target_city, row.get("city", ""))
        s_ind     = score_industry(target_industry, row.get("industry", ""))
        s_network = score_network(row.get("degree", 3))
        quality   = float(row.get("base_quality_score") or 0.7)

        raw = (active_w.get("geo",0)*s_geo + active_w.get("ind",0)*s_ind +
               active_w.get("network",0)*s_network)
        if s_size is not None:
            raw += active_w.get("size", 0) * s_size

        final = round(raw * quality, 4)
        if final < min_score:
            continue

        scored.append({
            **row,
            "match_score":  final,
            "match_pct":    round(final * 100, 1),
            "dim_geo":      round(s_geo, 3),
            "dim_industry": round(s_ind, 3),
            "dim_size":     round(s_size, 3) if s_size is not None else None,
            "dim_network":  round(s_network, 3),
            "quality":      round(quality, 3),
            "size_scored":  s_size is not None,
        })

    scored.sort(key=lambda r: r["match_score"], reverse=True)
    return scored


def compute_base_quality(row: dict) -> float:
    source_scores = {
        "verified":      0.45,
        "sec":           0.40,
        "wikipedia+sec": 0.35,
        "sec_bulk":      0.35,
        "forbes":        0.30,
        "pdl":           0.30,
        "dbpedia":       0.25,
        "google_xray":   0.15,
        "bing_xray":     0.15,
        "wikidata":      0.15,
    }
    score = source_scores.get(row.get("source", "wikidata"), 0.10)
    if row.get("city"):                                          score += 0.15
    if row.get("industry") and row["industry"] not in ("General", None): score += 0.15
    if row.get("company_size") and (row["company_size"] or 0) > 10:     score += 0.15
    if row.get("bio_snippet"):                                   score += 0.10
    if row.get("company"):                                       score += 0.05
    if (row.get("degree") or 3) <= 2:                           score += 0.05
    if (row.get("degree") or 3) == 1:                           score += 0.10
    return round(min(score, 1.0), 3)
