import json
import os
import re
import sqlite3
import time
from functools import wraps

from flask import Flask, jsonify, request, render_template, g
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv

from scoring import compute_expert_match_batch

load_dotenv()

app     = Flask(__name__)
CORS(app)
DB_PATH = os.getenv("CEO_DB_PATH", "ceos.db")

# ── Simple in-memory rate limiter (no extra package needed) ──
_rate_store: dict[str, list[float]] = {}
RATE_LIMIT   = 30      # requests
RATE_WINDOW  = 60.0    # seconds

def rate_limited(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        ip  = request.remote_addr or "unknown"
        now = time.time()
        hits = [t for t in _rate_store.get(ip, []) if now - t < RATE_WINDOW]
        if len(hits) >= RATE_LIMIT:
            return jsonify({"error": "Rate limit exceeded — 30 requests/minute"}), 429
        hits.append(now)
        _rate_store[ip] = hits
        return f(*args, **kwargs)
    return wrapper


# ── DB connection via Flask g (one per request, auto-closed) ──
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


# ── OpenRouter: NL query → structured filters ─────────────────
def parse_query(query: str) -> dict:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("[Warning] OPENROUTER_API_KEY not set — using default filters.")
        return _default_filters()

    # FIX: cap query length to prevent prompt injection / cost blowout
    query = query[:500]

    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        response = client.chat.completions.create(
            # FIX: valid free OpenRouter model 
            model="openai/gpt-oss-120b:free",
            extra_headers={
                "HTTP-Referer": os.getenv("APP_URL", "http://localhost:5000"),
                "X-Title": "CEO Finder Bot",
            },
            messages=[
                {
                    "role": "system",
                    "content": """
You extract search intent from a natural language CEO search query.
Return ONLY a valid JSON object with these exact keys:
{
  "city":         string or "",
  "industry":     string or "",
  "ideal_size":   integer or null,
  "min_degree":   integer 1-3 or 3,
  "weights": {
    "geo":     float,
    "ind":     float,
    "size":    float,
    "network": float
  }
}

Weight inference rules:
- City mentioned strongly → geo weight higher (0.45+)
- Industry/sector mentioned → ind weight higher (0.40+)
- Company size mentioned → size weight higher (0.35+)
- Weights must sum to 1.0. Default: geo=0.35, ind=0.30, size=0.25, network=0.10
- ideal_size: startup=50, small=200, mid-market=1000, large=5000, enterprise=20000
No explanation. No markdown. Raw JSON only.
"""
                },
                {"role": "user", "content": query},
            ],
            temperature=0,
            max_tokens=200,
        )
        # FIX: was response.choices.message (AttributeError) — choices is a list
        raw = response.choices[0].message.content.strip()

        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            parsed = json.loads(m.group())
            w = parsed.get("weights", {})
            total = sum(w.values())
            if total > 0:
                parsed["weights"] = {k: round(v / total, 4) for k, v in w.items()}
            return {**_default_filters(), **parsed}

    except Exception as e:
        print(f"[OpenRouter] parse error: {e}")

    return _default_filters()


def _default_filters() -> dict:
    return {
        "city":       "",
        "industry":   "",
        "ideal_size": None,
        "min_degree": 3,
        "weights":    {"geo": 0.35, "ind": 0.30, "size": 0.25, "network": 0.10},
    }


# ── DB fetch ──────────────────────────────────────────────────
def fetch_candidates(filters: dict) -> list[dict]:
    degree_limit = int(filters.get("min_degree", 3))
    db   = get_db()
    rows = db.execute(
        """
        SELECT id, name, company, city, industry,
               company_size, revenue, degree,
               bio_snippet, source, base_quality_score
        FROM   ceos
        WHERE  degree <= ?
          AND  name IS NOT NULL
        ORDER  BY base_quality_score DESC
        LIMIT  500
        """,
        (degree_limit,),
    ).fetchall()
    return [dict(r) for r in rows]


# ── Routes ────────────────────────────────────────────────────

# FIX: only ONE route on "/" (removed duplicate def index())
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/search")
@rate_limited
def search():
    query   = (request.args.get("q", "").strip() or "CEO USA")[:500]
    filters = parse_query(query)

    candidates = fetch_candidates(filters)
    scored = compute_expert_match_batch(
        rows            = candidates,
        target_city     = filters.get("city")       or None,
        target_industry = filters.get("industry")   or None,
        ideal_size      = filters.get("ideal_size") or None,
        weights         = filters.get("weights"),
        min_score       = 0.20,
    )[:50]

    output = [
        {
            "name":         r["name"],
            "company":      r.get("company", ""),
            "city":         r.get("city") or "",
            "industry":     r.get("industry", "General"),
            "company_size": r.get("company_size") or 0,
            "degree":       r.get("degree") or 3,
            "match_score":  r["match_pct"],
            "revenue":      r.get("revenue", ""),
            "bio_snippet":  r.get("bio_snippet", ""),
            "source":       r.get("source", ""),
            "dims": {
                "geo":      round(r["dim_geo"]      * 100, 1),
                "industry": round(r["dim_industry"] * 100, 1),
                "size":     round(r["dim_size"] * 100, 1) if r["dim_size"] is not None else 0,
                "network":  round(r["dim_network"]  * 100, 1),
            },
        }
        for r in scored
    ]
    return jsonify(output)


@app.route("/api/stats")
def stats():
    db    = get_db()
    # FIX: was fetchone() returning tuple — now [0] extracts the scalar
    total = db.execute("SELECT COUNT(*) FROM ceos").fetchone()[0]
    avg_q = db.execute("SELECT ROUND(AVG(base_quality_score),3) FROM ceos").fetchone()[0]
    by_src = db.execute(
        "SELECT source, COUNT(*) c FROM ceos GROUP BY source ORDER BY c DESC"
    ).fetchall()
    by_city = db.execute(
        "SELECT city, COUNT(*) c FROM ceos WHERE city IS NOT NULL"
        " GROUP BY city ORDER BY c DESC LIMIT 8"
    ).fetchall()
    return jsonify({
        "total_records":  total,
        "avg_quality":    avg_q,
        "by_source":      {r[0]: r[1] for r in by_src},
        "top_cities":     {r[0]: r[1] for r in by_city},
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
