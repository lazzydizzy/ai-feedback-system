# ============================================================
#  app.py  —  AI Customer Feedback System (Enhanced)
#  Features: Sentiment Analysis (VADER), Keyword Extraction,
#            Topic Classification, Emotion Detection,
#            Dashboard, Trend Analysis, Search/Filter,
#            Auto-Summary Generation
# ============================================================

import os, re, json, math
from datetime import datetime, timedelta
from collections import Counter
from flask import Flask, render_template, request, jsonify, redirect, url_for
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import sqlite3

# ── App Setup ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "database", "feedback.db")

app = Flask(__name__)
analyzer = SentimentIntensityAnalyzer()   # VADER — kept exactly as before


# ── Database Helpers ───────────────────────────────────────

def get_db():
    """Open a SQLite connection and return (conn, cursor)."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn, conn.cursor()


def init_db():
    """Create the feedback table if it doesn't exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn, cur = get_db()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            text        TEXT    NOT NULL,
            sentiment   TEXT    NOT NULL,
            compound    REAL    NOT NULL,
            pos         REAL    NOT NULL,
            neg         REAL    NOT NULL,
            neu         REAL    NOT NULL,
            category    TEXT    NOT NULL,
            emotion     TEXT    NOT NULL,
            keywords    TEXT    NOT NULL,   -- JSON list
            timestamp   TEXT    NOT NULL
        )
    """)
    conn.commit()
    conn.close()


# ── Sentiment (ORIGINAL — UNCHANGED) ───────────────────────

def classify_sentiment(compound_score):
    """
    VADER thresholds (unchanged from original):
      compound >= 0.05  → Positive
      compound <= -0.05 → Negative
      otherwise         → Neutral
    """
    if compound_score >= 0.05:
        return "Positive"
    elif compound_score <= -0.05:
        return "Negative"
    return "Neutral"


# ── Keyword Extraction (TF-IDF-style, no extra deps) ───────

STOP_WORDS = {
    "i","me","my","we","our","you","your","he","she","it","they","them",
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "is","was","are","were","be","been","have","has","had","do","did","will",
    "would","could","should","that","this","these","those","not","no","very",
    "so","just","about","from","as","by","if","then","than","its","am","been",
    "get","got","got","also","more","much","there","their","than","when","how",
    "what","which","who","all","some","any","over","up","out","into","after",
    "before","too","only","other","each","even","both","here","now"
}

ISSUE_PHRASES = [
    "delivery delay", "late delivery", "slow delivery", "shipping delay",
    "poor service", "bad service", "rude staff", "unhelpful",
    "bad packaging", "damaged package", "broken item", "poor quality",
    "wrong item", "missing item", "refund issue", "payment problem",
    "overpriced", "too expensive", "high price",
    "easy to use", "great quality", "fast delivery", "excellent service",
]

def extract_keywords(text, top_n=8):
    """
    Simple TF-IDF-inspired keyword extractor.
    1. Tokenise and lower-case
    2. Remove stop words
    3. Count frequency
    4. First check for known issue phrases (multi-word)
    5. Return top_n terms by frequency
    """
    lower_text = text.lower()

    # Check multi-word issue phrases first
    found_phrases = [p for p in ISSUE_PHRASES if p in lower_text]

    # Tokenise to single words
    words = re.findall(r"[a-z]{3,}", lower_text)
    words = [w for w in words if w not in STOP_WORDS]

    freq = Counter(words)
    top_words = [w for w, _ in freq.most_common(top_n)]

    # Merge phrases + single words, dedupe
    keywords = list(dict.fromkeys(found_phrases + top_words))[:top_n]
    return keywords


# ── Topic / Category Classification ────────────────────────

CATEGORY_RULES = {
    "Delivery": [
        "delivery","shipping","ship","courier","dispatch","arrived","late",
        "delayed","delay","transit","tracking","package","parcel","postage"
    ],
    "Service": [
        "service","support","staff","agent","representative","help","helpdesk",
        "customer service","rude","polite","response","wait","hold","resolve"
    ],
    "Pricing": [
        "price","pricing","cost","expensive","cheap","discount","refund","charge",
        "overpriced","value","money","payment","billing","invoice","fee","worth"
    ],
    "Product": [
        "product","quality","item","material","build","design","feature","broken",
        "defective","excellent","good","bad","durable","works","faulty","broken"
    ],
}

def classify_category(text):
    """Rule-based category classifier — scores each category by keyword hits."""
    lower = text.lower()
    scores = {cat: 0 for cat in CATEGORY_RULES}
    for cat, keywords in CATEGORY_RULES.items():
        for kw in keywords:
            if kw in lower:
                scores[cat] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "General"


# ── Emotion Detection ───────────────────────────────────────

EMOTION_RULES = {
    "Happy":        ["great","amazing","love","excellent","fantastic","wonderful",
                     "happy","joy","delighted","thrilled","pleased","awesome"],
    "Satisfied":    ["good","satisfied","fine","okay","decent","acceptable",
                     "works","reliable","solid","nice","thank"],
    "Frustrated":   ["frustrated","annoying","annoyed","irritating","useless",
                     "terrible","horrible","ridiculous","mess","waste","fail"],
    "Angry":        ["angry","furious","outraged","unacceptable","disgusting",
                     "worst","never again","demand","appalling","hate","awful"],
    "Disappointed": ["disappointed","expected","unfortunately","let down","sad",
                     "poor","not good","below","missing","wrong","broken"],
}

def detect_emotion(text, compound):
    """
    Rule-based emotion detector.
    Falls back to sentiment-based defaults if no strong keyword match.
    """
    lower = text.lower()
    scores = {e: 0 for e in EMOTION_RULES}
    for emotion, keywords in EMOTION_RULES.items():
        for kw in keywords:
            if kw in lower:
                scores[emotion] += 1
    best_emotion = max(scores, key=scores.get)
    if scores[best_emotion] > 0:
        return best_emotion
    # Fallback based on VADER compound
    if compound >= 0.5:   return "Happy"
    if compound >= 0.05:  return "Satisfied"
    if compound <= -0.5:  return "Angry"
    if compound <= -0.05: return "Disappointed"
    return "Satisfied"


# ── Feedback Summary (NLP fallback — no external API) ──────

def generate_summary(rows):
    """
    Produce a one-paragraph summary from all stored feedback rows.
    Uses simple heuristics: sentiment ratios + top category + top emotion.
    """
    if not rows:
        return "No feedback data available yet."

    total = len(rows)
    sentiments = Counter(r["sentiment"] for r in rows)
    categories = Counter(r["category"] for r in rows)
    emotions   = Counter(r["emotion"] for r in rows)

    pos_pct = round(sentiments.get("Positive", 0) / total * 100)
    neg_pct = round(sentiments.get("Negative", 0) / total * 100)
    neu_pct = 100 - pos_pct - neg_pct

    top_cat    = categories.most_common(1)[0][0]
    top_emo    = emotions.most_common(1)[0][0]
    avg_score  = round(sum(r["compound"] for r in rows) / total, 3)

    # Build summary sentence
    if pos_pct >= 60:
        tone = f"Overall customer sentiment is strongly positive ({pos_pct}% positive feedback)."
    elif neg_pct >= 50:
        tone = f"A majority of customers expressed negative sentiment ({neg_pct}% negative feedback)."
    elif neu_pct >= 50:
        tone = f"Most feedback is neutral ({neu_pct}%), suggesting customers have mixed impressions."
    else:
        tone = f"Feedback is mixed — {pos_pct}% positive and {neg_pct}% negative across {total} entries."

    cat_note = f"The most discussed topic is **{top_cat}**."
    emo_note = f"The dominant emotion detected is **{top_emo}**."
    score_note = f"Average sentiment compound score: {avg_score}."

    return f"{tone} {cat_note} {emo_note} {score_note}"


# ── Routes ─────────────────────────────────────────────────

@app.route("/", methods=["GET", "POST"])
def index():
    """
    Main route — handles feedback submission (POST) and display (GET).
    ORIGINAL behaviour preserved exactly; new fields added to result dict.
    """
    result = None

    if request.method == "POST":
        feedback_text = request.form.get("feedback", "").strip()

        if feedback_text:
            # ── 1. VADER Sentiment (ORIGINAL — UNCHANGED) ──
            scores   = analyzer.polarity_scores(feedback_text)
            compound = scores["compound"]
            label    = classify_sentiment(compound)

            # ── 2. Keyword Extraction ──────────────────────
            keywords = extract_keywords(feedback_text)

            # ── 3. Topic Classification ────────────────────
            category = classify_category(feedback_text)

            # ── 4. Emotion Detection ───────────────────────
            emotion = detect_emotion(feedback_text, compound)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # ── Save to DB ─────────────────────────────────
            conn, cur = get_db()
            cur.execute("""
                INSERT INTO feedback
                  (text, sentiment, compound, pos, neg, neu,
                   category, emotion, keywords, timestamp)
                VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (
                feedback_text, label, compound,
                scores["pos"], scores["neg"], scores["neu"],
                category, emotion,
                json.dumps(keywords), timestamp
            ))
            conn.commit()
            conn.close()

            result = {
                "feedback":  feedback_text,
                "label":     label,
                "compound":  round(compound, 4),
                "positive":  round(scores["pos"], 4),
                "negative":  round(scores["neg"], 4),
                "neutral":   round(scores["neu"], 4),
                "keywords":  keywords,
                "category":  category,
                "emotion":   emotion,
            }

    return render_template("index.html", result=result)


@app.route("/dashboard")
def dashboard():
    """Analytics dashboard — reads all stored feedback from SQLite."""
    conn, cur = get_db()

    # ── Filter params ──────────────────────────────────────
    sentiment_f = request.args.get("sentiment", "")
    category_f  = request.args.get("category",  "")
    date_f      = request.args.get("date",       "")
    search_f    = request.args.get("search",     "")

    query  = "SELECT * FROM feedback WHERE 1=1"
    params = []
    if sentiment_f: query += " AND sentiment=?"; params.append(sentiment_f)
    if category_f:  query += " AND category=?";  params.append(category_f)
    if date_f:      query += " AND DATE(timestamp)=?"; params.append(date_f)
    if search_f:    query += " AND text LIKE ?"; params.append(f"%{search_f}%")
    query += " ORDER BY timestamp DESC"

    cur.execute(query, params)
    rows = [dict(r) for r in cur.fetchall()]

    # ── All rows for charts (unfiltered) ──────────────────
    cur.execute("SELECT * FROM feedback ORDER BY timestamp")
    all_rows = [dict(r) for r in cur.fetchall()]
    conn.close()

    # ── Summary ───────────────────────────────────────────
    summary = generate_summary(all_rows)

    # ── Sentiment distribution (for pie chart) ────────────
    sent_counts = Counter(r["sentiment"] for r in all_rows)

    # ── Category distribution ─────────────────────────────
    cat_counts = Counter(r["category"] for r in all_rows)

    # ── Keyword frequency (aggregate from all rows) ───────
    all_keywords = []
    for r in all_rows:
        all_keywords.extend(json.loads(r["keywords"]))
    kw_freq = Counter(all_keywords).most_common(10)

    # ── Trend (daily sentiment for last 14 days) ──────────
    today   = datetime.now().date()
    dates   = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(13, -1, -1)]
    trend   = {d: {"Positive": 0, "Negative": 0, "Neutral": 0} for d in dates}
    for r in all_rows:
        day = r["timestamp"][:10]
        if day in trend:
            trend[day][r["sentiment"]] += 1

    trend_labels = dates
    trend_pos    = [trend[d]["Positive"] for d in dates]
    trend_neg    = [trend[d]["Negative"] for d in dates]
    trend_neu    = [trend[d]["Neutral"]  for d in dates]

    # ── Emotion distribution ──────────────────────────────
    emo_counts = Counter(r["emotion"] for r in all_rows)

    return render_template("dashboard.html",
        rows=rows,
        summary=summary,
        sent_counts=dict(sent_counts),
        cat_counts=dict(cat_counts),
        kw_freq=kw_freq,
        trend_labels=json.dumps(trend_labels),
        trend_pos=json.dumps(trend_pos),
        trend_neg=json.dumps(trend_neg),
        trend_neu=json.dumps(trend_neu),
        emo_counts=dict(emo_counts),
        filters={"sentiment": sentiment_f, "category": category_f,
                 "date": date_f, "search": search_f},
        total=len(all_rows),
    )


@app.route("/api/stats")
def api_stats():
    """JSON endpoint for quick stats (optional AJAX use)."""
    conn, cur = get_db()
    cur.execute("SELECT sentiment, category, emotion, compound FROM feedback")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return jsonify({
        "total":      len(rows),
        "sentiment":  dict(Counter(r["sentiment"] for r in rows)),
        "category":   dict(Counter(r["category"]  for r in rows)),
        "emotion":    dict(Counter(r["emotion"]    for r in rows)),
        "avg_score":  round(sum(r["compound"] for r in rows)/len(rows), 4) if rows else 0,
    })


# ── Bootstrap ──────────────────────────────────────────────

if __name__ == "__main__":
    init_db()
    app.run(debug=True, host="127.0.0.1", port=5000)
