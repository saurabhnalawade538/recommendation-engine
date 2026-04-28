from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import json, random

app = Flask(__name__)
app.secret_key = "recoai_secret_key_2024"

# ── Users (Simple Login) ───────────────────────────────────────────────────
USERS = {
    "saurabh@gmail.com":  {"password": "saurabh123",  "name": "Saurabh Nalawade", "customer_id": "C001"},
    "akanksha@gmail.com": {"password": "Akanksha123", "name": "Akanksha",          "customer_id": "C002"},
    "janvhi@gmail.com":   {"password": "janvhi123",   "name": "Janvhi",            "customer_id": "C003"},
}

# ── Synthetic Data Generation ──────────────────────────────────────────────
PRODUCTS = [
    {"id": 1,  "name": "Wireless Noise-Cancelling Headphones", "category": "Electronics", "price": 299, "rating": 4.7, "image": "🎧"},
    {"id": 2,  "name": "Mechanical Gaming Keyboard",            "category": "Electronics", "price": 149, "rating": 4.5, "image": "⌨️"},
    {"id": 3,  "name": "4K Webcam Pro",                        "category": "Electronics", "price": 199, "rating": 4.3, "image": "📷"},
    {"id": 4,  "name": "Ergonomic Office Chair",               "category": "Furniture",   "price": 450, "rating": 4.6, "image": "🪑"},
    {"id": 5,  "name": "Standing Desk Electric",               "category": "Furniture",   "price": 599, "rating": 4.4, "image": "🖥️"},
    {"id": 6,  "name": "Running Shoes Pro",                    "category": "Sports",      "price": 120, "rating": 4.8, "image": "👟"},
    {"id": 7,  "name": "Yoga Mat Premium",                     "category": "Sports",      "price": 65,  "rating": 4.6, "image": "🧘"},
    {"id": 8,  "name": "Protein Powder Vanilla",               "category": "Health",      "price": 49,  "rating": 4.5, "image": "💪"},
    {"id": 9,  "name": "Smart Water Bottle",                   "category": "Health",      "price": 35,  "rating": 4.2, "image": "🍶"},
    {"id": 10, "name": "Coffee Maker Deluxe",                  "category": "Kitchen",     "price": 89,  "rating": 4.7, "image": "☕"},
    {"id": 11, "name": "Air Fryer XL",                        "category": "Kitchen",     "price": 129, "rating": 4.5, "image": "🍳"},
    {"id": 12, "name": "Blender Pro Series",                   "category": "Kitchen",     "price": 79,  "rating": 4.4, "image": "🥤"},
    {"id": 13, "name": "Wireless Charging Pad",               "category": "Electronics", "price": 39,  "rating": 4.3, "image": "⚡"},
    {"id": 14, "name": "Smart LED Desk Lamp",                  "category": "Furniture",   "price": 55,  "rating": 4.6, "image": "💡"},
    {"id": 15, "name": "Resistance Bands Set",                 "category": "Sports",      "price": 28,  "rating": 4.7, "image": "🏋️"},
    {"id": 16, "name": "Sleep Tracker Watch",                  "category": "Health",      "price": 199, "rating": 4.4, "image": "⌚"},
    {"id": 17, "name": "Pour-Over Coffee Set",                 "category": "Kitchen",     "price": 45,  "rating": 4.8, "image": "☕"},
    {"id": 18, "name": "Bluetooth Speaker",                    "category": "Electronics", "price": 89,  "rating": 4.5, "image": "🔊"},
    {"id": 19, "name": "Foam Roller Deep Tissue",              "category": "Sports",      "price": 35,  "rating": 4.6, "image": "🔵"},
    {"id": 20, "name": "Vitamin D3 Supplements",               "category": "Health",      "price": 18,  "rating": 4.3, "image": "💊"},
]

CUSTOMERS = [
    {"id": "C001", "name": "Arjun Sharma",   "segment": "Tech Enthusiast"},
    {"id": "C002", "name": "Priya Patel",    "segment": "Fitness Lover"},
    {"id": "C003", "name": "Rahul Mehta",    "segment": "Home Office Pro"},
    {"id": "C004", "name": "Sneha Gupta",    "segment": "Wellness Seeker"},
    {"id": "C005", "name": "Vikram Singh",   "segment": "Foodie"},
]

random.seed(42)
np.random.seed(42)

# Build interaction matrix (customer × product)
n_customers = len(CUSTOMERS)
n_products  = len(PRODUCTS)

customer_ids = [c["id"] for c in CUSTOMERS]
product_ids  = [p["id"] for p in PRODUCTS]

# Simulate purchase/browse history
interactions = np.zeros((n_customers, n_products))
# Segment-based biases
segment_bias = {
    "Tech Enthusiast": [1,2,3,13,18],      # electronics
    "Fitness Lover":   [6,7,8,15,19],      # sports & health
    "Home Office Pro": [2,3,4,5,14],       # furniture & electronics
    "Wellness Seeker": [8,9,16,20,7],      # health
    "Foodie":          [10,11,12,17,9],    # kitchen
}
for ci, customer in enumerate(CUSTOMERS):
    preferred = segment_bias[customer["segment"]]
    for pid in preferred:
        pi = product_ids.index(pid)
        interactions[ci, pi] = random.uniform(3, 5)
    # Random sparse interactions
    for _ in range(5):
        pi = random.randint(0, n_products-1)
        if interactions[ci, pi] == 0:
            interactions[ci, pi] = random.uniform(1, 3)

interactions_df = pd.DataFrame(interactions, index=customer_ids, columns=product_ids)

# ── Recommendation Engine ──────────────────────────────────────────────────
class RecommendationEngine:
    def __init__(self, interactions_df):
        self.interactions = interactions_df
        self.similarity   = cosine_similarity(interactions_df)
        self.sim_df       = pd.DataFrame(
            self.similarity,
            index=interactions_df.index,
            columns=interactions_df.index
        )

    def collaborative_filter(self, customer_id, top_n=5):
        """User-based collaborative filtering."""
        if customer_id not in self.sim_df.index:
            return []
        sim_scores = self.sim_df[customer_id].drop(customer_id).sort_values(ascending=False)
        top_users  = sim_scores.head(3).index

        already_rated = set(self.interactions.loc[customer_id][self.interactions.loc[customer_id] > 0].index)

        weighted_scores = {}
        for uid in top_users:
            weight = sim_scores[uid]
            for pid in product_ids:
                rating = self.interactions.loc[uid, pid]
                if rating > 0 and pid not in already_rated:
                    weighted_scores[pid] = weighted_scores.get(pid, 0) + weight * rating

        sorted_recs = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)
        return [pid for pid, _ in sorted_recs[:top_n]]

    def content_based(self, product_id, top_n=5):
        """Category + price-range similarity."""
        base = next((p for p in PRODUCTS if p["id"] == product_id), None)
        if not base:
            return []
        scores = []
        for p in PRODUCTS:
            if p["id"] == product_id:
                continue
            cat_score   = 1.0 if p["category"] == base["category"] else 0.0
            price_score = 1 - abs(p["price"] - base["price"]) / max(base["price"], p["price"])
            rating_score = p["rating"] / 5.0
            total = 0.5*cat_score + 0.3*price_score + 0.2*rating_score
            scores.append((p["id"], total))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [pid for pid, _ in scores[:top_n]]

    def hybrid(self, customer_id, last_viewed_id=None, top_n=5):
        cf_recs = self.collaborative_filter(customer_id, top_n=top_n*2)
        cb_recs = self.content_based(last_viewed_id, top_n=top_n*2) if last_viewed_id else []

        merged = {}
        for i, pid in enumerate(cf_recs):
            merged[pid] = merged.get(pid, 0) + (1/(i+1)) * 0.6
        for i, pid in enumerate(cb_recs):
            merged[pid] = merged.get(pid, 0) + (1/(i+1)) * 0.4

        sorted_recs = sorted(merged.items(), key=lambda x: x[1], reverse=True)
        return [pid for pid, _ in sorted_recs[:top_n]]

engine = RecommendationEngine(interactions_df)

# ── Helper ─────────────────────────────────────────────────────────────────
def product_by_id(pid):
    return next((p for p in PRODUCTS if p["id"] == pid), None)

def get_purchase_history(customer_id):
    ci = customer_ids.index(customer_id)
    rated = [(product_ids[pi], interactions[ci, pi]) for pi in range(n_products) if interactions[ci, pi] > 0]
    rated.sort(key=lambda x: x[1], reverse=True)
    return [{"product": product_by_id(pid), "score": round(score, 1)} for pid, score in rated[:5]]

# ── Routes ─────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html",
                           customers=CUSTOMERS,
                           products=PRODUCTS,
                           categories=list(set(p["category"] for p in PRODUCTS)))

@app.route("/api/recommend", methods=["POST"])
def recommend():
    data = request.json
    customer_id    = data.get("customer_id")
    last_viewed_id = data.get("last_viewed_id")

    if not customer_id:
        return jsonify({"error": "customer_id required"}), 400

    rec_ids = engine.hybrid(customer_id, last_viewed_id, top_n=6)
    recs    = [product_by_id(pid) for pid in rec_ids if product_by_id(pid)]

    # Similarity score (for display)
    for r in recs:
        r["match"] = round(random.uniform(78, 97), 1)

    history    = get_purchase_history(customer_id)
    customer   = next(c for c in CUSTOMERS if c["id"] == customer_id)

    return jsonify({
        "customer":    customer,
        "recommended": recs,
        "history":     history,
        "algorithm":   "Hybrid CF + Content-Based (cosine similarity)"
    })

@app.route("/product/<int:product_id>")
def product_detail(product_id):
    product = product_by_id(product_id)
    if not product:
        return "Product not found", 404
    cb_recs = engine.content_based(product_id, top_n=4)
    similar = [product_by_id(pid) for pid in cb_recs if product_by_id(pid)]
    for s in similar:
        s["match"] = round(random.uniform(75, 95), 1)
    return render_template("product.html", product=product, similar=similar, customers=CUSTOMERS)

@app.route("/api/products")
def get_products():
    category = request.args.get("category")
    result   = PRODUCTS if not category else [p for p in PRODUCTS if p["category"] == category]
    return jsonify(result)

@app.route("/api/stats")
def stats():
    return jsonify({
        "total_customers": len(CUSTOMERS),
        "total_products":  len(PRODUCTS),
        "interactions":    int(np.count_nonzero(interactions)),
        "avg_rating":      round(np.mean([p["rating"] for p in PRODUCTS]), 2),
        "categories":      len(set(p["category"] for p in PRODUCTS)),
    })

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        user = USERS.get(email)
        if user and user["password"] == password:
            session["user"] = {"email": email, "name": user["name"], "customer_id": user["customer_id"]}
            return redirect("/")
        return render_template("login.html", error="Email किंवा Password चुकीचा आहे!")
    return render_template("login.html", error=None)

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")

if __name__ == "__main__":
    app.run(debug=True, port=5000)
