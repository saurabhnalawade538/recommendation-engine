# 🎯 RecoAI — Product Recommendation Engine

A machine learning-powered recommendation engine using **Collaborative Filtering + Content-Based** hybrid approach.

## 📋 Problem Statement
Build a recommendation engine that uses customer browsing and purchase data to suggest relevant products.

## 🛠️ Tech Stack
- **Python** — Core language
- **Flask** — Web server & REST API
- **Pandas** — Data manipulation & interaction matrices
- **Scikit-learn** — Cosine similarity for collaborative filtering
- **Vanilla JS** — Frontend interactivity

## 🧠 ML Algorithm: Hybrid Filtering

### 1. Collaborative Filtering (User-Based)
- Builds a **customer × product interaction matrix**
- Computes **cosine similarity** between customers
- Finds top-3 similar customers → aggregates their ratings (weighted)
- Recommends products the current user hasn't seen yet

### 2. Content-Based Filtering
- Scores products by: **category match (50%) + price proximity (30%) + rating (20%)**
- Triggered when a user clicks on a product (context-aware)

### 3. Hybrid Score
```
hybrid_score = 0.6 × CF_rank_score + 0.4 × CB_rank_score
```

## 🚀 Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the Flask server
python app.py

# 3. Open browser
http://localhost:5000
```

## 🎮 How to Use

1. **Select a Customer** from the left panel (5 profiles with different segments)
2. See **personalized recommendations** appear instantly
3. **Click any product** in the catalog to refine recommendations (hybrid kicks in)
4. View the customer's **Purchase History** strip
5. Filter catalog by **category tabs**

## 📊 Data

- **5 customers** with distinct segments (Tech Enthusiast, Fitness Lover, etc.)
- **20 products** across 5 categories
- **Synthetic interaction matrix** simulating real browsing/purchase behavior
- All prices displayed in INR (₹)

## 📁 Project Structure

```
recommendation-engine/
├── app.py              # Flask app + ML engine
├── requirements.txt    # Dependencies
├── README.md           # This file
└── templates/
    └── index.html      # Frontend UI
```
