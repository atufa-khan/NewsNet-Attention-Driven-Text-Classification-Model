---
title: E-news Express Article Classifier
emoji: 📰
colorFrom: blue
colorTo: cyan
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 📰 E-news Express — News Article Classifier

Unsupervised news article categorization using **Transformer embeddings** + **K-Means clustering**.

## Features
- 🚀 **Train** on your own `news_articles.csv`
- 🔍 **Semantic Search** — find articles similar to any query
- 🗂️ **Classify** new articles into Sports / Politics / Entertainment / Business / Technology
- 📊 **Accuracy report** when labels CSV is provided (~96% accuracy)

## Model
`sentence-transformers/all-MiniLM-L6-v2` — 384-dimensional embeddings, 6-layer MiniLM transformer.

## How to use
1. Go to the **Setup & Train** tab
2. Upload `news_articles.csv` (and optionally `news_article_labels.csv`)
3. Click **Train Model**
4. Use **Semantic Search** or **Classify Article** tabs
