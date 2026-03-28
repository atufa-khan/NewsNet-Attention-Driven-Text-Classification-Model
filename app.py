import gradio as gr
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from scipy.spatial.distance import cdist
import warnings
import os

warnings.filterwarnings("ignore")

# ── Global state ──────────────────────────────────────────────────────────────
model = None
embedding_matrix = None
kmeans = None
data = None
clustered_data = None

CATEGORY_DICT = {
    0: "Sports",
    1: "Politics",
    2: "Entertainment",
    3: "Business",
    4: "Technology",
}

CATEGORY_EMOJI = {
    "Sports": "⚽",
    "Politics": "🏛️",
    "Entertainment": "🎬",
    "Business": "💼",
    "Technology": "💻",
}

# ── Load model once ───────────────────────────────────────────────────────────
def load_model():
    global model
    if model is None:
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return model


# ── Train / load pipeline ─────────────────────────────────────────────────────
def train_pipeline(articles_file, labels_file):
    global model, embedding_matrix, kmeans, data, clustered_data

    if articles_file is None:
        return "❌ Please upload the **news_articles.csv** file.", None, None

    # Load data
    data = pd.read_csv(articles_file.name)
    data = data.drop_duplicates().reset_index(drop=True)

    # Load model & embed
    load_model()
    embedding_matrix = model.encode(
        data["Text"].tolist(), show_progress_bar=False, batch_size=64
    )

    # K-Means with k=5
    kmeans = KMeans(n_clusters=5, random_state=1)
    kmeans.fit(embedding_matrix)

    clustered_data = data.copy()
    clustered_data["Category"] = [CATEGORY_DICT[l] for l in kmeans.labels_]

    # Category distribution
    dist = clustered_data["Category"].value_counts().reset_index()
    dist.columns = ["Category", "Count"]
    dist["Emoji"] = dist["Category"].map(CATEGORY_EMOJI)
    dist["Category"] = dist["Emoji"] + "  " + dist["Category"]

    # Classification report if labels provided
    report_html = ""
    if labels_file is not None:
        labels_df = pd.read_csv(labels_file.name)
        clustered_data["Actual Category"] = labels_df["Label"].values
        report_txt = classification_report(
            clustered_data["Actual Category"], clustered_data["Category"]
        )
        report_html = f"""
<div style='background:#0f172a;color:#e2e8f0;padding:16px 20px;border-radius:10px;
font-family:monospace;font-size:13px;white-space:pre;overflow-x:auto;
border:1px solid #334155'>
<span style='color:#94a3b8'>Classification Report</span>
{report_txt}
</div>"""

    summary = f"""
<div style='display:flex;gap:12px;flex-wrap:wrap;margin-bottom:8px'>
  <div style='background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px 20px;flex:1;min-width:140px'>
    <div style='color:#64748b;font-size:12px;font-family:sans-serif'>ARTICLES</div>
    <div style='color:#f1f5f9;font-size:28px;font-weight:700;font-family:monospace'>{len(data):,}</div>
  </div>
  <div style='background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px 20px;flex:1;min-width:140px'>
    <div style='color:#64748b;font-size:12px;font-family:sans-serif'>CLUSTERS</div>
    <div style='color:#f1f5f9;font-size:28px;font-weight:700;font-family:monospace'>5</div>
  </div>
  <div style='background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px 20px;flex:1;min-width:140px'>
    <div style='color:#64748b;font-size:12px;font-family:sans-serif'>DIMENSIONS</div>
    <div style='color:#f1f5f9;font-size:28px;font-weight:700;font-family:monospace'>384</div>
  </div>
</div>
{report_html}
"""
    return summary, dist, clustered_data[["Text", "Category"]]


# ── Semantic Search ───────────────────────────────────────────────────────────
def semantic_search(query, top_k):
    if model is None or embedding_matrix is None:
        return "⚠️ Please train the model first (go to the **Setup** tab)."
    if not query.strip():
        return "⚠️ Please enter a search query."

    query_embedding = model.encode(query)
    scores = np.dot(embedding_matrix, query_embedding)
    top_indices = np.argsort(scores)[::-1][:int(top_k)]

    rows = []
    for rank, idx in enumerate(top_indices, 1):
        cat = clustered_data.loc[idx, "Category"]
        emoji = CATEGORY_EMOJI.get(cat, "📰")
        score = float(scores[idx])
        text = data.loc[idx, "Text"]
        preview = text[:300] + "..." if len(text) > 300 else text
        rows.append(
            f"""<div style='background:#1e293b;border:1px solid #334155;border-radius:10px;
padding:16px;margin-bottom:10px'>
  <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px'>
    <span style='background:#0f172a;color:#7dd3fc;font-size:12px;font-weight:700;
padding:3px 10px;border-radius:20px;font-family:monospace'>#{rank}</span>
    <span style='color:#94a3b8;font-size:12px;font-family:sans-serif'>{emoji} {cat} &nbsp;·&nbsp;
<span style='color:#4ade80'>score: {score:.3f}</span></span>
  </div>
  <p style='color:#cbd5e1;font-size:13px;line-height:1.6;margin:0;font-family:sans-serif'>{preview}</p>
</div>"""
        )

    return "".join(rows)


# ── Classify single article ───────────────────────────────────────────────────
def classify_article(text):
    if model is None or kmeans is None:
        return "⚠️ Please train the model first (go to the **Setup** tab)."
    if not text.strip():
        return "⚠️ Please enter article text."

    emb = model.encode(text).reshape(1, -1)
    label = kmeans.predict(emb)[0]
    category = CATEGORY_DICT[label]
    emoji = CATEGORY_EMOJI[category]

    distances = cdist(emb, kmeans.cluster_centers_, "euclidean")[0]
    total = distances.sum()
    confidences = [(1 - d / total) for d in distances]
    norm = sum(confidences)
    confidences = [c / norm for c in confidences]

    bars = ""
    for i, (cat, conf) in enumerate(zip(CATEGORY_DICT.values(), confidences)):
        em = CATEGORY_EMOJI[cat]
        pct = conf * 100
        color = "#7dd3fc" if i == label else "#334155"
        bars += f"""
<div style='margin-bottom:8px'>
  <div style='display:flex;justify-content:space-between;color:#94a3b8;
font-size:12px;font-family:sans-serif;margin-bottom:3px'>
    <span>{em} {cat}</span><span>{pct:.1f}%</span>
  </div>
  <div style='background:#0f172a;border-radius:4px;height:8px;overflow:hidden'>
    <div style='background:{color};width:{pct}%;height:100%;border-radius:4px;
transition:width 0.5s ease'></div>
  </div>
</div>"""

    return f"""
<div style='background:#1e293b;border:1px solid #334155;border-radius:12px;padding:20px'>
  <div style='text-align:center;margin-bottom:16px'>
    <div style='font-size:48px'>{emoji}</div>
    <div style='color:#f1f5f9;font-size:24px;font-weight:700;font-family:sans-serif'>{category}</div>
    <div style='color:#64748b;font-size:13px;font-family:sans-serif'>Predicted Category</div>
  </div>
  <hr style='border-color:#334155;margin:16px 0'>
  <div style='color:#94a3b8;font-size:12px;font-family:sans-serif;margin-bottom:10px;
text-transform:uppercase;letter-spacing:1px'>Similarity to all clusters</div>
  {bars}
</div>"""


# ── Build UI ──────────────────────────────────────────────────────────────────
css = """
body, .gradio-container { background: #020817 !important; }
.gradio-container { max-width: 960px !important; margin: 0 auto; }
h1, h2, h3, label, .label-wrap { color: #e2e8f0 !important; }
.tab-nav button { color: #94a3b8 !important; background: #0f172a !important;
  border: 1px solid #1e293b !important; border-radius: 8px !important; font-family: monospace; }
.tab-nav button.selected { color: #7dd3fc !important; border-color: #7dd3fc !important; }
.block { background: #0f172a !important; border: 1px solid #1e293b !important; border-radius: 12px !important; }
textarea, input[type=text], input[type=number] {
  background: #1e293b !important; color: #e2e8f0 !important;
  border: 1px solid #334155 !important; border-radius: 8px !important; }
button.primary { background: #0ea5e9 !important; border: none !important;
  color: #0f172a !important; font-weight: 700 !important; border-radius: 8px !important; }
button.secondary { background: #1e293b !important; border: 1px solid #334155 !important;
  color: #e2e8f0 !important; border-radius: 8px !important; }
.upload-button { background: #1e293b !important; border: 1px dashed #334155 !important; }
footer { display: none !important; }
"""

with gr.Blocks(css=css, title="📰 E-news Express · Article Classifier") as demo:
    gr.HTML("""
<div style='text-align:center;padding:32px 0 16px'>
  <div style='font-size:13px;letter-spacing:4px;color:#0ea5e9;font-family:monospace;
text-transform:uppercase;margin-bottom:8px'>E-news Express</div>
  <h1 style='font-size:36px;font-weight:800;color:#f1f5f9;font-family:sans-serif;margin:0'>
    News Article Classifier</h1>
  <p style='color:#64748b;font-size:15px;margin-top:10px;font-family:sans-serif'>
    Transformer embeddings · K-Means clustering · Semantic search</p>
</div>""")

    with gr.Tabs():
        # ── Tab 1: Setup ──────────────────────────────────────────────────────
        with gr.Tab("⚙️  Setup & Train"):
            gr.HTML("<p style='color:#64748b;font-size:13px;font-family:sans-serif'>"
                    "Upload your CSV files and click <b>Train</b> to generate embeddings and cluster the articles.</p>")
            with gr.Row():
                articles_file = gr.File(label="📂  news_articles.csv", file_types=[".csv"])
                labels_file   = gr.File(label="🏷️  news_article_labels.csv (optional)", file_types=[".csv"])
            train_btn  = gr.Button("🚀  Train Model", variant="primary")
            status_out = gr.HTML(label="Status")
            with gr.Row():
                dist_table = gr.Dataframe(label="Category Distribution", interactive=False)
            data_table = gr.Dataframe(label="Clustered Articles (preview)", interactive=False,
                                       max_rows=20)
            train_btn.click(
                fn=train_pipeline,
                inputs=[articles_file, labels_file],
                outputs=[status_out, dist_table, data_table],
            )

        # ── Tab 2: Semantic Search ────────────────────────────────────────────
        with gr.Tab("🔍  Semantic Search"):
            gr.HTML("<p style='color:#64748b;font-size:13px;font-family:sans-serif'>"
                    "Find the most relevant articles for any query using cosine similarity on embeddings.</p>")
            with gr.Row():
                query_box = gr.Textbox(placeholder="e.g. Budget for elections, Tech layoffs, Champions League...",
                                       label="Search Query", lines=2, scale=4)
                top_k_slider = gr.Slider(1, 10, value=3, step=1, label="Top K results", scale=1)
            search_btn     = gr.Button("Search", variant="primary")
            search_results = gr.HTML()
            search_btn.click(fn=semantic_search, inputs=[query_box, top_k_slider],
                             outputs=search_results)

        # ── Tab 3: Classify Article ───────────────────────────────────────────
        with gr.Tab("🗂️  Classify Article"):
            gr.HTML("<p style='color:#64748b;font-size:13px;font-family:sans-serif'>"
                    "Paste any news article text and the model will predict its category.</p>")
            article_input = gr.Textbox(placeholder="Paste a news article here...",
                                        label="Article Text", lines=8)
            classify_btn  = gr.Button("Classify", variant="primary")
            classify_out  = gr.HTML()
            classify_btn.click(fn=classify_article, inputs=article_input,
                               outputs=classify_out)

        # ── Tab 4: About ──────────────────────────────────────────────────────
        with gr.Tab("ℹ️  About"):
            gr.HTML("""
<div style='color:#94a3b8;font-family:sans-serif;line-height:1.8;font-size:14px;max-width:700px'>
  <h3 style='color:#e2e8f0'>How it works</h3>
  <ol>
    <li><b style='color:#7dd3fc'>Embedding</b> — Each article is encoded into a 384-dim vector
        using <code>all-MiniLM-L6-v2</code>, a lightweight but powerful transformer model.</li>
    <li><b style='color:#7dd3fc'>Clustering</b> — K-Means (k=5) groups articles by semantic
        similarity in the embedding space.</li>
    <li><b style='color:#7dd3fc'>Semantic Search</b> — Query text is embedded and compared
        against all article vectors using dot-product similarity.</li>
    <li><b style='color:#7dd3fc'>Classification</b> — New articles are assigned to the nearest
        cluster centroid.</li>
  </ol>
  <h3 style='color:#e2e8f0'>Model</h3>
  <p><code>sentence-transformers/all-MiniLM-L6-v2</code><br>
  6-layer MiniLM · 384 dimensions · ~22M parameters · Apache 2.0 license</p>
  <h3 style='color:#e2e8f0'>Categories</h3>
  <p>⚽ Sports &nbsp;·&nbsp; 🏛️ Politics &nbsp;·&nbsp; 🎬 Entertainment
     &nbsp;·&nbsp; 💼 Business &nbsp;·&nbsp; 💻 Technology</p>
</div>""")

if __name__ == "__main__":
    demo.launch()
