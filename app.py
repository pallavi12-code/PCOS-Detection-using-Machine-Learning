"""
╔══════════════════════════════════════════════════════════════════╗
║         PCOS FUSION MODEL — Streamlit Web App                   ║
║         Theme: Pastel Pink · White · Cream                      ║
╚══════════════════════════════════════════════════════════════════╝
Run:  streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import os
import warnings
import tempfile
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PCOS Predict",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────
# CUSTOM CSS — Patel Pink · White · Cream Theme
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root palette ── */
:root {
    --pink-deep:   #C2637A;
    --pink-mid:    #E8A0B0;
    --pink-light:  #F5C6D2;
    --pink-pale:   #FDE8ED;
    --cream:       #FDF6EE;
    --cream-dark:  #F4EBE0;
    --white:       #FFFFFF;
    --text-dark:   #3D2535;
    --text-mid:    #7A4D5F;
    --text-light:  #B08090;
    --success:     #6BBF8C;
    --warning:     #E8A85A;
    --danger:      #D95F72;
    --border:      #F0D8DF;
}

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--cream);
    color: var(--text-dark);
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem; max-width: 1200px; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #FDE8ED 0%, #FAD7E1 40%, #F5C6D2 100%);
    border-radius: 24px;
    padding: 3rem 3.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    border: 1px solid var(--border);
}
.hero::before {
    content: "🌸";
    position: absolute;
    font-size: 12rem;
    opacity: 0.08;
    right: -2rem;
    top: -2rem;
    transform: rotate(-15deg);
}
.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 700;
    color: var(--pink-deep);
    margin: 0 0 0.5rem;
    letter-spacing: -0.5px;
}
.hero p {
    font-size: 1.1rem;
    color: var(--text-mid);
    margin: 0;
    font-weight: 300;
    max-width: 500px;
}
.hero .badge {
    display: inline-block;
    background: var(--pink-deep);
    color: white;
    font-size: 0.75rem;
    font-weight: 500;
    padding: 4px 14px;
    border-radius: 100px;
    margin-bottom: 1rem;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* ── Section cards ── */
.section-card {
    background: var(--white);
    border-radius: 20px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    border: 1px solid var(--border);
    box-shadow: 0 4px 20px rgba(194,99,122,0.06);
}
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    font-weight: 600;
    color: var(--pink-deep);
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Streamlit widget overrides ── */
.stTextInput input, .stNumberInput input, .stSelectbox select {
    border: 1.5px solid var(--pink-light) !important;
    border-radius: 10px !important;
    background: var(--cream) !important;
    color: var(--text-dark) !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: var(--pink-mid) !important;
    box-shadow: 0 0 0 3px rgba(194,99,122,0.12) !important;
}

/* ── Checkbox overrides ── */
.stCheckbox label {
    font-size: 0.95rem !important;
    color: var(--text-dark) !important;
}

/* ── Slider ── */
.stSlider [data-baseweb="slider"] {
    padding-top: 0.5rem;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--pink-deep), #A0485E) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2.5rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 1rem !important;
    letter-spacing: 0.3px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 14px rgba(194,99,122,0.35) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(194,99,122,0.45) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 2px dashed var(--pink-light) !important;
    border-radius: 16px !important;
    background: var(--pink-pale) !important;
    padding: 1.5rem !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--pink-pale) 0%, var(--cream) 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Playfair Display', serif;
    color: var(--pink-deep);
}

/* ── Metric boxes ── */
[data-testid="stMetric"] {
    background: var(--pink-pale);
    border-radius: 14px;
    padding: 1rem;
    border: 1px solid var(--border);
}
[data-testid="stMetricLabel"] { color: var(--text-mid) !important; }
[data-testid="stMetricValue"] { color: var(--pink-deep) !important; font-family: 'Playfair Display', serif !important; }

/* ── Result banner ── */
.result-pcos {
    background: linear-gradient(135deg, #FFE4EA, #FFCCD6);
    border: 2px solid var(--danger);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    text-align: center;
}
.result-no-pcos {
    background: linear-gradient(135deg, #E8F8EF, #D2F0E0);
    border: 2px solid var(--success);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    text-align: center;
}
.result-label {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0;
}
.result-sub {
    font-size: 1rem;
    color: var(--text-mid);
    margin: 0.5rem 0 0;
    font-weight: 300;
}

/* ── Confidence pill ── */
.pill {
    display: inline-block;
    padding: 4px 18px;
    border-radius: 100px;
    font-size: 0.8rem;
    font-weight: 500;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-top: 0.8rem;
}
.pill-high   { background:#D4EDDA; color:#2D6A4F; }
.pill-medium { background:#FFF3CD; color:#856404; }
.pill-low    { background:#F8D7DA; color:#842029; }

/* ── Step numbers ── */
.step-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px; height: 28px;
    border-radius: 50%;
    background: var(--pink-deep);
    color: white;
    font-size: 0.85rem;
    font-weight: 600;
    flex-shrink: 0;
}

/* ── Symptom chip ── */
.chip-active {
    display: inline-block;
    background: var(--pink-light);
    color: var(--pink-deep);
    border-radius: 100px;
    padding: 3px 12px;
    font-size: 0.8rem;
    margin: 3px;
}
.chip-inactive {
    display: inline-block;
    background: var(--cream-dark);
    color: var(--text-light);
    border-radius: 100px;
    padding: 3px 12px;
    font-size: 0.8rem;
    margin: 3px;
}

/* ── Divider ── */
.pink-divider {
    height: 2px;
    background: linear-gradient(90deg, var(--pink-light), transparent);
    border: none;
    margin: 1.5rem 0;
}

/* ── Disclaimer ── */
.disclaimer {
    background: var(--cream-dark);
    border-left: 4px solid var(--pink-mid);
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.5rem;
    font-size: 0.85rem;
    color: var(--text-mid);
    margin-top: 2rem;
}

/* ── Progress bar override ── */
.stProgress > div > div { background-color: var(--pink-deep) !important; border-radius: 100px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# MODEL LOADING (cached)
# ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    """Load models once and cache. Returns None values if paths missing."""
    import joblib, tensorflow as tf

    CLINICAL_MODEL_PATH = "models/pcos_clinical_model.pkl"   # ← update path
    CNN_MODEL_PATH      = "models/pcos_cnn_model.keras"      # ← update path

    bundle    = joblib.load(CLINICAL_MODEL_PATH)
    model_c   = bundle["model"]
    imputer   = bundle["imputer"]
    scaler    = bundle["scaler"]
    selector  = bundle["selector"]
    cols      = bundle["feature_columns"]
    threshold = bundle.get("optimal_threshold", 0.20)
    cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)

    return model_c, imputer, scaler, selector, cols, threshold, cnn_model


# ─────────────────────────────────────────────────────────────────────
# SYMPTOM DEFINITIONS
# ─────────────────────────────────────────────────────────────────────
SYMPTOM_DEFINITIONS = [
    {"key": "irregular_periods",     "label": "Irregular or missed periods",         "weight": 0.25, "icon": "🔄"},
    {"key": "weight_gain",           "label": "Unexplained weight gain",              "weight": 0.10, "icon": "⚖️"},
    {"key": "hair_growth",           "label": "Excess facial / body hair",            "weight": 0.15, "icon": "🪒"},
    {"key": "hair_loss",             "label": "Hair thinning / scalp hair loss",      "weight": 0.10, "icon": "💆"},
    {"key": "acne_pimples",          "label": "Persistent acne or pimples",           "weight": 0.10, "icon": "🔴"},
    {"key": "skin_darkening",        "label": "Skin darkening (neck / armpits)",      "weight": 0.08, "icon": "🎨"},
    {"key": "fatigue",               "label": "Chronic fatigue / low energy",         "weight": 0.07, "icon": "😴"},
    {"key": "mood_changes",          "label": "Mood swings / depression / anxiety",   "weight": 0.05, "icon": "🧠"},
    {"key": "pelvic_pain",           "label": "Pelvic pain or bloating",              "weight": 0.05, "icon": "💊"},
    {"key": "difficulty_conceiving", "label": "Difficulty conceiving / infertility",  "weight": 0.05, "icon": "👶"},
]
SYMPTOM_WEIGHTS   = {s["key"]: s["weight"] for s in SYMPTOM_DEFINITIONS}
MAX_SYMPTOM_SCORE = sum(SYMPTOM_WEIGHTS.values())


# ─────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────
def symptom_score_to_prob(symptom_dict):
    raw  = sum(SYMPTOM_WEIGHTS[k] * symptom_dict.get(k, 0) for k in SYMPTOM_WEIGHTS)
    return round(float(raw / MAX_SYMPTOM_SCORE), 4)


def _match_col(key, cols):
    key = key.lower()
    for c in cols:
        if key in c.lower().split("_"): return c
    for c in cols:
        if key == c.lower():            return c
    for c in cols:
        if key in c.lower():            return c
    return None


def _confidence_label(prob, n_modalities):
    modality_cap = {1: "Low", 2: "Medium", 3: "High"}[n_modalities]
    prob_conf    = ("High"   if prob > 0.70 or prob < 0.30 else
                    "Medium" if prob > 0.55 or prob < 0.45 else "Low")
    order = ["Low", "Medium", "High"]
    return order[min(order.index(modality_cap), order.index(prob_conf))]


def adaptive_weights(prob_s, prob_c=None, prob_u=None):
    has_c = prob_c is not None
    has_u = prob_u is not None
    if not has_c and not has_u: return 1.0, 0.0, 0.0
    if has_c and not has_u:
        return (0.40, 0.60, 0.0) if abs(prob_s-prob_c)<0.20 else (0.30, 0.70, 0.0) if prob_c>prob_s else (0.50, 0.50, 0.0)
    if has_u and not has_c:
        return (0.40, 0.0, 0.60) if abs(prob_s-prob_u)<0.20 else (0.30, 0.0, 0.70) if prob_u>prob_s else (0.50, 0.0, 0.50)
    high_c=prob_c>=0.60; low_c=prob_c<=0.40; high_u=prob_u>=0.60; low_u=prob_u<=0.40
    if   high_c and high_u: return 0.20, 0.40, 0.40
    elif low_c  and low_u:  return 0.20, 0.40, 0.40
    elif high_u and low_c:  return 0.15, 0.20, 0.65
    elif high_c and low_u:  return 0.15, 0.55, 0.30
    else:                   return 0.20, 0.35, 0.45


def predict_clinical_prob(clinical_data, model_c, imputer, scaler, selector, cols):
    df  = pd.DataFrame([clinical_data])[cols]
    X   = imputer.transform(df.values.astype(float))
    X   = scaler.transform(X)
    Xs  = selector.transform(X)
    return float(model_c.predict_proba(Xs)[0][1])


def predict_cnn_prob(img_pil, cnn_model, img_size=(224, 224)):
    img  = img_pil.convert("RGB").resize(img_size)
    arr  = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, 0)
    pred = cnn_model.predict(arr, verbose=0)[0]
    out_units = cnn_model.output_shape[-1]
    if out_units == 1:
        return float(pred[0]), arr
    # auto-detect PCOS index: use index 1 as default (adjust if needed)
    return float(pred[1]), arr


def gradcam_heatmap(cnn_model, cnn_arr, img_size=(224, 224)):
    """Returns a GradCAM heatmap as numpy array."""
    import tensorflow as tf
    last_conv = None
    for layer in reversed(cnn_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer.name
            break
    if last_conv is None:
        return None
    grad_model = tf.keras.models.Model(
        inputs=cnn_model.inputs,
        outputs=[cnn_model.get_layer(last_conv).output, cnn_model.output],
    )
    img_tensor = tf.constant(cnn_arr)
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        conv_out, preds = grad_model(img_tensor)
        out_units = cnn_model.output_shape[-1]
        score = preds[:, 0] if out_units == 1 else preds[:, 1]
    grads   = tape.gradient(score, conv_out)
    pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_out[0] @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    hm_img = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(img_size, Image.LANCZOS)
    return np.array(hm_img) / 255.0


# ─────────────────────────────────────────────────────────────────────
# VISUALIZATION HELPERS
# ─────────────────────────────────────────────────────────────────────
PINK = "#C2637A"; PINK_LIGHT = "#F5C6D2"; CREAM = "#FDF6EE"; TEXT = "#3D2535"

def plot_gauge(prob, label="Risk Score"):
    """Semicircular gauge chart."""
    fig, ax = plt.subplots(figsize=(5, 3), subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor(CREAM)
    ax.set_facecolor(CREAM)

    theta = np.linspace(0, np.pi, 300)
    ax.set_xlim(0, np.pi); ax.set_ylim(0, 1)
    ax.set_theta_zero_location("W")
    ax.set_theta_direction(1)

    # Background track
    ax.plot(theta, [0.8]*300, color=PINK_LIGHT, linewidth=18, solid_capstyle="round")

    # Fill up to prob
    fill_theta = np.linspace(0, prob * np.pi, 300)
    color = "#6BBF8C" if prob < 0.45 else (PINK if prob < 0.70 else "#D95F72")
    ax.plot(fill_theta, [0.8]*len(fill_theta), color=color, linewidth=18, solid_capstyle="round")

    # Needle
    angle = prob * np.pi
    ax.annotate("", xy=(angle, 0.78), xytext=(angle, 0.2),
                arrowprops=dict(arrowstyle="-|>", color=TEXT, lw=2))

    ax.text(np.pi/2, 0.35, f"{prob*100:.1f}%",
            ha="center", va="center", fontsize=22, fontweight="bold",
            color=TEXT, fontfamily="serif")
    ax.text(np.pi/2, 0.08, label,
            ha="center", va="center", fontsize=9, color="#7A4D5F")

    ax.set_axis_off()
    ax.set_thetamin(0); ax.set_thetamax(180)
    plt.tight_layout(pad=0)
    return fig


def plot_symptom_bar(symptom_dict):
    """Horizontal bar chart of symptom contributions."""
    data = []
    for s in SYMPTOM_DEFINITIONS:
        if symptom_dict.get(s["key"], 0):
            data.append((s["label"], s["weight"] / MAX_SYMPTOM_SCORE * 100))
    if not data:
        return None
    data.sort(key=lambda x: x[1], reverse=True)
    labels, vals = zip(*data)

    fig, ax = plt.subplots(figsize=(6, max(2.5, len(data)*0.45)))
    fig.patch.set_facecolor(CREAM)
    ax.set_facecolor(CREAM)

    colors = [PINK if v > 15 else "#E8A0B0" if v > 8 else PINK_LIGHT for v in vals]
    bars = ax.barh(range(len(labels)), vals, color=colors, height=0.6, edgecolor="none")

    for i, (bar, val) in enumerate(zip(bars, vals)):
        ax.text(val + 0.5, i, f"{val:.1f}%", va="center",
                fontsize=9, color=TEXT, fontweight="500")

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([l[:38] for l in labels], fontsize=9, color=TEXT)
    ax.set_xlabel("Contribution (%)", fontsize=9, color="#7A4D5F")
    ax.set_xlim(0, max(vals) * 1.25)
    ax.invert_yaxis()
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(left=False, bottom=False)
    ax.set_title("Symptom Contribution to Score", fontsize=10, color=PINK,
                 fontweight="600", pad=10, fontfamily="serif")
    plt.tight_layout()
    return fig


def plot_modality_donut(weights, probs):
    """Donut chart of modality weights."""
    labels, vals, colors_d = [], [], []
    palette = {"Symptoms": "#C2637A", "Clinical": "#E8A0B0", "Ultrasound": "#F5C6D2"}

    if weights["symptoms"] > 0:
        labels.append(f"Symptoms\n({probs['symptom']:.0%})")
        vals.append(weights["symptoms"]); colors_d.append(palette["Symptoms"])
    if weights["clinical"] > 0 and probs["clinical"] is not None:
        labels.append(f"Clinical\n({probs['clinical']:.0%})")
        vals.append(weights["clinical"]); colors_d.append(palette["Clinical"])
    if weights["cnn"] > 0 and probs["cnn"] is not None:
        labels.append(f"Ultrasound\n({probs['cnn']:.0%})")
        vals.append(weights["cnn"]); colors_d.append(palette["Ultrasound"])

    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_facecolor(CREAM)
    ax.set_facecolor(CREAM)
    wedges, texts = ax.pie(vals, labels=labels, colors=colors_d,
                           wedgeprops=dict(width=0.5, edgecolor="white", linewidth=2),
                           textprops=dict(fontsize=8, color=TEXT),
                           startangle=90)
    ax.set_title("Model Weight Distribution", fontsize=10, color=PINK,
                 fontweight="600", fontfamily="serif")
    plt.tight_layout()
    return fig


def plot_gradcam_overlay(img_pil, heatmap, alpha=0.45):
    """Returns matplotlib figure with GradCAM overlay."""
    import matplotlib.cm as cm
    img_arr  = np.array(img_pil.convert("RGB").resize((224, 224)))
    colormap = cm.get_cmap("YlOrRd")
    heat_c   = (colormap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    overlay  = (img_arr * (1 - alpha) + heat_c * alpha).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    fig.patch.set_facecolor(CREAM)
    for ax_ in axes: ax_.axis("off")
    axes[0].imshow(img_arr);       axes[0].set_title("Original", fontsize=9, color=TEXT)
    axes[1].imshow(overlay);       axes[1].set_title("GradCAM Heatmap", fontsize=9, color=PINK)
    plt.tight_layout(pad=1)
    return fig


# ─────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────
def main():
    # ── Sidebar ──────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🌸 PCOS Predict")
        st.markdown("---")
        st.markdown("""
**About this tool**

An AI-powered screening tool that combines:
- 🩺 Symptom analysis
- 🧪 Clinical lab values
- 🔬 Ultrasound imaging

for multi-modal PCOS risk assessment.
        """)
        st.markdown("---")
        fusion_threshold = st.slider(
            "Fusion Threshold", 0.20, 0.70, 0.45, 0.01,
            help="Probability cutoff for PCOS prediction. Lower = more sensitive."
        )
        st.markdown("---")
        st.markdown("""
<small style="color:#B08090">
⚕ <b>Medical Disclaimer</b><br>
This tool is for screening purposes only and does not replace professional medical diagnosis. Always consult a qualified gynaecologist or endocrinologist.
</small>
""", unsafe_allow_html=True)

    # ── Hero ──────────────────────────────────────────────────────────
    st.markdown("""
<div class="hero">
    <div class="badge">AI-Powered Screening</div>
    <h1>PCOS Predict 🌸</h1>
    <p>Multi-modal polycystic ovary syndrome risk assessment using symptoms, clinical data, and ultrasound imaging.</p>
</div>
""", unsafe_allow_html=True)

    # ── Load models (with graceful fallback for demo) ─────────────────
    models_loaded = False
    try:
        model_c, imputer, scaler, selector, cols, threshold, cnn_model = load_models()
        models_loaded = True
        st.success("✅ Models loaded successfully", icon="✅")
    except Exception as e:
        st.warning(f"⚠️ Models not found at configured paths. Running in **Demo Mode**.\n\n`{e}`\n\nUpdate model paths in `app.py` → `load_models()`.", icon="⚠️")
        cols = []

    # ═══════════════════════════════════════════════════════════════
    # FORM SECTIONS
    # ═══════════════════════════════════════════════════════════════

    # ── STEP 1: Personal Info ─────────────────────────────────────────
    st.markdown("""
<div class="section-card">
<div class="section-title"><span class="step-num">1</span> Personal Information</div>
""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        name   = st.text_input("Full Name", placeholder="e.g. Priya Sharma")
    with c2:
        age    = st.number_input("Age (years)", min_value=10, max_value=60, value=25, step=1)
    with c3:
        st.markdown("<br>", unsafe_allow_html=True)

    c4, c5 = st.columns(2)
    with c4:
        weight = st.number_input("Weight (kg) — optional", min_value=0.0, max_value=200.0,
                                  value=0.0, step=0.1, format="%.1f",
                                  help="Leave at 0 to skip")
    with c5:
        height = st.number_input("Height (cm) — optional", min_value=0.0, max_value=230.0,
                                  value=0.0, step=0.1, format="%.1f",
                                  help="Leave at 0 to skip")

    bmi_display = None
    if weight > 0 and height > 0:
        bmi_display = round(weight / ((height/100)**2), 1)
        bmi_cat = ("Underweight" if bmi_display<18.5 else "Normal" if bmi_display<25
                   else "Overweight" if bmi_display<30 else "Obese")
        st.info(f"📊 Calculated BMI: **{bmi_display}** — {bmi_cat}")

    st.markdown("</div>", unsafe_allow_html=True)

    # ── STEP 2: Symptoms ──────────────────────────────────────────────
    st.markdown("""
<div class="section-card">
<div class="section-title"><span class="step-num">2</span> Symptom Checklist <span style="font-size:0.85rem; color:#B08090; font-family:'DM Sans'; font-weight:300"> — Select all that apply</span></div>
""", unsafe_allow_html=True)

    symptom_dict = {}
    cols_sym = st.columns(2)
    for i, s in enumerate(SYMPTOM_DEFINITIONS):
        with cols_sym[i % 2]:
            val = st.checkbox(f"{s['icon']}  {s['label']}", key=f"sym_{s['key']}")
            symptom_dict[s["key"]] = int(val)

    active_symptoms = [s["label"] for s in SYMPTOM_DEFINITIONS if symptom_dict.get(s["key"])]
    if active_symptoms:
        chips = " ".join([f'<span class="chip-active">✓ {l}</span>' for l in active_symptoms])
        st.markdown(f"<div style='margin-top:1rem'>{chips}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── STEP 3: Clinical Data ─────────────────────────────────────────
    st.markdown("""
<div class="section-card">
<div class="section-title"><span class="step-num">3</span> Clinical & Lab Values <span style="font-size:0.85rem; color:#B08090; font-family:'DM Sans'; font-weight:300"> — optional but improves accuracy</span></div>
""", unsafe_allow_html=True)

    with st.expander("🧪 Enter Lab Values", expanded=False):
        st.markdown("Leave fields at **0** to skip them.")
        cl1, cl2, cl3 = st.columns(3)

        with cl1:
            st.markdown("**Hormones**")
            amh  = st.number_input("AMH (ng/mL)",  0.0, 20.0, 0.0, 0.1)
            lh   = st.number_input("LH (mIU/mL)",  0.0, 60.0, 0.0, 0.1)
            fsh  = st.number_input("FSH (mIU/mL)", 0.0, 40.0, 0.0, 0.1)
            tsh  = st.number_input("TSH (mIU/L)",  0.0, 10.0, 0.0, 0.1)
            prl  = st.number_input("Prolactin (ng/mL)", 0.0, 100.0, 0.0, 0.1)

        with cl2:
            st.markdown("**Ultrasound Measurements**")
            foll_l = st.number_input("Follicle count — Left",  0, 30, 0)
            foll_r = st.number_input("Follicle count — Right", 0, 30, 0)
            endo   = st.number_input("Endometrial thickness (mm)", 0.0, 25.0, 0.0, 0.1)
            cycle  = st.number_input("Cycle length (days)", 0, 90, 0)

        with cl3:
            st.markdown("**Body & Blood**")
            waist    = st.number_input("Waist (cm)",       0.0, 160.0, 0.0, 0.1)
            hip      = st.number_input("Hip (cm)",         0.0, 180.0, 0.0, 0.1)
            bp_sys   = st.number_input("Systolic BP (mmHg)",  0, 200, 0)
            bp_dia   = st.number_input("Diastolic BP (mmHg)", 0, 130, 0)
            rbs      = st.number_input("Blood sugar (mg/dL)", 0.0, 400.0, 0.0, 1.0)
            hb       = st.number_input("Haemoglobin (g/dL)",  0.0, 20.0, 0.0, 0.1)

    # Build clinical dict
    clinical_raw = {
        "amh": amh, "lh": lh, "fsh": fsh, "tsh": tsh, "prl": prl,
        "follicle_no_l": foll_l, "follicle_no_r": foll_r,
        "endometrium": endo, "cycle_length": cycle,
        "waist": waist, "hip": hip,
        "bp_systolic": bp_sys, "bp_diastolic": bp_dia,
        "rbs": rbs, "hb": hb,
        "age": age,
        "weight": weight if weight > 0 else np.nan,
        "height": height if height > 0 else np.nan,
        "bmi": bmi_display if bmi_display else np.nan,
    }
    # Only include if at least one meaningful lab value entered
    has_clinical = any(v not in (0, 0.0, np.nan) and v is not None
                       for k, v in clinical_raw.items()
                       if k not in ("age",))

    st.markdown("</div>", unsafe_allow_html=True)

    # ── STEP 4: Ultrasound Image ──────────────────────────────────────
    st.markdown("""
<div class="section-card">
<div class="section-title"><span class="step-num">4</span> Ultrasound Image <span style="font-size:0.85rem; color:#B08090; font-family:'DM Sans'; font-weight:300"> — optional</span></div>
""", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload an ovarian ultrasound image (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    if uploaded_file:
        img_pil = Image.open(uploaded_file)
        st.image(img_pil, caption="Uploaded ultrasound image", width=280)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── PREDICT BUTTON ────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        predict_clicked = st.button("🌸 Analyse Now", use_container_width=True)

    # ═══════════════════════════════════════════════════════════════
    # PREDICTION LOGIC
    # ═══════════════════════════════════════════════════════════════
    if predict_clicked:
        if not name.strip():
            st.error("Please enter your name before proceeding.")
            st.stop()

        with st.spinner("🌸 Analysing your data..."):
            # --- Symptom prob ---
            prob_s = symptom_score_to_prob(symptom_dict)

            # --- Clinical prob ---
            prob_c = None
            if models_loaded and has_clinical:
                try:
                    clinical_input = {col: np.nan for col in cols}
                    for k, v in clinical_raw.items():
                        col_name = _match_col(k, cols)
                        if col_name and v not in (0, 0.0):
                            clinical_input[col_name] = float(v)
                    # Derived
                    for col_name in cols:
                        if "lh_fsh" in col_name.lower() and lh > 0 and fsh > 0:
                            clinical_input[col_name] = round(lh/fsh, 3)
                        elif "fsh_lh" in col_name.lower() and lh > 0:
                            clinical_input[col_name] = round(fsh/lh, 3)
                        elif "follicle_total" in col_name.lower():
                            clinical_input[col_name] = foll_l + foll_r
                        elif "polycystic_flag" in col_name.lower():
                            clinical_input[col_name] = 1 if (foll_l>=12 or foll_r>=12) else 0
                    prob_c = predict_clinical_prob(clinical_input, model_c, imputer, scaler, selector, cols)
                except Exception as ex:
                    st.warning(f"Clinical model error: {ex}")

            # --- CNN prob ---
            prob_u = None
            cnn_arr = None
            img_for_gradcam = None
            if models_loaded and uploaded_file is not None:
                try:
                    img_for_gradcam = img_pil
                    prob_u, cnn_arr = predict_cnn_prob(img_pil, cnn_model)
                except Exception as ex:
                    st.warning(f"CNN model error: {ex}")

            # --- Demo mode fallback ---
            if not models_loaded:
                # Simulate plausible values for demo
                import random
                prob_c = round(prob_s * 0.85 + random.uniform(0.02, 0.12), 4) if has_clinical else None
                prob_u = round(prob_s * 0.90 + random.uniform(-0.05, 0.10), 4) if uploaded_file else None
                prob_u = max(0, min(1, prob_u)) if prob_u else None

            # --- Fusion ---
            w_s, w_c, w_u = adaptive_weights(prob_s, prob_c, prob_u)
            final_prob = (w_s * prob_s
                          + (w_c * prob_c if prob_c is not None else 0)
                          + (w_u * prob_u if prob_u is not None else 0))
            final_prob = round(final_prob, 4)

            n_mod  = 1 + int(prob_c is not None) + int(prob_u is not None)
            active = (["Symptoms"] +
                      (["Clinical"]   if prob_c is not None else []) +
                      (["Ultrasound"] if prob_u is not None else []))
            confidence  = _confidence_label(final_prob, n_mod)
            is_pcos     = final_prob >= fusion_threshold

        # ═══════════════════════════════════════════════════════════════
        # RESULTS
        # ═══════════════════════════════════════════════════════════════
        st.markdown("<hr class='pink-divider'>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='font-family:Playfair Display,serif;color:#C2637A;'>Results for {name} 🌸</h2>",
                    unsafe_allow_html=True)

        # ── Main result card ──────────────────────────────────────────
        if is_pcos:
            pill_class = {"High":"pill-high","Medium":"pill-medium","Low":"pill-low"}[confidence]
            st.markdown(f"""
<div class="result-pcos">
    <p class="result-label" style="color:#D95F72;">⚠️ PCOS Risk Detected</p>
    <p class="result-sub">Probability score: <strong>{final_prob*100:.1f}%</strong> (threshold: {fusion_threshold*100:.0f}%)</p>
    <span class="pill {pill_class}">{confidence} Confidence</span>
    <p style="margin-top:1rem;font-size:0.9rem;color:#7A4D5F;">
        Please consult a gynaecologist or endocrinologist for a formal diagnosis.
    </p>
</div>
""", unsafe_allow_html=True)
        else:
            pill_class = {"High":"pill-high","Medium":"pill-medium","Low":"pill-low"}[confidence]
            st.markdown(f"""
<div class="result-no-pcos">
    <p class="result-label" style="color:#2D6A4F;">✅ Low PCOS Risk</p>
    <p class="result-sub">Probability score: <strong>{final_prob*100:.1f}%</strong> (threshold: {fusion_threshold*100:.0f}%)</p>
    <span class="pill {pill_class}">{confidence} Confidence</span>
    <p style="margin-top:1rem;font-size:0.9rem;color:#7A4D5F;">
        Low risk based on provided data. If symptoms persist, please consult a doctor.
    </p>
</div>
""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Metric row ────────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Final Score",     f"{final_prob*100:.1f}%")
        m2.metric("Symptom Score",   f"{prob_s*100:.1f}%")
        m3.metric("Confidence",      confidence)
        m4.metric("Data Sources",    f"{n_mod} / 3")

        # ── Charts ────────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        ch1, ch2, ch3 = st.columns([1.2, 1, 1])

        with ch1:
            fig_gauge = plot_gauge(final_prob, "PCOS Risk Probability")
            st.pyplot(fig_gauge, use_container_width=True)

        with ch2:
            fig_sym = plot_symptom_bar(symptom_dict)
            if fig_sym:
                st.pyplot(fig_sym, use_container_width=True)
            else:
                st.info("No symptoms selected.")

        with ch3:
            weights_dict = {"symptoms": w_s, "clinical": w_c, "cnn": w_u}
            probs_dict   = {"symptom": prob_s, "clinical": prob_c, "cnn": prob_u}
            fig_donut = plot_modality_donut(weights_dict, probs_dict)
            st.pyplot(fig_donut, use_container_width=True)

        # ── Modality breakdown ────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""<div class="section-card">
<div class="section-title">📊 Modality Breakdown</div>""", unsafe_allow_html=True)

        mb1, mb2, mb3 = st.columns(3)
        with mb1:
            st.markdown(f"""
<div style="text-align:center;padding:1rem;background:#FDE8ED;border-radius:14px;border:1px solid #F5C6D2;">
  <div style="font-size:1.8rem">🩺</div>
  <div style="font-family:'Playfair Display',serif;font-size:1.3rem;color:#C2637A;font-weight:600">{prob_s*100:.1f}%</div>
  <div style="font-size:0.85rem;color:#7A4D5F">Symptom Score</div>
  <div style="font-size:0.75rem;color:#B08090;margin-top:4px">Weight: {w_s*100:.0f}%</div>
</div>""", unsafe_allow_html=True)
        with mb2:
            c_val = f"{prob_c*100:.1f}%" if prob_c is not None else "Not provided"
            c_bg  = "#FDE8ED" if prob_c is not None else "#F4EBE0"
            st.markdown(f"""
<div style="text-align:center;padding:1rem;background:{c_bg};border-radius:14px;border:1px solid #F5C6D2;">
  <div style="font-size:1.8rem">🧪</div>
  <div style="font-family:'Playfair Display',serif;font-size:1.3rem;color:#C2637A;font-weight:600">{c_val}</div>
  <div style="font-size:0.85rem;color:#7A4D5F">Clinical Score</div>
  <div style="font-size:0.75rem;color:#B08090;margin-top:4px">Weight: {w_c*100:.0f}%</div>
</div>""", unsafe_allow_html=True)
        with mb3:
            u_val = f"{prob_u*100:.1f}%" if prob_u is not None else "Not provided"
            u_bg  = "#FDE8ED" if prob_u is not None else "#F4EBE0"
            st.markdown(f"""
<div style="text-align:center;padding:1rem;background:{u_bg};border-radius:14px;border:1px solid #F5C6D2;">
  <div style="font-size:1.8rem">🔬</div>
  <div style="font-family:'Playfair Display',serif;font-size:1.3rem;color:#C2637A;font-weight:600">{u_val}</div>
  <div style="font-size:0.85rem;color:#7A4D5F">Ultrasound Score</div>
  <div style="font-size:0.75rem;color:#B08090;margin-top:4px">Weight: {w_u*100:.0f}%</div>
</div>""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # ── GradCAM ───────────────────────────────────────────────────
        if uploaded_file is not None and cnn_arr is not None and models_loaded:
            with st.expander("🔬 View GradCAM Heatmap (CNN Explanation)", expanded=True):
                with st.spinner("Generating heatmap..."):
                    try:
                        heatmap = gradcam_heatmap(cnn_model, cnn_arr)
                        if heatmap is not None:
                            fig_cam = plot_gradcam_overlay(img_for_gradcam, heatmap)
                            st.pyplot(fig_cam, use_container_width=True)
                            st.caption("🟡 Warm regions = areas the CNN focused on for PCOS detection.")
                    except Exception as ex:
                        st.warning(f"GradCAM failed: {ex}")

        # ── Missing modalities tip ────────────────────────────────────
        if n_mod < 3:
            missing = []
            if prob_c is None: missing.append("clinical lab values")
            if prob_u is None: missing.append("an ultrasound image")
            st.info(f"💡 **Tip:** Adding {' and '.join(missing)} would increase prediction confidence from **{confidence}** toward **High**.")

        # ── Disclaimer ────────────────────────────────────────────────
        st.markdown("""
<div class="disclaimer">
⚕ <strong>Medical Disclaimer:</strong> This tool provides AI-based screening only and is not a substitute for clinical diagnosis. 
All results should be reviewed by a qualified gynaecologist or endocrinologist. 
PCOS diagnosis requires clinical examination, blood tests, and imaging by a healthcare professional.
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
