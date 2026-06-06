import streamlit as st
import requests
from PIL import Image
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CIFAR-10 Product Classifier",
    page_icon="🔍",
    layout="centered"
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .main-title {
        font-family: 'Space Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -1px;
        margin-bottom: 0;
    }
    .subtitle {
        color: #888;
        font-size: 0.95rem;
        margin-top: 4px;
        margin-bottom: 2rem;
    }
    .result-box {
        background: #f7f7f7;
        border-left: 4px solid #111;
        padding: 1.2rem 1.5rem;
        border-radius: 4px;
        margin-top: 1.5rem;
    }
    .predicted-label {
        font-family: 'Space Mono', monospace;
        font-size: 1.6rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .confidence-label {
        font-size: 0.9rem;
        color: #555;
        margin-top: 4px;
    }
    .stButton > button {
        background: #111;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.6rem 2rem;
        font-family: 'Space Mono', monospace;
        font-size: 0.85rem;
        letter-spacing: 1px;
        width: 100%;
    }
    .stButton > button:hover { background: #333; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">CIFAR-10 Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an image → get a real-time category prediction via CNN</div>', unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    api_url = st.text_input("Flask API URL", value="http://localhost:5000")
    st.markdown("---")
    st.markdown("**Supported Classes:**")
    classes = ['✈️ airplane', '🚗 automobile', '🐦 bird', '🐱 cat', '🦌 deer',
               '🐶 dog', '🐸 frog', '🐴 horse', '🚢 ship', '🚛 truck']
    for c in classes:
        st.markdown(f"- {c}")

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("PREDICT"):
        with st.spinner("Running inference..."):
            try:
                img_bytes = uploaded_file.getvalue()
                response = requests.post(
                    f"{api_url}/predict",
                    files={"file": (uploaded_file.name, img_bytes, uploaded_file.type)},
                    timeout=10
                )

                if response.status_code == 200:
                    result = response.json()
                    pred = result["predicted_class"].upper()
                    conf = result["confidence"]
                    probs = result["all_probabilities"]

                    st.markdown(f"""
                    <div class="result-box">
                        <div class="predicted-label">{pred}</div>
                        <div class="confidence-label">Confidence: <strong>{conf}%</strong></div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("#### All Class Probabilities")
                    labels = list(probs.keys())
                    values = [v * 100 for v in probs.values()]
                    colors = ['#111' if l == result["predicted_class"] else '#ddd' for l in labels]

                    fig = go.Figure(go.Bar(
                        x=values,
                        y=labels,
                        orientation='h',
                        marker_color=colors,
                        text=[f"{v:.1f}%" for v in values],
                        textposition='outside'
                    ))
                    fig.update_layout(
                        xaxis=dict(range=[0, 110], showgrid=False, title="Probability (%)"),
                        yaxis=dict(autorange="reversed"),
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        margin=dict(l=10, r=40, t=10, b=10),
                        height=350,
                        font=dict(family="DM Sans")
                    )
                    st.plotly_chart(fig, use_container_width=True)

                else:
                    st.error(f"API Error {response.status_code}: {response.json().get('error', 'Unknown')}")

            except requests.exceptions.ConnectionError:
                st.error("❌ Could not connect to Flask API. Make sure app.py is running.")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
