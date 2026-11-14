"""Streamlit UI.  Run with:  streamlit run app.py"""
import hashlib
import os

import pandas as pd
import streamlit as st
from PIL import Image
from google.genai.errors import APIError

from config import CHECKPOINT
from inference import PlantDiagnoser
from llm import generate_explanation
from knowledge_base import get_recommendations

st.set_page_config(page_title="Plant Health Diagnostics", page_icon="🌿", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def load_diagnoser(path: str) -> PlantDiagnoser:
    return PlantDiagnoser(path)


SEVERITY_ICON = {"none": "🟢", "moderate": "🟡", "moderate to high": "🟠",
                 "critical": "🔴", "unknown": "⚪"}

st.title("🌿 Plant Health, Nutrient & Post-Mortem Diagnostics")
st.caption("Upload a photo of a plant or leaf. The model estimates whether it is healthy, "
           "nutrient-deficient, or dead/necrotic, and suggests next steps.")

with st.sidebar:
    st.header("Settings")
    ckpt_path = st.text_input("Checkpoint path", str(CHECKPOINT))
    top_k = st.slider("Predictions to show", 2, 5, 3)
    gemini_api_key = st.text_input(
        "Gemini API key",
        value=os.environ.get("GEMINI_API_KEY", ""),
        type="password",
        help="Used only when you request an AI explanation. It is not saved in the project.",
    )
    st.markdown("---")
    st.info("This is a decision-support tool. Confirm with a soil/tissue test or an "
            "agronomist before making costly treatment decisions.")

try:
    diagnoser = load_diagnoser(ckpt_path)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()


uploaded = st.file_uploader("Upload plant image", type=["jpg", "jpeg", "png", "webp", "bmp"])

if uploaded is None:
    st.write("👆 Upload an image to begin.")
    st.stop()

image = Image.open(uploaded).convert("RGB")
image_id = hashlib.sha256(uploaded.getvalue()).hexdigest()
result = diagnoser.predict(image, top_k=top_k)
rec = result["recommendations"]
analysis_id = f"{image_id}:{top_k}"

previous_analysis = st.session_state.get("gemini_analysis")
if previous_analysis and previous_analysis["id"] != analysis_id:
    del st.session_state["gemini_analysis"]

col_img, col_res = st.columns([1, 1.2], gap="large")

with col_img:
    st.image(image, caption=uploaded.name, use_container_width=True)

with col_res:
    icon = SEVERITY_ICON.get(rec["severity"], "⚪")
    st.subheader(f"{icon} {rec['title']}")
    c1, c2 = st.columns(2)
    c1.metric("Confidence", f"{result['primary_confidence']:.1%}")
    c2.metric("Severity", rec["severity"].title())
    st.write(rec["summary"])
    for w in result["warnings"]:
        st.warning(w)

    st.markdown("#### Condition meters")
    ranked = sorted(result["all_probabilities"].items(), key=lambda kv: kv[1], reverse=True)
    for label, p in ranked:
        st.write(f"**{get_recommendations(label)['title']}**: {p:.1%}")
        st.progress(min(max(p, 0.0), 1.0))

st.markdown("---")
st.subheader("Probability breakdown")
df = pd.DataFrame({"class": [get_recommendations(l)["title"] for l, _ in ranked],
                   "probability": [p for _, p in ranked]}).set_index("class")
st.bar_chart(df)

st.subheader("Recommendations")
st.caption(
    "Optional Gemini explanation. When requested, the prediction and existing "
    "recommendations are sent to Google; the uploaded image is not."
)
if st.button("Generate AI explanation"):
    if not gemini_api_key.strip():
        st.error("Enter a Gemini API key in the sidebar first.")
    else:
        try:
            with st.spinner("Generating explanation with Gemini..."):
                explanation = generate_explanation(result, gemini_api_key)
        except APIError as error:
            st.error(f"Gemini request failed: {error}")
        else:
            st.session_state["gemini_analysis"] = {
                "id": analysis_id,
                "text": explanation,
            }

saved_analysis = st.session_state.get("gemini_analysis")
if saved_analysis and saved_analysis["id"] == analysis_id:
    st.markdown(saved_analysis["text"])

tabs = st.tabs(["🔍 Likely causes & symptoms", "🧪 Soil amendments",
                "⚠️ Precautions", "🛡️ Prevention"])


def bullets(items, empty="Nothing specific for this diagnosis."):
    if not items:
        st.write(empty)
    for item in items:
        st.markdown(f"- {item}")


with tabs[0]:
    st.markdown("**Typical symptoms**")
    bullets(rec["symptoms"])
    st.markdown("**Possible root causes**")
    bullets(rec["root_causes"])
with tabs[1]:
    bullets(rec["soil_amendments"])
    st.caption("Always follow the product label for rates and confirm with a soil test.")
with tabs[2]:
    bullets(rec["precautions"])
with tabs[3]:
    bullets(rec["prevention"])
