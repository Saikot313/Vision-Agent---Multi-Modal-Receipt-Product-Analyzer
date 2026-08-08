"""Streamlit demo UI for the Multi-Modal Vision + LLM Agent.

Run with: streamlit run streamlit_app.py
"""
import cv2
import numpy as np
import streamlit as st

from app.agent.tool_agent import run_agent

st.set_page_config(page_title="Receipt/Product Vision Agent", page_icon="🧾", layout="wide")

st.title("🧾 Multi-Modal Vision + LLM Agent")
st.caption(
    "Upload a receipt/invoice photo or a product photo. The LLM agent decides which "
    "tools (OCR, object detection, price check) to call — you can watch its trace below."
)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    cv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if cv_image is None:
        st.error("Couldn't decode this file as an image — please upload a JPG or PNG.")
        st.stop()

    col1, col2 = st.columns([1, 1.4])

    with col1:
        try:
            st.image(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB), caption="Uploaded image", use_container_width=True)
        except TypeError:
            # Older Streamlit versions (pre-1.39ish) don't have
            # use_container_width for st.image — fall back gracefully.
            st.image(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB), caption="Uploaded image")

    with col2:
        with st.spinner("Agent is analyzing the image (this calls tools + an LLM)..."):
            try:
                result = run_agent(cv_image)
            except RuntimeError as e:
                st.error(str(e))
                st.stop()

        st.subheader("Final summary")
        st.write(result.final_answer)

        with st.expander("🔍 Agent trace (tool calls, in order)", expanded=False):
            for i, step in enumerate(result.trace, start=1):
                st.markdown(f"**{i}. `{step.tool}`**")
                if step.arguments:
                    st.json(step.arguments)
                st.code(step.result_preview, language="json")

        # Structured views for specific tool outputs, when present
        if "check_price_mismatch" in result.tool_outputs:
            price = result.tool_outputs["check_price_mismatch"]
            st.subheader("💰 Price check")
            cols = st.columns(3)
            cols[0].metric("Computed total", price.get("computed_total"))
            cols[1].metric("Printed total", price.get("printed_total"))
            cols[2].metric("Mismatch?", "⚠️ Yes" if price.get("mismatch") else "✅ No")
            st.caption(price.get("note", ""))

        if "detect_objects" in result.tool_outputs:
            st.subheader("📦 Detected objects")
            detections = result.tool_outputs["detect_objects"].get("detections", [])
            if detections:
                st.table(detections)
            else:
                st.write("No objects detected above the confidence threshold.")
else:
    st.info("Upload a receipt or product image to get started.")
