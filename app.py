import streamlit as st
import google.generativeai as genai
import json
import time
import re
import cv2
import numpy as np

# 1. SETUP
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
else:
    st.error("Missing API Key.")
    st.stop()

st.set_page_config(page_title="eBay Video Lister", page_icon="🎥")

# --- HELPER: FRAME HUNTER ---
def get_burst_frames(video_path, timestamp_str):
    try:
        if not timestamp_str or "none" in timestamp_str.lower(): return []
        clean_time = re.search(r"(\d{1,2}:\d{2})", timestamp_str)
        if not clean_time: return []
        
        minutes, seconds = map(int, clean_time.group(1).split(':'))
        center_time = (minutes * 60) + seconds
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return []
        
        frames = []
        # Check -0.5s, 0.0s, +0.5s
        offsets = [-0.5, 0.0, 0.5]
        for offset in offsets:
            target_time = center_time + offset
            if target_time < 0: continue
            cap.set(cv2.CAP_PROP_POS_MSEC, target_time * 1000)
            ret, frame = cap.read()
            if ret: frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames
    except: return []

with st.sidebar:
    st.header("⚙️ Settings")
    currency_code = st.selectbox("Currency", ["£", "$", "€", "¥"])
    st.caption("Mode: Chameleon (Auto-Category Detect)")

st.title("🎥 eBay Video Auto-Lister")

uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

if uploaded_file:
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())
    st.video(uploaded_file)
    
    if st.button("✨ Analyze Video"):
        with st.spinner("Detecting Category & Reading Labels..."):
            try:
                # UPLOAD
                video_file = genai.upload_file(path="temp_video.mp4")
                while video_file.state.name == "PROCESSING":
                    time.sleep(1)
                    video_file = genai.get_file(video_file.name)

                model = genai.GenerativeModel('gemini-2.0-flash')

                # --- STEP 1: THE CHAMELEON DETECTIVE ---
                detective_prompt = f"""
                Act as a Universal Reseller.
                
                PHASE 1: CATEGORY DETECTION
                - Is this a Watch, Tool, Shoe, Camera, or Bag?
                - Adapt your visual search based on the category.
                
                PHASE 2: EVIDENCE GATHERING
                - WATCHES: Read Dial text. Look for "21 Jewels".
                - TOOLS: Read the RATING STICKER (Voltage/Model #). Don't guess by shape!
                - SHOES: Read the inner Size Tag.
                - CLOTHES: Read the Neck/Hip Tag.
                
                PHASE 3: SCREENSHOT SELECTION
                - Select 3 timestamps.
                - NAME them dynamically based on the item (e.g. "Rating Sticker" for tools, "Dial" for watches).
                
                Output JSON:
                {{
                    "category": "Detected Category",
                    "full_title": "Brand + Exact Model Number",
                    "id_evidence": "Text found (e.g. 'Read DCF887 on sticker')",
                    "condition_report": "Detailed flaws",
                    "visual_description": "Draft the sales text",
                    "shots": [
                        {{ "label": "Main View", "time": "00:00" }},
                        {{ "label": "ID Tag / Sticker / Dial", "time": "00:00" }},
                        {{ "label": "Defect / Texture", "time": "00:00" }}
                    ]
                }}
                """
                
                detective_resp = model.generate_content(
                    [detective_prompt, video_file],
                    generation_config=genai.types.GenerationConfig(temperature=0.2)
                )
                
                detective_data = json.loads(re.search(r"\{.*\}", detective_resp.text, re.DOTALL).group(0))
                
                # --- STEP 2: THE PRICER ---
                st.toast(f"Identified as {detective_data['category']}. Pricing...")
                
                valuator_prompt = f"""
                You are a Market Value Database.
                
                ITEM: {detective_data['full_title']}
                EVIDENCE: {detective_data['id_evidence']}
                CONDITION: {detective_data['condition_report']}
                
                TASK:
                - Provide the 'Sold' market price in {currency_code}.
                - Do not output currency symbols twice (e.g. just '55', not '£55').
                
                Output JSON:
                {{
                    "price_value": "55",
                    "price_note": "Reasoning"
                }}
                """
                
                valuator_resp = model.generate_content(
                    valuator_prompt,
                    generation_config=genai.types.GenerationConfig(temperature=0.0)
                )
                
                price_data = json.loads(re.search(r"\{.*\}", valuator_resp.text, re.DOTALL).group(0))

                # --- DISPLAY ---
                st.header(detective_data["full_title"])
                
                col1, col2 = st.columns(2)
                col1.metric("Market Value", f"{currency_code}{price_data['price_value']}")
                col2.success(f"ID Evidence: {detective_data['id_evidence']}")
                
                st.write(detective_data["visual_description"])
                st.caption(f"**Condition:** {detective_data['condition_report']}")
                
                st.markdown("---")
                st.subheader("📸 Proof of Item")
                
                # DYNAMIC PHOTO GALLERY
                # We loop through the list so the order is always Main -> ID -> Detail
                for shot in detective_data.get("shots", []):
                    label = shot.get("label", "Shot")
                    time_str = shot.get("time")
                    
                    if time_str and time_str != "null":
                        st.write(f"**{label}** ({time_str})")
                        photos = get_burst_frames("temp_video.mp4", time_str)
                        if photos:
                            c1, c2, c3 = st.columns(3)
                            if len(photos) > 0: c1.image(photos[0], caption="Early", use_container_width=True)
                            if len(photos) > 1: c2.image(photos[1], caption="Exact", use_container_width=True)
                            if len(photos) > 2: c3.image(photos[2], caption="Late", use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")
