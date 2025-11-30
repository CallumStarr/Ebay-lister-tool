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

# --- HELPER: ROBUST FRAME HUNTER ---
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
    currency = st.selectbox("Currency", ["£ (GBP)", "$ (USD)", "€ (EUR)"])
    st.caption("Mode: Expert Appraiser (High Value)")

st.title("🎥 eBay Video Auto-Lister")

uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

if uploaded_file:
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())
    st.video(uploaded_file)
    
    if st.button("✨ Analyze Video"):
        with st.spinner("Analyzing Dial Details & Market Value..."):
            try:
                # UPLOAD
                video_file = genai.upload_file(path="temp_video.mp4")
                while video_file.state.name == "PROCESSING":
                    time.sleep(1)
                    video_file = genai.get_file(video_file.name)

                model = genai.GenerativeModel('gemini-2.0-flash')

                # --- STEP 1: THE VISUAL EXPERT (Writes Description & ID) ---
                expert_prompt = f"""
                Act as a Professional eBay Appraiser.
                Watch the video closely.
                
                1. IDENTIFY VARIANT:
                   - If it's a Seiko SKX, look at the dial text at 6 o'clock. 
                   - "21 JEWELS" or "MADE IN JAPAN" = J Model (Higher Value).
                   - No "21 JEWELS" = K Model (Standard Value).
                   - If unsure, assume the Standard Model but mention the ambiguity.
                
                2. WRITE DESCRIPTION:
                   - Write a passionate, accurate sales description based on VISUALS.
                   - Do not use generic filler. Describe THIS specific item.
                
                3. FIND SHOTS:
                   - Find the best timestamps for Main View, Label/Dial, and Detail.
                
                Output JSON:
                {{
                    "full_title": "SEO Optimized Title",
                    "variant_identified": "J or K or Standard",
                    "visual_description": "The specific sales description text",
                    "condition_rating": "Mint / Good / Fair",
                    "shots": {{ "Main": "00:00", "Dial/Label": "00:00", "Side/Back": "00:00" }}
                }}
                """
                
                # Temp 0.4 allows just enough flexibility to read blurry text
                expert_resp = model.generate_content(
                    [expert_prompt, video_file],
                    generation_config=genai.types.GenerationConfig(temperature=0.4)
                )
                
                expert_data = json.loads(re.search(r"\{.*\}", expert_resp.text, re.DOTALL).group(0))
                
                # --- STEP 2: THE MARKET VALUATOR (Pricing Only) ---
                st.toast("Calculating market value...")
                
                valuator_prompt = f"""
                You are a Market Value Database.
                Determine the current average "Sold Listing" price on eBay for this item.
                
                ITEM: {expert_data['full_title']} ({expert_data['variant_identified']})
                CONDITION: {expert_data['condition_rating']}
                CURRENCY: {currency}
                
                INSTRUCTION:
                - Do not give a 'Pawn Shop' price. Give the 'Collector' price.
                - If it is a 'J' model, price higher.
                - Output a single fixed number.
                
                Output JSON:
                {{
                    "market_price": "{currency}XXX",
                    "price_note": "Short explanation of value"
                }}
                """
                
                # Temp 0.0 FORCES CONSISTENCY. Same input = Same Price.
                valuator_resp = model.generate_content(
                    valuator_prompt,
                    generation_config=genai.types.GenerationConfig(temperature=0.0)
                )
                
                price_data = json.loads(re.search(r"\{.*\}", valuator_resp.text, re.DOTALL).group(0))

                # --- DISPLAY ---
                st.header(expert_data["full_title"])
                
                col1, col2 = st.columns(2)
                col1.metric("Market Value", price_data["market_price"])
                col2.success(f"Variant: {expert_data['variant_identified']}")
                
                st.write(expert_data["visual_description"])
                st.caption(f"Pricing Logic: {price_data['price_note']}")
                
                st.markdown("---")
                st.subheader("📸 Proof of Item")
                
                for shot_name, time_str in expert_data.get("shots", {}).items():
                    if time_str and time_str != "null":
                        st.write(f"**{shot_name}** ({time_str})")
                        photos = get_burst_frames("temp_video.mp4", time_str)
                        if photos:
                            c1, c2, c3 = st.columns(3)
                            if len(photos) > 0: c1.image(photos[0], caption="Early", use_container_width=True)
                            if len(photos) > 1: c2.image(photos[1], caption="Exact", use_container_width=True)
                            if len(photos) > 2: c3.image(photos[2], caption="Late", use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")
