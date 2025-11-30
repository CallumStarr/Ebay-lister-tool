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
    st.caption("Mode: Hybrid (Manual Hint + AI Vision)")

st.title("🎥 eBay Video Auto-Lister")

# --- NEW: OPTIONAL INPUT ---
product_hint = st.text_input("Optional: Model Name / Code (e.g. 'SKX007J' or 'DeWalt DCF887')")

uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

if uploaded_file:
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())
    st.video(uploaded_file)
    
    if st.button("✨ Analyze Video"):
        with st.spinner("Analyzing..."):
            try:
                # UPLOAD
                video_file = genai.upload_file(path="temp_video.mp4")
                while video_file.state.name == "PROCESSING":
                    time.sleep(1)
                    video_file = genai.get_file(video_file.name)

                model = genai.GenerativeModel('gemini-2.0-flash')

                # --- STEP 1: THE HYBRID DETECTIVE ---
                detective_prompt = f"""
                Act as a Forensic Text Extractor.
                
                USER HINT: "{product_hint}"
                
                INSTRUCTIONS:
                1. IF HINT IS PROVIDED: Trust it. Use the video to confirm condition and find the best angles. Do not guess a different model.
                2. IF HINT IS EMPTY: Read text labels in the video to Identify the item (Look for Model Numbers/Voltage/Jewel count).
                
                TASK:
                - List 3 exact visible text strings found in the video.
                - Find 3 timestamps for photos (Main, Label, Detail).
                
                Output JSON:
                {{
                    "visible_text": ["String 1", "String 2"],
                    "full_title": "Precise Item Name (Use Hint if available)",
                    "condition_summary": "Short condition notes",
                    "visual_description": "Sales description matching the video visuals",
                    "shots": [
                        {{ "label": "Main View", "time": "00:00" }},
                        {{ "label": "Text/Label/Dial", "time": "00:00" }},
                        {{ "label": "Side/Detail", "time": "00:00" }}
                    ]
                }}
                """
                
                detective_resp = model.generate_content(
                    [detective_prompt, video_file],
                    generation_config=genai.types.GenerationConfig(temperature=0.2)
                )
                
                detective_data = json.loads(re.search(r"\{.*\}", detective_resp.text, re.DOTALL).group(0))
                
                # --- STEP 2: THE PRICER ---
                st.toast("Pricing...")
                
                valuator_prompt = f"""
                You are an eBay Power Seller.
                Set a "Buy It Now" listing price.
                
                ITEM: {detective_data['full_title']}
                EVIDENCE/HINT: {product_hint} (If provided, price THIS specific model).
                CONDITION: {detective_data['condition_summary']}
                
                STRATEGY:
                - Give the 'Target List Price' (Optimistic but realistic).
                - Use the Hint to ensure you price the correct variant (e.g. J model vs K model).
                
                Output JSON:
                {{
                    "list_price": "{currency_code}XXX",
                    "price_strategy": "Explain why this price was chosen"
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
                col1.metric("List Price", price_data["list_price"])
                col2.success(f"Strategy: {price_data['price_strategy']}")
                
                if product_hint:
                    st.caption(f"✅ Pricing locked to your hint: '{product_hint}'")
                else:
                    st.caption(f"🔎 Detected from video text: {detective_data['visible_text']}")

                st.write(detective_data["visual_description"])
                
                st.markdown("---")
                st.subheader("📸 Proof of Item")
                
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
