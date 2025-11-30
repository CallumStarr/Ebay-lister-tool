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
    currency = st.selectbox("Currency", ["£ (GBP)", "$ (USD)", "€ (EUR)"])
    st.info("Mode: Dual-Brain (Detective + Accountant)")

st.title("🎥 eBay Video Auto-Lister")

uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

if uploaded_file:
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())
    st.video(uploaded_file)
    
    if st.button("✨ Analyze Video"):
        with st.spinner("Step 1: The Detective is identifying the item..."):
            try:
                # UPLOAD
                video_file = genai.upload_file(path="temp_video.mp4")
                while video_file.state.name == "PROCESSING":
                    time.sleep(1)
                    video_file = genai.get_file(video_file.name)

                model = genai.GenerativeModel('gemini-2.0-flash')

                # --- STEP 1: THE DETECTIVE (High Detail, No Pricing) ---
                detective_prompt = """
                Act as a Forensic Authenticator. Watch the video.
                Identify the item with EXTREME precision.
                
                1. Read all text/labels.
                2. Identify specific Model Numbers (e.g. 'SKX007K' vs 'SKX007J').
                3. List every flaw (scratches, dents).
                4. Find the best timestamps for photos.
                
                Output JSON:
                {
                    "full_title": "Brand + Model + Variant",
                    "specific_variant": "e.g. Made in Japan version",
                    "condition_report": "Detailed list of flaws",
                    "specs": {"Material": "...", "Size": "..."},
                    "shots": {"Main": "00:00", "Label": "00:00", "Flaw": "00:00"}
                }
                """
                
                # Temp 0.4 allows it to "look around" and find small details
                detective_resp = model.generate_content(
                    [detective_prompt, video_file],
                    generation_config=genai.types.GenerationConfig(temperature=0.4)
                )
                
                detective_data = json.loads(re.search(r"\{.*\}", detective_resp.text, re.DOTALL).group(0))
                
                # --- STEP 2: THE ACCOUNTANT (Strict Logic, Pricing Only) ---
                st.toast("Step 2: The Accountant is calculating price...")
                
                accountant_prompt = f"""
                You are a strict pricing algorithm. You do not look at the video.
                You rely ONLY on this report:
                
                ITEM: {detective_data['full_title']}
                VARIANT: {detective_data['specific_variant']}
                CONDITION: {detective_data['condition_report']}
                
                TASK:
                Provide a single market price in {currency}.
                Logic: If condition has flaws, lower the price. If variant is rare, raise it.
                
                Output JSON:
                {{
                    "target_price": "{currency}XXX",
                    "price_reasoning": "Why this price?",
                    "final_description": "Write a sales description using the Detective's notes."
                }}
                """
                
                # Temp 0.1 forces consistency. Input A always equals Output B.
                accountant_resp = model.generate_content(
                    accountant_prompt,
                    generation_config=genai.types.GenerationConfig(temperature=0.1)
                )
                
                price_data = json.loads(re.search(r"\{.*\}", accountant_resp.text, re.DOTALL).group(0))

                # --- MERGE & DISPLAY ---
                st.header(detective_data["full_title"])
                
                col1, col2 = st.columns(2)
                col1.metric("Target Price", price_data["target_price"])
                col2.info(price_data["price_reasoning"])
                
                st.write(price_data["final_description"])
                st.caption(f"**Detailed Variant:** {detective_data['specific_variant']}")
                st.caption(f"**Condition Notes:** {detective_data['condition_report']}")
                
                st.markdown("---")
                st.subheader("📸 Proof of Item")
                
                # PHOTO EXTRACTION
                for shot_name, time_str in detective_data.get("shots", {}).items():
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
