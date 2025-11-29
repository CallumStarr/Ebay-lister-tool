import streamlit as st
import google.generativeai as genai
import json
import time
import re
import cv2
import numpy as np # Needed for the blur check

# 1. SETUP
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
else:
    st.error("Missing API Key.")
    st.stop()

st.set_page_config(page_title="eBay Video Lister", page_icon="🎥")

# --- HELPER: SMART FRAME HUNTER (Fixes Blurry Photos) ---
def get_sharpest_frame(video_path, timestamp_str):
    try:
        minutes, seconds = map(int, timestamp_str.split(':'))
        center_time = (minutes * 60) + seconds
        
        cap = cv2.VideoCapture(video_path)
        
        best_score = 0
        best_frame = None
        
        # Check 3 moments: Exactly at timestamp, 0.5s before, 0.5s after
        # This fixes "mid-motion" blur
        offsets = [0, -0.5, 0.5] 
        
        for offset in offsets:
            target_time = center_time + offset
            if target_time < 0: continue
            
            cap.set(cv2.CAP_PROP_POS_MSEC, target_time * 1000)
            ret, frame = cap.read()
            
            if ret:
                # Calculate "Sharpness" using Laplacian Variance
                # Higher number = Sharper image
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                score = cv2.Laplacian(gray, cv2.CV_64F).var()
                
                if score > best_score:
                    best_score = score
                    best_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        cap.release()
        return best_frame
    except:
        return None

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Settings")
    currency = st.selectbox("Currency", ["£ (GBP)", "$ (USD)", "€ (EUR)", "¥ (JPY)"])
    st.info("Mode: Logic Balanced (Temp 0.2)")

st.title("🎥 eBay Video Auto-Lister")

uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

if uploaded_file:
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())
    
    st.video(uploaded_file)
    
    if st.button("✨ Analyze Video"):
        with st.spinner("Analyzing Dial Text & Hunting for Clear Shots..."):
            try:
                # UPLOAD
                video_file = genai.upload_file(path="temp_video.mp4")
                while video_file.state.name == "PROCESSING":
                    time.sleep(1)
                    video_file = genai.get_file(video_file.name)

                if video_file.state.name == "FAILED":
                    st.error("Video processing failed.")
                    st.stop()

                # CONFIG
                model = genai.GenerativeModel('gemini-2.0-flash')
                
                # We use Temp 0.2: Just enough freedom to see details, but strict on pricing.
                generation_config = genai.types.GenerationConfig(
                    temperature=0.2, 
                    top_p=0.95, 
                    top_k=40
                )

                # --- THE "SHERLOCK HOLMES" PROMPT ---
                # We force it to list EVIDENCE before it decides the model name.
                prompt = f"""
                Act as a professional Watch & Tech authenticator.
                
                STEP 1: VISUAL EVIDENCE GATHERING
                - Read the text on the dial/label EXACTLY.
                - Look for "Made in Japan" or "21 Jewels" text.
                - Identify any serial numbers or reference codes.
                
                STEP 2: IDENTIFICATION
                - Based *only* on the evidence, what is the specific Model Reference? (e.g. SKX007K vs SKX007J)
                
                STEP 3: PRICING
                - Give a single target price in {currency} based on the condition seen.
                
                OUTPUT JSON ONLY:
                {{
                    "evidence_found": "Text found on item...",
                    "title": "Precise Model Title",
                    "target_price": "£XXX",
                    "condition": "Condition details",
                    "description": "Sales description",
                    "shots": {{
                        "Face": "00:00 (Find the steadiest shot)",
                        "Detail": "00:00 (Focus on text/logo)",
                        "Strap/Side": "00:00"
                    }}
                }}
                """
                
                response = model.generate_content(
                    [prompt, video_file],
                    generation_config=generation_config,
                    safety_settings=[{"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}]
                )
                
                match = re.search(r"\{.*\}", response.text, re.DOTALL)
                if match:
                    data = json.loads(match.group(0))

                    # DISPLAY
                    st.header(data.get("title"))
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Target Price", data.get("target_price"))
                    col2.caption(f"Evidence: {data.get('evidence_found')}")
                    
                    st.write(data.get("description"))
                    
                    st.markdown("---")
                    st.subheader("📸 Sharpest Screenshots Found")
                    
                    cols = st.columns(3)
                    idx = 0
                    for shot_name, time_str in data.get("shots", {}).items():
                        if time_str and time_str != "null":
                            # USE THE NEW SHARPNESS CHECKER
                            photo = get_sharpest_frame("temp_video.mp4", time_str)
                            if photo is not None:
                                with cols[idx % 3]:
                                    st.image(photo, caption=f"{shot_name} (Best Frame)", use_container_width=True)
                                    idx += 1
                else:
                    st.error("AI Error")
                    st.write(response.text)

            except Exception as e:
                st.error(f"Error: {e}")
