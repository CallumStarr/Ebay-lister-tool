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

# --- HELPER: GET 3 CANDIDATE FRAMES ---
def get_burst_frames(video_path, timestamp_str):
    """
    Returns a list of 3 images: 
    1. 1 second BEFORE the timestamp
    2. EXACT timestamp
    3. 1 second AFTER the timestamp
    """
    try:
        minutes, seconds = map(int, timestamp_str.split(':'))
        center_time = (minutes * 60) + seconds
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return []
        
        frames = []
        # Check -0.5s, 0.0s, +0.5s
        offsets = [-0.6, 0.0, 0.6] 
        
        for offset in offsets:
            target_time = center_time + offset
            if target_time < 0: continue
            
            cap.set(cv2.CAP_PROP_POS_MSEC, target_time * 1000)
            ret, frame = cap.read()
            
            if ret:
                # Convert to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(rgb_frame)
        
        cap.release()
        return frames
    except:
        return []

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Settings")
    currency = st.selectbox("Currency", ["£ (GBP)", "$ (USD)", "€ (EUR)", "¥ (JPY)"])
    st.info("Mode: Burst Capture (Select Best)")

st.title("🎥 eBay Video Auto-Lister")
st.write(f"Upload a video. We'll find the best shots.")

uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

if uploaded_file:
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())
    
    st.video(uploaded_file)
    
    if st.button("✨ Analyze Video"):
        with st.spinner("Analyzing & Taking Burst Photos..."):
            try:
                # UPLOAD
                video_file = genai.upload_file(path="temp_video.mp4")
                while video_file.state.name == "PROCESSING":
                    time.sleep(1)
                    video_file = genai.get_file(video_file.name)

                if video_file.state.name == "FAILED":
                    st.error("Video processing failed.")
                    st.stop()

                model = genai.GenerativeModel('gemini-2.0-flash')
                
                generation_config = genai.types.GenerationConfig(
                    temperature=0.2, 
                    top_p=0.95, 
                    top_k=40
                )

                prompt = f"""
                Act as an expert eBay Seller.
                
                STEP 1: EVIDENCE & ID
                - Read ALL text/tags/labels.
                - Identify Brand, Model, Size, Material.
                
                STEP 2: PRICING
                - Give a target price in {currency} based on condition.
                
                OUTPUT JSON ONLY:
                {{
                    "title": "SEO Title",
                    "target_price": "{currency}XXX",
                    "condition": "Condition notes",
                    "description": "Sales description",
                    "shots": {{
                        "Main View": "00:00 (Full item visible)",
                        "Label/Text": "00:00 (Close up of text)",
                        "Flaws/Detail": "00:00 (Any damage or feature)"
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

                    # DISPLAY LISTING
                    st.header(data.get("title"))
                    col1, col2 = st.columns(2)
                    col1.metric("Target Price", data.get("target_price"))
                    col2.success(f"Condition: {data.get('condition')}")
                    st.write(data.get("description"))
                    
                    st.markdown("---")
                    st.subheader("📸 Select The Best Shot")
                    st.caption("We took 3 photos for each moment. Pick the one with the best angle.")
                    
                    # LOOP THROUGH SHOTS
                    for shot_name, time_str in data.get("shots", {}).items():
                        if time_str and time_str != "null":
                            st.write(f"### 🎞️ {shot_name} (around {time_str})")
                            
                            # GET BURST (3 photos)
                            burst_photos = get_burst_frames("temp_video.mp4", time_str)
                            
                            if burst_photos:
                                c1, c2, c3 = st.columns(3)
                                # Display them side by side
                                if len(burst_photos) > 0: c1.image(burst_photos[0], caption="Early", use_container_width=True)
                                if len(burst_photos) > 1: c2.image(burst_photos[1], caption="Exact", use_container_width=True)
                                if len(burst_photos) > 2: c3.image(burst_photos[2], caption="Late", use_container_width=True)
                            else:
                                st.warning("Could not extract frames.")

                else:
                    st.error("AI Error")
                    st.write(response.text)

            except Exception as e:
                st.error(f"Error: {e}")
