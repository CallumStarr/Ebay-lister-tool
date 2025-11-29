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

# --- HELPER: SMART FRAME HUNTER ---
def get_sharpest_frame(video_path, timestamp_str):
    try:
        minutes, seconds = map(int, timestamp_str.split(':'))
        center_time = (minutes * 60) + seconds
        
        cap = cv2.VideoCapture(video_path)
        best_score = 0
        best_frame = None
        offsets = [0, -0.5, 0.5] 
        
        for offset in offsets:
            target_time = center_time + offset
            if target_time < 0: continue
            cap.set(cv2.CAP_PROP_POS_MSEC, target_time * 1000)
            ret, frame = cap.read()
            if ret:
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
    st.info("Mode: Universal Reseller (High Precision)")

st.title("🎥 eBay Video Auto-Lister")
st.write(f"Upload a video of ANY item. We'll list it in {currency}.")

uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

if uploaded_file:
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())
    
    st.video(uploaded_file)
    
    if st.button("✨ Analyze Video"):
        with st.spinner("Reading tags, labels & defects..."):
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

                # --- UNIVERSAL PROMPT ---
                # This works for Shoes, Tech, Tools, Clothes, etc.
                prompt = f"""
                Act as an expert eBay Power Seller for ALL categories (Electronics, Clothing, Collectibles).
                
                STEP 1: EVIDENCE GATHERING (The "Sherlock" Phase)
                - Read ANY text labels, tags, or stickers visible.
                - Look for Model Numbers (e.g., on the bottom of a drill, inside a shoe tongue, or back of a camera).
                - Identify materials (Leather, Plastic, Metal) and Brand Logos.
                
                STEP 2: IDENTIFICATION
                - Use the evidence to name the EXACT item.
                - If it's clothing, find the Size Tag.
                - If it's electronics, find the Model Number.
                
                STEP 3: PRICING
                - Give a target price in {currency} based on the condition seen.
                
                OUTPUT JSON ONLY:
                {{
                    "evidence_found": "e.g. Found 'Size 9' on tongue, 'Gore-Tex' label...",
                    "title": "SEO Title (Brand + Model + Key Specs)",
                    "target_price": "{currency}XXX",
                    "condition": "Specific condition notes (scratches, stains, wear)",
                    "description": "Professional sales description",
                    "shots": {{
                        "Main": "00:00 (Best full view)",
                        "Label/Tag": "00:00 (The crucial ID tag)",
                        "Defect/Detail": "00:00 (Close up of wear or features)"
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
                    col2.caption(f"🔎 Evidence: {data.get('evidence_found')}")
                    
                    st.success(f"**Condition:** {data.get('condition')}")
                    st.write(data.get("description"))
                    
                    st.markdown("---")
                    st.subheader("📸 Proof of Item")
                    
                    cols = st.columns(3)
                    idx = 0
                    for shot_name, time_str in data.get("shots", {}).items():
                        if time_str and time_str != "null":
                            photo = get_sharpest_frame("temp_video.mp4", time_str)
                            if photo is not None:
                                with cols[idx % 3]:
                                    st.image(photo, caption=f"{shot_name}", use_container_width=True)
                                    idx += 1
                else:
                    st.error("AI Error")
                    st.write(response.text)

            except Exception as e:
                st.error(f"Error: {e}")
