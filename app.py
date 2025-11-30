import streamlit as st
import google.generativeai as genai
import json
import time
import re
import cv2
import numpy as np
import pandas as pd
from io import StringIO

# 1. SETUP
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
else:
    st.error("Missing API Key.")
    st.stop()

st.set_page_config(page_title="eBay Auto-Lister", page_icon="🎥")

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
    currency_code = st.selectbox("Currency", ["GBP", "USD", "EUR", "JPY"])
    st.caption("Mode: Official eBay 'Prefill' Template")

st.title("🎥 eBay Auto-Lister")

product_hint = st.text_input("Product Name/Item Code (Optional - e.g. 'SKX007J')")

uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

if uploaded_file:
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())
    st.video(uploaded_file)
    
    if st.button("✨ Analyze & Generate CSV"):
        with st.spinner("Forensic Analysis in Progress..."):
            try:
                # UPLOAD
                video_file = genai.upload_file(path="temp_video.mp4")
                while video_file.state.name == "PROCESSING":
                    time.sleep(1)
                    video_file = genai.get_file(video_file.name)

                model = genai.GenerativeModel('gemini-2.0-flash')

                # --- STEP 1: FORENSIC DETECTIVE ---
                detective_prompt = f"""
                Act as an Expert eBay Lister.
                USER INPUT: "{product_hint}"
                
                1. IDENTIFY ITEM: Read text labels (Model #, Voltage, Jewels).
                2. GENERATE TITLE: Max 80 chars, keyword stuffed.
                3. CATEGORY: Predict the eBay category Name (e.g. "Wristwatches", "Impact Drivers").
                4. SPECIFICS: Extract Brand, Model, and Key Features.
                
                Output JSON:
                {{
                    "seo_title": "Full Title",
                    "category_name": "Category Name",
                    "specifics": {{ "Brand": "...", "Model": "..." }},
                    "visual_description": "Sales text",
                    "condition_text": "Condition details",
                    "shots": [
                        {{ "label": "Main", "time": "00:00" }},
                        {{ "label": "ID Tag", "time": "00:00" }},
                        {{ "label": "Detail", "time": "00:00" }}
                    ]
                }}
                """
                
                detective_resp = model.generate_content(
                    [detective_prompt, video_file],
                    generation_config=genai.types.GenerationConfig(temperature=0.3)
                )
                detective_data = json.loads(re.search(r"\{.*\}", detective_resp.text, re.DOTALL).group(0))
                
                # --- DISPLAY ---
                st.header(detective_data["seo_title"])
                st.write(detective_data["visual_description"])
                st.caption(f"Category: {detective_data['category_name']}")
                
                st.markdown("---")

                # --- BUILD THE EXACT TEMPLATE CSV ---
                # 1. Format "Aspects" as "Key:Value;Key:Value"
                aspects_str = ";".join([f"{k}:{v}" for k,v in detective_data["specifics"].items()])
                
                # 2. Generate SKU (Timestamp based for uniqueness)
                sku = f"AUTO-{int(time.time())}"
                
                # 3. Create the Data Rows
                csv_data = {
                    "Custom Label (SKU)": [sku],
                    "Item Photo URL": [""], # Leave blank, user adds photos manually
                    "Title": [detective_data["seo_title"]],
                    "Category": [detective_data["category_name"]],
                    "Aspects": [aspects_str]
                }
                df = pd.DataFrame(csv_data)
                
                # 4. Construct the file with the Special Header Row (#INFO...)
                # This matches your uploaded file version 1.0.0
                special_header = "#INFO,Version=1.0.0,,Template=eBay-taxonomy-mapping-template_US,"
                csv_buffer = StringIO()
                csv_buffer.write(special_header + "\n")
                df.to_csv(csv_buffer, index=False)
                
                st.download_button(
                    label="📥 Download 'Prefill' CSV",
                    data=csv_buffer.getvalue(),
                    file_name="ebay_upload.csv",
                    mime="text/csv",
                )
                
                st.info("ℹ️ Upload this to Seller Hub -> Reports -> Upload (Use the 'Create listing from file' option).")

                st.markdown("---")
                st.subheader("📸 Proof of Item")
                for shot in detective_data.get("shots", []):
                    time_str = shot.get("time")
                    if time_str and time_str != "null":
                        photos = get_burst_frames("temp_video.mp4", time_str)
                        if photos:
                            st.image(photos[1], caption=f"{shot.get('label')} ({time_str})", use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")
