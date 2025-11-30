import streamlit as st
import google.generativeai as genai
import json
import time
import re
import cv2
import numpy as np
import pandas as pd

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
    currency_code = st.selectbox("Currency", ["GBP", "USD", "EUR", "JPY"])
    st.caption("Mode: eBay CSV Exporter")

st.title("🎥 eBay Video Auto-Lister")

product_hint = st.text_input("Product Name/Code (Optional - e.g. 'SKX007J')")

uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

if uploaded_file:
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())
    st.video(uploaded_file)
    
    if st.button("✨ Analyze & Create CSV"):
        with st.spinner("Generating eBay Data..."):
            try:
                # UPLOAD
                video_file = genai.upload_file(path="temp_video.mp4")
                while video_file.state.name == "PROCESSING":
                    time.sleep(1)
                    video_file = genai.get_file(video_file.name)

                model = genai.GenerativeModel('gemini-2.0-flash')

                # --- STEP 1: ANALYSIS & DESCRIPTION ---
                detective_prompt = f"""
                Act as an eBay Listing Bot.
                
                USER HINT: "{product_hint}"
                
                1. IDENTIFY: Use visual evidence + hint to ID the item.
                2. WRITE TITLE: Create an SEO-rich title (Max 80 chars).
                3. WRITE DESCRIPTION: Professional HTML-ready description.
                4. SPECIFICS: List Brand, MPN (Model Number), and Type.
                
                Output JSON:
                {{
                    "seo_title": "Full Title",
                    "description": "HTML description",
                    "condition_id": "3000 (Used) or 1000 (New)",
                    "item_specifics": {{ "Brand": "...", "Model": "...", "Type": "..." }},
                    "shots": [
                        {{ "label": "Main", "time": "00:00" }},
                        {{ "label": "Label", "time": "00:00" }},
                        {{ "label": "Detail", "time": "00:00" }}
                    ]
                }}
                """
                
                detective_resp = model.generate_content(
                    [detective_prompt, video_file],
                    generation_config=genai.types.GenerationConfig(temperature=0.2)
                )
                
                detective_data = json.loads(re.search(r"\{.*\}", detective_resp.text, re.DOTALL).group(0))
                
                # --- STEP 2: PRICING ---
                valuator_prompt = f"""
                You are an eBay Power Seller.
                Item: {detective_data['seo_title']}
                Hint: {product_hint}
                
                Output a SINGLE number for the list price in {currency_code}.
                Do not include currency symbols. Just the number (e.g. 275.00).
                
                Output JSON: {{ "price": "275.00" }}
                """
                
                valuator_resp = model.generate_content(
                    valuator_prompt,
                    generation_config=genai.types.GenerationConfig(temperature=0.0)
                )
                price_data = json.loads(re.search(r"\{.*\}", valuator_resp.text, re.DOTALL).group(0))

                # --- DISPLAY ---
                st.header(detective_data["seo_title"])
                st.metric("List Price", f"{currency_code} {price_data['price']}")
                
                st.markdown("---")
                
                # --- CREATE CSV FOR EBAY ---
                # These headers map to eBay's official File Exchange format
                csv_data = {
                    "*Action": ["Add"],
                    "*Category": ["123456 (Check ID)"], # Placeholder
                    "*Title": [detective_data["seo_title"]],
                    "*Description": [detective_data["description"]],
                    "*ConditionID": [detective_data["condition_id"]],
                    "*StartPrice": [price_data["price"]],
                    "*Quantity": [1],
                    "*Format": ["FixedPrice"],
                    "Currency": [currency_code]
                }
                
                # Add Item Specifics (C:Brand, C:Model)
                for key, val in detective_data["item_specifics"].items():
                    csv_data[f"C:{key}"] = [val]

                df = pd.DataFrame(csv_data)
                csv = df.to_csv(index=False).encode('utf-8')

                st.download_button(
                    label="📥 Download eBay CSV",
                    data=csv,
                    file_name="ebay_draft.csv",
                    mime="text/csv",
                )
                
                st.info("ℹ️ Upload this CSV to eBay Seller Hub -> Reports -> Upload.")

                st.markdown("---")
                st.subheader("📸 Proof of Item (Save these!)")
                
                for shot in detective_data.get("shots", []):
                    label = shot.get("label", "Shot")
                    time_str = shot.get("time")
                    if time_str and time_str != "null":
                        photos = get_burst_frames("temp_video.mp4", time_str)
                        if photos:
                            st.write(f"**{label}** ({time_str})")
                            st.image(photos[1], caption="Best Shot", use_container_width=True) # Show middle shot

            except Exception as e:
                st.error(f"Error: {e}")
