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
    currency_code = st.selectbox("Currency", ["GBP", "USD", "EUR", "JPY"])
    st.caption("Mode: Forensic ID + CSV Export")

st.title("🎥 eBay Video Auto-Lister")

product_hint = st.text_input("Product Name/Code (Optional - e.g. 'SKX007J')")

uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

if uploaded_file:
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())
    st.video(uploaded_file)
    
    if st.button("✨ Analyze & Create CSV"):
        with st.spinner("Forensic ID in progress (Reading text on item)..."):
            try:
                # UPLOAD
                video_file = genai.upload_file(path="temp_video.mp4")
                while video_file.state.name == "PROCESSING":
                    time.sleep(1)
                    video_file = genai.get_file(video_file.name)

                model = genai.GenerativeModel('gemini-2.0-flash')

                # --- STEP 1: THE FORENSIC DETECTIVE (The logic that worked) ---
                detective_prompt = f"""
                Act as a Forensic Text Extractor.
                
                USER HINT: "{product_hint}"
                
                TASK 1: READ THE ITEM (Do not guess)
                - Watch the video.
                - List 3 exact phrases or numbers written on the item.
                - WATCHES: Look for "DIVER'S 200m" (SKX) vs "Automatic" (SRPD). Look for "21 JEWELS".
                - TOOLS: Look for Model Numbers (e.g. DCF887) and Voltage.
                
                TASK 2: IDENTIFY
                - Use those EXACT strings to name the item.
                - If "DIVER'S 200m" is present, it is an SKX, NOT an SRPE/5KX.
                
                TASK 3: FIND SHOTS
                - Select 3 timestamps showing the Item, the Text/Label, and any Detail.
                
                Output JSON:
                {{
                    "visible_text": ["String 1", "String 2"],
                    "full_title": "Precise Item Name",
                    "condition_summary": "Short condition notes",
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
                
                # --- STEP 2: THE LISTER (Formats for eBay CSV) ---
                st.toast(f"Identified: {detective_data['full_title']}. Generating CSV...")
                
                lister_prompt = f"""
                Act as an eBay File Exchange Bot.
                
                ITEM: {detective_data['full_title']}
                CONDITION: {detective_data['condition_summary']}
                USER HINT: {product_hint}
                
                1. PRICE: Give a realistic "Buy It Now" price in {currency_code} (Number only).
                2. TITLE: Write an 80-char SEO title.
                3. SPECIFICS: Extract Brand, Model, Type.
                4. DESCRIPTION: Write a sales description in HTML format.
                
                Output JSON:
                {{
                    "price": "275.00",
                    "title": "SEO Title",
                    "description": "<p>HTML Description</p>",
                    "specifics": {{ "Brand": "...", "Model": "..." }},
                    "condition_id": "3000"
                }}
                """
                
                lister_resp = model.generate_content(
                    lister_prompt,
                    generation_config=genai.types.GenerationConfig(temperature=0.0)
                )
                
                listing_data = json.loads(re.search(r"\{.*\}", lister_resp.text, re.DOTALL).group(0))

                # --- DISPLAY ---
                st.header(listing_data["title"])
                st.metric("List Price", f"{currency_code} {listing_data['price']}")
                st.caption(f"Verified via: {detective_data['visible_text']}")
                
                st.markdown("---")
                
                # --- CSV EXPORT ---
                csv_data = {
                    "*Action": ["Add"],
                    "*Category": ["123456 (Check ID)"], 
                    "*Title": [listing_data["title"]],
                    "*Description": [listing_data["description"]],
                    "*ConditionID": [listing_data["condition_id"]],
                    "*StartPrice": [listing_data["price"]],
                    "*Quantity": [1],
                    "*Format": ["FixedPrice"],
                    "Currency": [currency_code]
                }
                
                for key, val in listing_data["specifics"].items():
                    csv_data[f"C:{key}"] = [val]

                df = pd.DataFrame(csv_data)
                csv = df.to_csv(index=False).encode('utf-8')

                st.download_button(
                    label="📥 Download eBay CSV",
                    data=csv,
                    file_name="ebay_draft.csv",
                    mime="text/csv",
                )
                st.info("Upload to: Seller Hub > Reports > Uploads")

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
