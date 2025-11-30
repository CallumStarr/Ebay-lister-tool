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
    st.caption("Mode: Hybrid (Guide + Create)")

st.title("🎥 eBay Auto-Lister")

# --- INPUT ---
product_hint = st.text_input("Product Name/ID Code/Part Number (Optional - e.g. 'SKX007J')")

uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

if uploaded_file:
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())
    st.video(uploaded_file)
    
    if st.button("✨ Analyze Product"):
        with st.spinner("Analyzing Product & creating Ebay listing..."):
            try:
                # UPLOAD
                video_file = genai.upload_file(path="temp_video.mp4")
                while video_file.state.name == "PROCESSING":
                    time.sleep(1)
                    video_file = genai.get_file(video_file.name)

                model = genai.GenerativeModel('gemini-2.0-flash')

                # --- STEP 1: THE CREATIVE DETECTIVE ---
                detective_prompt = f"""
                Act as an Expert eBay Lister.
                
                USER INPUT: "{product_hint}"
                
                INSTRUCTIONS:
                1. USE THE INPUT (if provided) to confirm the model identity.
                2. WATCH THE VIDEO to see the specific condition, color, and features (e.g. "Scratched", "No battery").
                3. GENERATE A FULL TITLE: Do NOT just output the model code. Write a keyword-stuffed title (Max 80 chars).
                   - Bad: "SKX007J"
                   - Good: "Seiko SKX007J Diver Automatic Watch 200m Made in Japan Jubilee"
                
                TASK:
                - Find 3 text strings visible in the video (e.g. "21 Jewels").
                - Find 3 timestamps for photos.
                
                Output JSON:
                {{
                    "visible_text": ["String 1", "String 2"],
                    "seo_title": "Full Keyword Rich Title (Brand + Model + Features)",
                    "condition_summary": "Specific notes on wear/tear seen in video",
                    "visual_description": "Rich sales description describing THIS item's condition and features.",
                    "shots": [
                        {{ "label": "Main View", "time": "00:00" }},
                        {{ "label": "ID Tag / Dial", "time": "00:00" }},
                        {{ "label": "Detail / Flaw", "time": "00:00" }}
                    ]
                }}
                """
                
                detective_resp = model.generate_content(
                    [detective_prompt, video_file],
                    generation_config=genai.types.GenerationConfig(temperature=0.3)
                )
                
                detective_data = json.loads(re.search(r"\{.*\}", detective_resp.text, re.DOTALL).group(0))
                
                # --- STEP 2: THE PRICER ---
                st.toast("Calculating list price...")
                
                valuator_prompt = f"""
                You are an eBay Power Seller. Set the "Buy It Now" Price.
                
                ITEM TITLE: {detective_data['seo_title']}
                CONDITION: {detective_data['condition_summary']}
                USER HINT WAS: "{product_hint}"
                
                STRATEGY:
                - Use the Hint to identify the exact value (e.g. J model is worth more than K).
                - Use the Condition to adjust price.
                - Give a realistic fair List Price.
                
                Output JSON:
                {{
                    "list_price": "{currency_code}XXX",
                    "price_strategy": "Explain reasoning based on Model + Condition"
                }}
                """
                
                valuator_resp = model.generate_content(
                    valuator_prompt,
                    generation_config=genai.types.GenerationConfig(temperature=0.0)
                )
                
                price_data = json.loads(re.search(r"\{.*\}", valuator_resp.text, re.DOTALL).group(0))

                # --- DISPLAY ---
                st.header(detective_data["seo_title"])
                
                col1, col2 = st.columns(2)
                col1.metric("List Price", price_data["list_price"])
                col2.success(f"Strategy: {price_data['price_strategy']}")
                
                if product_hint:
                    st.caption(f"✅ Enhanced by your info: '{product_hint}'")
                
                st.write(detective_data["visual_description"])
                st.caption(f"**Condition:** {detective_data['condition_summary']}")
                
                st.markdown("---")
                st.subheader("📸 Photo's for Listing")
                
                for shot in detective_data.get("shots", []):
                    label = shot.get("label", "Shot")
                    time_str = shot.get("time")
                    
                    if time_str and time_str != "null":
                        st.write(f"**{label}** ({time_str})")
                        photos = get_burst_frames("temp_video.mp4", time_str)
                        if photos:
                            c1, c2, c3 = st.columns(3)
                            if len(photos) > 0: c1.image(photos[0], caption="Before", use_container_width=True)
                            if len(photos) > 1: c2.image(photos[1], caption="Best", use_container_width=True)
                            if len(photos) > 2: c3.image(photos[2], caption="After", use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")
