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

st.set_page_config(page_title="eBay Auto-Lister Pro", page_icon="🎥")

# --- HELPER: FRAME HUNTER (UPDATED) ---
def get_best_frame(video_path, timestamp_str):
    """Captures a single frame at the exact timestamp."""
    try:
        if not timestamp_str or "none" in str(timestamp_str).lower(): return None
        # Robust regex to catch MM:SS or M:SS
        clean_time = re.search(r"(\d{1,2}:\d{2})", str(timestamp_str))
        if not clean_time: return None
        
        minutes, seconds = map(int, clean_time.group(1).split(':'))
        target_time = (minutes * 60) + seconds
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return None
        
        # Grab frame at exact time
        cap.set(cv2.CAP_PROP_POS_MSEC, target_time * 1000)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            return None
    except: return None

with st.sidebar:
    st.header("⚙️ Settings")
    currency_code = st.selectbox("Currency", ["£", "$", "€", "¥"])
    st.caption("Mode: Deep Analysis (Gemini 2.0 Flash)")

st.title("🎥 eBay Auto-Lister Pro")

# --- INPUT ---
product_hint = st.text_input("Product Hint (Optional - e.g. 'Sony A6000', 'Nike Air Max')")

uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

if uploaded_file:
    # Save locally for CV2 frame extraction later
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())
    st.video(uploaded_file)
    
    if st.button("✨ Create Bulletproof Listing"):
        with st.spinner("Step 1: Optical Inspection & Identification..."):
            try:
                # UPLOAD TO GEMINI
                video_file = genai.upload_file(path="temp_video.mp4")
                while video_file.state.name == "PROCESSING":
                    time.sleep(1)
                    video_file = genai.get_file(video_file.name)

                model = genai.GenerativeModel('gemini-2.0-flash')

                # --- STEP 1: THE INSPECTOR (Visual Analysis) ---
                detective_prompt = f"""
                ROLE: You are an automated optical inspection system for eBay inventory.
                
                USER HINT: "{product_hint}"
                
                OBJECTIVE: Analyze the video frame-by-frame to extract technical specifications and visual condition.
                
                CRITICAL TASKS:
                1. IDENTIFY THE ITEM: Look for labels, model numbers on the bottom/back, and startup screens. Provide only an expert level understanding of the product
                2. ASSESS CONDITION: Look specifically for: scratches, fraying, dents, or missing parts.
                3. EXTRACT TEXT: OCR any visible text that helps identification (Serial numbers, Brand names).
                4. GENERATE SEO DATA: Create an 80-char max title using the formula: [Brand] [Model] [Key Feature] [Condition].
                
                OUTPUT REQUIREMENTS:
                Return ONLY raw JSON using this specific schema:
                {{
                    "detected_brand": "string",
                    "detected_model": "string",
                    "mpn_or_sku": "string (or null if not visible)",
                    "ebay_seo_title": "string (max 80 chars)",
                    "item_specifics": {{
                        "Color": "string",
                        "Type": "string"
                    }},
                    "condition_report": {{
                        "overall_grade": "Like New | Good | Fair | For Parts",
                        "specific_flaws": ["string", "string"],
                        "visual_reasoning": "string"
                    }},
                    "sales_description": "string (HTML formatted paragraph for eBay description)",
                    "listing_photos": [
                        {{ "label": "Main Front View", "timestamp": "MM:SS" }},
                        {{ "label": "Model/Label Macro", "timestamp": "MM:SS" }},
                        {{ "label": "Damage/Detail", "timestamp": "MM:SS" }}
                    ]
                }}
                """
                
                detective_resp = model.generate_content(
                    [detective_prompt, video_file],
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.2,
                        response_mime_type="application/json"
                    )
                )
                
                # Direct JSON load
                detective_data = json.loads(detective_resp.text)
                
                # --- STEP 2: THE APPRAISER (Logic & Pricing) ---
                st.toast(f"Identified: {detective_data['ebay_seo_title']}")
                st.spinner("Step 2: Market Analysis & Pricing...")
                
                valuator_prompt = f"""
                ROLE: You are a veteran eBay Market Analyst.
                
                INPUT DATA:
                - Item: {detective_data['detected_brand']} {detective_data['detected_model']}
                - MPN/SKU: {detective_data['mpn_or_sku']}
                - Condition: {detective_data['condition_report']['overall_grade']}
                - Flaws: {detective_data['condition_report']['specific_flaws']}
                - User Hint: "{product_hint}"
                
                TASK: Determine the optimal "Buy It Now" listing price.
                NOTE: As a specific used item, do not price at MSRP. Price based on current used market value depreciation.
                
                Output JSON:
                {{
                    "market_analysis": "string (Explain the value tier of this item)",
                    "recommended_list_price": "float (Just the number, e.g. 45.00)",
                    "pricing_strategy_note": "string (e.g. 'Priced lower due to screen scratch')"
                }}
                """
                
                valuator_resp = model.generate_content(
                    valuator_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        response_mime_type="application/json"
                    )
                )
                
                price_data = json.loads(valuator_resp.text)

                # --- DISPLAY RESULTS ---
                st.header(detective_data["ebay_seo_title"])
                
                col1, col2 = st.columns(2)
                col1.metric("Recommended Price", f"{currency_code}{price_data['recommended_list_price']}")
                col2.info(f"Strategy: {price_data['pricing_strategy_note']}")
                
                with st.expander("📝 Description & Condition", expanded=True):
                    st.markdown(detective_data["sales_description"], unsafe_allow_html=True)
                    st.markdown("---")
                    st.write(f"**Condition Grade:** {detective_data['condition_report']['overall_grade']}")
                    if detective_data['condition_report']['specific_flaws']:
                        st.warning("**Detected Flaws:**")
                        for flaw in detective_data['condition_report']['specific_flaws']:
                            st.write(f"- {flaw}")

                # --- STEP 3: EXPORT TO CSV ---
                st.markdown("---")
                
                # Clean price logic
                clean_price = str(price_data["recommended_list_price"]).replace(currency_code, '')
                
                # Flatten specs for CSV
                specs = detective_data.get("item_specifics", {})
                
                csv_dict = {
                    "*Action": ["Add"],
                    "*Title": [detective_data["ebay_seo_title"]],
                    "*Description": [detective_data["sales_description"]],
                    "*ConditionDescription": [detective_data['condition_report']['visual_reasoning']],
                    "C:Brand": [detective_data.get("detected_brand", "")],
                    "C:Model": [detective_data.get("detected_model", "")],
                    "C:MPN": [detective_data.get("mpn_or_sku", "")],
                    "*StartPrice": [clean_price],
                    "Currency": [currency_code],
                    "*Quantity": [1],
                    "*Format": ["FixedPrice"]
                }
                
                df = pd.DataFrame(csv_dict)
                csv = df.to_csv(index=False).encode('utf-8')
                
                st.download_button(
                    label="📥 Download eBay CSV",
                    data=csv,
                    file_name="ebay_listing.csv",
                    mime="text/csv",
                )
                
                st.markdown("---")
                st.subheader("📸 Optimal Photo's For Ebay Listing (Based on Video")
                
                # Display photos based on the timestamps identified by AI
                # This loop now displays only the single "Best Shot" for each timestamp.
                for shot in detective_data.get("listing_photos", []):
                    label = shot.get("label", "Shot")
                    time_str = shot.get("timestamp")
                    
                    if time_str:
                        st.write(f"**{label}** @ {time_str}")
                        # Use the new helper function to get the single best frame
                        best_frame = get_best_frame("temp_video.mp4", time_str)
                        
                        if best_frame is not None:
                            # Display the single image
                            st.image(best_frame, caption="Best Shot", use_container_width=True)

            except Exception as e:
                st.error(f"Analysis failed: {e}")
