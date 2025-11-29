import streamlit as st
import google.generativeai as genai
import json
import time
import re

# 1. SETUP
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
else:
    st.error("Missing API Key.")
    st.stop()

st.set_page_config(page_title="eBay Video Lister", page_icon="🎥")

# --- SIDEBAR SETTINGS ---
with st.sidebar:
    st.header("⚙️ Settings")
    currency = st.selectbox("Currency", ["£ (GBP)", "$ (USD)", "€ (EUR)", "¥ (JPY)"])
    model_mode = st.radio("Mode", ["Speed (Flash)", "Precision (Pro)"])
    # "Pro" mode uses a smarter model if available, otherwise defaults to Flash
    selected_model = 'gemini-2.0-flash' 

st.title("🎥 eBay Video Auto-Lister")
st.write(f"Upload a video. We'll list it in {currency}.")

# 2. VIDEO UPLOADER
uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

if uploaded_file:
    # Save video locally
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())
    
    st.video(uploaded_file)
    
    if st.button("✨ Analyze Video"):
        with st.spinner("Hunting for details (Dial text, Caseback, Condition)..."):
            try:
                # UPLOAD & WAIT
                video_file = genai.upload_file(path="temp_video.mp4")
                while video_file.state.name == "PROCESSING":
                    time.sleep(2)
                    video_file = genai.get_file(video_file.name)

                if video_file.state.name == "FAILED":
                    st.error("Video processing failed.")
                    st.stop()

                # CONFIG
                model = genai.GenerativeModel(selected_model)
                
                # --- THE UPDATED "SNIPER" PROMPT ---
                prompt = f"""
                You are an expert vintage reseller (Watches, Sneakers, Tech). 
                Watch this video frame-by-frame.
                
                CRITICAL INSTRUCTION:
                1. Look for tiny text (e.g., "Made in Japan", "21 Jewels", Model numbers on caseback).
                2. Distinguish variants (e.g., Seiko SKX007J vs SKX007K).
                3. If unsure, state "Unverified" for that specific detail.
                
                Output JSON ONLY:
                {{
                    "title": "SEO Title (Include exact variant if visible)",
                    "target_price": "Single value in {currency} (e.g. £250)",
                    "price_reasoning": "Why this price? (e.g. 'Scratches on bezel reduce value')",
                    "condition": "Strict condition report",
                    "specifics": {{
                        "Brand": "...",
                        "Model": "...",
                        "Variant": "..."
                    }},
                    "description": "Sales description",
                    "timestamps": ["00:04 (Dial shot)", "00:09 (Caseback)"]
                }}
                """
                
                # GENERATE
                response = model.generate_content(
                    [prompt, video_file],
                    safety_settings=[{"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}]
                )
                
                # PARSE JSON
                match = re.search(r"\{.*\}", response.text, re.DOTALL)
                if match:
                    data = json.loads(match.group(0))

                    # --- RESULTS UI ---
                    tab1, tab2 = st.tabs(["📝 Listing", "💾 Raw Data"])
                    
                    with tab1:
                        st.header(data.get("title"))
                        
                        # Price Box
                        col1, col2 = st.columns(2)
                        col1.metric("Target Price", data.get("target_price"))
                        col2.info(data.get("price_reasoning"))
                        
                        st.warning(f"**Condition:** {data.get('condition')}")
                        
                        st.subheader("Details identified:")
                        st.json(data.get("specifics"))
                        
                        st.write("### Description")
                        st.write(data.get("description"))
                        
                        st.write("### 📸 Capture these frames:")
                        for stamp in data.get("timestamps", []):
                            st.code(stamp)

                    with tab2:
                        st.json(data)
                else:
                    st.error("Could not read AI response.")
                    st.write(response.text)

            except Exception as e:
                st.error(f"Error: {e}")
