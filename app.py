import streamlit as st
import google.generativeai as genai
import json
import time
import re
import cv2

# 1. SETUP
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
else:
    st.error("Missing API Key.")
    st.stop()

st.set_page_config(page_title="eBay Video Lister", page_icon="🎥")

# --- HELPER: GRAB IMAGE ---
def capture_frame(video_path, timestamp_str):
    try:
        minutes, seconds = map(int, timestamp_str.split(':'))
        total_seconds = (minutes * 60) + seconds
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, total_seconds * 1000)
        ret, frame = cap.read()
        cap.release()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None
    except:
        return None

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Settings")
    currency = st.selectbox("Currency", ["£ (GBP)", "$ (USD)", "€ (EUR)", "¥ (JPY)"])
    st.caption("Mode: Strict (Temperature 0)")

st.title("🎥 eBay Video Auto-Lister")
st.write(f"Upload a video. We'll list it in {currency}.")

# 2. VIDEO UPLOADER
uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

if uploaded_file:
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())
    
    st.video(uploaded_file)
    
    if st.button("✨ Analyze Video"):
        with st.spinner("Analyzing... (Strict Mode Active)"):
            try:
                # UPLOAD
                video_file = genai.upload_file(path="temp_video.mp4")
                while video_file.state.name == "PROCESSING":
                    time.sleep(2)
                    video_file = genai.get_file(video_file.name)

                if video_file.state.name == "FAILED":
                    st.error("Video processing failed.")
                    st.stop()

                # CONFIG
                model = genai.GenerativeModel('gemini-2.0-flash')
                
                # --- 🔥 THE FIX IS HERE: GENERATION CONFIG 🔥 ---
                # temperature=0.0 means "Zero Randomness". It picks the most likely answer every time.
                generation_config = genai.types.GenerationConfig(
                    temperature=0.0,
                    top_p=1.0, 
                    top_k=1
                )

                prompt = f"""
                You are a strict pricing algorithm. Watch this video.
                
                TASK:
                1. Identify the item EXACTLY (Brand, Model, Ref Number).
                2. Provide a single, consistent market price in {currency}.
                3. Do NOT guess. If specific flaws are visible, deduct value.
                
                Output JSON ONLY:
                {{
                    "title": "SEO Title",
                    "target_price": "Single value e.g. £250",
                    "condition": "Condition Report",
                    "description": "Sales description",
                    "shots": {{
                        "Dial": "00:00",
                        "Detail": "00:00",
                        "Back": null
                    }}
                }}
                """
                
                response = model.generate_content(
                    [prompt, video_file],
                    generation_config=generation_config, # <--- Applying the lock
                    safety_settings=[{"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}]
                )
                
                match = re.search(r"\{.*\}", response.text, re.DOTALL)
                if match:
                    data = json.loads(match.group(0))

                    # DISPLAY
                    st.header(data.get("title"))
                    st.success(f"💰 Target Price: {data.get('target_price')}")
                    st.write(data.get("description"))
                    
                    st.markdown("---")
                    st.subheader("📸 Auto-Captured Screenshots")
                    
                    cols = st.columns(3)
                    idx = 0
                    for shot_name, time_str in data.get("shots", {}).items():
                        if time_str:
                            photo = capture_frame("temp_video.mp4", time_str)
                            if photo is not None:
                                with cols[idx % 3]:
                                    st.image(photo, caption=f"{shot_name} ({time_str})", use_container_width=True)
                                    idx += 1

                else:
                    st.error("AI Error")
                    st.write(response.text)

            except Exception as e:
                st.error(f"Error: {e}")
