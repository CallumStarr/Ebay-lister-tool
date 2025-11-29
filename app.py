import streamlit as st
import google.generativeai as genai
import json
import time
import re
import cv2  # The new tool for grabbing images
import os

# 1. SETUP
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
else:
    st.error("Missing API Key.")
    st.stop()

st.set_page_config(page_title="eBay Video Lister", page_icon="🎥")

# --- HELPER FUNCTION: GRAB IMAGE FROM VIDEO ---
def capture_frame(video_path, timestamp_str):
    """
    Goes to a specific time (e.g., '00:04') and grabs the photo.
    """
    try:
        # Convert "00:04" to 4 seconds
        minutes, seconds = map(int, timestamp_str.split(':'))
        total_seconds = (minutes * 60) + seconds
        
        cap = cv2.VideoCapture(video_path)
        # Jump to the specific millisecond
        cap.set(cv2.CAP_PROP_POS_MSEC, total_seconds * 1000)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # Convert colors from BGR (Video standard) to RGB (Image standard)
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None
    except Exception as e:
        return None

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Settings")
    currency = st.selectbox("Currency", ["£ (GBP)", "$ (USD)", "€ (EUR)"])
    st.info("💡 Tip: Ensure you show the dial, caseback, and defects clearly.")

st.title("🎥 eBay Video Auto-Lister")

# 2. VIDEO UPLOADER
uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

if uploaded_file:
    # Save video locally
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())
    
    st.video(uploaded_file)
    
    if st.button("✨ Analyze & Capture Photos"):
        with st.spinner("Watching video & extracting photos..."):
            try:
                # UPLOAD TO GOOGLE
                video_file = genai.upload_file(path="temp_video.mp4")
                while video_file.state.name == "PROCESSING":
                    time.sleep(2)
                    video_file = genai.get_file(video_file.name)

                if video_file.state.name == "FAILED":
                    st.error("Video processing failed.")
                    st.stop()

                # CALL GEMINI
                model = genai.GenerativeModel('gemini-2.0-flash')
                
                # --- UPDATED PROMPT: NO HALLUCINATIONS ---
                prompt = f"""
                You are an expert vintage reseller. Watch this video carefully.
                
                1. IDENTIFICATION: Look for specific text (e.g. '21 Jewels' = 'J' Model).
                2. TIMESTAMPS: Identify the BEST clear frame for:
                   - "Dial" (Face of item)
                   - "Detail" (A key feature or tag)
                   - "Defect" (If any damage exists)
                   - "Back" (ONLY if clearly shown. If not shown, return null)
                
                Output JSON ONLY:
                {{
                    "title": "SEO Title (Include J/K variant if Seiko)",
                    "price": "Target Price in {currency}",
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
                    safety_settings=[{"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}]
                )
                
                # PARSE JSON
                match = re.search(r"\{.*\}", response.text, re.DOTALL)
                if match:
                    data = json.loads(match.group(0))

                    # --- DISPLAY RESULTS ---
                    st.header(data.get("title"))
                    st.success(f"💰 Target Price: {data.get('price')}")
                    st.write(data.get("description"))
                    
                    st.markdown("---")
                    st.subheader("📸 Auto-Captured Screenshots")
                    
                    # GRID DISPLAY FOR PHOTOS
                    cols = st.columns(3)
                    idx = 0
                    
                    # Loop through the timestamps Gemini found
                    for shot_name, time_str in data.get("shots", {}).items():
                        if time_str:
                            # CALL OUR NEW HELPER FUNCTION
                            photo = capture_frame("temp_video.mp4", time_str)
                            
                            if photo is not None:
                                with cols[idx % 3]:
                                    st.image(photo, caption=f"{shot_name} ({time_str})", use_container_width=True)
                                    idx += 1
                        else:
                            # If Gemini returned null (e.g. for Caseback), we ignore it safely
                            pass

                else:
                    st.error("AI could not generate JSON data.")
                    st.write(response.text)

            except Exception as e:
                st.error(f"Error: {e}")
