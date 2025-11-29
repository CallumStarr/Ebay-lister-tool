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
st.title("🎥 eBay Video Auto-Lister")
st.write("Upload a video (10-45s) to generate a listing.")

# 2. VIDEO UPLOADER
uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

if uploaded_file:
    # Save video locally
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())
    
    st.video(uploaded_file)
    
    if st.button("✨ Analyze Video"):
        with st.spinner("Processing video... (This takes 10-20 seconds)"):
            try:
                # A. UPLOAD TO GOOGLE
                video_file = genai.upload_file(path="temp_video.mp4")
                
                # B. WAIT LOOP (Crucial for Video)
                while video_file.state.name == "PROCESSING":
                    time.sleep(2)
                    video_file = genai.get_file(video_file.name)

                if video_file.state.name == "FAILED":
                    st.error("Google failed to process the video file.")
                    st.stop()

                # C. CONFIGURE MODEL WITH SAFETY OFF
                # This prevents empty responses due to false alarms
                model = genai.GenerativeModel('gemini-2.0-flash')
                
                # "BLOCK_NONE" tells Google: "Don't hide the answer, I trust this content."
                safety_settings = [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                ]
                
                prompt = """
                You are an expert eBay Seller. Watch this video.
                Output valid JSON only. No markdown, no conversation.
                Structure:
                {
                    "title": "SEO Title",
                    "price_range": "Estimated Price",
                    "condition": "Condition details",
                    "description": "Sales description",
                    "suggested_timestamps": ["00:05", "00:12"]
                }
                """
                
                # D. GENERATE CONTENT
                response = model.generate_content(
                    [prompt, video_file], 
                    safety_settings=safety_settings
                )
                
                # E. ROBUST JSON PARSER
                # This Regex finds the {...} block even if the AI says "Here is the JSON:"
                match = re.search(r"\{.*\}", response.text, re.DOTALL)
                
                if match:
                    json_str = match.group(0)
                    data = json.loads(json_str)

                    # DISPLAY RESULTS
                    tab1, tab2 = st.tabs(["📝 Listing", "💾 Raw Data"])
                    
                    with tab1:
                        st.header(data.get("title", "No Title"))
                        st.success(f"💰 {data.get('price_range', 'N/A')}")
                        st.info(f"🔎 {data.get('condition', 'N/A')}")
                        st.write("### Description")
                        st.write(data.get("description", ""))
                        st.write("### Best Screenshots at:")
                        st.code(", ".join(data.get("suggested_timestamps", [])))

                    with tab2:
                        st.json(data)
                else:
                    st.error("AI returned text, but not JSON. Here is what it said:")
                    st.write(response.text)

            except Exception as e:
                st.error(f"Error: {e}")
                # Debugging aid: If response exists, show it
                if 'response' in locals() and response.prompt_feedback:
                    st.warning(f"Blocked Reason: {response.prompt_feedback}")
