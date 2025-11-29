import streamlit as st
import google.generativeai as genai
import json
import time

# 1. SETUP
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
else:
    st.error("Missing API Key.")
    st.stop()

st.set_page_config(page_title="eBay Video Lister", page_icon="🎥")
st.title("🎥 eBay Video Auto-Lister")
st.write("Upload a quick video walkthrough (10-30s). We'll do the rest.")

# 2. VIDEO UPLOADER
uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov"])

if uploaded_file:
    # We need to save the video to a temporary file first
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())
    
    st.video(uploaded_file)
    
    if st.button("✨ Analyze Video"):
        with st.spinner("Uploading to Gemini (this takes a moment)..."):
            try:
                # Upload to Google's Server
                video_file = genai.upload_file(path="temp_video.mp4")
                
                # Wait for processing (Video isn't instant like images)
                while video_file.state.name == "PROCESSING":
                    time.sleep(2)
                    video_file = genai.get_file(video_file.name)

                if video_file.state.name == "FAILED":
                    st.error("Video processing failed.")
                    st.stop()

                # CALL GEMINI 2.0 FLASH
                model = genai.GenerativeModel(model_name="gemini-2.0-flash")
                
                prompt = """
                You are an expert eBay Power Seller. Watch this video carefully.
                Output a valid JSON object with this structure:
                {
                    "title": "SEO Optimized Title (Max 80 chars)",
                    "price_range": "Low - High",
                    "condition": "Specific flaws mentioned or seen",
                    "item_specifics": {
                        "Brand": "...",
                        "Color": "...",
                        "Type": "..."
                    },
                    "description": "Persuasive sales description",
                    "suggested_timestamps": [
                        "00:02 (Front view)", 
                        "00:05 (Back view)", 
                        "00:10 (Tag/Label)",
                        "00:15 (Defects if any)"
                    ]
                }
                """
                
                response = model.generate_content([prompt, video_file])
                
                # CLEANUP
                clean_text = response.text.replace("```json", "").replace("```", "")
                data = json.loads(clean_text)

                # DISPLAY TABS
                tab1, tab2 = st.tabs(["📝 Listing Draft", "💾 Data for eBay"])

                with tab1:
                    st.header(data["title"])
                    st.success(f"💰 Est. Price: {data['price_range']}")
                    st.warning(f"🔎 Condition: {data['condition']}")
                    
                    st.subheader("📸 Suggested Screenshots")
                    st.write("Gemini found the best angles at these times:")
                    for stamp in data["suggested_timestamps"]:
                        st.code(stamp)

                    st.write("### Description")
                    st.write(data["description"])

                with tab2:
                    st.json(data)
                    # This is where we would add the "Export CSV" button
                    st.download_button(
                        label="Download JSON for eBay",
                        data=json.dumps(data, indent=2),
                        file_name="ebay_listing.json",
                        mime="application/json"
                    )

            except Exception as e:
                st.error(f"Error: {e}")
