import streamlit as st
import google.generativeai as genai
from PIL import Image

# 1. SECURITY: Grab the key from the Cloud's "Safe"
# If the key is missing, stop the app.
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
else:
    st.error("Missing API Key. Please configure secrets.")
    st.stop()

# 2. THE APP INTERFACE
st.set_page_config(page_title="eBay Lister", page_icon="⚡")
st.title("⚡ eBay Speed-Lister")

# 3. UPLOAD AND ANALYZE
uploaded_file = st.file_uploader("Upload Item Photo", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Your Item", use_container_width=True)
    
    if st.button("Generate Listing"):
        with st.spinner("Asking Gemini..."):
            try:
                model = genai.GenerativeModel('gemini-1.5-flash')
                prompt = "Analyze this image for an eBay listing. Give me a Title, Item Specifics, Condition, and Description."
                response = model.generate_content([prompt, image])
                st.markdown(response.text)
            except Exception as e:
                st.error(f"Error: {e}")
