import streamlit as st
import google.generativeai as genai
from PIL import Image

# 1. SETUP: Securely get the key
# We check if the key is in the "Vault" (Secrets)
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
else:
    st.error("Missing API Key. Go to 'Manage App' -> 'Secrets' to add it.")
    st.stop()

# 2. THE INTERFACE
st.set_page_config(page_title="eBay AI Lister", page_icon="📸")
st.title("📸 eBay Speed-Lister")
st.write("Take a photo -> Get a Title, Price, and Description.")

# 3. CAMERA & UPLOAD BUTTON
# This allows you to use your phone camera OR upload a file
uploaded_file = st.file_uploader("Take a photo of your item", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Show the image to the user
    image = Image.open(uploaded_file)
    st.image(image, caption="Ready to list!", use_container_width=True)
    
    # 4. THE MAGIC BUTTON
    if st.button("✨ Write My Listing"):
        with st.spinner("Analyzing... (Using Gemini 2.0)"):
            try:
                # ✅ WE UPDATED THIS TO THE WORKING MODEL
                model = genai.GenerativeModel('gemini-2.0-flash')
                
                prompt = """
                You are an expert eBay Power Seller.
                Analyze this image and output a JSON-style report with:
                1. SEO Optimized Title (max 80 chars)
                2. Estimated Price Range (Low-High)
                3. Item Specifics (Brand, Color, Size, etc)
                4. A Condition Report
                5. A Persuasive Description
                """
                
                response = model.generate_content([prompt, image])
                
                # Show the result
                st.markdown("---")
                st.markdown(response.text)
                st.success("Done! Copy and paste to eBay.")
                
            except Exception as e:
                st.error(f"Error: {e}")
