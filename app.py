 


import streamlit as st
import pandas as pd
from transformers import pipeline
import re

st.set_page_config(page_title="India Crop & Rainfall As Per Data by Devesh", layout="wide")

st.markdown("""
    <style>
    
    [data-testid="stAppViewContainer"] {
        background-image: url("https://preview.redd.it/need-advice-about-rain-sim-with-slow-motion-v0-1nkb022xlxkf1.gif?width=658&auto=webp&s=47bcdfc2926c93f21afb216be671550fdc4e7f51");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

     
    .glass-card {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 20px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 2rem;
        margin: 1rem 0;
        color: #fff;
        box-shadow: 0 4px 30px rgba(0,0,0,0.2);
    }

     
    h1 {
        color: #ffffff;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.8);
        text-align: center;
        font-size: 3rem;
    }

    
    [data-testid="stSidebar"] {
        background: rgba(0, 0, 0, 0.4);
        color: white;
    }

     
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #00c6ff, #0072ff);
    }

    </style>
""", unsafe_allow_html=True)

 
@st.cache_data
def load_data():
    rainfall = pd.read_csv("Sub_Division_IMD_2017.csv")
    crops = pd.read_csv("crop_yield.csv")
    return rainfall, crops

rainfall_df, crop_df = load_data()

 

generator = pipeline("text-generation", model="distilgpt2")

 
st.title("India Crop and Rainfall Chat Prototype By Devesh")
st.write("Ask any question about rainfall or crop data in India. Example: *'Tell me how much crop Goa and Maharashtra produce.'*")
st.write(" Data used from https://www.data.gov.in/ ")


query = st.text_input("Type your question")

if query:
    query_lower = query.lower()
    is_rainfall = bool(re.search(r"rain|precipitation", query_lower))
    is_crop = bool(re.search(r"crop|yield|production", query_lower))

   
    states_mentioned = [state for state in rainfall_df['SUBDIVISION'].unique() if state.lower() in query_lower] or \
                       [state for state in crop_df['State'].unique() if state.lower() in query_lower]

    if is_rainfall and not is_crop:
        st.write("🌧 **Rainfall Data**:")
        for state in states_mentioned:
            rain_data = rainfall_df[rainfall_df["SUBDIVISION"].str.contains(state, case=False, na=False)]
            if not rain_data.empty:
                avg_rain = rain_data["ANNUAL"].mean()
                st.write(f"{state} → Avg Rainfall: {avg_rain:.2f} mm")

    elif is_crop and not is_rainfall:
        st.write("🌾 **Crop Data**:")
        for state in states_mentioned:
            crop_data = crop_df[crop_df["State"].str.contains(state, case=False, na=False)]
            if not crop_data.empty:
                avg_prod = crop_data["Production"].mean()
                avg_yield = crop_data["Yield"].mean()
                st.write(f"{state} → Avg Production: {avg_prod:.2f}, Avg Yield: {avg_yield:.2f}")

    else:
        st.write("🤖 **AI Analysis:**")
        prompt = f"Question: {query}\nUse Indian crop and rainfall data contextually to give insights."
        output = generator(prompt, max_new_tokens=100, num_return_sequences=1)
        st.write(output[0]["generated_text"])
