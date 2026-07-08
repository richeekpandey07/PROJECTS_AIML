import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import joblib

# ==========================================

# PAGE CONFIGURATION

# ==========================================

st.set_page_config(
page_title="AI Gender Classification System",
page_icon="🧠",
layout="wide"
)

# ==========================================

# CUSTOM CSS

# ==========================================

st.markdown("""

<style>

.main {
    background-color: #f8fafc;
}

.big-title {
    text-align:center;
    font-size:48px;
    font-weight:bold;
    color:#4F46E5;
}

.sub-title {
    text-align:center;
    font-size:20px;
    color:#64748b;
}

.prediction-box {
    padding:20px;
    border-radius:15px;
    background:#eef2ff;
    box-shadow:0px 4px 15px rgba(0,0,0,0.1);
}

.footer {
    text-align:center;
    padding:20px;
    margin-top:30px;
}

</style>

""", unsafe_allow_html=True)

# ==========================================

# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():
    return joblib.load("male_female_model .pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Error Loading Model: {e}")
    st.stop()

IMG_SIZE = 64

# ==========================================
# SIDEBAR

# ==========================================

st.sidebar.title("📌 Project Information")

st.sidebar.info("""

### AI Gender Classification

This application uses Machine Learning to predict whether an uploaded image belongs to:

👨 Male

👩 Female

Built using:

* Python
* Scikit-Learn
* Streamlit
* NumPy
  """)

st.sidebar.success("Model: Logistic Regression")

# ==========================================

# HEADER

# ==========================================

st.markdown("""

<div class="big-title">
🧠 AI-Powered Gender Classification System
</div>

<div class="sub-title">
Upload a face image and let Machine Learning predict the gender with confidence scores.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ==========================================

# FILE UPLOAD

# ==========================================

uploaded_file = st.file_uploader(
"📤 Upload an Image",
type=["jpg", "jpeg", "png"]
)

# ==========================================
# PREDICTION
# ==========================================

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📷 Uploaded Image")

            st.image(
                image,
                use_container_width=True
            )

    except Exception as e:
        st.error(f"❌ Prediction Error: {e}")
    # PREPROCESSING

    resized = image.resize((IMG_SIZE, IMG_SIZE))
    resized = np.array(resized)

    resized = resized / 255.0

    resized = resized.flatten()

    # PREDICTION

    prediction = model.predict([resized])[0]

    probability = None

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba([resized])[0]

    with col2:

        st.subheader("🎯 Prediction Result")

        if prediction == 0:
            st.success("👨 MALE")
        else:
            st.success("👩 FEMALE")

        if probability is not None:

            male_prob = probability[0] * 100
            female_prob = probability[1] * 100

if uploaded_file is not None:

    try:
        # your prediction code

        confidence = max(male_prob, female_prob)

        st.metric(
            label="Confidence Score",
            value=f"{confidence:.2f}%"
        )

        st.markdown("### 📊 Probability Analysis")

        st.write(f"👨 Male Probability: **{male_prob:.2f}%**")
        st.progress(float(probability[0]))

        st.write(f"👩 Female Probability: **{female_prob:.2f}%**")
        st.progress(float(probability[1]))

        chart_data = pd.DataFrame({
            "Gender": ["Male", "Female"],
            "Probability": [male_prob, female_prob]
        })

        st.bar_chart(chart_data.set_index("Gender"))

    except Exception as e:
        st.error(f"❌ Prediction Error: {e}")

# ==========================================
# FEATURES
# ==========================================
st.markdown("---")

st.markdown("""

## ✨ Features

✅ Real-Time Gender Prediction

✅ Machine Learning Based Classification

✅ Confidence Score Visualization

✅ Interactive User Interface

✅ Streamlit Deployment Ready

✅ GitHub Portfolio Project
""")

# ==========================================

# DEVELOPER SECTION

# ==========================================

st.markdown("---")

st.markdown("""

## 👨‍💻 Developer

### Richeek Pandey

🎓 Artificial Intelligence & Machine Learning Enthusiast

🔗 LinkedIn:
https://www.linkedin.com/in/richeek-pandey-9954783a9

🔗 GitHub:
https://github.com/richeekpandey07
""")

# ==========================================

# FOOTER

# ==========================================

st.markdown("---")

st.markdown(
"""

<div class="footer">

⭐ Thank you for using the AI Gender Classification System ⭐

Made with ❤️ using Streamlit & Scikit-Learn

</div>
""",
unsafe_allow_html=True
)
