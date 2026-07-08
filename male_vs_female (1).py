import streamlit as st
import numpy as np
from PIL import Image
import joblib

# ----------------------------------
# Page Configuration
# ----------------------------------
st.set_page_config(
    page_title="Male vs Female Classifier",
    page_icon="👨👩",
    layout="centered"
)

# ----------------------------------
# Load Model
# ----------------------------------
@st.cache_resource
def load_model():
    return joblib.load("male_female_model .pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

IMG_SIZE = 64

# ----------------------------------
# Header
# ----------------------------------
st.markdown("""
<h1 style='text-align:center;color:#4F46E5;'>
👨 Male vs Female Image Classifier 👩
</h1>
""", unsafe_allow_html=True)

st.markdown("""
### Upload an image and let the model predict whether the face belongs to a Male or Female.
""")

st.divider()

# ----------------------------------
# File Upload
# ----------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload Image",
    type=["jpg", "jpeg", "png"]
)

# ----------------------------------
# Prediction Section
# ----------------------------------
if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

        # Preprocessing
        resized = image.resize((IMG_SIZE, IMG_SIZE))
        resized = np.array(resized)

      
        resized = resized / 255.0

        resized = resized.flatten()

        # Prediction
        prediction = model.predict([resized])[0]

        # Probability
        probability = None
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba([resized])[0]

        with col2:

            st.subheader("Prediction Result")

            if prediction == 0:
                st.success("👨 Male")
            else:
                st.success("👩 Female")

            if probability is not None:

                st.subheader("Confidence Score")

                st.write(
                    f"👨 Male Probability: **{probability[0] * 100:.2f}%**"
                )
                st.progress(float(probability[0]))

                st.write(
                    f"👩 Female Probability: **{probability[1] * 100:.2f}%**"
                )
                st.progress(float(probability[1]))

    except Exception as e:
        st.error(f"Prediction Error: {e}")

# ----------------------------------
# Footer
# ----------------------------------

st.markdown(
    """
    <div style='text-align:center'>
        <h3>🚀 About the Developer</h3>
        <p><b>Richeek Pandey</b></p>
        <p>
            <a href="https://www.linkedin.com/in/richeek-pandey-9954783a9">
                LinkedIn Profile
            </a>
            |
            <a href="https://github.com/richeekpandey07" >
                GitHub Profile
            </a>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


