import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Gender Classifier",
    page_icon="🧠",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg,#0f172a,#1e293b);
}

.title {
    text-align:center;
    font-size:55px;
    font-weight:800;
    background: linear-gradient(to right,#38bdf8,#22c55e);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.subtitle{
    text-align:center;
    color:#cbd5e1;
    font-size:20px;
}

.result-box{
    padding:20px;
    border-radius:15px;
    text-align:center;
    font-size:30px;
    font-weight:bold;
    background:#1e293b;
    border:2px solid #38bdf8;
}

.footer{
    text-align:center;
    color:gray;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_my_model():
    return load_model("gender_cnn_model.keras")

model = load_my_model()
st.write("Model Input Shape:", model.input_shape)

# ---------------- HEADER ----------------
st.markdown(
    '<p class="title">👨‍💼 AI Gender Recognition System 👩‍💼</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Deep Learning Powered Male vs Female Image Classification</p>',
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------- SIDEBAR ----------------
with st.sidebar:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/4712/4712027.png",
        width=150
    )

    st.header("🚀 Project Overview")

    st.info("""
    This AI model predicts whether an uploaded face belongs to:

    ✅ Male

    ✅ Female

    using a Convolutional Neural Network (CNN).
    """)

    st.markdown("---")

    st.subheader("👨‍💻 Developer")

    st.markdown("""
    **Richeek Pandey**
    
🔗 GitHub: https://github.com/richeekpandey07

🔗 LinkedIn: https://www.linkedin.com/in/richeek-pandey

    🎓 AIML Enthusiast
    """)

# ---------------- MAIN SECTION ----------------
col1, col2 = st.columns([1,1])

with col1:

    uploaded_file = st.file_uploader(
        "📤 Upload an Image",
        type=["jpg","jpeg","png"]
    )

with col2:

    st.info("""
    📸 Upload a clear face image.

    🤖 AI will analyze the image.

    🎯 Prediction confidence will be displayed.
    """)

# ---------------- PREDICTION ----------------
# ---------------- PREDICTION ----------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    img = image.resize((64, 64))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Debug information
    st.write("Model Input Shape:", model.input_shape)
    st.write("Image Shape:", img.shape)

    try:
        prediction = model.predict(img)

    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.stop()

    confidence = float(np.max(prediction)) * 100

    if prediction[0][0] > 0.5:
        gender = "👩 Female"
        female_prob = confidence
        male_prob = 100 - confidence
    else:
        gender = "👨 Male"
        male_prob = confidence
        female_prob = 100 - confidence

    with col2:

        st.markdown(
            f"""
            <div class="result-box">
            Prediction <br><br>
            {gender}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.metric(
            "🎯 Confidence Score",
            f"{confidence:.2f}%"
        )

        st.success("Prediction Completed")

    st.markdown("---")

    st.subheader("📊 Prediction Analysis")

    c1, c2 = st.columns(2)

    with c1:
        st.write(f"👨 Male Probability : **{male_prob:.2f}%**")
        st.progress(int(male_prob))

    with c2:
        st.write(f"👩 Female Probability : **{female_prob:.2f}%**")
        st.progress(int(female_prob))

    if confidence > 90:
        st.balloons()

# ---------------- FOOTER ----------------
st.markdown("---")

st.markdown(
    """
    <div class="footer">
        Developed with ❤️ by <b>Richeek Pandey</b><br><br>

        🚀 Deep Learning | Computer Vision | Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
