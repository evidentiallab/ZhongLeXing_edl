import os

import requests
import streamlit as st
from PIL import Image
from matplotlib import pyplot as plt
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
import plotly.graph_objects as go
from plotly.subplots import make_subplots



def get_plotly_figure(img, responses):
    uncertainty = responses["uncertainty"]
    normal = responses["normal"]
    classes = list(range(10))
    uncertainty["probs"] = [float("{:.3f}".format(i)) for i in uncertainty["probs"]]
    normal["probs"] = [float("{:.3f}".format(i)) for i in normal["probs"]]

    fig = make_subplots(1,
                        2,
                        subplot_titles=("Input Image", "Prediction Comparation"),
                        column_widths=[0.4, 1.5])
    fig.add_trace(go.Image(z=img), 1, 1)
    fig.add_trace(
        go.Bar(name=f'uncertainty:{round(uncertainty["uncertainty"][0], 2)}',
               x=classes,
               y=uncertainty["probs"],
               text=uncertainty["probs"],
               textposition='outside'
               ), 1, 2)
    fig.add_trace(
        go.Bar(name=f'traditional:NULL',
               x=classes,
               y=normal["probs"],
               text=normal["probs"],
               textposition='outside'
               ), 1, 2)

    fig.update_layout(barmode='group')
    return fig



SIZE = 192
det_url = os.getenv("BACKEND_URL", "localhost")

st.title("EDL VS TNN")
st.subheader("Introduction")
video_file = open('videos/presentation.mp4', 'rb')
video_bytes = video_file.read()
st.video(video_bytes)

st.subheader("Try it out")
col1, col2 = st.columns(2)
with col1:
    canvas_result = st_canvas(
        fill_color="#ffffff",
        stroke_width=10,
        stroke_color='#ffffff',
        background_color="#000000",
        height=150,
        width=150,
        drawing_mode='freedraw',
        key="canvas",
    )
    if canvas_result.image_data is not None:
        img_show = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
        st.image(img_show, "Write Image")

with col2:
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
         # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        img_show = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        img_show = cv2.resize(img_show, (28, 28), interpolation=cv2.INTER_NEAREST)

        st.image(img_show, "uploaded image")


if st.button("Predict"):
    url = f"http://{det_url}:7510/predict"
    data_sent = cv2.imencode('.jpg', img_show)[1].tobytes()
    files = {'img': data_sent}
    responses = requests.post(url, files=files).json()
    if len(set(responses['uncertainty']['probs'])) == 1:
        uncertainty_prediction = "I don't know"
    else:
        uncertainty_prediction = responses['uncertainty']['predict']

    fig = get_plotly_figure(img_show, responses)
    st.subheader(f"uncertainty model predict it as '{uncertainty_prediction}' with "
             f"{round(responses['uncertainty']['uncertainty'][0], 2)} uncertainty")

    st.subheader(f"tradictional model predict it as '{responses['normal']['predict']}' with "
                 f"full of confidence")
    st.subheader("Results as follow:")
    st.plotly_chart(fig)

