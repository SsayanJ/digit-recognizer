"""
To run the app
$ streamlit run digit_app.py
"""


import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model
import cv2

digit_recognizer = load_model('models/CNN_1.0_98781.h5')


# Specify canvas parameters in application
stroke_width = st.sidebar.slider("Stroke width: ", 1, 50, 20)
stroke_color = st.sidebar.color_picker("Stroke color hex: ", "#fff")
bg_color = st.sidebar.color_picker("Background color hex: ", "#000000")
bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform")
)
realtime_update = st.sidebar.checkbox("Update in realtime", True)

"""
# Draw a digit
"""
# Create a canvas component
canvas_result = st_canvas(
    # fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    background_image=Image.open(bg_image) if bg_image else None,
    update_streamlit=realtime_update,
    height=280,
    width=280,
    drawing_mode=drawing_mode,
    key="canvas",
)

submit_button = st.button("Predict digit")

if submit_button:
    st.write(np.array(canvas_result.image_data).shape)
    grayscale = canvas_result.image_data[:, :, :-1].mean(axis=2)
    grayscale = abs(grayscale - 255)
    # darkest = np.max(grayscale)
    small_img = np.zeros((28, 28))
    original_img = np.array(canvas_result.image_data, dtype='uint8')
    img = cv2.resize(original_img, (28, 28))
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    img = cv2.bitwise_not(img)
    cv2.imwrite("temp/cv2.jpg", img)
    img = img.reshape((1, 28, 28, 1))

    for i in range(28):
        for j in range(28):
            small_img[i, j] = int(grayscale[i*10:i*10+9, j*10: j*10+9].mean())
    small_img = cv2.resize(small_img, (28, 28))
    cv2.imwrite("temp/numpy.jpg", small_img)
    st.image("temp/numpy.jpg", caption="Numpy resize")
    small_img = small_img.reshape((1, 28, 28, 1))
    st.write(small_img.shape)

    preds = digit_recognizer.predict(small_img)
    st.write(preds)

    st.image("temp/cv2.jpg", caption="CV2 resize")
    preds2 = digit_recognizer.predict(img)
    st.write(preds2)
    # st.write(small_img)

# Do something interesting with the image data and paths
# if canvas_result.image_data is not None:
#     a = st.image(canvas_result.image_data)
    # b = Image(a)
# if canvas_result.json_data is not None:
#     st.dataframe(pd.json_normalize(canvas_result.json_data["objects"]))
