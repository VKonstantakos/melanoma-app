"""Streamlit web app for melanoma detection"""

import time
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from classification import *

from tempfile import NamedTemporaryFile
temp_file = NamedTemporaryFile(delete=False)


#---------------------------------#

# Page layout
## Page expands to full width

st.set_page_config(page_title='Melanoma detection - Deep Learning',
                   layout='wide')


#---------------------------------#

## Sidebar options

st.sidebar.title("Prediction Settings")
st.sidebar.text('')

st.sidebar.write("""
:warning: NOT FOR MEDICAL USE!

If you continue, you assume all liability when using the system.""")
agree = st.sidebar.checkbox("I agree.")


models = ['EfficientNetB3 (85%)', 'InceptionV3 (84%)', 'ResNet50 (79%)',
            'InceptionResNetV2 (79%)', 'DenseNet201 (69%)', 'NASNet (69%)']

if agree:
    model_choice = []
    st.sidebar.write('Do you want to use a single or multiple models?')
    build_choice = st.sidebar.radio(
    '', ('Single', 'Multiple'))

    if build_choice == 'Single':
        st.sidebar.write('Which model do you want to use for the prediction?')
        model_choice.append(st.sidebar.radio(
            '', models))
    else:
        model_choice = []
        st.sidebar.write('Which models do you want to include?')
        option_1 = st.sidebar.checkbox('EfficientNetB3 (85%)')
        option_2 = st.sidebar.checkbox('InceptionV3 (84%)')
        option_3 = st.sidebar.checkbox('ResNet50 (79%)')
        option_4 = st.sidebar.checkbox('InceptionResNetV2 (79%)')
        option_5 = st.sidebar.checkbox('DenseNet201 (69%)')
        option_6 = st.sidebar.checkbox('NASNet (69%)')
        options = [option_1, option_2, option_3, option_4, option_5, option_6]

        for idx, option in enumerate(options):
            if option:
                model_choice.append(models[idx])

        if len(model_choice) < 2:
            st.sidebar.warning('Select at least 2 models.')


#---------------------------------#

## Main Page options

col1, col2, col3 = st.beta_columns([1, 6, 1])


with col2:
    st.title("Melanoma detection using Convolutional Neural Networks")

with col2:
    st.write("""
    This is a prototype system for detecting melanoma or other skin conditions from images using CNNs.

    Please upload a dermoscopic or camera image for classification. Dermoscopic images are more accurate. Here's an example:
    """)

example_image = np.array(Image.open("media/melanoma.jpg"))
with col2:
    st.image(example_image, caption="An example input of melanoma.", width=300)

if (agree == True and build_choice == 'Single') or (agree == True and build_choice == 'Multiple' and len(model_choice) > 1):
    with col2:
        file = st.file_uploader("Upload an image of the affected area.", type=['jpg', 'png'])


    with col2:

        if file is not None:
            img = Image.open(file)
            temp_file.write(file.getvalue())
            st.image(img, caption='Uploaded skin lesion.', width=300)

            if st.button('Predict'):
                st.write("")
                st.write("Classifying...")

                start_time = time.time()
                scores = np.zeros((1, 7))

                for choice in model_choice:
                    label, score = image_classification(choice, temp_file.name)
                    scores += score

                    execute_bar = st.progress(0)
                    for percent_complete in range(100):
                        time.sleep(0.001)
                        execute_bar.progress(percent_complete + 1)

                scores = scores/len(model_choice)
                label = np.argmax(scores)

                if label == 0:
                    st.write("Diagnosis: Actinic keratosis / Bowen's disease")
                    st.write("Lesion Type: Pre-Malignant / Malignant")
                elif label == 1:
                    st.write("Diagnosis: Basal cell carcinoma")
                    st.write("Lesion Type: Malignant")
                elif label == 2:
                    st.write("Diagnosis: Benign keratosis (Solar lentigo / Seborrheic keratosis / Lichen planus-like keratosis)")
                    st.write("Lesion Type: Benign")
                elif label == 3:
                    st.write("Diagnosis: Dermatofibroma")
                    st.write("Lesion Type: Benign")
                elif label == 4:
                    st.write("Diagnosis: Melanoma")
                    st.write("Lesion Type: Malignant")
                elif label == 5:
                    st.write("Diagnosis: Melanocytic nevus")
                    st.write("Lesion Type: Benign")
                elif label == 6:
                    st.write("Diagnosis: Vascular lesion")
                    st.write("Lesion Type: Mostly benign but can be malignant")


                # Show predictions
                results = pd.DataFrame({
                    'Diagnosis': ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC'],
                    'Confidence Score': [float(x)*100 for x in scores[0]]
                })

                st.dataframe(results.style.format({"Confidence Score": "{:.2f}"}))

                st.write("Took {} seconds to run.".format(
                    round(time.time() - start_time, 2)))

#---------------------------------#
