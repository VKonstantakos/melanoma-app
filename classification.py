import os
import time
import gdown
import numpy as np
import streamlit as st
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout


# Streamlit sharing is CPU only
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


@st.cache(show_spinner=False)
def download_weights(model_choice):
    '''
    Downloads model weights for deployment purposes.
    '''

    # Create directory
    save_dest = Path('models')
    save_dest.mkdir(exist_ok=True)

    # Download weights for the chosen model
    if model_choice == 'DenseNet201 (69%)':
        url = 'https://drive.google.com/uc?id=1HfETcESJHpnEZ-x0dj2AZfqM3dCGs8EZ'
        output = 'models/DenseNet201.h5'

        if not Path(output).exists():
            with st.spinner("Model weights were not found, downloading them. This may take a while."):
                gdown.download(url, output, quiet=False)


    elif model_choice == 'ResNet50 (79%)':
        url = 'https://drive.google.com/uc?id=16-L3FaZUhWedQy2WagEfILv2-XwGazZO'
        output = 'models/ResNet50.h5'

        if not Path(output).exists():
            with st.spinner("Model weights were not found, downloading them. This may take a while."):
                gdown.download(url, output, quiet=False)

    elif model_choice == 'InceptionV3 (84%)':
        url = 'https://drive.google.com/uc?id=1wmNirs6NwvLEamvGSwLzRlJHLwHivPit'
        output = 'models/Inceptionv3.h5'

        if not Path(output).exists():
            with st.spinner("Model weights were not found, downloading them. This may take a while."):
                gdown.download(url, output, quiet=False)

    elif model_choice == 'EfficientNetB3 (85%)':
        url = 'https://drive.google.com/uc?id=1lebK-70tcon9hUWjfTm-sqsjqH2Ny83f'
        output = 'models/EfficientNetB3.h5'

        if not Path(output).exists():
            with st.spinner("Model weights were not found, downloading them. This may take a while."):
                gdown.download(url, output, quiet=False)

    elif model_choice == 'InceptionResNetV2 (79%)':
        url = 'https://drive.google.com/uc?id=14xPWqyeiz4S2XPiizEeDTTEQn5sSYBbE'
        output = 'models/InceptionResNetv2.h5'

        if not Path(output).exists():
            with st.spinner("Model weights were not found, downloading them. This may take a while."):
                gdown.download(url, output, quiet=False)

    elif model_choice == 'NASNet (69%)':
        url = 'https://drive.google.com/uc?id=1OGV-ZS2VtJJ_EK2oAB1IXp-jFqgWC1zU'
        output = 'models/NasNetLarge.h5'

        if not Path(output).exists():
            with st.spinner("Model weights were not found, downloading them. This may take a while."):
                gdown.download(url, output, quiet=False)

    else:
        raise ValueError('Unknown model: {}'.format(model_choice))


@st.cache(show_spinner=False)
def get_model(base_model, img_size, weights_file):
    '''
    Builds the final model to be used for predictions.
    '''

    # Define model architecture
    base_pretrained_model = base_model(input_shape=(img_size, img_size, 3), include_top=False, weights=None)
    model = base_pretrained_model.output
    model = GlobalAveragePooling2D()(model)
    model = Dense(256, activation='relu')(model)
    model = Dropout(0.2)(model)
    model = Dense(128, activation='relu')(model)
    model = Dropout(0.1)(model)
    predictions = Dense(7, activation='softmax')(model)

    model = Model(inputs=base_pretrained_model.input, outputs=predictions)

    # Load weights
    model.load_weights(weights_file)

    return model


def get_prediction(model, file, img_size, preprocessing_function):
    '''
    Obtains prediction for a specified file and model.
    '''

    # Load and process image into the suitable format
    img = image.load_img(file, target_size=(img_size, img_size))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocessing_function(x)
    x = x/255

    # Make prediction
    preds = model.predict(x)

    return np.argmax(preds), preds


def image_classification(model_choice, file):
    '''
    Main function to classify a skin lesion, including confidence scores.
    '''

    # Download weights for the chosen model
    download_weights(model_choice)

    # Obtain prediction
    if model_choice == 'DenseNet201 (69%)':
        from tensorflow.keras.applications.densenet import DenseNet201 as PTModel, preprocess_input
        img_size = 224
        weights = 'models/DenseNet201.h5'
        preprocess = preprocess_input
        model = get_model(PTModel, img_size, weights)
        label, scores = get_prediction(model, file, img_size, preprocessing_function=preprocess)

    elif model_choice == 'ResNet50 (79%)':
        from tensorflow.keras.applications.resnet import ResNet50 as PTModel, preprocess_input
        img_size = 224
        weights = 'models/ResNet50.h5'
        preprocess = preprocess_input
        model = get_model(PTModel, img_size, weights)
        label, scores = get_prediction(model, file, img_size, preprocessing_function=preprocess)

    elif model_choice == 'InceptionV3 (84%)':
        from tensorflow.keras.applications.inception_v3 import InceptionV3 as PTModel, preprocess_input
        img_size = 299
        weights = 'models/Inceptionv3.h5'
        preprocess = preprocess_input
        model = get_model(PTModel, img_size, weights)
        label, scores = get_prediction(model, file, img_size, preprocessing_function=preprocess)

    elif model_choice == 'EfficientNetB3 (85%)':
        from tensorflow.keras.applications.efficientnet import EfficientNetB3 as PTModel, preprocess_input
        img_size = 224
        weights = 'models/EfficientNetB3.h5'
        preprocess = preprocess_input
        model = get_model(PTModel, img_size, weights)
        label, scores = get_prediction(model, file, img_size, preprocessing_function=preprocess)

    elif model_choice == 'InceptionResNetV2 (79%)':
        from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2 as PTModel, preprocess_input
        img_size = 299
        weights = 'models/InceptionResNetv2.h5'
        preprocess = preprocess_input
        model = get_model(PTModel, img_size, weights)
        label, scores = get_prediction(model, file, img_size, preprocessing_function=preprocess)

    elif model_choice == 'NASNet (69%)':
        from tensorflow.keras.applications.nasnet import NASNetLarge as PTModel, preprocess_input
        img_size = 331
        weights = 'models/NasNetLarge.h5'
        preprocess = preprocess_input
        model = get_model(PTModel, img_size, weights)
        label, scores = get_prediction(model, file, img_size, preprocessing_function=preprocess)

    else:
        raise ValueError('Unknown model: {}'.format(model_choice))

    return label, scores
