# Application for melanoma diagnosis using CNNs


The objective of this project is to build a web application to identify melanoma in images of skin lesions.

Motivated by the [ISIC 2018 challenge](https://challenge2018.isic-archive.com/task3/), the goal was to build an accurate model with good generalization
that can predict the diagnosis of any skin lesion. That is, given a dermoscopic or camera image,
we want to correctly classify it, especially when involving a possible melanoma diagnosis.

To accomplish that, we used the provided [HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) dataset with additional external data from
[BCN20000](https://arxiv.org/abs/1908.02288) and [PAD-UFES-20](https://data.mendeley.com/datasets/zr7vgbcyr2/1),
and implemented transfer learning using various CNN architectures (e.g., Inception-v3, ResNet-50, EfficientNet-B3).


Launch the web app:

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/vkonstantakos/melanoma-app/main/app.py)

## Demo

![Demo](https://github.com/VKonstantakos/melanoma-app/blob/main/demo/Melanoma%20Detection%20App.gif)

## Support

For support, email vkonstantakos@iit.demokritos.gr
