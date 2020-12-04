# Visual-to-Audio aid for visually impaired
A system to process visual input on timed frames to produce sensible audio aid in accordance with human information processing limits, using image captioning, semantic text comparison and text-to-speech modules. 

# Requirements

install requirements from requirements.txt

# Running application

change below lines for video source and frame selection interval, (set video_path to 0 for camera capture)

> video_path = 'videoplayback.mp4'

> frame_interval = 30

run *aid.py* to see output

# Model details: (will be replaced with a HED + GAN architecture soon)

> Image captioning model is trained with Bahdanau attention, with a CNN encoder and GRU based RNN decoder

> The dataset is 50,000 random images out of >4,70,000 images in MS-COCO dataset due to colab infrastructure limitations, for better results train your own model with code from colab notebook "train_captioning_model.ipynb" (!!! migrate to own infrastructure or deep learning vms to do so as this is the maximium capability of colab !!!)

> The Images are preprocessed with a pre-trained inceptionV3 application trained on ImageNet, the limiters are in place for colab limitations, feel free to lift them up if you are training yourself

> The caption unique word store is limited using top_k, which is set to top 5000 words, feel free to change if you are training yourself

> The tokeniser.pkl store in pickles folder is the one with limitations, if you decide to train yourself then make sure to pickle the tokeniser and replace the same here with same name.

> architecture diagram and overview are in design folder

# final text-to-speech

pyttsx3 - https://pypi.org/project/pyttsx3/


# Pending Work:

> semantic text similarity for comparing older and newer caption to avoid output congestion
