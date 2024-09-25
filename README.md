
# Image Caption Generator

This project implements an image caption generator using Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks. The model generates captions for input images by extracting features using a pre-trained InceptionV3 model and predicting captions using a trained LSTM model.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)

## Features
- Generates descriptive captions for images.
- Utilizes a pre-trained InceptionV3 model for feature extraction.
- Uses an LSTM network for caption generation.
- Easy to use with minimal setup.

## Installation
To run this project, you need to have Python installed along with the necessary libraries. Follow the steps below:

1. **Clone the repository:**
   ```bash
   git clone https://github.com//ahmdmohamedd/image-caption-generator.git
   cd image-caption-generator
   ```

2. **Install required packages:**
   You can install the necessary packages using pip. It's recommended to use a virtual environment:
   ```bash
   pip install tensorflow
   pip install numpy
   ```

3. **Download Model and Tokenizer:**
   Ensure you have the trained LSTM model (`caption_model.h5`) and the tokenizer (`tokenizer.pkl`) in the project directory.

## Usage
1. Place your image in the project directory or specify the path to the image in the `main.py` file:
   ```python
   image_path = r'C:\path\to\your\image.jpg'
   ```

2. Run the program:
   ```bash
   python main.py
   ```

3. The generated caption will be printed to the console.

## Model Training
To train your own model:
1. Prepare your dataset with images and corresponding captions.
2. Create and fit the tokenizer using `Tokenizer` from `tensorflow.keras.preprocessing.text`.
3. Train the LSTM model on your dataset.
4. Save the model and tokenizer using `pickle`.
