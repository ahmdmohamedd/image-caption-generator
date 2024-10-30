# Image Caption Generator

This project implements an image caption generator using a Convolutional Neural Network (CNN) for feature extraction and a Long Short-Term Memory (LSTM) network for generating captions. The model is trained to take an image as input and generate a descriptive caption that reflects the contents of the image.

This project was developed using Python and TensorFlow, and it includes a detailed `Jupyter Notebook` file, `image_caption.ipynb`, where each step of the process is documented.

## Repository Structure

- **`image_caption.ipynb`**: Jupyter Notebook containing the project code, model architecture, and explanations.
- **Data**: Instructions on how to download the dataset and the steps for preprocessing.
- **Saved Models**: Placeholder for model files if training is completed or resumed by future users.

## Project Overview

### Model Architecture

1. **Feature Extraction**:
   - We use a pre-trained InceptionV3 model as the CNN for feature extraction. This model captures high-level image features that are passed into the LSTM model for caption generation.
2. **Caption Generation**:
   - The LSTM model serves as the language model. It takes the features extracted by the CNN and generates a sequence of words to describe the image. 
3. **Image Preprocessing**:
   - Each image is resized and preprocessed to match the input requirements of the InceptionV3 model.
   
The architecture follows a typical image-captioning pipeline, where the CNN extracts the image features, and the LSTM model leverages those features to create a meaningful caption.

### Dataset

This project uses the [Flickr8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k), which consists of 8,000 images with multiple captions per image. The dataset includes:
   - **Images**: Used as inputs for the feature extraction CNN.
   - **Captions**: Text descriptions used as target outputs for the LSTM.

## Setup Instructions

### Prerequisites

- Python 3.7+
- TensorFlow
- Keras
- NumPy, Pandas, Matplotlib
- NLTK (for text preprocessing)
  
Install the required packages:

```bash
pip install tensorflow keras numpy pandas matplotlib nltk
```

### Running the Project

1. **Download the Dataset**:
   - Download and unzip the Flickr8k dataset from the [Kaggle link](https://www.kaggle.com/datasets/adityajn105/flickr8k).
   - Set the path to the images directory in `image_caption.ipynb`.
   
2. **Feature Extraction**:
   - Run the feature extraction code block to preprocess and extract features from each image using InceptionV3.
   
3. **Training**:
   - Run the training code block in `image_caption.ipynb`. 

### Limitations and Training Constraints

Due to computational limitations, **the training process for this model was not completed**. Running this notebook locally on a CPU estimated a training time of approximately **22 hours per epoch**, which is impractical for most setups without dedicated hardware acceleration. This is primarily due to:

1. **Complex Model Architecture**: The combination of InceptionV3 and LSTM requires significant computational resources.
2. **Dataset Size and Complexity**: Processing and training on the Flickr8k dataset with multiple captions per image adds to the load.
3. **Hardware Constraints**: The model ideally requires a GPU or TPU for feasible training times.

#### Suggested Solutions for Future Training

For users interested in training the model, here are some possible optimizations:

1. **Use Pre-Extracted Features**: Extract and save the InceptionV3 features once and reuse them to avoid reprocessing images during each epoch.
   
2. **Reduce Batch Size**: Lowering the batch size can make each epoch shorter, though it may increase overall training time.
   
3. **Train on a GPU or TPU**: Consider using Google Colab or a similar service with GPU or TPU support.
   
4. **Implement Early Stopping**: Adding an early stopping mechanism can prevent unnecessary training once the model converges.

## Future Work

- **Optimize Training**: Explore smaller CNN architectures like MobileNet for faster feature extraction.
- **Additional Preprocessing**: Experiment with different tokenization or embedding techniques for captions.
- **Expand Dataset**: Implement the model on a larger dataset like MS COCO for more robust results.

## Results and Observations

Although the model was not fully trained, the structure and initial setup of the caption generator are well-defined and ready for users with sufficient resources to continue the process. The notebook contains outputs and intermediate results from successful model initialization and preprocessing steps.

## How to Contribute

If you'd like to contribute by optimizing or completing the model training, please fork the repository and submit a pull request. Contributions are always welcome!
