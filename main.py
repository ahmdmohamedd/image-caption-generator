import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model  # Ensure to import Model from tensorflow.keras

# Load the pre-trained InceptionV3 model and use it to extract image features
def extract_features(image_path):
    model = InceptionV3(weights='imagenet')
    model = Model(inputs=model.input, outputs=model.layers[-2].output)  # Last hidden layer
    image = load_img(image_path, target_size=(299, 299))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)
    return feature

# Load the tokenizer and trained caption generation model
def load_models():
    # Load the tokenizer from the correct path
    tokenizer = pickle.load(open(r'C:\Users\ahmed\Downloads\Image caption generator\tokenizer.pkl', 'rb'))
    # Load the trained LSTM model from the correct path
    model = load_model(r'C:\Users\ahmed\Downloads\Image caption generator\caption_model.h5')
    max_length = 34  # Maximum caption length (should be consistent with training)
    return tokenizer, model, max_length

# Generate caption using the trained LSTM model and the extracted image features
def generate_caption(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat, None)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    final_caption = in_text.split()[1:-1]  # Remove 'startseq' and 'endseq'
    return ' '.join(final_caption)

# Main function to take user input and generate caption
if __name__ == "__main__":
    # Load the models
    tokenizer, model, max_length = load_models()

    # Get image path from the user
    image_path = r'C:\Users\ahmed\Downloads\Image caption generator\image.jpg'  # Use raw string for the image path

    # Extract features from the image
    photo = extract_features(image_path)

    # Generate the caption
    caption = generate_caption(model, tokenizer, photo, max_length)
    print("Generated Caption: ", caption)
