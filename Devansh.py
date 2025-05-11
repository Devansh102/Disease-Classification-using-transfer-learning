import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import streamlit as st
import base64
from io import BytesIO

# Load pre-trained ResNet model and modify the final layer for 3 classes
model = models.resnet50(pretrained=True)  # Using ResNet50 here, you can change it to ResNet18 or any other
# Replace the final fully connected layer to output 3 classes (normal, tuberculosis, pneumonia)
model.fc = nn.Linear(model.fc.in_features, 7)  # 3 output classes

# Load the model weights if you've trained it on your dataset
device = torch.device("cpu")  # For CPU-only inference, change to "cuda" if you have a GPU
model.load_state_dict(torch.load('best_model_7.pth', map_location=device))  # Load model weights
model.eval()  # Set model to evaluation mode

# Define image transformations for ResNet (same as pre-trained models)
data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

# Define the class index mapping
# class_index = {0: 'NORMAL', 1: 'PNEUMONIA', 2: 'TUBERCULOSIS'}
class_index = {
    0: 'Cyst',
    1: 'Normal_Kidney',
    2: 'Normal_Lung',
    3: 'Pneumonia',
    4: 'Stone',
    5: 'Tuberculosis',
    6: 'Tumor'
}

# Function to preprocess the image and perform prediction
def preprocess_image(image: Image.Image):
    # Ensure the image is in RGB format (in case it's grayscale)
    if image.mode != 'RGB':
        image = image.convert('RGB')  # Convert grayscale to RGB
    
    image = data_transform(image)  # Apply the transformations
    image = image.unsqueeze(0)  # Add a batch dimension
    return image

# Prediction function
def predict_image(image: Image.Image):
    image_tensor = preprocess_image(image)  # Preprocess the image
    with torch.no_grad():  # Disable gradient calculations for inference
        outputs = model(image_tensor)  # Get model outputs (logits)
    # Get the predicted class (highest probability)
    prediction_index = torch.argmax(outputs, dim=1).item()
    predicted_class = class_index[prediction_index]  # Map the index to the label
    return predicted_class

# Function to convert image to base64
def convert_image_to_base64(image: Image.Image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")  # Save the image as PNG to the buffer
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")  # Base64 encode
    return img_str

# Streamlit UI components
st.set_page_config(page_title="Lung Disease Detection", page_icon="🩺")

# Custom CSS for styling: White UI with background image
st.markdown("""
    <style>
        body {
            background-image: url('https://media.istockphoto.com/id/1293974981/vector/human-lungs-icons-on-white-background.jpg?s=612x612&w=0&k=20&c=dc4aShbSDLkr5SwWLcSrd4SF5j8sr7N6xaC_CgAeEDI=');  /* Replace with your image URL */
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            color: #333;
            font-family: 'Arial', sans-serif;
        }

        .stApp {
            background-color: rgba(255, 255, 255, 0.8);  /* Semi-transparent white background */
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 15px rgba(0, 0, 0, 0.1);
        }

        .stTitle {
            color: #1ABC9C;  /* Teal color for the title */
            font-size: 2.5rem;
            font-weight: bold;
            text-align: center;
        }

        .stButton>button {
            background-color: #1ABC9C;
            color: white;
            padding: 12px 24px;
            font-size: 18px;
            font-weight: bold;
            border-radius: 5px;
            border: none;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }

        .stButton>button:hover {
            background-color: #16A085;  /* Darker teal on hover */
        }

        .stFileUploader {
            display: flex;
            justify-content: center;
            margin-top: 20px;
        }

        /* Full-width container with centered image preview */
        .image-container {
            width: 100%;
            display: flex;
            justify-content: center;  /* Center the image horizontally */
            padding-top: 20px;
        }

        .stImage {
            width: 10rem;  /* Fixed width of 5rem for image preview */
            height: 10rem;  /* Fixed height of 5rem for image preview */
            object-fit: cover;  /* Ensure the image scales properly */
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }

        .stMarkdown {
            text-align: center;
            font-size: 1.25rem;
        }

        .result {
            font-size: 1.5rem;
            font-weight: bold;
            color: #1ABC9C;
            text-align: center;
            margin-top: 20px;
        }

        /* Center the button horizontally */
        .stButton {
            display: flex;
            justify-content: center;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)


# Page title
st.markdown("<h1 class='stTitle'>AI Medical Lab</h1>", unsafe_allow_html=True)
st.markdown("""
    <div style="color: blue;">
        Upload a Chest X-ray or a CT scan image of kidney to detect the disease.
    </div>
""", unsafe_allow_html=True)


# File uploader widget
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)

    # Convert image to base64
    img_base64 = convert_image_to_base64(image)

    # Display the uploaded image in a full-width container with centered preview
    st.markdown("""
        <div class="image-container">
            <img src="data:image/png;base64,{}" class="stImage" />
        </div>
    """.format(img_base64), unsafe_allow_html=True)

    # Button to get prediction
    if st.button("Get Prediction"):
        prediction = predict_image(image)

        # Display the result
        st.markdown(f"<div class='result'>Prediction: {prediction}</div>", unsafe_allow_html=True)