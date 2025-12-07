import warnings
warnings.filterwarnings('ignore')
import os
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import streamlit as st
import google.generativeai as genai



train = [
    'Tomato___Late_blight', 'Tomato___healthy', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Potato___healthy',
    'Corn_(maize)___Northern_Leaf_Blight', 'Tomato___Early_blight',
    'Tomato___Septoria_leaf_spot', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Strawberry___Leaf_scorch', 'Peach___healthy', 'Apple___Apple_scab',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Bacterial_spot',
    'Apple___Black_rot', 'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew', 'Peach___Bacterial_spot',
    'Apple___Cedar_apple_rust', 'Tomato___Target_Spot',
    'Pepper,_bell___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Potato___Late_blight', 'Tomato___Tomato_mosaic_virus',
    'Strawberry___healthy', 'Apple___healthy', 'Grape___Black_rot',
    'Potato___Early_blight', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Common_rust_', 'Grape___Esca_(Black_Measles)',
    'Raspberry___healthy', 'Tomato___Leaf_Mold',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Pepper,_bell___Bacterial_spot', 'Corn_(maize)___healthy'
]

# Sort the array alphabetically
train.sort()



# for moving data into GPU (if available)
def get_default_device():
    # """Pick GPU if available, else CPU"""
    # if torch.cuda.is_available:
    #     return torch.device("cuda")
    # else:
        return torch.device("cpu")


# for moving data to device (CPU or GPU)
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)



# for calculating the accuracy
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):

    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))


# convolution block with BatchNormalization
def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
             nn.BatchNorm2d(out_channels),
             nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)


# resnet architecture
class CNN_NeuralNet(ImageClassificationBase):
    def __init__(self, in_channels, num_diseases):
        super().__init__()

        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))

        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True)
        # self.conv5 = ConvBlock(256, 256, pool=True)
        # self.conv6 = ConvBlock(256, 512, pool=True)
        # self.conv7 = ConvBlock(512, 512, pool=True)

        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                        nn.Flatten(),
                                        nn.Linear(512, num_diseases))

    def forward(self, x):  # x is the loaded batch
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        # out = self.conv5(out)
        # out = self.conv6(out)
        # out = self.conv7(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out


device = get_default_device()
# defining the model and moving it to the GPU
# 3 is number of channels RGB, len(train.classes()) is number of diseases.
model = to_device(CNN_NeuralNet(3, len(train)), device)

# # Assuming the model path is stored in FILE
FILE = "plant-disease-model.pth"

# Load the model, map it to the correct device (GPU or CPU)
# Make sure the model architecture is the same as during training
model.load_state_dict(torch.load(FILE, weights_only=True, map_location=device))
model.eval()  # Ensure the model is in evaluation mode

# Set page configuration
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f8ff;
    }
    .title h1 {
        color: #2e8b57;
        font-size: 48px;
        font-weight: bold;
        text-align: center;
    }
    .subtitle p {
        color: #228b22;
        font-size: 20px;
        text-align: center;
    }
    .upload-box {
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        color: white;
        padding: 0.5em 1em;
        border-radius: 10px;
        font-size: 18px;
    }
    .prediction-box {
        background-color: #d1e7dd;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .prediction-box p {
        color: #2f855a;
        font-size: 24px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


# Title and description
st.markdown("<div class='title'><h1>🍃 Plant Leaf Disease Detection 🌱</h1></div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'><p>Upload an image of a plant leaf, and let us help you identify any disease "
            "that may be affecting it. 🌿 The app will predict the disease based on the uploaded image of the leaf."
            "</p></div>", unsafe_allow_html=True)
st.write("---")


# Uploading the image
plant_image = st.file_uploader("Choose an image...", type="jpg", label_visibility="collapsed")
submit = st.button('🔍 Predict Disease', use_container_width=True)

# Updated predict function
def predict_image(img, model):
    """Converts image to tensor and returns the predicted class with the highest probability."""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to the appropriate size
        transforms.ToTensor()           # Same transformation as used in training
    ])

    img = transform(img)
    xb = to_device(img.unsqueeze(0), device)  # Convert to batch format and move to the correct device
    yb = model(xb)
    _, preds = torch.max(yb, dim=1)
    return train[preds[0].item()]

# App logic
if submit:
    if plant_image is not None:
        # Open the uploaded image
        image = Image.open(plant_image).convert('RGB')

        # Display the uploaded image in the Streamlit app with a shadow effect
        st.image(image, caption='Uploaded Image', use_container_width=True)

        # Make predictions using the predict_image function
        predicted_class = predict_image(image, model)

        # Extract the crop name and disease name
        parts = predicted_class.split('___')
        crop_name = parts[0].replace('_', ' ')  # Replace underscores in the crop name with spaces
        disease_name = ' '.join(parts[1].split('_')).title()  # Format the disease name

        # Remove the crop name from the disease name if it's repeated
        if disease_name.startswith(crop_name):
            disease_name = disease_name[len(crop_name):].strip()

        # Combine the crop name with the adjusted disease name
        predicted_disease = f"{crop_name} {disease_name}"

        st.markdown(f"""
            <div style="text-align:center; background-color:#d1e7dd; padding:10px; border-radius: 10px; box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);">
                <p style="color: #2f855a; font-size: 20px; font-weight: bold; margin: 0;">
                    <strong>Predicted Disease:</strong> {predicted_disease}
                </p>
            </div>
            <br><br>
        """, unsafe_allow_html=True)

        os.environ["API_KEY"] = "AIzaSyApglh4b5fxexq0gylUhDo6BiX9EkfsI9w"
        genai.configure(api_key=os.environ["API_KEY"])

        # Create the model
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }

        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            generation_config=generation_config,
        )



        chat_session = model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": [
                        "I am building a website that predicts plant leaf diseases. You will be given a plant disease name, and your task is to explain what the disease is, how it affects the plant, the symptoms to look out for, the environmental factors that contribute to its spread, and potential treatments or preventive measures. Additionally, provide helpful tips for gardeners on how to manage and prevent the disease, such as proper watering, fertilizing, or spacing practices. If relevant, include any natural remedies or eco-friendly solutions. if it is given healthy instead you will do the same excluding the disease part.",
                    ],
                },
                {
                    "role": "model",
                    "parts": [
                        "Please provide the plant disease name. I need the name of the specific disease to give you the requested information.\n",
                    ],
                },
            ]
        )

        # Use the predicted disease for response generation
        response = chat_session.send_message(predicted_disease)

        # Ensure Markdown headings in the response are rendered correctly
        response_text = response.text
        lines = response_text.split("\n")  # Split into lines

        # Process each line: add extra newline after headings for Markdown continuity
        processed_text = ""
        for line in lines:
            if line.startswith("##"):  # Identify Markdown headings
                processed_text += f"{line}\n\n"  # Add extra newline after heading
            else:
                processed_text += f"{line}\n"

        # Render the processed text
        st.markdown(processed_text)