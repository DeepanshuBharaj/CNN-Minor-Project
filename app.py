import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐾", layout="centered")

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("cat_dog_cnn.pth", map_location=device))
    model = model.to(device)
    model.eval()
    return model, device

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

st.title("🐱 Cat vs Dog Classifier 🐶")
st.write("Upload an image of a cat or a dog, and the ResNet18 model will classify it!")

model, device = load_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        confidence, predicted_class = torch.max(probabilities, 0)
        
    classes = ["Cat 🐱", "Dog 🐶"]
    pred_label = classes[predicted_class.item()]
    conf_score = confidence.item() * 100
    
    st.markdown("---")
    st.subheader(f"Prediction: **{pred_label}**")
    st.write(f"**Confidence:** {conf_score:.2f}%")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Cat Probability", f"{probabilities[0].item() * 100:.2f}%")
        st.progress(float(probabilities[0].item()))
    with col2:
        st.metric("Dog Probability", f"{probabilities[1].item() * 100:.2f}%")
        st.progress(float(probabilities[1].item()))