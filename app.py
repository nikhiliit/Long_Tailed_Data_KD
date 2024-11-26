# app.py
from flask import Flask, request, jsonify, render_template
from PIL import Image
import os
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights

# Initialize Flask app
app = Flask(__name__)

# Directory for predefined images
PREDEFINED_IMAGES_DIR = '/Users/gauravs/Desktop/files_for_gaurav_for_UI/static/images'

# CIFAR-100 Classes
cifar100_classes = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
    'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
    'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
    'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

# Image transformation
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

# Load the model function
def load_model(model_type='teacher'):
    model_path = './models/teacher_model.pth' if model_type == 'teacher' else './models/student_model.pth'
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, 100)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Prediction function
def predict_image(model, image):
    image = transform(image).unsqueeze(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted_class = torch.max(output, 1)
    return cifar100_classes[predicted_class.item()]

# Route for the home page
@app.route('/')
def home():
    # List predefined images from the directory
    image_list = os.listdir(PREDEFINED_IMAGES_DIR)
    image_list = [f for f in image_list if f.lower().endswith(('png', 'jpg', 'jpeg'))]  # Filter images
    return render_template('index.html', image_list=image_list)

# Route for handling image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    model_type = request.form.get('model_type')
    selected_image = request.form.get('image-select')
    image_file = request.files.get('image')
    threshold = request.form.get('threshold') #threshold
    
    # Check if model type is provided
    if not model_type:
        return jsonify({'error': 'Model type is required'}), 400

    # Determine image source (predefined or uploaded)
    if selected_image:
        # Use predefined image
        image_path = os.path.join(PREDEFINED_IMAGES_DIR, selected_image)
        image = Image.open(image_path).convert('RGB')
    elif image_file:
        # Use uploaded image
        image = Image.open(image_file).convert('RGB')
    else:
        return jsonify({'error': 'An image is required'}), 400

    # Load model and predict
    model = load_model(model_type)
    predicted_class = predict_image(model, image)
    return jsonify({'predicted_class': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)