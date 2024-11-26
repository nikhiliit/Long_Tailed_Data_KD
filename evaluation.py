import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights

# CIFAR-100 Class Names
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

# Define the transformation pipeline for the input image
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

def load_model(model_type='teacher'):
    """Load the trained teacher or student model."""
    model_path = './saved_model/teacher_model.pth' if model_type == 'teacher' else './saved_model/student_model.pth'
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, 100)  # Adjust final layer for CIFAR-100
    
    # Load the saved weights on CPU
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    print(f"{model_type.capitalize()} model loaded for evaluation on CPU.")
    return model

def predict_image(model, image_path):
    """Predict the class of an input image using the trained model."""
    # Load the image
    image = Image.open(image_path).convert('RGB')

    # Apply transformations
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension

    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    image = image.to(device)

    # Make prediction
    with torch.no_grad():
        output = model(image)
        _, predicted_class = torch.max(output, 1)

    # Get class name
    predicted_class_name = cifar100_classes[predicted_class.item()]
    return predicted_class_name

if __name__ == "__main__":
    # Set image path and model type directly in the code
    image_path = "/Users/gauravs/Desktop/files_for_gaurav_for_UI/cifar100_images/bus.png"
    model_type = "student"

    # Load the specified model
    model = load_model(model_type=model_type)

    # Predict and print the class name
    predicted_class_name = predict_image(model, image_path)
    print(f"The predicted class is: {predicted_class_name}")