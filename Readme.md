# Long Tail Data Distillation with Flask

This project is a web-based image classification tool that utilizes a ResNet50 model trained on CIFAR-100 classes. The app allows users to classify images by either uploading their own images or selecting from a set of predefined images.

## Features

* **Model Selection**: Choose between the "Teacher Model" and "Student Model".
* **Image Input**: Toggle between uploading an image or selecting from predefined images.
* **Preview**: View a preview of the selected image before classification.
* **Result**: Display the predicted class of the image.

## Demo
(Replace with an actual screenshot if available)

## Prerequisites

* **Python 3.6 or above**
* **Flask web framework**
* **PyTorch for model inference**
* **Torchvision for model and transformation utilities**

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/image-classification-flask.git
    cd image-classification-flask
    ```

2. **Create a virtual environment (recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Add the saved models:**

    Place your `teacher_model.pth` and `student_model.pth` files in a `saved_model` directory in the project root.

## Predefined Images

Add predefined images in the `static/images` directory. Ensure images are in `.png`, `.jpg`, or `.jpeg` format.

## Add a Logo

Place your logo as `logo.png` in the `static` directory.

## Usage

1. **Run the Flask server:**

    ```bash
    python app.py
    ```

2. **Access the Web App:**

    Open your browser and navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000).

## Project Structure

```plaintext
image-classification-flask/
├── app.py                 # Main application script
├── saved_model/           # Folder for model .pth files
├── static/
│   ├── images/            # Folder for predefined images
│   ├── logo.png           # App logo
│   └── styles.css         # Styles for the app
├── templates/
│   └── index.html         # HTML template
├── requirements.txt       # List of dependencies
└── README.md              # Project documentation