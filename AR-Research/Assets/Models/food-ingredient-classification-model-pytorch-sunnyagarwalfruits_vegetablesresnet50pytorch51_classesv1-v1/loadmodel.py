import torch
from torchvision import models, transforms
from PIL import Image

# Load the ResNet-50 model architecture
model = models.resnet50(pretrained=False)
num_classes = 51
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# Load the saved model weights
model.load_state_dict(torch.load('fruits_vegetables_51.pth'))
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load and preprocess the image
image_path = r"D:\SpringBoard\huggingface model\val\Jalepeno\Jalepeno_1.jpg"  # Replace with your image path
image = Image.open(image_path)
image = transform(image)
image = image.unsqueeze(0)

# Move the image to the same device as the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image = image.to(device)
model = model.to(device)

# Perform the prediction
with torch.no_grad():
    outputs = model(image)
    _, predicted_class = torch.max(outputs, 1)

# Map predicted class index to class name
class_names = [
               'Amaranth',
               'Apple',
               'Banana',
               'Beetroot',
               'Bell pepper',
               'Bitter Gourd',
               'Blueberry',
               'Bottle Gourd',
               'Broccoli',
               'Cabbage',
               'Cantaloupe',
               'Capsicum',
               'Carrot',
               'Cauliflower',
               'Chilli pepper',
               'Coconut',
               'Corn',
               'Cucumber',
               'Dragon_fruit',
               'Eggplant',
               'Fig',
               'Garlic',
               'Ginger',
               'Grapes',
               'Jalepeno',
               'Kiwi',
               'Lemon',
               'Mango',
               'Okra',
               'Onion',
               'Orange',
               'Paprika',
               'Pear',
               'Peas',
               'Pineapple',
               'Pomegranate',
               'Potato',
               'Pumpkin',
               'Raddish',
               'Raspberry',
               'Ridge Gourd',
               'Soy beans',
               'Spinach',
               'Spiny Gourd',
               'Sponge Gourd',
               'Strawberry',
               'Sweetcorn',
               'Sweetpotato',
               'Tomato',
               'Turnip',
               'Watermelon'
               ]

# Output predicted class
print(f'Predicted class: {class_names[predicted_class.item()]}')
