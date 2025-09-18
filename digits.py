import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pygame
import numpy as np
import cv2


class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# load the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

model = ANN()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
episodes = 10

for epoch in range(episodes):
    running_loss = 0.0
    for images, labels in trainloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()  # Backpropagation
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(trainloader):.4f}")

# Test the model
correct = 0
total = 0
model.eval()  # set the model to evaluation mode

with torch.no_grad():
    for images, labels in testloader:
        outputs = model(images)
        _, prediction = torch.max(outputs, 1)  # values, indexes = torch.max(outputs, 1) 0-9
        # outputs = torch.tensor([[2.1, 0.3, 5.6, 1.2, 0.9, 3.4, 0.8, 0.5, 4.2, 1.0]]), prediction = 2
        total += labels.size(0)
        correct += (prediction == labels).sum().item()
print(f"Accuracy: {100 * correct / total:.2f}%")


# pygame

def draw_digit():
    pygame.init()
    window_size = 280  # 10X scale of 28x28
    display_height = window_size + 50
    screen = pygame.display.set_mode((window_size, display_height))
    pygame.display.set_caption("Draw Digit")
    clock = pygame.time.Clock()
    screen.fill((0, 0, 0))
    drawing = False
    prediction = None

    font = pygame.font.Font(None, 36)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True
            if event.type == pygame.MOUSEBUTTONUP:
                drawing = False
                # call function for prediction
                prediction = predict_digit(screen)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    screen.fill((0, 0, 0))
                    prediction = None

            if event.type == pygame.MOUSEMOTION and drawing:
                pygame.draw.circle(screen, (255, 255, 255), event.pos, 8)

        # display
        if prediction is not None:
            text = font.render(f"Prediction: {prediction}", True, (0, 255, 0))
            screen.blit(text, (10, window_size + 10))

        pygame.display.flip()
        clock.tick(60)


# Process the drawing
def process_drawing(screen):
    # Grab the pixel array (shape: width x height x 3)
    surface = pygame.surfarray.array3d(screen)

    # Pygame's array3d yields shape (width, height, 3). We want (height, width, 3)
    surface = np.transpose(surface, (1, 0, 2))

    # Convert to grayscale using proper luminance weights
    gray = cv2.cvtColor(surface, cv2.COLOR_RGB2GRAY)

    # Crop to the bounding box of the drawn content to remove extra background
    # Threshold to find non-background pixels (assume drawing is bright on dark bg)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        roi = gray[y:y + h, x:x + w]
    else:
        # nothing drawn; return a blank tensor
        roi = gray

    # Resize while keeping aspect ratio, pad to 28x28
    h, w = roi.shape
    if h > w:
        new_h = 20
        new_w = int(w * (20.0 / h))
    else:
        new_w = 20
        new_h = int(h * (20.0 / w))

    if new_w == 0:
        new_w = 1
    if new_h == 0:
        new_h = 1

    resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create a 28x28 canvas and paste the resized digit centered
    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    # Some smoothing helps (optional)
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0)

    # In MNIST, background is black (0) and digit strokes are white (255).
    # If the drawing has inverted colors, invert it.
    # Use mean intensity to guess which is background.
    if np.mean(canvas) > 127:
        canvas = 255 - canvas

    # Normalize to the same pipeline used for training: ToTensor then Normalize((0.5,), (0.5,))
    array = canvas.astype(np.float32) / 255.0  # [0,1]
    array = (array - 0.5) / 0.5  # normalize to [-1, 1]

    tensor = torch.tensor(array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1,1,28,28]
    return tensor


# Predict
def predict_digit(screen):
    image = process_drawing(screen)
    if image is None:
        return None

    model.eval()
    with torch.no_grad():
        output = model(image)
        _, prediction = torch.max(output, 1)
    return prediction.item()


draw_digit()