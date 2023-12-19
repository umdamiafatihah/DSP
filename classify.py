import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

import warnings
warnings.filterwarnings('ignore')

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=43):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def get_model():
    checkpoint_path = 'model/final_resnet18_model_1.pth'
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model = ResNet18()
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def get_model_adv(model):
    if model == 'standard':
        checkpoint_path = 'model/final_resnet18_model_1.pth'
    if model == 'madry':
        checkpoint_path = 'model/final_robust_resnet18_model_madry_3.pth'
    if model == 'trades':
        checkpoint_path = 'model/final_robust_resnet18_model_trades_3.pth'
    if model == 'awp':
        checkpoint_path = 'model/final_robust_resnet18_model_awp_4.pth'
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model = ResNet18()
    model.load_state_dict(checkpoint['model_state_dict'])
    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=0.001)
    return model, criterion, opt

def generate_adversarial_example(image, model_type, epsilon):
    image = image.resize((32, 32))
    image_array = np.array(image)
    image_array = image_array.astype(np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    image_array = np.transpose(image_array, (0, 3, 1, 2))
    model, criterion, opt = get_model_adv(model_type)

    classifier = PyTorchClassifier(
        model=model,
        clip_values=(0.0, 1.0),
        loss=criterion,
        optimizer=opt,
        input_shape=(3, 32, 32),
        nb_classes=43
    )

    test_attack = FastGradientMethod(
        classifier,
        norm=np.inf,
        eps=epsilon,
        targeted=False,
        batch_size=32
    )

    adv_image = test_attack.generate(image_array)
    adv_image = np.transpose(adv_image, (0, 2, 3, 1))
    return adv_image

def predict(image_path):
    # Load the model
    model = get_model()
    model.eval()
    # Define the same transformations as used during training
    transform = transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path)
    image = torch.unsqueeze(transform(image), 0)

    with torch.no_grad():
        output = model(image)
        probabilities = nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    # Return both the predicted class and the confidence percentage
    return predicted.item(), confidence.item()


def predict_adv(image, model_type):
    try:
        # Load the model
        model, criterion, opt = get_model_adv(model_type)
        model.eval()

        # Define the same transformations as used during training
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([32, 32]),
            transforms.ToTensor(),
        ])

        image = np.transpose(image, (0, 3, 1, 2))
        image = image.squeeze(0)
        image = torch.from_numpy(image)
        image = torch.unsqueeze(transform(image), 0)

        with torch.no_grad():
            output = model(image)
            probabilities = nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        # Return both the predicted class and the confidence percentage
        return predicted.item(), confidence.item()
    except Exception as e:
        # Print the error
        print(f"An error occurred: {e}")
        # Optionally, you can re-raise the exception to halt the program
        raise e

