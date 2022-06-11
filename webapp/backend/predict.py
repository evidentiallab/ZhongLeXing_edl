import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


def relu_evidence(y):
    return F.relu(y)

class LeNet(nn.Module):
    def __init__(self, dropout=False):
        super().__init__()
        self.use_dropout = dropout
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.fc1 = nn.Linear(20000, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 1))
        x = F.relu(F.max_pool2d(self.conv2(x), 1))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        if self.use_dropout:
            x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class LeNet5(nn.Module):
    def __init__(self, dropout=False):
        super().__init__()
        self.use_dropout = dropout

        self.conv1 = nn.Conv2d(1, 30, kernel_size=5, padding=2)
        # self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(30, 100, kernel_size=5)
        # self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(57600, 800)
        self.fc2 = nn.Linear(800, 400)
        self.fc3 = nn.Linear(400, 10)

    def forward(self, x):

        # print(x.shape)
        x = torch.relu(self.conv1(x))
        # print(x.shape)
        x = torch.relu(self.conv2(x))
        # print(x.shape)
        x = torch.flatten(x, start_dim=1)
        # print(x.shape)
        x = torch.relu(self.fc1(x))
        # print(x.shape)
        if self.use_dropout:
            x = F.dropout(x, training=self.training)
            # print(x.shape)
            x = F.dropout(torch.relu(self.fc2(x)), training=self.training)
            # print(x.shape)
        else:
            x = torch.relu(self.fc2(x))
            # print(x.shape)
        x = self.fc3(x)
        # print(x.shape)
        return x

def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return device

def get_model(model_name):
    device = get_device()
    model = LeNet()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters())
    checkpoint = torch.load(f"./models/{model_name}.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    model.eval()
    return model


def prediction(model, img_path, uncertainty=False, device=None):
    img = Image.open(img_path).convert("L")
    print(img)
    if not device:
        device = get_device()
    num_classes = 10
    trans = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])
    img_tensor = trans(img)
    img_tensor.unsqueeze_(0)
    img_variable = Variable(img_tensor)
    img_variable = img_variable.to(device)


    if uncertainty:
        output = model(img_variable)
        evidence = relu_evidence(output)
        alpha = evidence + 1
        uncertainty = num_classes / torch.sum(alpha, dim=1, keepdim=True)
        _, preds = torch.max(output, 1)
        prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
        output = output.flatten()
        prob = prob.flatten()
        preds = preds.flatten()
        print("Predict:", preds[0])
        print("Probs:", prob.cpu().detach().numpy())
        print("Uncertainty:", uncertainty)
    else:
        output = model(img_variable)
        _, preds = torch.max(output, 1)
        prob = F.softmax(output, dim=1)
        output = output.flatten()
        prob = prob.flatten()
        preds = preds.flatten()
        uncertainty = torch.zeros(1)
        print("Predict:", preds[0])
        print("Probs:", prob.cpu().detach().numpy())
        print("Uncertainty:", uncertainty)

    return {"predict": preds[0].tolist(),
            "probs": prob.cpu().detach().numpy().tolist(),
            "uncertainty": uncertainty[0].tolist()}
