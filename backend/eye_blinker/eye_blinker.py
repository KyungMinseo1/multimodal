import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class EyeBlinker(nn.Module):
    def __init__(self):
        super(EyeBlinker, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(1536, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = x.reshape(-1, 1536)
        x = F.relu(self.fc1(x))
        output = self.fc2(x)
        return output, x

def load_model():
    # Determine the device to use
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = EyeBlinker()
    module_dir = os.path.dirname(__file__)
    model_path = os.path.join(module_dir, '../../models/eye_blinker.pth')
    state_dict = torch.load(model_path, map_location=device)
    if isinstance(state_dict, EyeBlinker):
        return state_dict
    model.load_state_dict(state_dict)
    model = model.eval()
    return model