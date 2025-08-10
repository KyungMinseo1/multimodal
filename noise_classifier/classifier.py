import torch
import torch.nn as nn

# defining NoiseClassifier

class NoiseClassifier(nn.Module):
    def __init__(self, input_size=25, hidden_sizes=[128, 512, 256], num_classes=6, dropout=0.3):
        super(NoiseClassifier, self).__init__()
        
        layers = []
        prev_size = input_size

        # 마지막 은닉층 전까지 구성
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size

        # 은닉층 부분
        self.feature_extractor = nn.Sequential(*layers)

        # 출력층 따로 구성
        self.classifier = nn.Linear(prev_size, num_classes)

    def forward(self, x, return_features=True):
        features = self.feature_extractor(x)
        out = self.classifier(features)
        if return_features:
            return features, out  # 마지막 은닉층 출력 반환
        return out
      
# loading NoiseClassifier Model

def load_model():
    model = NoiseClassifier()
    state_dict = torch.load('model/audio_noise_classifier.pth', map_location=torch.device('cuda'))
    if isinstance(state_dict, NoiseClassifier):
        return state_dict
    model.load_state_dict(state_dict)
    model.eval()
    return model