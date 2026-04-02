import torch
import torch.nn as nn
import torchvision.models as models


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)
        return out.mean(dim=1)


class PretrainedExtractor(nn.Module):
    def __init__(self, model_name='resnet50', pretrained=True):
        super(PretrainedExtractor, self).__init__()
        if model_name == 'resnet50':
            base = models.resnet50(pretrained=pretrained)
            self.feature_extractor = nn.Sequential(*list(base.children())[:-1])
            self.out_features = 2048
        else:
            base = models.mobilenet_v2(pretrained=pretrained)
            self.feature_extractor = nn.Sequential(*list(base.children())[:-1])
            self.out_features = 1280

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        return x


class SequenceModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_classes=5, rnn_type='LSTM', use_attention=False):
        super(SequenceModel, self).__init__()
        self.rnn_type = rnn_type
        self.use_attention = use_attention

        if rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        else:
            self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)

        self.attention = Attention(hidden_size) if use_attention else None
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        if self.use_attention:
            out = self.attention(out)
        else:
            out = out[:, -1, :]
        out = self.classifier(out)
        return out
