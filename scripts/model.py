import torch
import torch.nn as nn
import torchvision.models as models

class ConvLSTMClassifier(nn.Module):
    def __init__(self,hidden_dim=256,num_layers = 1,num_classes=2):
        super(ConvLSTMClassifier,self).__init__()


        # Using a pretrained CNN (MobilenetV2) as feature extractor
        mobilenet = models.mobilenet_v2(pretrained=True)
        self.cnn = mobilenet.features # output batch (batch,1280,7,7)
        self.pool = nn.AdaptiveMaxPool2d((1,1))  # maket it (batch,180,1,1)


        self.featur_dim  = 1280 #output channels of mobilenet

        self.lstm = nn.LSTM(
            input_size=self.featur_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # Final layer
        self.fc = nn.Linear(hidden_dim,num_classes)

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.shape
        cnn_features = []

        # Loop over each frame in the sequence
        for t in range(seq_len):
            frame = x[:, t]  # shape: (batch, C, H, W)
            f = self.cnn(frame)  # (batch, 1280, 7, 7)
            f = self.pool(f)     # (batch, 1280, 1, 1)
            f = f.view(batch_size, -1)  # (batch, 1280)
            cnn_features.append(f)


        # stack into sequence: (batch, seq_len, feature_dim)
        features = torch.stack(cnn_features, dim=1)  # (batch, seq_len, 1280)

        # pass through LSTM: output shape (batch, seq_len, hidden_dim)
        _, (h_n, _) = self.lstm(features)

        # use last hidden state for classification
        logits = self.fc(h_n[-1])  # shape (batch, num_classes)

        return logits