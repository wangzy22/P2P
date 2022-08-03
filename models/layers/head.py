import torch.nn as nn


class MLPHead(nn.Module):
    def __init__(self, num_features, num_classes, mid_channels, dropout_ratio):
        super().__init__()
        self.mlp_head = nn.Sequential(
                nn.Linear(num_features, mid_channels[0]),
                nn.BatchNorm1d(mid_channels[0]),
                nn.ReLU(),
                nn.Dropout(dropout_ratio),
                nn.Linear(mid_channels[0], mid_channels[1]),
                nn.BatchNorm1d(mid_channels[1]),
                nn.ReLU(),
                nn.Dropout(dropout_ratio),
                nn.Linear(mid_channels[1], num_classes)
            )
        
    def forward(self, feats):
        return self.mlp_head(feats)