import torch.nn as nn
import torch


class network(nn.Module):
    def __init__(self, numclass, feature_extractor):
        super(network, self).__init__()
        self.feature = feature_extractor
        #self.hidden = nn.Linear(512, 256, bias=True) 
        self.hidden = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 128)
                )
        self.fc = nn.Linear(128, numclass, bias=True)
    def forward(self, input):
        import pudb; pu.db
        x = self.feature(input)
        x = self.hidden(x)
        x = self.fc(x)
        return x

    def Incremental_learning(self, numclass):
        weight = self.fc.weight.data
        bias = self.fc.bias.data
        in_feature = self.fc.in_features
        out_feature = self.fc.out_features

        self.fc = nn.Linear(in_feature, numclass, bias=True)
        self.fc.weight.data[:out_feature] = weight[:out_feature]
        self.fc.bias.data[:out_feature] = bias[:out_feature]

    def feature_extractor(self,inputs):
        x = self.feature(inputs)
        return self.hidden(x)
