from torch import nn


class MLP(nn.Module):
    """
    作为基线的多层感知机模型
    """

    def __init__(self, feature_num, target_num):
        super().__init__()
        self.feature_num = feature_num
        self.target_num = target_num
        # add 特征选择
        self.linear0 = nn.Linear(feature_num, feature_num, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.linear1 = nn.Linear(feature_num, 16)
        self.linear3 = nn.Linear(16, 8)
        self.linear5 = nn.Linear(8, target_num)
        self.relu = nn.ReLU()

    def forward(self, full_data):
        # add
        layer0 = self.linear0(full_data)
        activate0 = self.sigmoid(layer0)

        full_data = full_data.mul(activate0)

        layer1 = self.linear1(full_data)
        activate1 = self.relu(layer1)

        layer3 = self.linear3(activate1)
        activate3 = self.relu(layer3)

        prediction = self.linear5(activate3)
        return prediction
