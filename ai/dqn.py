# 导入模块
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """深度Q网络模型"""

    def __init__(self, input_size, hidden_size=128, use_gpu=False):
        super(DQN, self).__init__()
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, input_size)  # 输出对应每个动作的Q值

        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
