# 导入模块
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, input_size, action_size, hidden_size=256, use_gpu=False):
        super(DQN, self).__init__()
        self.device = torch.device("cuda" if use_gpu else "cpu")

        # 网络层定义
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

        self.to(self.device)

    def forward(self, x):
        """前向传播（已集成玩家特征）"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
