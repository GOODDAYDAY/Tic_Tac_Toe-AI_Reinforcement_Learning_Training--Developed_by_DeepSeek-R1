# 导入模块
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, input_size, hidden_size=256, use_gpu=False):
        super(DQN, self).__init__()
        self.device = torch.device("cuda" if use_gpu else "cpu")

        # 增加玩家位置编码层
        self.player_embed = nn.Embedding(2, 4)  # 两种玩家状态

        self.fc1 = nn.Linear(input_size + 4, hidden_size)  # 增加特征维度
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, input_size)

        self.to(self.device)

    def forward(self, x, player):
        # 玩家特征嵌入
        player_tensor = self.player_embed(torch.tensor([0 if player == 1 else 1]).to(self.device))
        x = torch.cat([x.view(-1), player_tensor.squeeze(0)])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
