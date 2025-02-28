# 导入模块
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DQN(nn.Module):
    def __init__(self, input_size, action_size, hidden_size=256, use_gpu=False):
        super(DQN, self).__init__()
        # 统一设备名称格式，显式指定设备索引
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        else:
            self.device = torch.device("cpu")

        # 添加参数验证
        if input_size <= 0 or action_size <= 0:
            raise ValueError(f"无效的模型参数：input_size={input_size}, action_size={action_size}")

        # 网络层定义（增加参数打印）
        logger.debug(f"构建DQN网络结构：输入层{input_size} -> 隐藏层{hidden_size} -> 输出层{action_size}")
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

        self.to(self.device)
        logger.info(f"模型已初始化到设备：{self.device}")

    def forward(self, x):
        """前向传播（已集成玩家特征）"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
