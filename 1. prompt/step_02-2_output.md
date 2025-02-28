# 井字棋AI强化学习优化

### 🧑‍💻 User

### 0. 上下文文件设置

#### 0.1 附带的二方包等无法引用内容

#### 0.2 文件引用

Tic_Tac_Toe-AI_Reinforcement_Learning_Training--Developed_by_DeepSeek-R1/
├── 1. prompt/
│ ├── file-path-output.py
│ ├── step_01-1_input-init_generate.md
│ └── step_01-2_output-init_generate.md
├── ai/
│ ├── __pycache__/
│ │ ├── __init__.cpython-311.pyc
│ │ ├── agent.cpython-311.pyc
│ │ └── dqn.cpython-311.pyc
│ ├── __init__.py
│ ├── agent.py
│ └── dqn.py
├── game/
│ ├── __pycache__/
│ │ ├── __init__.cpython-311.pyc
│ │ ├── config.cpython-311.pyc
│ │ └── core.cpython-311.pyc
│ ├── __init__.py
│ ├── config.py
│ └── core.py
├── gui/
│ ├── __pycache__/
│ │ ├── __init__.cpython-311.pyc
│ │ └── interface.cpython-311.pyc
│ ├── __init__.py
│ └── interface.py
├── utils/
│ ├── __init__.py
│ └── logger.py
├── LICENSE
├── README.md
├── main.py
├── requirements.txt
└── toy.py

### 1. 业务说明

- 带有界面的井字棋，好看的界面，动感的交互方式
- 带有强化学习

#### 1.1 核心业务说明

1. 需要使用强化学习，让 AI 自我训练 1000 局
2. 井字棋核心就是 3\*3，我希望能比较轻松地改成 4\*4 或者 n\*n 的一个形式，n超过5时，使用五子棋规则
3. 让AI强化学习训练1000局后，与我进行对练
4. 我要看到AI训练的参数
5. 应该需要明显区分训练时候AI下棋进行学习，和与人类下棋需要使用已有训练来下棋。需要知道，与人类下棋是检验成果的时候，不是用来训练的时候
6. 在AI训练的时候，应该设定自己先手或者后手，然后根据先手或者后手来增加削减的分数
7. 这里明显的一个错误就是，在AI训练的时候，没有失败的情况，因为此时AI是双方
8. 所以应该增加一个AI先手或者后手，然后根据先后手，增加成功失败。也就是在AI训练中，虽然下棋都是使用AI已有的训练结果，但是要根据先后手确定是胜利还是失败

#### 1.2 业务约束

1. 可调节参数n用于设定棋盘大小
2. 日志打印完善
3. 尽量使用GPU进行训练
3. 尽量使用GPU进行训练
3. 尽量使用GPU进行训练
3. 尽量使用GPU进行训练
4. 需要保证训练若干盘之后有明显提升
5. 需要可以保存\读取训练的结果
6. 二次训练时，需要使用之前的训练结果
7. 保存模型数据时，应标识体现棋盘大小n，否则会因为修改数字n而出现错误

#### 1.3 核心逻辑说明

### 2. 可复用资源

#### 2.1 复用方法

#### 2.2 复用数据结构

#### 开发约束

- 高质量代码
- 高可读性代码
- 高质量类注释、高质量方法注释、高质量行注释
- 大量注释
- 慎用 log.error 打印日志
- 组装逻辑里，要尽可能一次性查询，然后再进行组装
- 能复用方法的方法尽量复用
- 应优化文件结构，使用最佳文件结构打印并输出，方便我后续做文件修改
- 严谨使用略等字眼，要把需要的部分全部打出来
- 包含 import 信息

#### 开发预期

- 符合业务要求
- 尽可能打印关键日志
- 尽可能抽出可复用方法

#### 使用技术

- Python

---

### 🤖 Assistant

根据分析，代码需要以下改进以满足业务需求：

**1. 修复日志模块的datetime导入问题**

```python
# logger.py
from datetime import datetime  # 添加导入


def configure_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"tictactoe_{datetime.now().strftime('%Y%m%d%H%M')}.log"),
            logging.StreamHandler()
        ]
    )
```

**2. 增强模型加载的兼容性检查**

```python
# agent.py
def load_model(self):
    """加载模型（增加维度检查）"""
    if os.path.exists(self.model_file):
        checkpoint = torch.load(self.model_file)
        if checkpoint.get('input_size') != self.input_size:
            logging.warning("Model dimension mismatch, skipping load")
            return
        # 其余加载逻辑...
```

**3. 改进训练逻辑（支持双AI对抗训练）**

```python
# interface.py
def train_ai(self, episodes=1000):
    """训练循环（支持双AI对抗）"""
    for episode in range(episodes):
        game = GameLogic(self.n)
        total_reward = {1: 0, -1: 0}  # 分别记录双方奖励

        while not game.game_over:
            # 当前玩家获取状态
            state = self.ai.get_state(game)
            valid_moves = game.get_valid_moves()

            # AI决策
            action = self.ai.act(state, valid_moves)
            row, col = action

            # 记录执行前状态
            prev_state = self.ai.get_state(game).clone()
            game.make_move(row, col)
            next_state = self.ai.get_state(game)

            # 计算奖励（区分玩家）
            reward = 0
            if game.game_over:
                if game.winner == game.current_player:
                    reward = 1  # 当前玩家获胜
                elif game.winner is not None:
                    reward = -1  # 当前玩家失败
            total_reward[game.current_player] += reward

            # 存储经验（包含玩家标识）
            self.ai.remember(
                prev_state,
                row * self.n + col,
                reward,
                next_state,
                game.game_over,
                player=game.current_player
            )

            # 经验回放
            self.ai.replay()
```

**4. 改进DQN网络结构（增加玩家位置编码）**

```python
# dqn.py
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
```

**5. 完善获胜条件校验**

```python
# core.py
def check_win(self, row, col):
    """改进的胜利条件校验"""
    player = self.board[row][col]
    required = GameConfig.get_win_condition(self.n)

    # 动态步长检查
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dx, dy in directions:
        count = 1
        # 双向检查
        for d in [1, -1]:
            step = 1
            while True:
                x = row + dx * d * step
                y = col + dy * d * step
                if 0 <= x < self.n and 0 <= y < self.n:
                    if self.board[x][y] == player:
                        count += 1
                        step += 1
                    else:
                        break
                else:
                    break
        if count >= required:
            return True
    return False
```

**6. 增强训练参数配置**

```python
# config.py
class TrainingConfig:
    BATCH_SIZE = 128
    GAMMA = 0.99
    EPS_START = 1.0
    EPS_END = 0.01
    EPS_DECAY = 0.999
    TARGET_UPDATE = 10  # 目标网络更新频率
```

**需要新增的代码文件：**

**models/base_model.py**

```python
import torch


class BaseModel(torch.nn.Module):
    """模型基类（实现公共方法）"""

    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def save(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.__dict__
        }, path)

    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path)
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
```

**训练执行改进建议：**

1. 使用双网络机制（Policy Net + Target Net）提高稳定性
2. 增加优先经验回放（Prioritized Experience Replay）
3. 实现完整的ε-greedy策略衰减调度
4. 增加TensorBoard日志支持

完整改进方案需要结合具体业务场景进行参数调优，建议通过以下命令启动训练：

```bash
python main.py --train --n=5 --episodes=10000 --gpu
```
