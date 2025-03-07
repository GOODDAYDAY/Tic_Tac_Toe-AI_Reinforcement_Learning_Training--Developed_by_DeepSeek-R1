"""
图形用户界面模块，处理用户交互和游戏流程
"""
import logging
import random
import tkinter as tk
from tkinter import messagebox
from typing import Optional

from ai.agent import RLAgent
from ai.trainer import AITrainer
from game.config import GameConfig
from game.core import GameLogic

logger = logging.getLogger(__name__)


class GameGUI:
    """井字棋图形界面，管理用户交互流程"""

    def __init__(self, n: int = GameConfig.DEFAULT_BOARD_SIZE):
        # 初始化游戏状态
        self.n = n
        self.game: Optional[GameLogic] = None
        self.ai: Optional[RLAgent] = None
        self.trainer: Optional[AITrainer] = None
        self.is_human_turn: bool = True

        # 初始化界面组件
        self.root = tk.Tk()
        self.root.title(f"Tic-Tac-Toe {n}x{n}")
        self._init_ui()

        # 启动新游戏
        self.new_game()

    def _init_ui(self) -> None:
        """初始化用户界面组件"""
        # 棋盘画布
        self.canvas = tk.Canvas(
            self.root,
            width=self.n * GameConfig.CELL_SIZE,
            height=self.n * GameConfig.CELL_SIZE,
            bg=GameConfig.COLORS['bg']
        )
        self.canvas.pack(pady=20)
        self.canvas.bind("<Button-1>", self.on_click)

        # 控制面板
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10)

        # 功能按钮
        buttons = [
            ("新游戏", self.new_game),
            ("训练AI", self.start_training),
            ("人机对战", self.start_pvp)
        ]
        for text, cmd in buttons:
            tk.Button(control_frame, text=text, command=cmd).pack(side=tk.LEFT, padx=5)

    def new_game(self) -> None:
        """初始化新游戏"""
        self.game = GameLogic(self.n)
        self.ai = RLAgent(self.n, training_mode=False)
        self.trainer = AITrainer(self.ai, self.n)
        self.is_human_turn = True
        self.update_board()

    def update_board(self) -> None:
        """更新棋盘显示"""
        self.canvas.delete("all")

        # 绘制棋盘格线
        for i in range(self.n):
            for j in range(self.n):
                x0 = j * GameConfig.CELL_SIZE
                y0 = i * GameConfig.CELL_SIZE
                x1 = x0 + GameConfig.CELL_SIZE
                y1 = y0 + GameConfig.CELL_SIZE
                self.canvas.create_rectangle(x0, y0, x1, y1, outline="black")

                # 绘制棋子
                if self.game.board[i][j] == 1:
                    self._draw_x(x0, y0, x1, y1)
                elif self.game.board[i][j] == -1:
                    self._draw_o(x0, y0, x1, y1)

        # 检查游戏状态
        if self.game.game_over:
            self._handle_game_end()

    def _draw_x(self, x0: int, y0: int, x1: int, y1: int) -> None:
        """绘制X标记"""
        self.canvas.create_line(x0 + 10, y0 + 10, x1 - 10, y1 - 10, width=2, fill="blue")
        self.canvas.create_line(x0 + 10, y1 - 10, x1 - 10, y0 + 10, width=2, fill="blue")

    def _draw_o(self, x0: int, y0: int, x1: int, y1: int) -> None:
        """绘制O标记"""
        self.canvas.create_oval(x0 + 10, y0 + 10, x1 - 10, y1 - 10, width=2, outline="red")

    def on_click(self, event) -> None:
        """处理玩家点击事件"""
        if self.game.game_over or not self.is_human_turn:
            return

        col = event.x // GameConfig.CELL_SIZE
        row = event.y // GameConfig.CELL_SIZE

        if self.game.make_move(row, col):
            self.update_board()
            if not self.game.game_over:
                # 从人类移动中学习
                self._learn_from_human_move(row, col)
                self.is_human_turn = False
                self._ai_move()

    def _learn_from_human_move(self, row: int, col: int) -> None:
        """从人类移动中学习"""
        prev_state = self.ai.get_state(self.game)
        action_index = row * self.n + col
        next_state = self.ai.get_state(self.game)

        # 动态计算即时奖励
        instant_reward = self._calculate_instant_reward_for_human_move(row, col)
        done = self.game.game_over

        # 将经验存入AI的经验回放缓冲区
        print(f"action_index: {action_index}, reward: {instant_reward}, done: {done}")
        self.ai.remember(prev_state, action_index, instant_reward, next_state, done)

        # 如果经验池足够大，执行一次模型更新
        if len(self.ai.memory) >= self.ai.batch_size:
            self.ai.replay()

    def _calculate_instant_reward_for_human_move(self, row: int, col: int) -> float:
        """计算人类玩家移动的即时奖励"""
        return self.trainer._calculate_instant_reward(self.game, 1, row, col)

    def _ai_move(self) -> None:
        """执行AI移动"""
        state = self.ai.get_state(self.game)
        valid_moves = self.game.get_valid_moves()
        row, col = self.ai.act(state, valid_moves)
        self.game.make_move(row, col)
        self.update_board()
        self.is_human_turn = True

    def start_training(self) -> None:
        """启动AI训练流程"""
        self.ai = RLAgent(self.n, training_mode=True)
        self.trainer = AITrainer(self.ai, self.n)

        try:
            stats = self.trainer.train(episodes=10000)
            messagebox.showinfo("训练完成",
                                f"训练结果：\n胜率: {stats['wins'] / 100:.2%}\n败率: {stats['losses'] / 100:.2%}")
        except Exception as e:
            messagebox.showerror("训练错误", str(e))
        finally:
            self.ai.save_model()
            self.new_game()

    def start_pvp(self) -> None:
        """启动人机对战模式"""
        self.new_game()
        if random.choice([True, False]):
            self.is_human_turn = False
            self._ai_move()

    def _handle_game_end(self) -> None:
        """处理游戏结束场景"""
        if self.game.winner:
            winner = "玩家" if self.game.winner == 1 else "AI"
            messagebox.showinfo("游戏结束", f"{winner} 获胜！")
        else:
            messagebox.showinfo("游戏结束", "平局！")
        self.new_game()

    def run(self) -> None:
        """启动主事件循环"""
        self.root.mainloop()
