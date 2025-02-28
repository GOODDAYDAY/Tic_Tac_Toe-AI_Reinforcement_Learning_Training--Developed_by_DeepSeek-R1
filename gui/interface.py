# 导入模块
import logging
import random
import tkinter as tk
from tkinter import messagebox

from ai.agent import RLAgent
from game.config import GameConfig
from game.core import GameLogic


class GameGUI:
    def __init__(self, n=GameConfig.DEFAULT_BOARD_SIZE):
        self.n = n
        self.cell_size = GameConfig.CELL_SIZE

        self.game = GameLogic(n)
        self.root = tk.Tk()  # 使用tk别名
        self.root.title(f"Tic-Tac-Toe {n}x{n}")

        self.ai = RLAgent(n, use_gpu=True)
        self.is_human_turn = True
        self.train_mode = False

        self.create_widgets()
        self.update_board()

    def create_widgets(self):
        """创建界面组件"""
        self.canvas = tk.Canvas(  # 显式使用tk.前缀
            self.root,
            width=self.n * self.cell_size,
            height=self.n * self.cell_size
        )
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.click_handler)

        control_frame = tk.Frame(self.root)  # 使用tk.Frame
        control_frame.pack(pady=10)

        tk.Button(  # 使用tk.Button
            control_frame,
            text="New Game",
            command=self.new_game
        ).pack(side=tk.LEFT, padx=5)  # 使用tk.LEFT

    def draw_board(self):
        """绘制棋盘"""
        self.canvas.delete("all")
        for i in range(self.n):
            for j in range(self.n):
                # 使用配置中的单元格尺寸
                x0 = j * self.cell_size
                y0 = i * self.cell_size
                x1 = x0 + self.cell_size
                y1 = y0 + self.cell_size
                self.canvas.create_rectangle(x0, y0, x1, y1, outline="black")

                if self.game.board[i][j] == 1:
                    self.canvas.create_line(x0 + 10, y0 + 10, x1 - 10, y1 - 10, width=2, fill="blue")
                    self.canvas.create_line(x0 + 10, y1 - 10, x1 - 10, y0 + 10, width=2, fill="blue")
                elif self.game.board[i][j] == -1:
                    self.canvas.create_oval(x0 + 10, y0 + 10, x1 - 10, y1 - 10, width=2, outline="red")

    def update_board(self):
        """更新棋盘状态"""
        self.draw_board()
        if self.game.game_over:
            if self.game.winner:
                winner = "Human" if self.game.winner == 1 else "AI"
                messagebox.showinfo("Game Over", f"{winner} wins!")
            else:
                messagebox.showinfo("Game Over", "It's a tie!")
            self.new_game()

    def click_handler(self, event):
        """处理用户点击"""
        if not self.is_human_turn or self.game.game_over or self.train_mode:
            return

        col = event.x // self.cell_size
        row = event.y // self.cell_size

        if self.game.make_move(row, col):
            self.update_board()
            if not self.game.game_over:
                self.is_human_turn = False
                self.root.after(500, self.ai_move)

    def ai_move(self):
        """AI移动"""
        state = self.ai.get_state(self.game)
        valid_moves = self.game.get_valid_moves()
        if valid_moves:
            row, col = self.ai.act(state, valid_moves)
            self.game.make_move(row, col)
        self.update_board()
        self.is_human_turn = True

    def start_training(self):
        """开始训练AI"""
        self.train_mode = True
        self.new_game()
        logging.info("Starting AI training...")
        self.train_ai()
        messagebox.showinfo("Training Complete", "AI training completed!")
        self.train_mode = False

    def train_ai(self, episodes=1000):
        """训练循环"""
        for episode in range(episodes):
            game = GameLogic(self.n)
            state = self.ai.get_state(game)
            total_reward = 0

            while not game.game_over:
                valid_moves = game.get_valid_moves()
                action = self.ai.act(state, valid_moves)
                row, col = action
                prev_state = state.clone()
                game.make_move(row, col)
                next_state = self.ai.get_state(game)

                # 计算奖励
                if game.game_over:
                    if game.winner == 1:  # AI作为先手时获胜
                        reward = 1
                    else:
                        reward = -1 if game.winner == -1 else 0
                else:
                    reward = 0

                self.ai.remember(prev_state, row * self.n + col, reward, next_state, game.game_over)
                total_reward += reward
                state = next_state

                # 经验回放
                self.ai.replay()

            if (episode + 1) % 100 == 0:
                logging.info(f"Episode: {episode + 1}, Reward: {total_reward:.2f}, Epsilon: {self.ai.epsilon:.2f}")
                self.ai.save_model()

    def start_human_game(self):
        """开始人机对战"""
        self.train_mode = False
        self.new_game()
        if random.choice([True, False]):  # AI随机选择先手或后手
            self.ai_move()

    def new_game(self):
        """开始新游戏"""
        self.game.reset()
        self.is_human_turn = True
        self.update_board()

    def run(self):
        """运行主循环"""
        self.root.mainloop()
