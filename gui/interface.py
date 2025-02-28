# 导入模块
import logging
import random
import tkinter as tk
from tkinter import messagebox

import torch

from ai.agent import RLAgent
from game.config import GameConfig
from game.core import GameLogic

logger = logging.getLogger(__name__)


class GameGUI:
    """井字棋图形界面主类，负责处理用户交互和AI训练流程"""

    def __init__(self, n: int = GameConfig.DEFAULT_BOARD_SIZE):
        """初始化游戏界面"""
        logger.info(f"Initializing game GUI with {n}x{n} board")
        self.n = n
        self.cell_size = GameConfig.CELL_SIZE

        self.game = self._init_game_logic()
        self.root = tk.Tk()
        self.root.title(f"Tic-Tac-Toe {n}x{n} - DeepSeek RL")

        self.ai = self._init_ai_agent()
        self.is_human_turn = True
        self.train_mode = False

        self._create_widgets()
        self.update_board()
        logger.debug("GUI initialization completed")

    def _init_game_logic(self) -> GameLogic:
        """初始化游戏逻辑核心"""
        logger.debug("Creating game logic instance")
        return GameLogic(self.n)

    def _init_ai_agent(self) -> RLAgent:
        """初始化强化学习智能体"""
        logger.info("Initializing RL agent with GPU support")
        return RLAgent(self.n, use_gpu=True)

    def _create_widgets(self) -> None:
        """创建界面组件"""
        logger.debug("Building UI components")

        # 棋盘画布
        self.canvas = tk.Canvas(
            self.root,
            width=self.n * self.cell_size,
            height=self.n * self.cell_size,
            bg=GameConfig.COLORS['bg']
        )
        self.canvas.pack(pady=20)
        self.canvas.bind("<Button-1>", self._handle_click)  # 修复事件绑定

        # 控制面板
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10)

        # 功能按钮
        buttons = [
            ("New Game", self.new_game),
            ("Train AI", self.start_training),
            ("Human vs AI", self.start_human_game)
        ]

        for text, cmd in buttons:
            tk.Button(
                control_frame,
                text=text,
                command=cmd,
                width=15
            ).pack(side=tk.LEFT, padx=5)

        logger.info("UI components created successfully")

    def _handle_click(self, event) -> None:  # 添加事件处理方法
        """处理用户点击事件（私有方法）"""
        logger.debug(f"Processing click at ({event.x}, {event.y})")
        if not self.is_human_turn or self.game.game_over or self.train_mode:
            logger.debug("Ignoring click in current state")
            return

        col = event.x // self.cell_size
        row = event.y // self.cell_size

        logger.debug(f"Processing move at ({row}, {col})")
        if self.game.make_move(row, col):
            self.update_board()
            if not self.game.game_over:
                self.is_human_turn = False
                logger.debug("Scheduling AI move")
                self.root.after(500, self.ai_move)
        else:
            logger.warning(f"Invalid move attempted at ({row}, {col})")

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
        # 确保所有组件在相同设备
        self.ai.model.to(self.ai.model.device)
        logging.info("Starting AI training...")
        self.train_ai()
        messagebox.showinfo("Training Complete", "AI training completed!")
        self.train_mode = False

    def train_ai(self, episodes=1000):
        """训练循环（支持双AI对抗）"""
        win_count = 0
        lose_count = 0
        draw_count = 0

        for episode in range(episodes):
            # 获取模型参数的实际设备
            param_device = next(self.ai.model.parameters()).device

            # 使用设备类型比较代替字符串匹配
            if self.ai.model.device.type != param_device.type:
                raise RuntimeError(
                    f"设备类型不匹配：模型声明设备 {self.ai.model.device}，参数实际设备 {param_device}"
                )

            game = GameLogic(self.n)
            ai_player = random.choice([1, -1])
            game.current_player = ai_player

            while not game.game_over:
                # 添加状态维度验证
                state = self.ai.get_state(game)
                if state.size(0) != self.ai.input_size:
                    raise ValueError(f"状态维度错误：预期{self.ai.input_size}，实际{state.size(0)}")
                current_player_before = game.current_player
                state = self.ai.get_state(game)
                valid_moves = game.get_valid_moves()

                # 获取动作前的状态
                prev_state = state.clone() if isinstance(state, torch.Tensor) else state

                # AI选择动作（无论是哪个玩家）
                row, col = self.ai.act(state, valid_moves)

                # 执行动作
                move_success = game.make_move(row, col)
                if not move_success:
                    continue

                next_state = self.ai.get_state(game)
                done = game.game_over

                # 计算奖励（仅当当前玩家是AI角色时）
                reward = 0
                if current_player_before == ai_player:
                    if done:
                        if game.winner == ai_player:
                            reward = 1  # 胜利
                        elif game.winner is not None:
                            reward = -1  # 被对手击败
                        else:
                            reward = 0  # 平局

                    # 存储经验
                    action_index = row * self.n + col
                    self.ai.remember(
                        prev_state,
                        action_index,
                        reward,
                        next_state,
                        done
                    )
                    self.ai.replay()

                if done:
                    if game.winner == ai_player:
                        win_count += 1
                    elif game.winner is not None:
                        lose_count += 1
                    else:
                        draw_count += 1

                if (episode + 1) % 100 == 0:
                    logger.info(
                        f"Win Rate: {win_count / 100:.2%} | Loss Rate: {lose_count / 100:.2%} | Draw Rate: {draw_count / 100:.2%}")
                    win_count = lose_count = draw_count = 0

    def start_human_game(self):
        """开始人机对战（明确关闭训练模式）"""
        self.train_mode = False
        self.new_game()
        # 随机AI先手（使用训练好的模型）
        if random.choice([True, False]):
            self.ai_move()

    def new_game(self):
        """开始新游戏"""
        self.game.reset()
        self.is_human_turn = True
        self.update_board()

    def run(self) -> None:
        """运行主循环"""
        logger.info("Starting GUI main loop")
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            logger.info("Application closed by user")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}", exc_info=True)
            raise
