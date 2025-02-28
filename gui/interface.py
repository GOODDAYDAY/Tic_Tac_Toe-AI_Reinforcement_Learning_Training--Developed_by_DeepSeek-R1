# 导入模块
import logging
import random
import tkinter as tk
from tkinter import messagebox

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

        logger.info(f"Processing move at ({row}, {col})")
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
        logging.info("Starting AI training...")
        self.train_ai()
        messagebox.showinfo("Training Complete", "AI training completed!")
        self.train_mode = False

    def train_ai(self, episodes=1000):
        """训练循环（支持双AI对抗）"""
        for episode in range(episodes):
            game = GameLogic(self.n)
            total_reward = {1: 0, -1: 0}

            while not game.game_over:
                state = self.ai.get_state(game)
                valid_moves = game.get_valid_moves()

                row, col = self.ai.act(state, valid_moves)
                prev_state = self.ai.get_state(game)
                game.make_move(row, col)
                next_state = self.ai.get_state(game)
                done = game.game_over

                reward = 0
                if done:
                    if game.winner == game.current_player:
                        reward = 1
                    elif game.winner is not None:
                        reward = -1
                total_reward[game.current_player] += reward

                # 修改后的remember调用（移除player参数）
                self.ai.remember(
                    prev_state,
                    row * self.n + col,
                    reward,
                    next_state,
                    done
                )

                self.ai.replay()

            if (episode + 1) % 100 == 0:
                avg_reward = sum(total_reward.values()) / 2
                logging.info(f"Episode: {episode + 1}, Avg Reward: {avg_reward:.2f}, Epsilon: {self.ai.epsilon:.2f}")
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
