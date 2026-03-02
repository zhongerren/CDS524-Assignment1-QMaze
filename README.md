# CDS524-Assignment1-QMaze
Q-Learning Maze Game | CDS524 Reinforcement Learning
# CDS524 Assignment 1 — Q-Learning Maze Game
**Course:** CDS524 Reinforcement Learning  
**Author:** ZHU Zhongxi  
**Date:** March 2026

---
📹 YouTube Demo: https://youtu.be/5OIgb70PICY
📓 Google Colab: https://colab.research.google.com/drive/1NK4iXke-3JB6rcjwbIRhO94TRDMoPgN4?usp=sharing


## 🎮 Game Overview
A 10×10 grid maze game where the **human player** navigates to the goal
while a **trained Q-Learning AI chaser** tries to catch them.

| Role | Colour | Control |
|------|--------|---------|
| Player | 🔵 Blue | WASD / Arrow Keys |
| AI Chaser | 🔴 Red | Q-Learning Agent |

**Win:** Reach the gold square  
**Lose:** Get caught by the AI

---

## 🧠 Q-Learning Details
- **State space:** (chaser_row, chaser_col, player_row, player_col) = 10,000 states  
- **Q-Table shape:** `[10, 10, 10, 10, 4]`  
- **Algorithm:** Tabular Q-Learning + Epsilon-Greedy  
- **Bellman Equation:** `Q(s,a) ← Q(s,a) + α[r + γ·maxQ(s',a') − Q(s,a)]`  
- **Hyperparameters:** α=0.1, γ=0.95, ε: 1.0 → 0.08 over 500 episodes

---

## 📦 Files
| File | Description |
|------|-------------|
| `maze_game.py` | Main game source code |
| `CDS524_Report.docx` | Written report (1000-1500 words) |
| `training_rewards.png` | AI training reward curve |

---

## ▶️ How to Run
```bash  
pip install pygame numpy matplotlib  
python maze_game.py  
