
import os
import pygame
import numpy as np
import random
import matplotlib.pyplot as plt

# ==========================================
# FIXED MAZE LAYOUT
# 0 = path, 1 = wall
# ==========================================
MAZE_GRID = np.array([
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 1, 0, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
], dtype=int)


# ==========================================
# 1. MAZE ENVIRONMENT
# ==========================================
class MazeEnv:
    def __init__(self):
        self.size         = 10
        self.grid         = MAZE_GRID.copy()
        self.player_start = (0, 0)
        self.goal_pos     = (9, 9)
        self.chaser_start = (8, 0)
        self.player_pos   = self.player_start
        self.chaser_pos   = self.chaser_start
        self.visited_by_player = set()

    def reset(self):
        self.player_pos = self.player_start
        self.chaser_pos = self.chaser_start
        self.visited_by_player = {self.player_pos}
        return self.get_chaser_state()

    def is_valid(self, r, c):
        return 0 <= r < self.size and 0 <= c < self.size and self.grid[r, c] == 0

    def move_player(self, action):
        r, c = self.player_pos
        dr, dc = {0:(-1,0), 1:(1,0), 2:(0,-1), 3:(0,1)}[action]
        nr, nc = r+dr, c+dc
        if self.is_valid(nr, nc):
            self.player_pos = (nr, nc)
            self.visited_by_player.add(self.player_pos)
            return False   # no wall hit
        return True        # wall hit

    def move_chaser(self, action):
        r, c = self.chaser_pos
        dr, dc = {0:(-1,0), 1:(1,0), 2:(0,-1), 3:(0,1)}[action]
        nr, nc = r+dr, c+dc
        if not self.is_valid(nr, nc):
            return self.get_chaser_state(), -5, False
        self.chaser_pos = (nr, nc)
        if self.chaser_pos == self.player_pos:
            return self.get_chaser_state(), +100, True
        prev = abs(r  - self.player_pos[0]) + abs(c  - self.player_pos[1])
        new  = abs(nr - self.player_pos[0]) + abs(nc - self.player_pos[1])
        reward = +2 if new < prev else (-2 if new > prev else -1)
        return self.get_chaser_state(), reward, False

    def get_chaser_state(self):
        return (*self.chaser_pos, *self.player_pos)

    def manhattan(self):
        return abs(self.chaser_pos[0]-self.player_pos[0]) + \
               abs(self.chaser_pos[1]-self.player_pos[1])


# ==========================================
# 2. Q-LEARNING AGENT
# ==========================================
class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.95,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min   = epsilon_min
        self.q_table = np.zeros((10, 10, 10, 10, 4))

    def choose_action(self, state, training=True):
        cr, cc, pr, pc = state
        if training and random.random() < self.epsilon:
            return random.randint(0, 3)
        return int(np.argmax(self.q_table[cr, cc, pr, pc]))

    def update(self, s, a, r, s2):
        cr,cc,pr,pc   = s
        cr2,cc2,pr2,pc2 = s2
        best = np.max(self.q_table[cr2,cc2,pr2,pc2])
        td   = r + self.gamma*best - self.q_table[cr,cc,pr,pc,a]
        self.q_table[cr,cc,pr,pc,a] += self.alpha * td

    def decay(self):
        self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)

    def stats(self):
        return np.max(self.q_table), np.mean(self.q_table)

    def reset(self):
        self.q_table = np.zeros((10, 10, 10, 10, 4))
        self.epsilon = 1.0


# ==========================================
# 3. UI
# ==========================================
class GameUI:
    def __init__(self, env, agent):
        pygame.init()
        self.env   = env
        self.agent = agent
        self.W, self.H = 700, 750
        self.screen = pygame.display.set_mode((self.W, self.H))
        pygame.display.set_caption("Q-Maze: Player vs AI Chaser")

        self.CELL = 60
        self.OX   = 50

        self.CW  = (51,51,51)
        self.CE  = (245,245,245)
        self.CS  = (0,170,0)
        self.CG  = (255,215,0)
        self.CP  = (0,85,255)
        self.CR  = (220,50,50)
        self.CV  = (173,216,230)
        self.CBG = (20,20,20)
        self.CT  = (230,230,230)
        self.CWN = (255,100,50)

        self.flg = pygame.font.SysFont("Consolas", 22, bold=True)
        self.fsm = pygame.font.SysFont("Consolas", 15)
        self.fxl = pygame.font.SysFont("Consolas", 34, bold=True)

    def _px(self, r, c):
        return (self.OX + c*self.CELL + self.CELL//2,
                         r*self.CELL + self.CELL//2)

    def draw(self, ep, step, status, extra=""):
        self.screen.fill(self.CBG)

        # Grid
        for r in range(10):
            for c in range(10):
                x = self.OX + c*self.CELL
                y =           r*self.CELL
                rect = pygame.Rect(x, y, self.CELL, self.CELL)
                if   self.env.grid[r,c]==1:                      col = self.CW
                elif (r,c)==self.env.goal_pos:                   col = self.CG
                elif (r,c)==self.env.player_start:               col = self.CS
                elif (r,c) in self.env.visited_by_player:        col = self.CV
                else:                                            col = self.CE
                pygame.draw.rect(self.screen, col, rect)
                pygame.draw.rect(self.screen, (120,120,120), rect, 1)

        # Player
        px,py = self._px(*self.env.player_pos)
        pygame.draw.circle(self.screen, self.CP, (px,py), self.CELL//3)
        self.screen.blit(self.fsm.render("P",True,(255,255,255)), (px-5,py-8))

        # Chaser
        cx,cy = self._px(*self.env.chaser_pos)
        pygame.draw.circle(self.screen, self.CR, (cx,cy), self.CELL//3)
        self.screen.blit(self.fsm.render("AI",True,(255,255,255)), (cx-9,cy-8))

        # Danger
        dist = self.env.manhattan()
        if dist<=2 and "PLAY" in status:
            self.screen.blit(
                self.fxl.render("!! DANGER !!",True,self.CWN),
                (self.W//2-110, 255))

        # Panel
        pygame.draw.line(self.screen,(100,100,100),(0,600),(self.W,600),2)
        y0 = 608

        sc = (100,255,100) if "WIN" in status else \
             (255,100,100) if ("CAUGHT" in status or "TRAIN" in status) else \
             (100,200,255)

        mq,aq = self.agent.stats()
        rows = [
            (self.flg, f"Status  : {status}",                      sc,               20,  y0),
            (self.fsm, f"Train Ep: {ep}/500",                      self.CT,          20,  y0+30),
            (self.fsm, f"Steps   : {step}",                        self.CT,          20,  y0+50),
            (self.fsm, f"Distance: {dist} cells",                  self.CWN if dist<=3 else self.CT, 20, y0+70),
            (self.fsm, f"Epsilon : {self.agent.epsilon:.3f}",      self.CT,          400, y0+30),
            (self.fsm, f"MaxQ:{mq:.1f} AvgQ:{aq:.2f}",            self.CT,          400, y0+50),
            (self.fsm, extra,                                       (200,200,100),    400, y0+70),
            (self.fsm, "[WASD/Arrows]Move  [R]Retrain  [Q]Quit",  (140,140,140),    20,  y0+112),
        ]
        for font,txt,col,x,y in rows:
            self.screen.blit(font.render(txt,True,col),(x,y))

        pygame.display.flip()

    def overlay(self, msg, col):
        surf = pygame.Surface((self.W,600), pygame.SRCALPHA)
        surf.fill((0,0,0,160))
        self.screen.blit(surf,(0,0))
        for i,line in enumerate(msg.split("\n")):
            t = self.fxl.render(line,True,col)
            self.screen.blit(t, t.get_rect(center=(self.W//2, 220+i*60)))
        pygame.display.flip()


# ==========================================
# 4. TRAINER
# ==========================================
class Trainer:
    def __init__(self, env, agent, ui):
        self.env   = env
        self.agent = agent
        self.ui    = ui
        self.clock = pygame.time.Clock()
        self.hist  = []

    def _rand_player(self):
        r,c = self.env.player_pos
        opts = [(r+dr,c+dc) for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]
                if self.env.is_valid(r+dr,c+dc)]
        if opts:
            self.env.player_pos = random.choice(opts)

    def train(self, eps=500, max_steps=200):
        self.hist = []
        print("--- Training AI Chaser ---")
        for ep in range(1, eps+1):
            state = self.env.reset()
            total = 0
            for step in range(max_steps):
                for e in pygame.event.get():
                    if e.type == pygame.QUIT:
                        return False
                self._rand_player()
                a = self.agent.choose_action(state, training=True)
                s2, r, done = self.env.move_chaser(a)
                self.agent.update(state, a, r, s2)
                state = s2
                total += r
                if ep % 50 == 0:
                    self.ui.draw(ep, step, "TRAINING AI...", f"Reward:{total}")
                    self.clock.tick(60)
                if done:
                    break
            self.agent.decay()
            self.hist.append(total)
            if ep % 50 == 0:
                avg = np.mean(self.hist[-50:])
                print(f"Ep {ep:3d}/500 | AvgReward:{avg:7.1f} | Eps:{self.agent.epsilon:.3f}")
        print("--- Training Complete ---")
        return True

    def plot(self):
        if not self.hist: return
        plt.figure(figsize=(10,5))
        plt.plot(self.hist, alpha=0.3, color='blue', label='Reward')
        ma = [np.mean(self.hist[max(0,i-20):i+1]) for i in range(len(self.hist))]
        plt.plot(ma, color='red', lw=2, label='20-ep MA')
        plt.title('AI Chaser Training')
        plt.xlabel('Episode'); plt.ylabel('Reward')
        plt.legend(); plt.grid(); plt.tight_layout()
        plt.savefig("training_rewards.png"); plt.close()
        print("Saved training_rewards.png")


# ==========================================
# 5. MAIN
# ==========================================
def main():
    env     = MazeEnv()
    agent   = QLearningAgent()
    ui      = GameUI(env, agent)
    trainer = Trainer(env, agent, ui)
    clock   = pygame.time.Clock()

    # ── KEY MAP ──────────────────────────────────────────────────
    KEY_MAP = {
        pygame.K_w:     0, pygame.K_UP:    0,
        pygame.K_s:     1, pygame.K_DOWN:  1,
        pygame.K_a:     2, pygame.K_LEFT:  2,
        pygame.K_d:     3, pygame.K_RIGHT: 3,
    }

    is_running     = True
    needs_training = True

    while is_running:

        # ── PHASE 1: TRAIN ───────────────────────────────────────
        if needs_training:
            ui.draw(0, 0, "TRAINING...", "Please wait")
            if not trainer.train():
                break
            np.save("q_table.npy", agent.q_table)
            trainer.plot()
            needs_training = False

            env.reset()
            ui.draw(500, 0, "TRAINING DONE!", "Blue=You  Red=AI")
            ui.overlay("AI READY!\nPress SPACE to Play", (100,255,100))

            # ── IMPORTANT: clear stale events then wait for SPACE ──
            pygame.event.clear()
            waiting = True
            while waiting and is_running:
                clock.tick(30)
                for e in pygame.event.get():
                    if e.type == pygame.QUIT:
                        is_running = False
                        waiting    = False
                    elif e.type == pygame.KEYDOWN:
                        if e.key in (pygame.K_q, pygame.K_ESCAPE):
                            is_running = False
                            waiting    = False
                        else:
                            waiting = False   # any key starts game

        if not is_running:
            break

        # ── PHASE 2: PLAY ────────────────────────────────────────
        state = env.reset()
        step  = 0

        # Draw first frame immediately so player sees clean maze
        ui.draw(500, 0, "PLAY! Reach the GOLD square!", "Avoid the RED AI!")
        pygame.event.clear()   # flush again right before game loop

        game_over = False

        while not game_over and is_running:
            clock.tick(30)   # 30 FPS is plenty

            player_moved = False

            # ── ALL INPUT handled inside KEYDOWN events ──────────
            # This is the most reliable method on Windows + PyCharm
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    is_running = False
                    game_over  = True

                elif e.type == pygame.KEYDOWN:

                    # Quit / retrain
                    if e.key in (pygame.K_q, pygame.K_ESCAPE):
                        is_running = False
                        game_over  = True

                    elif e.key == pygame.K_r:
                        agent.reset()
                        needs_training = True
                        game_over      = True

                    # ── MOVEMENT ─────────────────────────────────
                    elif e.key in KEY_MAP:
                        env.move_player(KEY_MAP[e.key])
                        player_moved = True

            if not is_running or game_over:
                break

            # ── Check WIN (even without player_moved, handles edge) ──
            if env.player_pos == env.goal_pos:
                ui.draw(500, step, "YOU WIN!", f"Cleared in {step} steps!")
                ui.overlay("YOU WIN!\nPress any key...", (100,255,100))
                pygame.event.clear()
                _any_key()
                game_over = True
                break

            # ── AI moves only after player moves ─────────────────
            if player_moved:
                step += 1
                ai_a = agent.choose_action(state, training=False)
                state, _, caught = env.move_chaser(ai_a)

                if caught or env.chaser_pos == env.player_pos:
                    ui.draw(500, step, "CAUGHT! GAME OVER",
                            f"Survived {step} steps")
                    ui.overlay("CAUGHT!\nPress any key...", (255,80,80))
                    pygame.event.clear()
                    _any_key()
                    game_over = True
                    break

            # ── Render ───────────────────────────────────────────
            ui.draw(500, step,
                    "PLAY! Reach the GOLD square!",
                    "Avoid the RED AI chaser!")

        # ── Post-game menu ────────────────────────────────────────
        if is_running and not needs_training:
            env.reset()
            ui.draw(500, 0,
                    "SPACE=Play Again  R=Retrain  Q=Quit", "")
            pygame.event.clear()
            choosing = True
            while choosing and is_running:
                clock.tick(30)
                for e in pygame.event.get():
                    if e.type == pygame.QUIT:
                        is_running = False; choosing = False
                    elif e.type == pygame.KEYDOWN:
                        if e.key in (pygame.K_q, pygame.K_ESCAPE):
                            is_running = False; choosing = False
                        elif e.key == pygame.K_SPACE:
                            choosing = False
                        elif e.key == pygame.K_r:
                            agent.reset()
                            needs_training = True
                            choosing = False

    pygame.quit()
    print("Thanks for playing!")


def _any_key():
    """Block until any key or window-close."""
    while True:
        for e in pygame.event.get():
            if e.type in (pygame.QUIT, pygame.KEYDOWN):
                return
        pygame.time.wait(20)


if __name__ == "__main__":
    main()
