"""Minimal 2D pedestrian-avoidance testbed.

World is a WIDTH x HEIGHT meter box. The robot starts on the left and its goal
is the right edge. Pedestrians navigate with ORCA (pedestrians.py) toward
random goals, avoiding each other and the robot.

Each step, a predictor (predictor.py) guesses every pedestrian's position
1..HORIZON steps ahead, and ACI (aci.py) calibrates one disk radius per
(lookahead, pedestrian) online. Each disk targets alpha/HORIZON miscoverage,
so by the union bound a pedestrian's whole tube of disks contains its true
path ~(1 - alpha) of the time. The disks are drawn as white circles — an
expanding tube along each pedestrian's predicted path.

Gym-style API:
    obs = env.reset()
    obs, reward, done = env.step(action)   # action = robot (vx, vy) in m/s

obs is a dict:
    "robot": (2,) robot position
    "peds":  (N, 4) pedestrian positions and velocities [x, y, vx, vy]

reward is -1 while the robot overlaps a pedestrian, else 0.
done when the robot reaches the right edge.
"""

import numpy as np
import pygame

from aci import MaxACI, ACI
from pedestrians import Crowd
from policies import FlowPolicy, OrcaExpert, WalkForward
from predictor import ConstantVelocity

WIDTH, HEIGHT = 12.0, 8.0   # meters
DT = 0.1                    # seconds per step
N_PEDS = 3
PED_SPEED = 1.2             # m/s
RADIUS = 0.25               # agent radius, meters (robot and pedestrians)
HORIZON = 16                # prediction horizon, steps (= TRAJ_LEN, so every
                            # generated waypoint has a calibrated disk)
MAX_SPEED = 1.0             # robot actuator limit, m/s
PX_PER_M = 80               # rendering scale
# A wall across the middle with a doorway at mid-height: everyone funnels
# through the gap. Rectangles as (xmin, xmax, ymin, ymax).
WALLS = [(5.9, 6.1, 0.0, 3.2), (5.9, 6.1, 4.8, 8.0)]


def in_wall(p, margin=RADIUS):
    return any(x0 - margin <= p[0] <= x1 + margin and y0 - margin <= p[1] <= y1 + margin
               for x0, x1, y0, y1 in WALLS)


class Env:
    def __init__(self, seed=0, deference=0.4):
        # deference: the radius pedestrians assign to the robot in their ORCA
        # sim — how much the crowd yields to it. Lower = more assertive crowd.
        self.rng = np.random.default_rng(seed)
        self.deference = deference
        self.screen = None

    def reset(self, robot_start=(0.5, HEIGHT / 2)):
        self.robot = np.array(robot_start, dtype=float)
        self.crowd = Crowd(N_PEDS, (WIDTH, HEIGHT), DT, RADIUS, PED_SPEED, self.rng, WALLS,
                           robot_agent_radius=self.deference)
        return self._obs()

    def step(self, action):
        vel = np.asarray(action, dtype=float)
        speed = np.linalg.norm(vel)
        if speed > MAX_SPEED:
            vel = vel * (MAX_SPEED / speed)
        for axis in (0, 1):  # per-axis, so the robot slides along walls
            moved = self.robot.copy()
            moved[axis] += vel[axis] * DT
            if not in_wall(moved):
                self.robot = moved
        self.robot[1] = np.clip(self.robot[1], 0, HEIGHT)
        self.crowd.step(self.robot, vel)

        collision = np.any(np.linalg.norm(self.crowd.positions() - self.robot, axis=1) < 2 * RADIUS)
        reward = -1.0 if collision else 0.0
        done = self.robot[0] >= WIDTH - 0.5
        return self._obs(), reward, done

    def _obs(self):
        return {"robot": self.robot.copy(),
                "peds": np.hstack([self.crowd.positions(), self.crowd.velocities()])}

    def render(self, disks=None, plan=None, certified=0):
        """disks: optional (prediction, radii) — conformal disks to draw. The
        first `certified` lookaheads are enforced during sampling: drawn bright,
        at the true keep-out radius for the robot's center (r + both bodies).
        The rest are unenforced predictor uncertainty: drawn dim, sized to
        contain the pedestrian's body (r + RADIUS).
        plan: optional (T, 2) robot trajectory to draw as dots."""
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((int(WIDTH * PX_PER_M), int(HEIGHT * PX_PER_M)))
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit

        def px(p):
            return (int(p[0] * PX_PER_M), int(p[1] * PX_PER_M))

        self.screen.fill((30, 30, 30))
        for x0, x1, y0, y1 in WALLS:
            pygame.draw.rect(self.screen, (110, 110, 110),
                             (*px((x0, y0)), int((x1 - x0) * PX_PER_M), int((y1 - y0) * PX_PER_M)))
        for p in self.crowd.positions():
            pygame.draw.circle(self.screen, (220, 180, 60), px(p), int(RADIUS * PX_PER_M))
        pygame.draw.circle(self.screen, (80, 160, 255), px(self.robot), int(RADIUS * PX_PER_M))
        if disks is not None:
            prediction, radii = disks  # (HORIZON, N, 2), (HORIZON, N)
            for k, (centers, rs) in enumerate(zip(prediction, radii)):
                for c, r in zip(centers, rs):
                    if np.isfinite(r):
                        if k < certified:
                            pygame.draw.circle(self.screen, (240, 240, 240), px(c),
                                               int((r + 2 * RADIUS) * PX_PER_M), width=2)
                        else:
                            pygame.draw.circle(self.screen, (90, 90, 90), px(c),
                                               int((r + RADIUS) * PX_PER_M), width=1)
        if plan is not None:
            for p in plan:
                pygame.draw.circle(self.screen, (80, 160, 255), px(p), 3)
        pygame.display.flip()
        self.clock.tick(1 / DT)


if __name__ == "__main__":
    env = Env()
    policy = FlowPolicy()           # swap in any object with .act(obs)
    # policy = OrcaExpert(goal=(WIDTH - 0.3, HEIGHT / 2), dt=DT, radius=0.25, walls=WALLS)
    # policy = WalkForward()
    predictor = ConstantVelocity(DT, HORIZON)
    aci = ACI(alpha=0.1, horizon=HORIZON, n_peds=N_PEDS)  # swap in ACI(...) for the union-bound method

    obs = env.reset()
    done = False
    total_reward, misses, steps = 0.0, 0.0, 0
    while not done and steps < 600:  # cap, in case a policy gets stuck at the wall
        prediction = predictor.predict(obs["peds"])
        obs, reward, done = env.step(policy.act(obs, (prediction, aci.radii())))
        misses += aci.update(prediction, obs["peds"][:, :2])[0].mean()  # k=1, avg over peds
        total_reward += reward
        steps += 1
        env.render(disks=(prediction, aci.radii()), plan=getattr(policy, "plan", None),
                   certified=getattr(policy, "replan_every", 0))
    radii = aci.radii()
    print(f"total reward {total_reward}, k=1 coverage {1 - misses / steps:.3f} "
          f"(tube target {1 - aci.alpha}), "
          f"mean radii k=1 {radii[0].mean():.3f} m ... k={HORIZON} {radii[-1].mean():.3f} m")
