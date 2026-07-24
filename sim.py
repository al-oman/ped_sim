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

from pedestrians import Crowd
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

        self.screen.fill((255, 255, 255))  # white floor (paper-friendly)
        for x0, x1, y0, y1 in WALLS:
            pygame.draw.rect(self.screen, (150, 150, 150),  # grey walls
                             (*px((x0, y0)), int((x1 - x0) * PX_PER_M), int((y1 - y0) * PX_PER_M)))
        for p in self.crowd.positions():
            pygame.draw.circle(self.screen, (220, 180, 60), px(p), int(RADIUS * PX_PER_M))
        pygame.draw.circle(self.screen, (80, 160, 255), px(self.robot), int(RADIUS * PX_PER_M))
        if disks is not None:
            prediction, radii = disks  # (HORIZON, N, 2), (HORIZON, N)
            for k, (centers, rs) in enumerate(zip(prediction, radii)):
                for c, r in zip(centers, rs):
                    if np.isfinite(r):
                        if k < certified:  # enforced keep-out: bold dark outline
                            pygame.draw.circle(self.screen, (40, 40, 40), px(c),
                                               int((r + 2 * RADIUS) * PX_PER_M), width=2)
                        else:              # unenforced uncertainty: light grey
                            pygame.draw.circle(self.screen, (180, 180, 180), px(c),
                                               int((r + RADIUS) * PX_PER_M), width=1)
            # The predicted trajectory the disks are centered on: a path from
            # each pedestrian through its HORIZON predicted positions.
            for n, start in enumerate(self.crowd.positions()):
                pts = [px(start)] + [px(p) for p in prediction[:, n]]
                pygame.draw.lines(self.screen, (220, 140, 60), False, pts, 2)
                for p in pts[1:]:
                    pygame.draw.circle(self.screen, (220, 140, 60), p, 3)
        if plan is not None:
            for p in plan:
                pygame.draw.circle(self.screen, (80, 160, 255), px(p), 3)
        pygame.display.flip()
        self.clock.tick(1 / DT)


class HM3DViz:
    """Pygame renderer for a real Social-HM3D floor, same visual language as
    the toy Env.render (yellow humans, blue robot + plan dots, bright certified
    keep-out disks / dim uncertainty disks, orange predicted paths) but over
    the scene's navmesh instead of the toy box. The floor is drawn once from
    the occupancy grid; everything else is redrawn each step."""

    MAXPX = 900  # fit the map's longer side to this many pixels

    def __init__(self, grid):
        self.grid = grid
        nrows, ncols = grid.nav.shape
        self.ppm = min(60.0, self.MAXPX / (max(nrows, ncols) * grid.mpp))  # pixels per metre
        self.W, self.H = int(ncols * grid.mpp * self.ppm), int(nrows * grid.mpp * self.ppm)
        self.screen = None

    def _init(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.W, self.H))
        self.clock = pygame.time.Clock()
        nrows, ncols = self.grid.nav.shape
        rgb = np.full((ncols, nrows, 3), 150, np.uint8)  # surfarray is [x=col, y=row]
        rgb[self.grid.nav.T] = (255, 255, 255)           # walkable floor white, non-nav grey
        self.floor = pygame.transform.scale(
            pygame.surfarray.make_surface(rgb), (self.W, self.H))

    def _px(self, p):
        q = (np.asarray(p, float) - self.grid.origin) * self.ppm
        return (int(q[0]), int(q[1]))

    def render(self, env, disks=None, plan=None, certified=0):
        if self.screen is None:
            self._init()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit
        R, px = env.radius, self._px
        self.screen.blit(self.floor, (0, 0))
        pygame.draw.circle(self.screen, (90, 220, 120), px(env.goal),
                           max(5, int(R * self.ppm)), width=2)          # goal
        for p in env.crowd.positions():
            pygame.draw.circle(self.screen, (220, 180, 60), px(p), int(R * self.ppm))
        pygame.draw.circle(self.screen, (80, 160, 255), px(env.robot), int(R * self.ppm))
        if disks is not None:
            prediction, radii = disks
            for k, (centers, rs) in enumerate(zip(prediction, radii)):
                for c, r in zip(centers, rs):
                    if np.isfinite(r):
                        # enforced keep-out: bold dark; uncertainty: light grey
                        col, w, extra = (((40, 40, 40), 2, 2 * R) if k < certified
                                         else ((180, 180, 180), 1, R))
                        pygame.draw.circle(self.screen, col, px(c),
                                           int((r + extra) * self.ppm), width=w)
            for n, start in enumerate(env.crowd.positions()):
                pts = [px(start)] + [px(p) for p in prediction[:, n]]
                pygame.draw.lines(self.screen, (220, 140, 60), False, pts, 2)
                for p in pts[1:]:
                    pygame.draw.circle(self.screen, (220, 140, 60), p, 3)
        if plan is not None:
            for p in plan:
                pygame.draw.circle(self.screen, (80, 160, 255), px(p), 3)
        pygame.display.flip()
        self.clock.tick(1 / DT)


def demo_loop(env, policy, aci, predictor, render, max_steps):
    """One rendered episode, shared by the toy and HM3D demos (both envs share
    the gym API). Returns (steps, reward, k=1 coverage, tube coverage). `render`
    is called each step with the current disks; disks are None on the steps a
    scene happens to have no humans."""
    obs = env.reset()
    for p in {id(p): p for p in (predictor, getattr(policy, "predictor", None))
              if hasattr(p, "reset")}.values():
        p.reset()  # stateful predictors (SocialLSTM) are per-episode
    done, steps, total_reward = False, 0, 0.0
    misses, miss_n, tube_miss, tube_n, pending = 0.0, 0, 0.0, 0, []
    while not done and steps < max_steps:
        disks = prediction = None
        if len(obs["peds"]):
            prediction = predictor.predict(obs["peds"])
            disks = (prediction, aci.radii())
        obs, reward, done = env.step(policy.act(obs, disks))
        if disks is not None:
            missed = aci.update(prediction, obs["peds"][:, :2])
            misses += missed[0].mean()  # k=1, avg over peds
            miss_n += 1
            # Tube bookkeeping as in eval.py: a prediction's tube fails if ANY
            # of its HORIZON lookaheads missed — the alpha-level guarantee event.
            pending.insert(0, np.zeros((HORIZON, missed.shape[1]), bool))
            for j in range(len(missed)):
                pending[j][j] = missed[j]
            if len(pending) == HORIZON:  # oldest prediction fully checked
                tube_miss += pending.pop().any(axis=0).mean()
                tube_n += 1
            disks = (prediction, aci.radii())  # post-update radii, as the toy drew
        total_reward += reward  # 0/-1 per step; kept for the toy print
        steps += 1
        render(disks)
    k1 = 1 - misses / miss_n if miss_n else float("nan")
    tube = 1 - tube_miss / tube_n if tube_n else float("nan")
    return steps, total_reward, k1, tube


if __name__ == "__main__":
    import argparse

    from aci import CALIBRATORS, make_calibrator

    ap = argparse.ArgumentParser(
        description="Live demo: one rendered episode. White circles are the "
                    "calibrated keep-out disks, orange the predicted pedestrian "
                    "paths they are centered on, blue dots the robot's plan. "
                    "Default world is the toy box; --hm3d renders a real "
                    "Social-HM3D floor instead.")
    ap.add_argument("--cal", choices=list(CALIBRATORS), default="aci",
                    help="disk calibrator — union-bound per-lookahead trackers "
                         "(aci, dtaci) vs max-over-horizon aggregation (max, max+, "
                         "maxdt+); dtaci/maxdt+ self-tune gamma, max+/maxdt+ drop the lag")
    ap.add_argument("--policy", default="flow",
                    help="toy: flow|orca|walk; hm3d: astar|orca|rvo|flow")
    ap.add_argument("--alpha", type=float, default=0.1,
                    help="target tube miscoverage (default 0.1)")
    ap.add_argument("--hm3d", action="store_true", help="render a real HM3D scene")
    ap.add_argument("--map-root", default="data")
    ap.add_argument("--ep-root", default="/Users/alexoman/workspaces/diffusion/"
                    "socialnav_map_gen/pointnav/social-hm3d/minival_falcon")
    ap.add_argument("--scene", default=None, help="scene id (default: first with humans)")
    ap.add_argument("--episode", type=int, default=0, help="which humans-episode in the scene")
    ap.add_argument("--reactive", type=float, default=0.0,
                    help="crowd's robot-avoidance strength (0 = Falcon-blind)")
    ap.add_argument("--predictor", choices=["cv", "slstm"], default="cv",
                    help="trajectory predictor behind the disks: cv = constant "
                         "velocity, slstm = the trained Social-LSTM "
                         "(social_lstm.pt). A futures-conditioned flow "
                         "checkpoint overrides this with its own predictor, so "
                         "conditioning and disks always share one forecast.")
    ap.add_argument("--save-frame", default=None, metavar="PATH",
                    help="save the rendered frame at --frame-step to PATH and exit")
    ap.add_argument("--frame-step", type=int, default=200,
                    help="step at which --save-frame captures (default 200)")
    ap.add_argument("--ckpt", default=None,
                    help="flow checkpoint to demo (default: flow.pt in the toy, "
                         "hm3d_flow.pt on --hm3d; use hm3d_flow_slstm.pt for "
                         "the Social-LSTM-conditioned model)")
    args = ap.parse_args()

    def frame_saver(screen_of):
        """Wrap a render closure: capture the pygame surface at --frame-step."""
        count = [0]

        def hook():
            count[0] += 1
            if args.save_frame and count[0] == args.frame_step:
                pygame.image.save(screen_of(), args.save_frame)
                print(f"saved frame at step {count[0]} -> {args.save_frame}")
                raise SystemExit
        return hook

    if args.predictor == "slstm":
        from predictor import SocialLSTM
        predictor = SocialLSTM(DT, HORIZON)  # loads social_lstm.pt
    else:
        predictor = ConstantVelocity(DT, HORIZON)

    if not args.hm3d:
        from policies import FlowPolicy, OrcaExpert, WalkForward
        env = Env()
        policy = {"flow": lambda: FlowPolicy(path=args.ckpt or "flow.pt"),
                  "orca": lambda: OrcaExpert(goal=(WIDTH - 0.3, HEIGHT / 2), dt=DT,
                                             radius=0.25, walls=WALLS),
                  "walk": WalkForward}[args.policy]()
        aci = make_calibrator(args.cal, alpha=args.alpha, horizon=HORIZON, n_peds=N_PEDS)

        save_hook = frame_saver(lambda: env.screen)

        def render(disks):
            env.render(disks=disks, plan=getattr(policy, "plan", None),
                       certified=getattr(policy, "replan_every", 0))
            save_hook()

        steps, reward, k1, tube = demo_loop(env, policy, aci, predictor, render, max_steps=600)
        radii = aci.radii()
        print(f"policy {args.policy}, calibrator {args.cal}: total reward {reward}, "
              f"k=1 coverage {k1:.3f}, tube coverage {tube:.3f} (target {1 - aci.alpha}), "
              f"mean radii k=1 {radii[0].mean():.3f} m ... k={HORIZON} {radii[-1].mean():.3f} m")
    else:
        from hm3d import HM3DEnv
        from hm3d_eval import scene_episodes, scene_index
        from hm3d_policies import AStarPolicy, OrcaPolicy, RvoPolicy
        from hm3d_policies import FlowPolicy as HM3DFlowPolicy

        idx = scene_index(args.map_root, args.ep_root)
        if not idx:
            raise SystemExit(f"no scenes with both maps and episodes under "
                             f"{args.map_root} / {args.ep_root}")
        # default: first scene that actually has a humans-episode (some minival
        # scenes are human-free); a named --scene is used as-is.
        scenes = [args.scene] if args.scene else list(idx)
        pairs, scene = [], args.scene
        for s in scenes:
            pairs = [(g, e) for g, e in scene_episodes(*idx[s]) if len(e["humans"]) > 0]
            if pairs:
                scene = s
                break
        if not pairs:
            raise SystemExit(f"no episodes with humans in "
                             f"{'scene ' + args.scene if args.scene else 'any scene'}")
        grid, ep = pairs[args.episode % len(pairs)]
        print(f"scene {scene}: {len(pairs)} humans-episodes, showing #{args.episode % len(pairs)} "
              f"({len(ep['humans'])} humans); policy {args.policy}, calibrator {args.cal}")

        factory = {"astar": lambda g: AStarPolicy(g),
                   "orca": lambda g: OrcaPolicy(g),
                   "rvo": lambda g: RvoPolicy(g, dt=DT),
                   "flow": lambda g: HM3DFlowPolicy(g, path=args.ckpt or "hm3d_flow.pt")}
        if args.policy not in factory:
            raise SystemExit(f"--policy for --hm3d must be one of {list(factory)}")
        policy = factory[args.policy](grid)
        # futures-conditioned flow checkpoints carry their own predictor; use it
        # for the disks too, so both see the same forecast
        predictor = getattr(policy, "predictor", None) or predictor
        print(f"disk predictor: {type(predictor).__name__}"
              + (" (from checkpoint cond_mode, overrides --predictor)"
                 if getattr(policy, "predictor", None) is not None else ""))
        env = HM3DEnv(grid, ep, dt=DT, reactive=args.reactive)
        aci = make_calibrator(args.cal, alpha=args.alpha, horizon=HORIZON,
                              n_peds=len(ep["humans"]))
        viz = HM3DViz(grid)

        save_hook = frame_saver(lambda: viz.screen)

        def render(disks):
            viz.render(env, disks=disks, plan=getattr(policy, "plan", None),
                       certified=getattr(policy, "replan_every", 0))
            save_hook()

        steps, reward, k1, tube = demo_loop(env, policy, aci, predictor, render, max_steps=1000)
        radii = aci.radii()
        print(f"reached {getattr(env, 'reached', False)} in {steps} steps, "
              f"k=1 coverage {k1:.3f}, tube coverage {tube:.3f} (target {1 - aci.alpha}), "
              f"mean radii k=1 {radii[0].mean():.3f} m ... k={HORIZON} {radii[-1].mean():.3f} m")
