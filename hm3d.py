"""Social-HM3D/MP3D world loading for the 2D testbed.

Maps come from map_gen.py (navmesh occupancy at true walkable geometry);
episodes come straight from the benchmark's json.gz files: robot start/goal
plus each human's fixed waypoints (waypoint_0 is the start, the rest are the
patrol goals — the same episode.info fields Falcon's humans read).

Also a grid A* for global guidance. Real floorplans are concave, so every
agent follows waypoints toward its goal with local avoidance in between —
the same division of labor as the benchmark, where agents get geodesic
waypoints from the navmesh pathfinder.

Positions everywhere are (x, z) world meters; habitat's y (height) only
selects the floor.

Visual check (draws episodes + A* paths onto the map):
    python hm3d.py <scene_map.npz> [episodes.json.gz] [n_episodes]
"""

import gzip
import heapq
import json

import numpy as np

from vendor_falcon import update_rel_targ_obstacle


class GridMap:
    """Occupancy map with world <-> cell transforms (format: map_gen.py)."""

    def __init__(self, path):
        d = np.load(path)
        self.nav = d["nav"]  # [row=z, col=x], True = walkable
        self.mpp = float(d["mpp"])
        self.origin = np.array([float(d["origin_x"]), float(d["origin_z"])])
        self.height = float(d["height"])

    def to_cell(self, xz):
        col, row = np.round((np.asarray(xz, float) - self.origin) / self.mpp).astype(int)
        return row, col

    def to_world(self, cell):
        return self.origin + np.array([cell[1], cell[0]]) * self.mpp

    def navigable(self, xz):
        r, c = self.to_cell(xz)
        return (0 <= r < self.nav.shape[0] and 0 <= c < self.nav.shape[1]
                and bool(self.nav[r, c]))


def load_episodes(path):
    """List of episodes: robot start/goal, floor height, and each human's
    waypoints as a (k, 2) array (row 0 = start, rest = patrol goals)."""
    def xz(p):
        return np.array([p[0], p[2]])

    with gzip.open(path, "rt") as f:
        raw = json.load(f)["episodes"]
    episodes = []
    for e in raw:
        humans = []
        for i in range(e["info"].get("human_num", 0)):
            waypoints = []
            j = 0
            while f"human_{i}_waypoint_{j}_position" in e["info"]:
                waypoints.append(xz(e["info"][f"human_{i}_waypoint_{j}_position"]))
                j += 1
            humans.append(np.array(waypoints))
        episodes.append({"robot_start": xz(e["start_position"]),
                         "robot_goal": xz(e["goals"][0]["position"]),
                         "height": e["start_position"][1],
                         "humans": humans})
    return episodes


def _visible(nav, a, b):
    """Straight line between two cells stays on walkable ground."""
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) + 2
    rows = np.linspace(a[0], b[0], n).round().astype(int)
    cols = np.linspace(a[1], b[1], n).round().astype(int)
    return bool(nav[rows, cols].all())


def _nearest_free(nav, cell):
    free = np.argwhere(nav)
    return tuple(free[np.abs(free - cell).sum(axis=1).argmin()])


def astar(grid, start, goal):
    """8-connected A* from world start to world goal. Returns a short list of
    world waypoints (line-of-sight pruned), or None if no path exists."""
    nav = grid.nav
    s, g = grid.to_cell(start), grid.to_cell(goal)
    if not nav[s]:
        s = _nearest_free(nav, s)
    if not nav[g]:
        g = _nearest_free(nav, g)

    def h(c):  # octile distance
        dr, dc = abs(c[0] - g[0]), abs(c[1] - g[1])
        return max(dr, dc) + 0.4142 * min(dr, dc)

    dist, prev = {s: 0.0}, {}
    frontier = [(h(s), s)]
    while frontier:
        _, cur = heapq.heappop(frontier)
        if cur == g:
            break
        for dr, dc in ((-1, -1), (-1, 0), (-1, 1), (0, -1),
                       (0, 1), (1, -1), (1, 0), (1, 1)):
            nxt = (cur[0] + dr, cur[1] + dc)
            if not (0 <= nxt[0] < nav.shape[0] and 0 <= nxt[1] < nav.shape[1]):
                continue
            if not nav[nxt]:
                continue
            if dr and dc and not (nav[cur[0] + dr, cur[1]] and nav[cur[0], cur[1] + dc]):
                continue  # don't cut corners diagonally
            d = dist[cur] + (1.4142 if dr and dc else 1.0)
            if d < dist.get(nxt, np.inf):
                dist[nxt] = d
                prev[nxt] = cur
                heapq.heappush(frontier, (d + h(nxt), nxt))
    else:
        return None

    cells = [g]
    while cells[-1] != s:
        cells.append(prev[cells[-1]])
    cells = cells[::-1]

    # Greedy line-of-sight pruning: keep only the waypoints that matter.
    pruned, i = [cells[0]], 0
    while i < len(cells) - 1:
        j = i + 1
        while j + 1 < len(cells) and _visible(nav, cells[i], cells[j + 1]):
            j += 1
        pruned.append(cells[j])
        i = j
    return [grid.to_world(c) for c in pruned]


def carrot(waypoints, pos, lookahead=0.15):
    """Pure-pursuit target: project pos onto the waypoint polyline, then return
    the point `lookahead` metres ahead along it. Keeps a follower on the (LOS-
    clear) A* corridor instead of cutting corners toward a far waypoint. The
    lookahead must stay below the tightest doorway width or the robot cuts the
    corner into a wall — real floorplans need ~0.15 m; larger drifts."""
    wp = np.asarray(waypoints, float)
    if len(wp) == 1:
        return wp[0]
    best_d, best_i, best_proj = np.inf, 0, wp[0]
    for i in range(len(wp) - 1):
        a, ab = wp[i], wp[i + 1] - wp[i]
        L2 = ab @ ab
        t = 0.0 if L2 == 0 else np.clip((pos - a) @ ab / L2, 0, 1)
        proj = a + t * ab
        d = np.linalg.norm(pos - proj)
        if d < best_d:
            best_d, best_i, best_proj = d, i, proj
    remaining, cur, i = lookahead, best_proj, best_i
    while i < len(wp) - 1:
        step = wp[i + 1] - cur
        d = np.linalg.norm(step)
        if d >= remaining:
            return cur + step * (remaining / d)
        remaining -= d
        cur = wp[i + 1]
        i += 1
    return wp[-1]


class WaypointCrowd:
    """Humans following their episode waypoints with Falcon's local-avoidance
    nudge. Same interface as the toy Crowd (positions/velocities/step), so the
    predictor, calibrator, and eval code consume it unchanged.

    A* gives each human a global route to its current patrol goal; between
    subwaypoints, update_rel_targ_obstacle bends the heading around nearby
    agents. Robot-blind by default (reactive=0, exactly Falcon); reactive>0
    also bends humans away from the robot — the Experiment 2 knob."""

    NEIGHBOR = 5.0   # only avoid agents within this range, m
    REACH = 0.3      # subwaypoint reached, m

    def __init__(self, humans, grid, dt, speed=1.0, reactive=0.0,
                 rng=None, start_jitter=0.0, speed_jitter=0.0):
        self.grid, self.dt, self.reactive = grid, dt, reactive
        self.goals = [np.asarray(h, float) for h in humans]  # (k, 2) patrol waypoints
        self.pos = np.array([h[0] for h in self.goals])      # start at waypoint 0
        # Per-rollout stochasticity (fidelity_check --repeats): jitter each
        # human's start and speed so repeats of one episode differ, the way
        # habitat's stochastic humans gave Falcon ~15 distinct rollouts per
        # episode. rng=None (default) => deterministic, exactly as before.
        self.speeds = np.full(self.n, speed, float)
        if rng is not None:
            if start_jitter:
                for i in range(self.n):
                    p = self.pos[i] + rng.normal(scale=start_jitter, size=2)
                    self.pos[i] = p if grid.navigable(p) else \
                        grid.to_world(_nearest_free(grid.nav, grid.to_cell(p)))
            if speed_jitter:
                self.speeds *= 1 + rng.uniform(-speed_jitter, speed_jitter, self.n)
        self.prev = self.pos.copy()
        self.goal_idx = [1 % len(h) for h in self.goals]     # heading to waypoint 1
        self.path = [self._plan(i) for i in range(self.n)]

    @property
    def n(self):
        return len(self.pos)

    def positions(self):
        return self.pos.copy()

    def velocities(self):
        return (self.pos - self.prev) / self.dt

    def _plan(self, i):
        p = astar(self.grid, self.pos[i], self.goals[i][self.goal_idx[i]])
        return list(p[1:]) if p and len(p) > 1 else [self.goals[i][self.goal_idx[i]]]

    def _heading(self, i, robot_pos):
        # Follow the cached A* route to the current patrol goal.
        while self.path[i] and np.linalg.norm(self.path[i][0] - self.pos[i]) < self.REACH:
            self.path[i].pop(0)
        if not self.path[i]:
            self.goal_idx[i] = (self.goal_idx[i] + 1) % len(self.goals[i])  # cycle
            self.path[i] = self._plan(i)
        rel = self.path[i][0] - self.pos[i]
        rel = rel / (np.linalg.norm(rel) + 1e-9)

        # Local avoidance among nearby other humans (Falcon, robot-blind).
        d = np.linalg.norm(self.pos - self.pos[i], axis=1)
        others = [j for j in range(self.n) if j != i and 1e-3 < d[j] < self.NEIGHBOR]
        if others:
            rel = update_rel_targ_obstacle(rel, self.pos[i], self.pos[others], self.prev[others])
            rel = rel / (np.linalg.norm(rel) + 1e-9)

        # Reactivity to the robot (Experiment 2): same exp(-dist^2/std) kernel.
        if self.reactive > 0:
            away = self.pos[i] - robot_pos
            dist = np.linalg.norm(away)
            rel = rel + self.reactive * 8.0 * np.exp(-dist ** 2 / 8.0) * away / (dist + 1e-9)
            rel = rel / (np.linalg.norm(rel) + 1e-9)
        return rel

    def step(self, robot_pos, robot_vel=None):
        new = self.pos.copy()
        for i in range(self.n):
            move = self._heading(i, robot_pos) * self.speeds[i] * self.dt
            if self.grid.navigable(self.pos[i] + move):
                new[i] = self.pos[i] + move  # else blocked: hold (A* shouldn't, nudge might)
        self.prev, self.pos = self.pos, new


class HM3DEnv:
    """One Social-HM3D/MP3D episode as a gym-style env, matching the toy Env:

        obs = env.reset()
        obs, reward, done = env.step(action)   # action = robot (vx, vy) m/s

    obs adds a "goal" key (episodes have per-episode goals, unlike the toy):
        "robot": (2,), "peds": (N, 4) [x, z, vx, vz], "goal": (2,)

    reward is -1 while the robot overlaps a human; done when the robot reaches
    the goal. Timeouts are left to the caller's step cap, as in the toy. The
    robot slides along walls (navmesh) per-axis; walls are not penalized, only
    blocking. `reactive` is the crowd's robot-avoidance knob (0 = Falcon)."""

    def __init__(self, grid, episode, dt=0.1, robot_speed=1.0, human_speed=1.0,
                 radius=0.25, goal_thresh=0.3, reactive=0.0, collision_ends=False,
                 rng=None, start_jitter=0.0, speed_jitter=0.0):
        self.grid, self.episode, self.dt = grid, episode, dt
        self.robot_speed, self.human_speed = robot_speed, human_speed
        self.radius, self.goal_thresh, self.reactive = radius, goal_thresh, reactive
        # Falcon ends the episode (as a failure) on the first human collision;
        # set True to match its human_collision / success semantics.
        self.collision_ends = collision_ends
        # per-rollout human stochasticity (see WaypointCrowd); rng=None = off
        self.rng, self.start_jitter, self.speed_jitter = rng, start_jitter, speed_jitter

    def reset(self):
        start = self.episode["robot_start"]
        if not self.grid.navigable(start):  # snap onto the navmesh if slightly off
            start = self.grid.to_world(_nearest_free(self.grid.nav, self.grid.to_cell(start)))
        self.robot = np.array(start, float)
        self.goal = self.episode["robot_goal"]
        self.crowd = WaypointCrowd(self.episode["humans"], self.grid, self.dt,
                                   self.human_speed, self.reactive, rng=self.rng,
                                   start_jitter=self.start_jitter,
                                   speed_jitter=self.speed_jitter)
        return self._obs()

    def step(self, action):
        vel = np.asarray(action, float)
        speed = np.linalg.norm(vel)
        if speed > self.robot_speed:
            vel = vel * (self.robot_speed / speed)
        full = self.robot + vel * self.dt
        if self.grid.navigable(full):
            self.robot = full
        else:  # blocked: slide along the wall, one axis at a time
            for axis in (0, 1):
                cand = self.robot.copy()
                cand[axis] += vel[axis] * self.dt
                if self.grid.navigable(cand):
                    self.robot = cand
        self.crowd.step(self.robot, vel)

        peds = self.crowd.positions()
        collision = bool(len(peds) and np.any(np.linalg.norm(peds - self.robot, axis=1) < 2 * self.radius))
        reward = -1.0 if collision else 0.0
        self.reached = np.linalg.norm(self.robot - self.goal) < self.goal_thresh
        done = self.reached or (self.collision_ends and collision)
        return self._obs(), reward, done

    def _obs(self):
        peds = np.hstack([self.crowd.positions(), self.crowd.velocities()]) \
            if self.crowd.n else np.zeros((0, 4))
        return {"robot": self.robot.copy(), "peds": peds, "goal": self.goal.copy()}


if __name__ == "__main__":
    import sys

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for thing in sys.argv:
        print(thing)
    print(len(sys.argv))

    grid = GridMap(sys.argv[1])
    nrows, ncols = grid.nav.shape
    extent = [grid.origin[0], grid.origin[0] + ncols * grid.mpp,
              grid.origin[1], grid.origin[1] + nrows * grid.mpp]
    plt.figure(figsize=(10, 10 * nrows / ncols))
    plt.imshow(grid.nav, origin="lower", extent=extent, cmap="gray")

    if len(sys.argv) > 2:  # draw real episodes on their map
        episodes = [e for e in load_episodes(sys.argv[2])
                    if abs(e["height"] - grid.height) < 0.5]
        print(f"{len(episodes)} episodes on this floor")
        for e in episodes[:int(sys.argv[3]) if len(sys.argv) > 3 else 3]:
            path = astar(grid, e["robot_start"], e["robot_goal"])
            if path:
                plt.plot(*np.array(path).T, "-o", color="tab:blue", ms=3)
            plt.plot(*e["robot_start"], "b^", ms=10)
            plt.plot(*e["robot_goal"], "g*", ms=14)
            for wps in e["humans"]:
                for a, b in zip(wps[:-1], wps[1:]):
                    p = astar(grid, a, b)
                    if p:
                        plt.plot(*np.array(p).T, "--", color="tab:orange", lw=1)
                plt.plot(*wps.T, "o", color="tab:orange", ms=5)
    else:  # no episodes: demo A* between random free points
        rng = np.random.default_rng(0)
        free = np.argwhere(grid.nav)
        for _ in range(3):
            a, b = free[rng.integers(len(free), size=2)]
            path = astar(grid, grid.to_world(tuple(a)), grid.to_world(tuple(b)))
            if path:
                plt.plot(*np.array(path).T, "-o", ms=3)
    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.savefig("hm3d_check.png", dpi=140, bbox_inches="tight")
    print("saved hm3d_check.png")
