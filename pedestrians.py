"""ORCA pedestrians, simulated with RVO2.

Pedestrians walk toward random goal points while avoiding each other, the
walls, AND the robot. The robot lives inside the RVO2 sim as one extra agent
whose position and velocity we overwrite every step from the outside — so
pedestrians react to it, but ORCA never controls it.
"""

import numpy as np
import rvo2


class Crowd:
    def __init__(self, n, bounds, dt, radius, speed, rng, walls=()):
        self.n, self.bounds, self.speed, self.rng = n, np.array(bounds), speed, rng
        self.radius, self.walls = radius, walls
        # args: dt, neighborDist, maxNeighbors, timeHorizon, timeHorizonObst, radius, maxSpeed
        self.sim = rvo2.PyRVOSimulator(dt, 5.0, 10, 2.0, 2.0, radius, speed)
        for x0, x1, y0, y1 in walls:
            self.sim.addObstacle([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        self.sim.processObstacles()
        for _ in range(n):
            self.sim.addAgent(tuple(self.free_point()))
        # Robot agent gets an inflated radius so pedestrians give it a wide
        # berth even though it never reciprocates the avoidance.
        self.robot_id = self.sim.addAgent((0, 0), 5.0, 10, 2.0, 2.0, 0.4, speed, (0, 0))
        self.goals = np.array([self.free_point() for _ in range(n)])
        self.stuck = np.zeros(n)  # steps spent barely moving

    def free_point(self):
        """Random point in the arena that isn't inside (or too near) a wall."""
        while True:
            p = self.rng.uniform(1, self.bounds - 1)
            if not any(x0 - self.radius <= p[0] <= x1 + self.radius and
                       y0 - self.radius <= p[1] <= y1 + self.radius
                       for x0, x1, y0, y1 in self.walls):
                return p

    def set_speed(self, speed):
        """Change walking speed mid-run (e.g., to create a behavior shift)."""
        self.speed = speed
        for i in range(self.n):
            self.sim.setAgentMaxSpeed(i, speed)

    def positions(self):
        return np.array([self.sim.getAgentPosition(i) for i in range(self.n)])

    def velocities(self):
        return np.array([self.sim.getAgentVelocity(i) for i in range(self.n)])

    def step(self, robot_pos, robot_vel):
        # Each pedestrian prefers to walk straight at its goal; ORCA turns
        # that preference into a collision-free velocity.
        to_goal = self.goals - self.positions()
        dist = np.linalg.norm(to_goal, axis=1, keepdims=True)
        pref = self.speed * to_goal / (dist + 1e-6)
        for i in range(self.n):
            self.sim.setAgentPrefVelocity(i, tuple(pref[i]))

        # Pin the robot agent to its externally-computed state.
        self.sim.setAgentPosition(self.robot_id, tuple(robot_pos))
        self.sim.setAgentVelocity(self.robot_id, tuple(robot_vel))
        self.sim.setAgentPrefVelocity(self.robot_id, tuple(robot_vel))

        self.sim.doStep()

        # New goal for pedestrians that arrived — or that are stuck pushing
        # against a wall (ORCA is local; a goal behind the wall can pin them).
        speeds = np.linalg.norm(self.velocities(), axis=1)
        self.stuck = np.where(speeds < 0.1 * self.speed, self.stuck + 1, 0)
        for i in np.where((dist[:, 0] < 0.3) | (self.stuck > 30))[0]:
            self.goals[i] = self.free_point()
            self.stuck[i] = 0
