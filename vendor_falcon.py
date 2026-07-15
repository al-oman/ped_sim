"""Human local-avoidance nudge, lifted from Falcon
(github.com/Zeying-Gong/Falcon, falcon/additional_action.py::update_rel_targ_obstacle).

Falcon's humans follow navmesh waypoints toward a goal and call this to bend
the immediate heading away from nearby agents — a soft velocity-obstacle-style
push whose strength grows with how close and how fast-moving the neighbours are.

Verbatim except: the agent's own position (read there from
self.cur_articulated_agent.base_transformation.translation[[0, 2]]) is passed
in as `curr_pos`, already 2D (x, z) in our sim, so the [[0, 2]] slice is gone.
"""

import numpy as np


def compute_orca_velocity(pos, vel, others_pos, others_vel, max_speed,
                          time_horizon=4.0, combined_radius=0.6):
    """Falcon's ORCA-baseline avoidance velocity, from
    orca_policy.py::ORCAPolicy.compute_orca_velocity — a velocity-obstacle-
    style push away from each nearby agent, averaged and speed-capped.

    Verbatim math; adapted from torch to numpy, and — since our sim carries
    velocity vectors directly — the other agents' velocities are used as-is
    instead of reconstructed from a heading (their [sin, cos] rotation trick),
    and everything is in 2D (x, z) so the [[0, 2]] slices are gone."""
    combined = np.zeros(2)
    for opos, ovel in zip(others_pos, others_vel):
        rel_pos = opos - pos
        rel_vel = vel - ovel
        dist = np.linalg.norm(rel_pos)
        if dist > combined_radius:
            combined += rel_vel + (rel_pos / dist) * (combined_radius - dist) / time_horizon
    adjusted = vel + combined / max(len(others_pos), 1)
    n = np.linalg.norm(adjusted)
    if n > max_speed:
        adjusted = adjusted / n * max_speed
    return adjusted


def update_rel_targ_obstacle(rel_targ, curr_pos, new_human_pos, old_human_pos=None):
    if old_human_pos is None or len(old_human_pos) == 0:
        human_velocity_scale = 0.0
    else:
        # take the norm of the distance between old and new human position
        human_velocity_scale = (
            np.linalg.norm(new_human_pos - old_human_pos) / 0.25
        )  # 0.25 is a magic number
        # set a minimum value for the human velocity scale
        human_velocity_scale = max(human_velocity_scale, 0.1)

    std = 8.0
    # scale the amplitude by the human velocity
    amp = 8.0 * human_velocity_scale

    # Get the position of the other agents
    other_agent_rel_pos, other_agent_dist = [], []
    curr_agent_T = np.asarray(curr_pos)

    other_agent_rel_pos.append(rel_targ[None, :])
    other_agent_dist.append(0.0)  # dummy value
    rel_pos = new_human_pos - curr_agent_T
    dist_pos = np.linalg.norm(rel_pos, ord=2, axis=-1)
    # normalized relative vector
    rel_pos = rel_pos / dist_pos[:, np.newaxis]
    other_agent_dist.extend(dist_pos)
    other_agent_rel_pos.append(-rel_pos)

    rel_pos = np.concatenate(other_agent_rel_pos)
    rel_dist = np.array(other_agent_dist)
    weight = amp * np.exp(-(rel_dist**2) / std)
    weight[0] = 1.0
    weight_norm = weight[:, None] / weight.sum()
    # weighted sum of the old target position and
    # relative position that avoids human
    final_rel_pos = (rel_pos * weight_norm).sum(0)
    return final_rel_pos
