# Project: ACI-Safe Flow Matching for Social Navigation — HM3D Deployment

## What this is
Porting a 2D pedestrian-avoidance testbed (`~/workspaces/diffusion/ped_sim/`) to run on the
**Social-HM3D / Social-MP3D** benchmark (from the Falcon repo). Core research: an
adaptive-conformal-inference (ACI) safety layer that sizes uncertainty "bubbles" around
predicted pedestrians and enforces them *inside* a flow-matching model's sampling loop. The
2D toy world is the analysis microscope; HM3D is the realism/scale validation. Method uses
**privileged state** (ground-truth ped positions/velocities) and a **holonomic** robot
(quadruped target), so it is NOT directly comparable to Falcon's egocentric learned numbers —
only to their A*/ORCA classical baselines, and even that is a loose ranking check.

## Repo layout (`ped_sim/`)
- **Toy sim** (working, done): `sim.py`, `pedestrians.py`, `policies.py`, `predictor.py`,
  `aci.py` (`ACI` union-bound + `MaxACI` max-over-horizon), `flow_model.py` (vendored
  Diffuser/SafeFlowMatcher `TemporalUnet` as `BaselineUnet`, ~4M params), `flow.pt` (trained),
  plus `eval.py`, `sweep.py`, `deference.py`, `ablation.py`, `compare_cp.py`,
  `shift_experiment.py`.
- **HM3D port** (in progress): `hm3d.py` (`GridMap`, `load_episodes`, `astar`, `carrot`,
  `WaypointCrowd`, `HM3DEnv`), `hm3d_policies.py` (`AStarPolicy`, `OrcaPolicy`, adding
  `RvoPolicy`), `hm3d_eval.py` (`scene_index`/`scene_episodes`/`evaluate`/`run_episode`),
  `fidelity_check.py`, `vendor_falcon.py` (Falcon's `update_rel_targ_obstacle` +
  `compute_orca_velocity`, verbatim).
- **Maps**: `~/workspaces/diffusion/socialnav_map_gen/` — `map_gen.py`/`map_gen_all.py`
  extract per-floor navmesh occupancy `.npz` (human + spot clearance profiles) from HM3D/MP3D
  scenes. Maps symlinked into `ped_sim/data/`. Episodes (json.gz) under
  `socialnav_map_gen/pointnav/social-{hm3d,mp3d}/{split}/content/`.

## Key facts / decisions
- Falcon's humans are **robot-blind** (avoid each other via a weak ORCA nudge, not the robot);
  episode json.gz store human **waypoints** (fixed patrol goals), not trajectories.
- Falcon's A*/ORCA baselines are **discrete** (turn/forward); we run **holonomic** → ORCA
  ranking may invert (discretization amplifies avoidance). Falcon's `compute_orca_velocity` is
  genuinely weak head-on (confirmed) → adding a competent **RVO2** baseline (`RvoPolicy`,
  half-done in `hm3d_policies.py`).
- `HM3DEnv(collision_ends=True)` matches Falcon's semantics (collision terminates episode as
  failure). `human_collision` metric is per-episode binary + terminal.
- `carrot()` pure-pursuit lookahead must stay <=0.15 m (narrow real doorways).
- Encounters ARE abundant (~46% of MP3D episodes bring naive A* within collision range) —
  earlier "low density" worry was a sampling artifact.

## Fidelity validation: DONE (2026-07-16)
Ran `fidelity_check.py data ../socialnav_map_gen/pointnav/social-hm3d/minival_falcon 150 40
--falcon-astar astar_minival.log --falcon-orca orca_minival.log` (logs copied into `ped_sim/`).

**Key discovery: minival is only 10 episodes** (one per content json.gz; top-level
`minival.json.gz` is empty). Falcon's "~150 episodes" = habitat cycling those 10 ~15× with
stochastic humans (`test_episode_count=150` > available). So our n=10 vs their n=150-with-repeats.
- `minival_falcon/` is an exact rsync of the cluster set (`speed:/speed-scratch/al_oman/VLA/
  falcon/data/datasets/pointnav/social-hm3d/minival/`); the old local `minival/` had
  `zUG6FL9TYeR_ep1_1` gunzipped to plain `.json`, which the `*.json.gz` glob silently skipped.
- 2 of the 10 episodes have **zero humans**; Falcon scores them, so `evaluate()` grew a
  `min_humans` flag (fidelity passes 0; research evals keep the default 1).

Results (success / human_collision / spl):
| | A* | ORCA | RVO2 (ours only) |
|---|---|---|---|
| Falcon | .453 / .547 / .444 | .380 / .427 / .325 | — |
| ours | .700 / .300 / .700 | .700 / .300 / .700 | .800 / .000 / .770 |

Reading: (a) not degenerate; (b) ours higher than Falcon's ≈ expected holonomic advantage;
(c) our ORCA == our A* to the decimal — Falcon's weak `compute_orca_velocity` changes nothing
on 10 holonomic episodes (the anticipated action-model gap; noted, not chased);
(d) new `RvoPolicy` behaves exactly as a competent baseline should: zero collisions, small SPL
cost, and its 2 failures are non-collision (timeout/hard), not crashes.

## Next steps
1. ~~Finish `RvoPolicy`~~ DONE — in `hm3d_policies.py` (one-step RVO2 solve around A*-carrot
   pref velocity, agent radius 0.3 > env 0.25 for margin), wired into `hm3d_eval.py` main.
2. **Rebuild the flow model for HM3D** (big ML lift): goal-conditioning (`condition()` in
   `flow_model.py`), variable pedestrian count (pad K-nearest), A*-waypoint-guided ORCA expert
   for data collection, retrain. Validate on toy first (regression), then HM3D.
3. Wire `FlowPolicy` + calibrators (`ACI`/`MaxACI`) into `hm3d_eval.py` — disk-constrained
   sampling carries over unchanged.
4. Experiment 1: A* / RVO2-ORCA / raw-flow / flow+ACI × calibrators on HM3D.
   Experiment 2: add human->robot reactivity via `HM3DEnv(reactive=...)`, re-run.

## Environment notes
- Local venv `ped_sim/.venv` (numpy, torch, pygame, rvo2, matplotlib, einops); run headless
  with `SDL_VIDEODRIVER=dummy` and pipe `2>&1 | grep -v -i pygame`. Checkpoints load with
  `map_location="cpu"`; default device CPU (MIG/MPS slower for this tiny model). `--device=cuda`
  flag exists on scripts.
- Cluster is **tcsh** + habitat-sim env. SLURM triggers habitat "distributed mode" error — fix
  with `setenv SLURM_PROCID 0` / `SLURM_NTASKS 1` / `SLURM_LOCALID 0` / `SLURM_NNODES 1`. MIG
  GPUs can't do habitat EGL rendering (need a full GPU). Correct module invocation is
  `python -m habitat_baselines.run` (underscore); `run.py` does `import falcon` so falcon repo
  root must be on PYTHONPATH. Baseline eval overrides: `habitat.dataset.split=minival`,
  `habitat_baselines.test_episode_count=150`, `habitat_baselines.eval.video_option="[]"`.
