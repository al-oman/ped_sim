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
  `aci.py` (`ACI` union-bound + `MaxACI` max-over-horizon), `flow.pt` (trained),
  plus `eval.py`, `sweep.py`, `deference.py`, `ablation.py`, `compare_cp.py`,
  `shift_experiment.py`.
- **Flow model** (restructured 2026-07-16 to mirror the official Diffuser/SafeDiffuser/
  SafeFlowMatcher repos, per their "architectures strictly adhere to" claim):
  `diffuser/models/{temporal,helpers}.py` (vendored verbatim), `diffuser/models/cfm.py`
  (`CFM`: loss + `conditional_sample` with the CBF brake/projection/warm-start inside the
  Euler loop, plus the one documented deviation — obs conditioning enters via the time
  embedding (`_CondTimeMLP`), since peds aren't in the generated trajectory and the user
  chose NOT to generate them jointly), `diffuser/datasets/{sequence,normalization}.py`
  (trajectories use `IsotropicNormalizer`, scalar std, because ACI disks must stay circles),
  `diffuser/utils/training.py` (real-EMA `Trainer`; note SafeFlowMatcher's own EMA is
  accidentally a no-op — `ema_model = model` aliased), `config/toy.py`, `scripts/train.py`.
  Old `flow_model.py`/`train.py`/`vendor_diffuser/` are superseded (deletion pending user
  OK); pre-restructure checkpoint backed up as `flow_legacy.pt` (loads via key-remap;
  verified bit-identical outputs old vs new path).
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

## Calibrators (updated 2026-07-17)
Five classes in `aci.py`: `ACI`/`DtACI` (union bound; DtACI self-tunes gamma via a dial
bank) and `MaxACI`/`PartialMaxACI`/`MaxDtACI` (max-over-horizon; partial removes the
horizon-step feedback lag, MaxDtACI = partial + DtACI's self-tuning gamma — the two
orthogonal fixes combined). Two design axes: aggregation (union vs max) × self-tuning
(fixed gamma vs dt gamma bank); max+/maxdt+ also drop the feedback lag. Registry
`aci.CALIBRATORS` = {aci, dtaci, max, max+, maxdt+} + `make_calibrator()` (DtACI and
MaxDtACI take no gamma). MaxDtACI grades its whole dial bank on the shared displayed-disk
event (keeps the max family's operational guarantee); its auto-tuning is therefore
principled-but-heuristic (DtACI's regret bound assumes per-dial grading) — documented in
the class. Synthetic check (n_peds 3, 3000 steps): tube coverage aci 0.953 / dtaci 0.949 /
max 0.900 / max+ 0.899 / maxdt+ 0.899, max-family disks ~10% smaller. Wired everywhere: `eval.py` (one flow row per calibrator),
`deference.py` (all four + raw), `compare_cp.py`/`shift_experiment.py` (all six incl.
split/online CP), `sweep.py`/`ablation.py` (`--cal=` flag, separate output CSVs),
`sim.py` (`--cal=`), `hm3d_eval.py` (`--cal=` for the flow+cal row).
`sim.py` is the live demo: `python sim.py [--cal=aci|dtaci|max|max+] [--policy=flow|orca|walk]
[--alpha=0.1]`; renders keep-out disks + the predicted pedestrian paths inside them (orange)
+ the robot's plan, prints k=1 and tube coverage.

## Next steps
1. ~~Finish `RvoPolicy`~~ DONE — in `hm3d_policies.py` (one-step RVO2 solve around A*-carrot
   pref velocity, agent radius 0.3 > env 0.25 for margin), wired into `hm3d_eval.py` main.
2. ~~Rebuild the flow model for HM3D~~ IN PROGRESS (2026-07-17): `diffuser/datasets/hm3d.py`
   (`HM3DSequenceDataset`: RvoPolicy expert on train split, cond = carrot vector + K=4
   nearest peds padded at 10 m, cond_dim 18) + `config/hm3d.py`;
   `scripts/train.py --config=hm3d --device=mps` -> `hm3d_flow.pt` (training launched).
3. ~~Wire `FlowPolicy` + calibrators into `hm3d_eval.py`~~ DONE (untested until the model
   lands): `hm3d_policies.FlowPolicy` subclasses the toy policy, overriding only
   `_condition()`; `hm3d_eval.run_episode/evaluate` take an optional calibrator name —
   fresh calibrator per episode (crowd size varies; radii start inf = disks disabled until
   warmed). Flow rows appear in `hm3d_eval` automatically once `hm3d_flow.pt` exists.
   **Validate the HM3D flow model when training finishes** (sane SR vs A*/RVO, then flow+cal).

   SCALE-UP (2026-07-17): 120 scenes x 10 eps, 100k steps (`config/hm3d.py`), ~1 h MPS
   (~6x faster than CPU; HPC not needed at this model size). Held-out scenes[120:140]
   (n=70, collision_ends=True): astar 46/53%coll, rvo 50/9%, flow 49/46%, flow+aci
   **61/26%**, flow+max+ 51/30%, flow+maxdt+ 57/27%. Same-scenes checkpoint comparison:
   30sc model flow 39%/flow+aci 49% -> 120sc model 49%/61% — data scaling and the safety
   layer each add ~+10-12 SR and stack; flow+aci now beats the RVO2 expert it imitated
   (disks avoid the expert's deadlock stalls). 30sc checkpoint kept as `hm3d_flow_30sc.pt`.
   NOTE: eval held-out sets must start at shuffled scene index >=120 now.

   TUNING LOG (2026-07-17): three iterations on minival (n=8, treat as directional only).
   (a) v1 model crawled (336 steps vs expert 72) — cause: RVO2 expert deadlocks vs
   robot-blind humans (~16% of episodes burn 1000 standing-still steps; median training
   window displacement was 0 m). (b) Deadlock detector added in `diffuser/datasets/hm3d.py`
   (<0.1 m progress over 50 steps ends collection); full trim taught the model to plow
   through people (102 coll-steps) — now keeps ~20 stall steps so stopping-near-people
   survives in the data. (c) Per-episode calibrators were cold at first encounter —
   `hm3d_eval.warm_calibrator()` pre-fills buffers on a robot-free replay of the episode's
   humans (valid because they're robot-blind). Current minival snapshot
   (collision_ends=True): astar 62%/38% coll, rvo 75%/0%, flow 62%/38%, flow+aci 50%/12%,
   flow+max+ 50%/12% with closest-med 0.97 m; max+ disks (no union bound) roughly double
   SR over union-aci when timeouts dominate. Remaining gaps: ~3/8 disk-episodes time out
   (infeasibility brake in corridors), flow raw ≈ astar profile — likely wants more
   training scenes/episodes (config currently 30x10).
4. Experiment 1: A* / RVO2-ORCA / raw-flow / flow+cal × {aci, dtaci, max, max+} on HM3D.
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
