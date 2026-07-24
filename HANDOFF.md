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

## Calibrators (updated 2026-07-24)
Five classes in `aci.py`: `ACI`/`DtACI` (union bound; DtACI self-tunes gamma via a dial
bank) and `MaxACI`/`PartialMaxACI`/`MaxDtACI` (max-over-horizon; partial removes the
horizon-step feedback lag, MaxDtACI = partial + DtACI's self-tuning gamma). Two design axes:
aggregation (union vs max) × self-tuning (fixed gamma vs dt gamma bank); max+/maxdt+ also
drop the feedback lag. Registry `aci.CALIBRATORS` = {aci, dtaci, max, max+, maxdt+} +
`make_calibrator()` (DtACI and MaxDtACI take no gamma).
**Paper role (2026-07-24):** the method is aci + max+ (Partial-Max). dtaci is a baseline;
**maxdt+ is demoted to future-work** — it combines two individually-fine pieces that
interact badly (see "Angle change" in Paper kit). Keep the classes in the registry; just
don't headline maxdt+.
**Debt-queue fix (2026-07-24):** `PartialMaxACI`/`MaxDtACI` now rate-limit tube-violation
penalties to one -gamma per step (excess queued as `self.debt`, drained later). Restores the
one-move-per-step invariant every ACI variant assumes (what bounds alpha_t near [0,1]); total
delivered is unchanged, so the fixed-point argument survives. Precedent: multi-step conformal
PID (arXiv:2410.13115) does the same "one update per step / cap at largest score" limiting.
This slightly changed `max+` numbers (gamma=0.01) — **re-run any max+/maxdt+ result rows.**
MaxDtACI grades its whole dial bank on the shared displayed-disk event (keeps the max family's
operational guarantee); its auto-tuning is therefore principled-but-heuristic (DtACI's regret
bound assumes per-dial grading) — documented in the class. Synthetic check (n_peds 3, 3000
steps): tube coverage aci 0.953 / dtaci 0.949 / max 0.900 / max+ 0.899 / maxdt+ 0.899,
max-family disks ~10% smaller. Wired everywhere: `eval.py` (one flow row per calibrator),
`deference.py` (all four + raw), `compare_cp.py`/`shift_experiment.py` (all six incl.
split/online CP), `sweep.py`/`ablation.py` (`--cal=` flag, separate output CSVs),
`sim.py` (`--cal=`), `hm3d_eval.py` (`--cal=` for the flow+cal row).
`sim.py` is the live demo, two worlds:
- toy box (default): `python sim.py [--cal=...] [--policy=flow|orca|walk] [--alpha=0.1]`.
- real HM3D floor: `python sim.py --hm3d [--policy=astar|orca|rvo|flow] [--cal=...]
  [--scene=<id>] [--episode=N] [--reactive=0.0] [--map-root=data] [--ep-root=<...>]
  [--predictor=cv|slstm] [--ckpt=<flow ckpt>]`. `--predictor=slstm` puts Social-LSTM
  forecasts behind the disks (visibly tighter far-horizon disks than cv);
  `--ckpt=hm3d_flow_slstm.pt` demos the futures-conditioned model — its checkpoint's own
  predictor then feeds BOTH conditioning and disks, overriding `--predictor` (printed).
  Renders the navmesh + agents + keep-out disks + orange predicted paths + robot plan;
  prints reached/steps and k=1 & tube coverage. `--policy=flow` loads `hm3d_flow.pt`.
  Shared `demo_loop()` drives both; HM3D drawing is `HM3DViz` (navmesh floor surface,
  world→pixel from grid origin/mpp, auto-fit to 900 px). Default ep-root is minival_falcon
  (held out from training). Verified headless on scene ZVPMj4YoZtK (rvo reaches goal, 70
  steps); frame saved as `hm3d_demo_frame.png`.

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
4. ~~Experiment 1~~ DONE (2026-07-19): `hm3d_exp1.py` -> `hm3d_exp1.csv` (merged), both
   flow models × {raw + all 5 calibrators} + astar/rvo, scenes[120:160], n=140/cell,
   collision_ends=True, per-episode-seeded sampling. SR/coll headline:
   astar 54/46, rvo 59/6; flow[state] 54/43, +aci 64/17 (best SR), +dtaci 63/19,
   max-family 61-62/19-21; flow[slstm] 51/39, +aci & +dtaci 60/9 (best safety),
   max-family 60-61/12-14. Reads: (a) every calibrated flow row beats rvo's SR;
   state+aci surpasses the expert (+5 SR) though rvo keeps the collision floor (6%);
   (b) slstm conditioning uniformly halves-ish collisions at ~equal SR within the
   max-family and lands 9% with union — the Pareto pair is state+aci (64/17) vs
   slstm+aci (60/9); (c) dtaci ≡ aci and maxdt+ ≈ max+ within noise in both models:
   the self-tuning variants cost nothing — report them as the knob-free defaults.
   ~~Experiment 2~~ DONE (2026-07-17): `hm3d_exp2.py` -> `hm3d_exp2.csv`, reactivity sweep
   {0 (baseline run), 0.1, 0.3, 1.0} × {astar, rvo, flow, flow+aci, flow+maxdt+}, held-out
   scenes[120:140], n=70/cell, collision_ends=True. SR/coll: flow+aci 61/26 -> 61/20 ->
   87/3 -> 87/3 — best or tied-best SR at every level; raw flow 49/46 -> 84/7; astar
   recovers SR when humans dodge but keeps colliding (16-17%); rvo stays conservative
   (64-67 SR). Reactive>0 is a distribution shift for both model and warmup, and ACI's
   online adaptation absorbs it (collision rate falls, disks track the dodging crowd).

## Futures conditioning (2026-07-18, in progress)
Motivation: the model planned against the *present* (peds' pos+vel) while the disks
constrain the *future* — the projection kept overriding it. `hm3d_condition()` now has a
futures mode: each of the K=4 nearest peds contributes its predictor's horizon x 2 forecast
(robot-relative, raw-flattened; cond_dim 130) instead of (pos, vel). Same embedding
side-door, zero architecture change. Checkpoints carry `cond_mode`
("state" | "futures-<Predictor>"); `hm3d_policies.FlowPolicy` reads it and builds the
matching predictor (`make_predictor()`); `hm3d_eval`/`sim.py` share the policy's predictor
instance for the ACI disks (mandatory for stateful predictors) and reset it per episode.
- Baselines: `hm3d_flow.pt` = state-cond 120sc/100k (user-trained);
  `hm3d_flow_30sc.pt` = old 30-scene snapshot.
- `hm3d_flow_cvk.pt` = ConstantVelocity-futures, matched 120/100k budget
  (`config/hm3d_cvk.py`) — TRAINING (mps). Note: CV futures add no *information* over
  pos+vel (deterministic function of them), only an inductive bias — the informative test
  is Social-LSTM.
- Social-LSTM: vendored verbatim in `vendor_social_lstm/` from quancore/social-lstm
  (**upstream has NO license file** — ok for prototyping, resolve before code release).
  `scripts/train_social_lstm.py` trains on crowd-only WaypointCrowd rollouts (8 obs +
  16 pred, their hyperparams; dims=[1,1] + neighborhood 2.0 = 4 m social box; one
  documented fix: their train.py grades output[t] vs pos[t] — known target-alignment bug —
  we grade vs pos[t+1], matching their own inference). `predictor.SocialLSTM` = stateful
  mean-rollout wrapper (reset() per episode; missing history back-extrapolated with
  current velocity) — TRAINING (cpu, 80 scenes x 30 epochs).
  GATE before conditioning flow on it: held-out ADE must beat ConstantVelocity
  (CV = 0.32 m over 16 steps; crowds are A*-driven, so CV is strong here).
- Then `config/hm3d_slstm.py` -> `hm3d_flow_slstm.pt`; A/B/C eval (state vs cvk vs slstm)
  on held-out scenes (exclude the 120 training scenes!), Falcon semantics, same seeds.

RESULTS (2026-07-19, scenes[120:140], n=70/cell, collision_ends=True, SR%/coll%):
| cond  | raw   | +aci      | +maxdt+  |
| state | 47/50 | 59/24     | 53/33    |
| cvk   | 44/47 | 47/27     | 49/31    |
| slstm | 44/44 | 50/11     | 53/16    |
Verdicts: (a) CV-futures = null-to-negative, as predicted (no information over pos+vel,
130-dim mostly-padding cond just dilutes; also slower). Keep as the ablation line
"futures conditioning helps only when the predictor adds information".
(b) Social-LSTM predictor gate: ADE 0.211 m vs CV 0.322 m (35% better), val NLL -7.0.
(c) SLSTM-futures trades some SR for large safety gains: at matched SR 53%, coll 16% vs
state's 33% (maxdt+); slstm+aci has the best safety overall (11% coll, closest 0.67 m)
vs state+aci's 24% at 59% SR. Same predictor feeds cond AND disks (shared instance).
n=70 -> SR diffs ~borderline (+-6pt), collision diffs solid. Note slstm rows are
wall-clock slow (LSTM rollout every policy step + disk update).
Open choice for Experiment 1: state+aci (max SR) vs slstm+aci/maxdt+ (max safety) as
the headline system — or report both as a Pareto pair.

## Abstract numbers (2026-07-19)
- X (adaptation delay): `adaptation_delay.py` -> `adaptation_delay.csv` (toy speed_mid
  shift, n=30, mean all-k curves, W=20 smoothing): max 98 steps -> max+ 65 steps =
  **34% reduction**; coverage dip also halves (0.065 -> 0.030).
- Y/Z (ACI vs static CP under shift): `hm3d_shift.py` -> `hm3d_shift.csv` (HM3D held-out,
  warm at human_speed 1.0 / deploy at 1.5, n=97-100). Tube coverage (target 0.90):
  **aci holds 0.912** under shift; **split-cp degrades to 0.810** (from 0.951 unshifted).
  max+ 0.895 at shift; maxdt+ 0.857-0.868 (slightly under at per-episode timescale —
  the guarantee is asymptotic; fine, but quote ACI for Y). Caveat: split-cp's *collision*
  rate under shift is lower (19% vs 29%) because its frozen 99.4%-quantile-of-150-points
  disks are huge -> over-conservative (SR 46% vs 56-59%) — so phrase Y/Z as COVERAGE,
  not collisions (the coverage guarantee is also what the theory actually states).
- Paper snippets in `paper/results_exp1.tex` (exp1 booktabs table + prose).
- ~~Abstract flag: Social-MP3D untested~~ resolved: full MP3D grid done (see MP3D section).

## MP3D (2026-07-19, in progress)
Setup found complete: `data/mp3d` maps pair with all 60 train + 11 val scenes; val is a
REAL held-out split (no scene-index bookkeeping). **Zero-shot transfer of the HM3D-trained
models works** (MP3D val, n=91, collision_ends): astar 67/32, rvo 65/2,
flow[state] 68/26, flow[state]+aci 74/13, **flow[slstm]+aci 76/7 — best SR on the board
AND near-expert collisions; the HM3D Pareto tension dissolves here** (MP3D floors are more
open). No native MP3D training needed -> the paper claim upgrades to zero-shot cross-
dataset transfer (all conditioning is robot-relative/local). `config/mp3d.py` staged
anyway if native training is ever wanted. Full MP3D grid DONE -> `mp3d_exp1.csv`
(merged; both models × raw+5 calibrators + classical, n=91/cell). Highlights (SR/coll):
astar 67/32, rvo 65/2; state: raw 68/26, aci=dtaci 74/13, max+ 75/13, maxdt+ 73/11;
slstm: raw 68/25, aci 76/7, **maxdt+ 76/4 — best cell of the project** (top SR + SPL
0.70, collisions ~at the expert's floor). The full contribution stack (slstm cond +
partial-max + dt tuning) compounds. MP3D table + prose appended to
`paper/results_exp1.tex`. Remaining optional: MP3D shift run (`hm3d_shift.py` needs the
same --ep-root treatment) if per-dataset Y/Z is wanted.

## Paper kit (2026-07-20) — everything needed to start writing

### Contributions (in claim order)
1. **Partial-Max nonconformity score**: max-over-horizon CP without the horizon-step
   feedback lag. Two mechanisms: pending predictions contribute running maxes to the
   quantile buffer, and the dial update is split into +gamma*alpha at birth /
   -gamma at first tube violation (penalties debt-queued, one per step). Measured: 34%
   faster post-shift recovery, half the coverage dip (`adaptation_delay.csv`). Verdict
   (2026-07-24): handles adaptation well; the only residual is a ~1pt transient
   under-coverage under short per-episode warmups (running maxes are lower bounds; toy
   sim with a long calibration stream sits ABOVE target), which is a good trade for
   smaller, plannable disks. NOT a lag problem — the lag is fixed.
2. **ACI-constrained flow-matching sampling**: calibrated keep-out disks enforced inside
   the Euler loop (CBF-style velocity brake + annealed projection; warm-start replanning;
   only the executed prefix `hard=replan_every=4` is constrained). Same enforcement slot
   as SafeDiffuser/SafeFlowMatcher, but the constraint is a statistically calibrated
   uncertainty set around a *predicted human*, not a fixed obstacle.
3. **Predictor-shared conditioning**: the flow model is conditioned on the same
   Social-LSTM forecasts the disks calibrate (one predictor instance feeds both) —
   halves collisions at matched SR. CV-futures null result is the mechanism ablation.
4. Secondary: DtACI is a knob-free BASELINE (dtaci ≡ aci everywhere), not part of our
   method — see angle change below.

### Angle change (2026-07-24): MaxDtACI dropped, DtACI demoted to baseline
Decision: our method is **ACI + Partial-Max** (fixed gamma). DtACI is now cited only as a
step-size-free *baseline*; MaxDtACI (partial + DtACI bank) moves to a one-sentence
future-work note. Why: partial-max and DtACI each work alone, but combined they interact
badly. Partial-max concentrates a regime change into up to H penalties on ONE dial (union
spreads them over H independent dials, one each); harmless at gamma=0.01 (~-0.16) but the
DtACI bank's large gamma (0.128) turns the same event into ~-2, cratering the dial negative
and pinning radii at the buffer max for ~30 steps (`disk_radius_trace.py` spikes to ~5 m).
The debt-queue fix (one -gamma/step, added to `PartialMaxACI`/`MaxDtACI` 2026-07-24) equalizes
partial-max's worst case with MaxACI's and is worth keeping, but it only spreads the maxdt+
plunge, it doesn't shrink it (total is preserved by design). Shared-event grading also removes
the self-correction DtACI's regret bound assumes. Net: maxdt+ is ≤ aci/max+ everywhere,
below target in the shift table, ugliest radius curve — cutting it removes a liability and
protects the weakest claim. Future-work sentence: "Partial-Max composes naturally with
DtACI's step-size bank, but a single tube-break delivers up to H correlated penalties to one
dial, which the bank's large step sizes amplify; a principled combination is future work."

Paper rescoping that follows (proposed, NOT yet applied to draft1.tex):
- Intro: "ACI", not "DtACI" (line ~127).
- Prelim: shrink DtACI subsection to ~3 sentences (cite Gibbs & Candes 2024 + note it is a
  baseline); drop the pinball-loss eq, gradient-step eq, and BOTH algorithm blocks from the
  main text (Alg. 2 optionally to appendix). `\ref{eq:dtaci}` will dangle — grep it.
- Deterministic-mean deviation becomes a footnote on the DtACI baseline row, not Alg. 2.
- Results: one sentence, "step-size-free DtACI matches fixed-gamma ACI, so no per-env tuning".
- Naming: label the gamma=0 row "rolling CP (fixed level)" / "online CP (no dial)", NOT bare
  "online CP" (that term is the whole adaptive family in the literature, incl. ACI/DtACI).
- Zaffran (AgACI) != DtACI (Gibbs & Candes) — don't let the prelim conflate them.

### Partial-Max guarantee argument (write this up properly; currently chat-only)
ACI's guarantee is a long-run average: telescoping the dial recursion gives
avg-miss = target + (alpha_0 - alpha_T)/(gamma*T), and boundedness of alpha_T (the dial
is self-correcting) sends the second term to 0 — no distributional assumptions.
Partial-Max changes only WHEN the dial moves, never the total: per prediction,
+gamma*alpha at birth plus -gamma iff the displayed tube is ever escaped sums to exactly
MaxACI's single delayed update, graded on the same operational event (escape from the
disks actually shown to the robot). Total dial displacement therefore matches MaxACI's up
to the <=horizon predictions in flight (bounded, O(gamma*horizon)), so the same
telescoping argument gives the same fixed point: displayed-tube miss rate -> alpha.
Caveats to state: asymptotic (not finite-sample/per-episode); transient disks lean small
(running maxes underestimate) — the prompt miss penalty is the correcting feedback.
MaxDtACI extra caveat: dial bank grades on the shared displayed event, so DtACI's regret
bound doesn't formally carry over (principled-but-heuristic; empirically ≡).

### Framing decisions (agreed in discussion, 2026-07-19/20)
- **Do NOT headline "zero-shot"**: say "transfers across scan datasets without
  retraining or recalibration — by design (egocentric, scene-agnostic conditioning; the
  ACI guarantee is distribution-free and transfers by theorem)". Measured dataset gap to
  cite: navigable area 58 vs 136 m^2/floor (2.3x), opt path 8.0 vs 10.3 m, humans/ep 3.1
  vs 4.6 (+48%, pressures the K=4 truncation), clearance 0.31 vs 0.36 m. Honest caveats:
  same Matterport-scan family, identical crowd generator (geometry+density shift, not
  behavior shift). `paper/results_exp1.tex` still says "zero-shot" — soften when pasting.
- **System positioning**: the FM model is a *local social maneuver generator* (16 steps
  = 1.6 s) inside a standard hierarchical stack — A* carrot does global routing, exactly
  like the benchmark's own agents. Related-work axes: Diffuser-lineage = monolithic
  full-horizon generation with scene-in-weights (doesn't transfer); NoMaD/ViNT-style
  hierarchy = closest precedent; guidance-based obstacle costs = soft analogue of our
  hard disks. Claim "certified-safe local social navigation", not global planning.
- **Y/Z phrasing**: coverage, never collision rate (split-CP is over-conservative on
  collisions under shift; and coverage is what the theorem states).
- Comparability: privileged state + holonomic robot -> compare only to our own
  A*/ORCA/RVO2 implementations, NOT Falcon's learned egocentric numbers (loose ranking
  vs their classical logs only; see fidelity section).

### Method numbers (single source of truth for the paper's setup section)
dt 0.1 s; horizon 16 (1.6 s plans); replan_every 4 (certified prefix); alpha 0.1;
disk margin 0.5 m (two 0.25 m body radii); collision <0.5 m center distance;
robot/human speed 1.0 m/s (toy peds 1.2); K=4 nearest peds, pad at 10 m rel;
cond_dim: toy 14, hm3d-state 18, futures 130; calibrator warmup 150 robot-free steps
(fresh per episode; humans robot-blind so replay is exact); window 100 (split-CP 1e6);
DtACI gamma grid 0.001..0.128, span 100; TemporalUnet dim 32, mults (1,2,4,8), ~4M
params; CFM: 10 Euler steps, sigma=0 straight-line flow; training 100k steps batch 32
grad-accum 2 lr 2e-4 EMA 0.995 (config/hm3d*.py); Social-LSTM: quancore verbatim,
rnn 128 / embed 64 / grid 4 / 4 m neighborhood, 8 obs + 16 pred, Adagrad, 30 epochs,
ADE 0.211 vs CV 0.322 (held-out); FlowPolicy: n_samples 4 + warm-start tau 0.6,
best-of-N by jerk + plan-consistency + 1e3/violation, infeasibility brake.

### Reproducibility map (table/figure -> source -> command)
- Tab exp1 (HM3D): `hm3d_exp1.csv` <- `python hm3d_exp1.py` (resumable; --models/--out
  for parallel halves). Tab MP3D: `mp3d_exp1.csv` <- same + `--ep-root=<mp3d val>
  --n-per-scene=10`. Prose+tabs drafted in `paper/results_exp1.tex`.
- Exp2 reactivity: `hm3d_exp2.csv` <- `python hm3d_exp2.py`.
- Shift Y/Z: `hm3d_shift.csv` <- `python hm3d_shift.py`. Delay X + fig curves:
  `adaptation_delay.csv` <- `python adaptation_delay.py`.
- Toy: eval.py table (6 rows), `compare_cp.py` (6 calibrators warmed identically),
  `shift_experiment.py` -> `shift_coverage.png` (2x2 fig), `sweep.csv`, `ablation.csv`,
  `deference.csv` (wired for all 5 cals; deference not re-run since 4-cal version).
- Stats noise: SR standard error ~±4 pts at n=140, ~±5 at n=91 — SR gaps <8 pts are
  suggestive, collision-rate gaps are the solid ones.
- Figures on hand: `shift_coverage.png`, `hm3d_demo_frame.png`, `sim_demo_frame.png`,
  toy CSVs for plots; `sim.py --hm3d` renders live demos for video/GIFs.
- **Paper figures (2026-07-24):** `paper/make_figs.py` -> `paper/figs/{adaptation,
  reactivity,shift_coverage,disk_radii}.{pdf,png}` from the CSVs above; run its producer
  first if a CSV is missing. New: `disk_radius_trace.py` -> `disk_radius_trace.csv` (fig 5,
  union vs max disk size, parked-robot warmup so streams aren't reset-poisoned). New:
  `sim.py --save-frame PATH --frame-step N` (+ `--hm3d`) saves the qualitative frames
  (`paper/figs/qualitative_{toy,hm3d}.png`). NOTE fig data predates the debt-queue fix for
  the max+/maxdt+ curves except disk_radii — re-run before final.
- **Fig 5 caveat:** maxdt+ radius curve has large conservative spikes (dial plunges on
  regime change); consider plotting only aci vs max+ (one axis: aggregation), mention maxdt+
  transients in text. Consistent with dropping maxdt+ to future-work.

### Citation checklist (from memory — verify before submitting)
Janner+ 22 Diffuser; Xiao+ SafeDiffuser; SafeFlowMatcher (Takahashi+); Lipman+ flow
matching; torchcfm/Tong+; Gibbs & Candes 21 (ACI) + 22 (DtACI/AgACI); Cleaveland+
max-over-horizon conformal prediction; Lindemann+/Dixit+ conformal prediction for
planners (MPC lineage); Alahi+ 16 Social-LSTM (+ quancore impl, NO LICENSE — resolve);
van den Berg+ ORCA/RVO2; Falcon benchmark paper (Social-HM3D/MP3D); HM3D + MP3D dataset
papers; NoMaD/ViNT (hierarchy precedent); Chi+ Diffusion Policy (receding-horizon
precedent); ETH/UCY for Social-LSTM's origins.

### Limitations paragraph fodder
Privileged ground-truth human states (no perception); holonomic point robot; 2D
projection of 3D scenes; constant-velocity + Social-LSTM predictors only (no learned
multimodal forecaster); guarantee is asymptotic long-run coverage (max+ reads ~0.885
tube at per-episode timescale on HM3D, ~1pt under target — the honest number to show;
union aci meets target with built-in slack); humans
robot-blind by benchmark design (exp2 sweeps reactivity); n=91-140 per cell; single
training seed per model; adaptation-delay metric definition (W=20 smoothing, TOL 0.01,
SUSTAIN 5) is ours — state it explicitly.

## TODO / tests to run
Open items, most-actionable first. (Done items are folded into the sections above.)

### Ablations still owed
- [ ] **Re-run all max+/maxdt+ result rows after the debt-queue fix (2026-07-24).** The
  penalty rate-limit changed `PartialMaxACI`/`MaxDtACI` output slightly. Affected:
  `hm3d_shift.csv`, `hm3d_exp2.csv` (max+/maxdt+ rows), any exp tables with max+/maxdt+.
  `disk_radius_trace.csv` already re-run.
- [ ] **DtACI / MaxDtACI: sampled-vs-mean output (step 2).** We deploy the *deterministic
  weighted mean* `alpha_t = sum_i p_i alpha_i` (`aci.py radii()`), NOT the paper's randomized
  expert-sampling (`alpha_t = alpha_i w.p. p_i`). Currently justified only by reasoning —
  Jensen: the mean's pinball (tracking) loss <= the expected sampled loss, and determinism
  gives reproducible disk sizes so the planner's path is stable across identical runs — but
  NOT measured. To test: add a `sample=False|True` switch to `DtACI`+`MaxDtACI`, run both
  through `compare_cp.py` (tube + per-disk coverage) and one HM3D cell; expect mean to match
  sampled on coverage and beat it on variance/task metrics. Outcome is a one-line footnote
  either way; cite the paper's *sampled* version for the formal regret/coverage guarantee
  regardless (the mean variant is ours and not covered by their theorem). Applies to both
  self-tuning calibrators.
- [ ] `deference.py`: re-run with all 5 calibrators (last run was the 4-cal version — wired,
  just not executed; ~2 h).
- [ ] (optional) MP3D distribution-shift run for per-dataset Y/Z (`hm3d_shift.py` with the
  MP3D `--ep-root`).

### Writing / release
- [ ] Write up the Partial-Max guarantee properly (argument sketched in Paper kit; currently
  chat-only).
- [ ] Soften "zero-shot" -> "transfers without retraining/recalibration" when pasting
  `paper/results_exp1.tex`.
- [ ] Verify the citation checklist (drafted from memory).
- [ ] **Social-LSTM license**: `vendor_social_lstm/` (quancore) has NO license file — resolve
  before any code release.

## Checks
Notation/consistency items to keep straight when writing the CBF-QP section.
- **Disk coordinate frame.** The centers `c_{k,n}` and radii `r_{k,n}` are in the model's
  *normalized robot-relative* coordinates, the same frame as the generated trajectory `tau`
  (`cfm.py` `conditional_sample`, disks arg). Don't mix them with world coordinates in the math.
- **Sampling-step index must not be `t`.** The FM prelim reserves `t` (flow time) and
  `sigma_t = 1-t`. Use `i` for the Euler sampling-step index in the CBF-QP math to avoid the
  clash (the code already uses `i`).

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
