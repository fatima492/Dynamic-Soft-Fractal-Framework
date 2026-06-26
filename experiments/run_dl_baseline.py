#!/usr/bin/env python3
"""
run_dl_baseline.py  --  MEASURED deep-learning comparison harness

This script does NOT contain or generate any results. It is a template that runs a
real grasp model on REAL (or high-fidelity simulated) trial records and feeds the
measured per-trial binary outcomes into the same binomial GLMM used in the manuscript.

It deliberately refuses to run without:
  (1) a real model checkpoint, and
  (2) a real trial-manifest CSV describing the shared, paired trial set.

If either is missing it exits with an error rather than inventing numbers, so that no
output of this script can be mistaken for a measured result it did not produce.

Required inputs
---------------
--model {contact_graspnet,dexnet,graspnet1b}   which baseline to evaluate
--checkpoint PATH                              path to the real model weights
--trials PATH                                  CSV manifest of the shared trial set:
        columns: trial_id, domain, object_id, pose_x, pose_y, theta, approach_phi,
                 lux, scene_path
--method-under-test {DSF, ...}                 second method, evaluated on the SAME trials
--out PATH                                     where measured per-trial outcomes are written

Output (only ever written from measured evaluations)
----------------------------------------------------
  measured_trials.csv   trial_id, domain, method, success(0/1)
  glmm_dl.csv           odds ratios, cluster-robust 95% CI, p, for DSF vs each DL baseline

Pairing: every method is evaluated on the identical trial_id set so the GLMM can use a
trial-level random effect and within-trial pairing.
"""
import argparse, os, sys, csv

def fail(msg: str) -> "NoReturn":
    sys.stderr.write("ERROR: " + msg + "\n")
    sys.stderr.write("This harness does not fabricate results; provide real inputs.\n")
    sys.exit(2)

def load_model(kind: str, checkpoint: str):
    if not checkpoint or not os.path.exists(checkpoint):
        fail(f"model checkpoint not found: {checkpoint!r}. "
             f"Download the official weights for {kind} and pass --checkpoint.")
    # Intentionally not implemented here: import the real package and load weights, e.g.
    #   contact_graspnet -> from contact_graspnet import inference
    #   dexnet           -> from gqcnn import get_gqcnn_model
    # Loading must use the genuine library; no surrogate is provided on purpose.
    fail(f"loader for {kind} is a required integration point. Wire in the real "
         f"{kind} inference call here, then re-run. No surrogate model is shipped, "
         f"so that this script cannot emit a fake comparison.")

def read_trials(path: str):
    if not path or not os.path.exists(path):
        fail(f"trial manifest not found: {path!r}. Provide the shared, paired trial set.")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        fail("trial manifest is empty.")
    needed = {"trial_id", "domain", "scene_path"}
    if not needed.issubset(rows[0].keys()):
        fail(f"trial manifest missing required columns {needed}.")
    return rows

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    choices=["contact_graspnet", "dexnet", "graspnet1b"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--trials", required=True)
    ap.add_argument("--method-under-test", default="DSF")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    trials = read_trials(args.trials)          # real, shared trial set
    model = load_model(args.model, args.checkpoint)   # exits unless wired to real model

    # The lines below run ONLY once a real model and real scenes are supplied:
    #   for t in trials:
    #       scene = load_scene(t["scene_path"])         # real sensor capture / sim frame
    #       grasp = model.predict(scene)                # real inference
    #       success = execute_and_score(grasp, t)       # real or HIL execution, measured
    #       write_row(args.out, t["trial_id"], t["domain"], args.model, success)
    # then fit the GLMM (reuse dsf_analysis.fit_glmm) on the measured outcomes.
    fail("reached the measurement loop with no real model wired in. "
         "Integrate the real inference + execution calls above to produce measured "
         "outcomes; until then there is intentionally nothing to write.")

if __name__ == "__main__":
    main()
