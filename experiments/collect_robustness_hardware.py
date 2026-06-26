#!/usr/bin/env python3
"""
collect_robustness_hardware.py -- MEASURED real-environment robustness

Template for collecting robustness data on the physical UR5 rig. It records the dispersion
of D_s(t) and the classical D(t) under controlled real perturbations, so that the reported
SD values are ESTIMATED from repeated measured runs rather than assigned a priori.

It writes only measured values. With no robot/sensor backend connected it exits without
producing output, by design.

Perturbation protocol (per condition, repeated R times)
-------------------------------------------------------
  - sensor noise:   record at native gain; no synthetic noise injection
  - lighting:       block-randomised over {200,400,600,800} lux (real luminaire)
  - pose jitter:    object pose offset (x,y) ~ U[-30,30] mm, theta ~ U[-pi/4,pi/4]
  - vibration:      mechanical excitation profile P applied to the jaw

For each repeat the script reads the live force/depth/vibration tensors, builds the
membership fields with the SAME pipeline as dsf_analysis.py, computes D_s(t) and D(t),
and logs them. SD is then computed across the R repeats per condition.

Output (only from measured runs)
--------------------------------
  robustness_measured.csv   condition, repeat, Ds_mean, D_mean
  robustness_summary.csv    condition, n_repeats, SD_Ds, SD_D, CI_lo, CI_hi
"""
import argparse, os, sys

def fail(msg: str):
    sys.stderr.write("ERROR: " + msg + "\n")
    sys.stderr.write("This template logs only measured data; no values are synthesised.\n")
    sys.exit(2)

def connect_robot(backend: str):
    # Integration point: open the real ROS 2 / UR5 SDK session and sensor streams.
    fail(f"no hardware backend '{backend}' connected. Wire in the real ROS 2 / UR5 SDK "
         f"session and the force/depth/vibration sensor readers here. No simulated "
         f"fallback is provided, so this script cannot emit fabricated robustness numbers.")

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", required=True, help="hardware backend id (ros2/ur5)")
    ap.add_argument("--repeats", type=int, default=20,
                    help="measured repeats per condition (>=10 recommended for an SD)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    if args.repeats < 2:
        fail("an SD requires at least 2 measured repeats per condition.")
    session = connect_robot(args.backend)   # exits unless real hardware is wired in

    # Once connected, for each condition and repeat:
    #   tensors = session.read_sensors()                 # measured
    #   mu = build_membership(tensors)                   # same pipeline as dsf_analysis.py
    #   Ds = weighted_soft_box_count_dim(mu); D = classical_box_count_dim(mu)
    #   log(condition, repeat, Ds, D)
    # then SD_Ds, SD_D, and bootstrap CIs are computed across repeats and written.
    fail("reached the measurement loop with no hardware wired in; nothing to record.")

if __name__ == "__main__":
    main()
