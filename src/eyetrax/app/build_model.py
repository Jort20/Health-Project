"""CLI utility to calibrate and export an EyeTrax gaze estimation model.

This script initializes a GazeEstimator,
runs an blue-noise calibration session via webcam, and saves the
trained model to disk as a .pkl file.
"""

import argparse
from pathlib import Path

from eyetrax.calibration.adaptive import run_adaptive_calibration
from eyetrax.gaze import GazeEstimator


def _cli() -> argparse.Namespace:
    """Parse command-line arguments for model calibration and output settings.

    Returns:
        argparse.Namespace: Parsed CLI configuration arguments.
    """
    p = argparse.ArgumentParser("Build and save a calibrated gaze model")
    p.add_argument("--camera", type=int, default=0, help="Camera index")
    p.add_argument(
        "--random", type=int, default=60, help="Number of random blue-noise points"
    )
    p.add_argument(
        "--retrain-every", type=int, default=10, help="Retrain after this many points"
    )
    p.add_argument(
        "--show-pred",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Display live prediction during calibration",
    )
    p.add_argument("--outfile", required=True, help="Destination .pkl file")
    p.add_argument("--base", help="Optional: start from an existing model")
    p.add_argument("--model", default="ridge", help="Backend regression model")
    return p.parse_args()


def main() -> None:
    """Initialize the estimator, run adaptive calibration, and save the model."""
    args = _cli()

    # Initialize estimator and load pre-trained weights
    if args.base:
        print(f"[build_model] Loading base model from {args.base}")
        gaze = GazeEstimator(model_name=args.model)
        gaze.load_model(args.base)
    else:
        gaze = GazeEstimator(model_name=args.model)

    # Run interactive webcam calibration using blue-noise target points
    run_adaptive_calibration(
        gaze,
        num_random_points=args.random,
        retrain_every=args.retrain_every,
        show_predictions=args.show_pred,
        camera_index=args.camera,
    )

    # Ensure target directory exists before exporting the serialized model
    Path(args.outfile).parent.mkdir(parents=True, exist_ok=True)
    gaze.save_model(args.outfile)
    print(f"[build_model] Saved calibrated model → {args.outfile}")


if __name__ == "__main__":
    main()
