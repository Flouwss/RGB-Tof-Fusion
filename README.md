# RGB–ToF Fusion

Research code for RGB-guided time-of-flight depth completion, quantitative evaluation, and depth-based reconstruction.

The project intentionally excludes datasets and generated experiment outputs. Place a dataset outside the repository or in an ignored `data/` directory, then pass its path through each command's `--root` argument.

## Layout

- `src/rgb_tof_fusion/` — reusable implementation modules and command entry points.
- `scripts/` — small executable wrappers for the main workflows.
- `configs/calibration_parameters.json` — camera calibration used by VO and SLAM workflows.
- `docs/RGB_ToF_Depth_Fusion_Report.pdf` — project report.

## Dataset contract

The processing pipelines expect numeric PNG frame names and the following directories beneath `--root`:

```text
rgb_rect/        # rectified RGB images
tof_rect/        # rectified ToF depth images, uint16
amplitude_rect/  # ToF amplitude images, uint16
depthmaps/       # optional ground-truth depth maps, uint16
```

## Install and run

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
python scripts/run_weighted_median.py --root D:\path\to\dataset
```

Run `python scripts/<name>.py --help` to inspect the remaining parameters.
