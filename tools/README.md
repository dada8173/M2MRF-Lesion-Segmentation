Plotting utilities for mmseg experiments
=====================================

This folder contains a small plotting helper `tools/plot_experiment.py` that parses
MMSeg training logs and outputs PNGs showing training loss and validation metrics.
The LA-M2MRF report-generation scripts also default to writing figures under
`work_dirs/la_m2mrf/report_figures/` so generated artifacts stay out of the repo
root unless an explicit `--output-dir` is provided.

What it produces
- `plots/loss_iter.png` — training loss (epoch means) plotted against iteration numbers
- `plots/classification_report_iter.png` — validation summary metrics and per-class F1, plotted against iteration numbers

Quick start
-----------
1. Ensure `matplotlib` is installed in the active Python environment:

```bash
python -m pip install matplotlib
```

2. Run the plotting script pointing at an experiment folder (the folder that holds the `.log` and `.log.json` files):

```bash
python tools/plot_experiment.py /home/dachen/projects/M2MRF-Lesion-Segmentation/work_dirs/la_m2mrf/idrid_official_scratch_bs3_fp16
```

Files will be written to the `plots/` subfolder inside the experiment folder. Example outputs:

- `work_dirs/la_m2mrf/idrid_official_scratch_bs3_fp16/plots/loss_iter.png`
- `work_dirs/la_m2mrf/idrid_official_scratch_bs3_fp16/plots/classification_report_iter.png`

Notes and options
-----------------
- The script extracts iteration numbers from the plain text log (lines like `Iter [5000/40000] ...`) and uses those iteration values for the x-axis (5000, 10000, ...). This gives exact, interpretable x-axis ticks.
- The script also parses `.log.json` (if present) to collect per-epoch training loss values and maps each epoch to a representative iteration (median iteration for that epoch) so the loss curve can be shown against iteration as well.
- If you prefer to display epochs on the x-axis instead, you can either:
  - Use the epoch fields available in `.log.json` when plotting (the script already reads them); or
  - Modify the script to plot `epoch` instead of `iter` by replacing the x arrays with the epoch numbers.

Extending or packaging as a skill
--------------------------------
If you want to formalize this as a repeatable skill in this repo:

1. The repo-level memory already stores a short description at `/memories/repo/plot_experiment_skill.md`.
2. To make this an installable CLI-style tool you could add an entry in `setup.py` or create a tiny wrapper that installs required deps and exposes the command.

Contact
-------
If you want, I can:

- Add a short `tools/usage.md` with annotated example outputs; or
- Add flags to the script (e.g. `--xaxis iteration|epoch`, `--outdir`) and re-run it to regenerate figures.

Files
-----
- `tools/plot_experiment.py` — main script (already added)
- `tools/README.md` — this file
