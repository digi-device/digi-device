# digi-device

## Setup

Run these from the root

```bash
conda create -n digi-device python=3.10 -y
```

```bash
conda activate digi-device
```

```bash
conda install -c conda-forge numpy "scipy<1.15" pandas matplotlib -y
```

Install the simulator, then this project:

```bash
pip install -e ./sim
pip install -e .
```

Extras are defined under `[project.optional-dependencies]` in `pyproject.toml`. Add them in brackets:

```bash
pip install -e ".[dev]"            # pytest
pip install -e ".[kaggle]"         # kagglehub + pandas datasets
pip install -e ".[dev,kaggle]"     # both
```

## Run a simulation

```bash
python run_sim.py
```

Results (a PNG graph and a CSV) are saved to `results/`. Add `--show` to also open the graph at the end.
