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


## Modular research portal

Local synthetic data research only. 

Install the existing simulation dependencies as documented above, and install the
portal with `python -m pip install -e ".[portal]"`. Start `python main.py` and open
http://127.0.0.1:5000. Register with username and a password of at
least 12 characters. The database is created in `data/research.sqlite3`. The
portal is local only; sessions reset on restart unless DIGI_SECRET_KEY is set.

History accepts synthetic dimensionless response observations. Profile evaluation
compares the most recent 12 observations in three consecutive groups of four;
users can save a version after reviewing results. Saved profiles never change simulation control. 
Existing simulator CSV output can be imported for descriptive review using `python simulation_import.py PATH USERNAME`.
Run `python -m unittest discover -s tests -v` for backend tests.

### Changes in this upgrade
- `main.py`, `config.py`: local application and configuration.
- `users/`, `database/`: account authentication, data access and schema.
- `personalization/`: synthetic observations, notes, trend evaluation, profile versions.
- `interface/`: web routes, separate templates and CSS.
- `simulation_import.py`: authenticated offline CSV summary import.
- `tests/`: research backend tests.
- `pyproject.toml`, `.gitignore`, `run_sim.py`: packaging, local data exclusions,
  and removal of the default intentional meal-bolus stress test.
- Existing `sim/` and terminal UI retained unchanged.

## Browser-based virtual simulation and missed meal announcements

Restart `python main.py`, log in, and open Simulation. Select a virtual patient,
duration, seed and reporting mode. The browser waits for a completed simulation,
then displays glucose, insulin, glucagon and actual/reported carbohydrate graphs.
Each run is saved to the logged-in account. The original meal schedule is unchanged (see
`DEFAULT_MEALS`); random factors are drawn once per contiguous nonzero meal episode.
CLI examples:

```
python run_sim.py --mode 2 --carb-factor 0.5
python run_sim.py --mode 2 --random-carb-reporting --seed 42
```

Use the same patient, duration and seed for comparisons. The CSV includes
`CHO` (actual g/min), `CHO_reported` (announced g/min), and `carb_reporting_factor`.
CLI output filenames are reused unless you specify `--output-dir`.
The database creates the new graph table automatically; do not delete existing data.
