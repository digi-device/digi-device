"""Run the bundled virtual simulator in an isolated subprocess."""
import csv
import json
import math
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATIENTS = tuple(f'{group}#{i:03d}' for group in ('adolescent', 'adult', 'child') for i in range(1, 11))


def run_experiment(patient, hours, seed, factor, randomized):
    if patient not in PATIENTS or hours not in (12, 24, 48):
        raise ValueError('Choose an available virtual patient and duration.')
    if not 0 <= seed <= 2147483647 or not math.isfinite(factor) or not 0 <= factor <= 1:
        raise ValueError('Invalid seed or carbohydrate reporting factor.')
    with tempfile.TemporaryDirectory(prefix='digi-research-') as tmp:
        cmd = [sys.executable, str(ROOT / 'run_sim.py'), '--mode', '2',
               '--patient', patient, '--hours', str(hours), '--seed', str(seed),
               '--carb-factor', str(factor), '--output-dir', tmp]
        if randomized:
            cmd.append('--random-carb-reporting')
        try:
            subprocess.run(cmd, cwd=ROOT, env=dict(os.environ, MPLBACKEND='Agg'),
                           capture_output=True, text=True, check=True, timeout=180)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError('The simulation timed out.') from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError('Simulation failed: ' + (exc.stderr or exc.stdout or 'Unknown error')[-1500:]) from exc
        stem = patient.replace('#', '_') + '_dual'
        path = Path(tmp)
        with (path / (stem + '.csv')).open(newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        if len(rows) < 2:
            raise RuntimeError('Simulation returned insufficient data.')
        dt = (datetime.fromisoformat(rows[1]['Time']) - datetime.fromisoformat(rows[0]['Time'])).total_seconds()/60
        bg = [float(r['BG']) for r in rows]
        if not all(math.isfinite(x) for x in bg) or dt <= 0:
            raise RuntimeError('Invalid simulation results.')
        actual = sum(float(r['CHO'] or 0) * dt for r in rows[:-1])
        reported = sum(float(r['CHO_reported'] or 0) * dt for r in rows[:-1])
        summary = dict(patient=patient, hours=hours, simulated_hours=(len(rows)-1)*dt/60,
                       seed=seed, factor=None if randomized else factor, randomized=randomized,
                       mean=sum(bg)/len(bg), minimum=min(bg), maximum=max(bg),
                       TIR=100*sum(70 <= x <= 180 for x in bg)/len(bg),
                       TBR=100*sum(x < 70 for x in bg)/len(bg),
                       TAR=100*sum(x > 180 for x in bg)/len(bg),
                       actual_carbs=actual, reported_carbs=reported,
                       unreported_carbs=actual-reported,
                       glucagon_ug=sum(float(r['glucagon'] or 0)*dt for r in rows[:-1]))
        image = (path / (stem + '.png')).read_bytes()
        return summary, image


def save_experiment(conn, uid, summary, image):
    label = f"{summary['patient']} — {summary['hours']} h"
    cur = conn.execute('INSERT INTO simulation_runs(user_id,label,summary) VALUES (?,?,?)',
                       (uid,label,json.dumps(summary,allow_nan=False)))
    conn.execute('INSERT INTO simulation_graphs(run_id,png) VALUES (?,?)', (cur.lastrowid,image))
    return cur.lastrowid
