"""Offline analysis of synthetic, dimensionless response observations only.
No CGM input, dose calculation, or actuator interface.
"""
import statistics
import math

FIELDS = ('insulin_index', 'glucagon_index')

def evaluate(rows):
    """Compare a recent median against an earlier median on held-out observations.

    Rows must be ordered chronologically. This is a descriptive benchmark,
    not an estimator of human hormone sensitivity.
    """
    if len(rows) < 12:
        return None
    rows = list(rows)[-12:]
    result = {}
    for field in FIELDS:
        vals = [float(row[field]) for row in rows]
        if not all(math.isfinite(v) and 0 < v <= 100 for v in vals):
            return None
        baseline = statistics.median(vals[:4])
        candidate = statistics.median(vals[4:8])
        holdout = vals[8:]
        old_error = statistics.mean(abs(v-baseline) for v in holdout)
        new_error = statistics.mean(abs(v-candidate) for v in holdout)
        result[field] = dict(baseline=baseline, candidate=candidate,
                             baseline_error=old_error, candidate_error=new_error,
                             improved=new_error < old_error)
    return result
