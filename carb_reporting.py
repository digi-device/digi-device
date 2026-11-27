"""Synthetic meal-announcement disturbance; no device interfaces."""
import math
import random

class CarbReporting:
    def __init__(self, factor=1.0, randomized=False, seed=1):
        self.factor = float(factor)
        if not math.isfinite(self.factor) or not 0 <= self.factor <= 1:
            raise ValueError('Factor must be between 0 and 1.')
        self.randomized = randomized
        self.rng = random.Random(seed)
        self.active = False
        self.current = self.factor
        self.history = []

    def report(self, actual):
        actual = float(actual)
        if not math.isfinite(actual) or actual < 0:
            raise ValueError('Invalid carbohydrate amount.')
        if actual > 0 and not self.active:
            self.current = self.rng.random() if self.randomized else self.factor
        self.active = actual > 0
        reported = actual * self.current if self.active else 0.0
        self.history.append((reported, self.current if self.active else None))
        return reported
