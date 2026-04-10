"""
Traffic Data Generator v2
Generates realistic synthetic traffic sensor data for 6 Delhi road segments.
Features a multi-modal time-pattern model with weather, incidents, and seasonality.
"""

import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
import random

ROAD_SEGMENTS = [
    {"id": "NH48_Delhi",   "name": "NH-48 Delhi-Gurgaon",     "lat": 28.535, "lng": 77.391, "cap": 3000},
    {"id": "Ring_Road",    "name": "Ring Road Delhi",          "lat": 28.614, "lng": 77.209, "cap": 2500},
    {"id": "Expressway_N", "name": "Delhi-Meerut Expressway",  "lat": 28.669, "lng": 77.454, "cap": 2800},
    {"id": "Airport_Rd",   "name": "Airport Express Road",     "lat": 28.556, "lng": 77.100, "cap": 2200},
    {"id": "Outer_Ring",   "name": "Outer Ring Road",          "lat": 28.630, "lng": 77.380, "cap": 3200},
    {"id": "NH9_Noida",    "name": "NH-9 Delhi-Noida",         "lat": 28.571, "lng": 77.322, "cap": 2600},
]


class TrafficDataGenerator:
    """
    Generates (X, y) pairs for CNN training.
    X shape: (N, 24, 10, 3)  — 24 time steps, 10 features, 3 channels
    y shape: (N,)            — congestion class 0-3
    """

    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)

    # ── Congestion ratio model ──────────────────────────────────────
    def _base_ratio(self, hour, dow, weather=1.0):
        """Returns a congestion ratio [0,1] based on time and weather."""
        wd = dow < 5  # weekday
        if wd:
            if 7 <= hour <= 10:    base = 0.82
            elif 17 <= hour <= 20: base = 0.88
            elif 12 <= hour <= 14: base = 0.55
            elif 22 <= hour or hour <= 5: base = 0.08
            else:                  base = 0.38
        else:
            if 11 <= hour <= 19:   base = 0.38
            elif 22 <= hour or hour <= 6: base = 0.06
            else:                  base = 0.22

        weather_penalty = (1 - weather) * 0.30
        noise = np.random.normal(0, 0.07)
        return float(np.clip(base + weather_penalty + noise, 0, 1))

    def _label(self, r):
        """Map ratio → 4-class label"""
        if r < 0.26: return 0
        if r < 0.54: return 1
        if r < 0.77: return 2
        return 3

    # ── Feature vector ──────────────────────────────────────────────
    def _features(self, hour, dow, weather=1.0):
        r = self._base_ratio(hour, dow, weather)
        max_speed = 120
        speed   = np.clip(max_speed * (1 - r * 0.88) + np.random.normal(0, 4), 5, max_speed)
        volume  = np.clip(55 * r * weather + np.random.normal(0, 2.5), 0, 60)
        occ     = np.clip(r * 0.96 + np.random.normal(0, 0.04), 0, 1)
        inc     = int(np.random.poisson(r * 2.5))
        wx      = np.clip(weather + np.random.normal(0, 0.04), 0, 1)
        t_sin   = np.sin(2 * np.pi * hour / 24)
        t_cos   = np.cos(2 * np.pi * hour / 24)
        cap_u   = np.clip(r * 1.12 + np.random.normal(0, 0.05), 0, 1)
        temp    = np.clip((28 + np.random.normal(0, 8) + 20) / 80, 0, 1)
        vis     = np.clip((1 - 0.35 * (1 - weather)) + np.random.normal(0, 0.04), 0, 1)
        return np.array([
            speed/120, volume/60, occ, min(inc,10)/10,
            wx, t_sin, t_cos, cap_u, temp, vis
        ], dtype=np.float32), r

    # ── Build (24,10,3) tensor ───────────────────────────────────────
    def _build_sample(self, start_hour, dow, weather):
        seq = []
        ratios = []
        for t in range(24):
            h = (start_hour + t) % 24
            f, r = self._features(h, dow, weather)
            seq.append(f)
            ratios.append(r)
        arr = np.array(seq, dtype=np.float32)  # (24,10)
        # 3 channels: raw, slightly noisy copy, inverted-speed emphasis
        ch2 = arr * 0.92 + np.random.normal(0, 0.015, arr.shape).astype(np.float32)
        ch3 = arr.copy()
        ch3[:, 0] = 1 - arr[:, 0]  # invert speed → emphasises slow = high congestion
        sample = np.stack([arr, ch2, ch3], axis=-1)  # (24,10,3)
        label  = self._label(ratios[-1])
        return sample, label

    # ── Public API ──────────────────────────────────────────────────
    def generate(self, n_samples=5000):
        """Generate n_samples (X,y) pairs."""
        X, y = [], []
        for _ in range(n_samples):
            dow     = random.randint(0, 6)
            hour    = random.randint(0, 23)
            weather = random.uniform(0.55, 1.0)
            s, lbl  = self._build_sample(hour, dow, weather)
            X.append(s)
            y.append(lbl)
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)
        print(f"Generated {n_samples} samples | shape: {X.shape} | classes: {np.bincount(y)}")
        return X, y

    def split(self, n_samples=5000, val=0.2):
        X, y = self.generate(n_samples)
        return train_test_split(X, y, test_size=val, random_state=42, stratify=y)

    def realtime_sample(self, hour=None, dow=None, weather=1.0):
        """Generate one real-time sample for live inference."""
        if hour is None: hour = datetime.now().hour
        if dow  is None: dow  = datetime.now().weekday()
        s, _ = self._build_sample(hour, dow, weather)
        return s

    def all_segments_realtime(self, weather=1.0):
        """Return realtime samples for all 6 segments."""
        now = datetime.now()
        out = []
        for seg in ROAD_SEGMENTS:
            w = weather * random.uniform(0.88, 1.0)
            sample = self.realtime_sample(now.hour, now.weekday(), w)
            out.append({"segment": seg, "sample": sample})
        return out
