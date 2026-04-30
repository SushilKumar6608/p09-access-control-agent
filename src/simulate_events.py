"""
simulate_events.py
P09 — Access Control Anomaly Detection Agent

Synthetic access control event stream generator.
Models badge/door access events as a marked Temporal Point Process:
  each event = (timestamp, user_id, door_id, event_type, success)

Normal behaviour is learned from configurable building schedules.
Anomalies are injected with controlled probability and type.
"""

import numpy as np
import pandas as pd
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import IntEnum
import random
from datetime import datetime, timedelta


# ── Event type vocabulary ─────────────────────────────────────────────────────
# Kept as IntEnum so the TPP mark space is a clean integer range [0, N_TYPES)

class EventType(IntEnum):
    ACCESS_GRANTED     = 0   # Normal: badge accepted, door opens
    ACCESS_DENIED      = 1   # Normal: wrong zone/time, door stays closed
    DOOR_FORCED        = 2   # Anomaly: door opened without badge
    DOOR_HELD_OPEN     = 3   # Anomaly: door held open beyond threshold
    BADGE_TIMEOUT      = 4   # Normal: badge presented, no entry within 10s
    MULTI_BADGE        = 5   # Anomaly: badge used twice in <5s (tailgating)
    ACCESS_AFTER_HOURS = 6   # Anomaly: access outside allowed schedule
    FAILED_ATTEMPT     = 7   # Normal/Anomaly: wrong PIN or unrecognised badge
    SYSTEM_ALARM       = 8   # Anomaly: triggered by consecutive failures
    DOOR_OPENED        = 9   # Normal: door physically opened after grant
    DOOR_CLOSED        = 10  # Normal: door closed and latched

N_EVENT_TYPES = len(EventType)


# ── Building configuration ────────────────────────────────────────────────────

@dataclass
class Door:
    door_id: str
    zone: str                          # e.g. "lobby", "server_room", "office"
    allowed_roles: List[str]           # roles permitted access
    allowed_hours: Tuple[int, int]     # (start_hour, end_hour) in 24h
    avg_daily_cycles: int              # expected number of access cycles per day


@dataclass
class User:
    user_id: str
    role: str                          # "employee", "contractor", "admin", "security"
    usual_arrival: int                 # hour of day, e.g. 8
    usual_departure: int               # hour of day, e.g. 17
    usual_doors: List[str]             # doors this user normally accesses


@dataclass
class BuildingConfig:
    doors: List[Door]
    users: List[User]
    workday_start: int = 7
    workday_end: int   = 19
    anomaly_rate: float = 0.05         # fraction of cycles that are anomalous


# ── Default building: Axis-style office building ──────────────────────────────

def default_building() -> BuildingConfig:
    doors = [
        Door("D01_LOBBY",       "lobby",       ["employee","contractor","admin","security"], (6, 22),  200),
        Door("D02_OFFICE_A",    "office",      ["employee","admin","security"],              (7, 19),  150),
        Door("D03_OFFICE_B",    "office",      ["employee","admin","security"],              (7, 19),  120),
        Door("D04_SERVER_ROOM", "server_room", ["admin","security"],                         (0, 24),   30),
        Door("D05_LAB",         "lab",         ["employee","admin"],                         (7, 20),   60),
        Door("D06_PARKING",     "parking",     ["employee","contractor","admin","security"], (5, 23),  180),
        Door("D07_ROOF",        "roof",        ["security","admin"],                         (8, 18),   10),
        Door("D08_CANTEEN",     "canteen",     ["employee","contractor","admin","security"], (7, 15),   90),
    ]

    users = []
    # Employees
    for i in range(1, 41):
        users.append(User(
            user_id=f"EMP{i:03d}",
            role="employee",
            usual_arrival=random.randint(7, 9),
            usual_departure=random.randint(16, 18),
            usual_doors=["D01_LOBBY", "D06_PARKING",
                         random.choice(["D02_OFFICE_A", "D03_OFFICE_B"]),
                         "D08_CANTEEN"]
        ))
    # Admins
    for i in range(1, 6):
        users.append(User(
            user_id=f"ADM{i:03d}",
            role="admin",
            usual_arrival=8,
            usual_departure=17,
            usual_doors=["D01_LOBBY","D02_OFFICE_A","D03_OFFICE_B",
                         "D04_SERVER_ROOM","D05_LAB","D06_PARKING","D08_CANTEEN"]
        ))
    # Security
    for i in range(1, 4):
        users.append(User(
            user_id=f"SEC{i:03d}",
            role="security",
            usual_arrival=0,
            usual_departure=24,
            usual_doors=[d.door_id for d in doors]
        ))
    # Contractors
    for i in range(1, 11):
        users.append(User(
            user_id=f"CON{i:03d}",
            role="contractor",
            usual_arrival=9,
            usual_departure=16,
            usual_doors=["D01_LOBBY", "D05_LAB"]
        ))

    return BuildingConfig(doors=doors, users=users)


# ── Inter-arrival time sampling ───────────────────────────────────────────────
# Within a door cycle (badge → open → close), inter-event times follow
# a Weibull-like distribution — matching the TPP training assumption.

def sample_inter_arrival(mean_seconds: float, shape: float = 2.0) -> float:
    """Sample inter-arrival time from Weibull distribution.
    shape=2 gives a unimodal distribution with realistic door cycle timing.
    Minimum 0.5s enforced (jitter floor, analogous to thesis 10ms floor scaled up).
    """
    scale = mean_seconds / (math.gamma(1 + 1/shape) if shape > 0 else mean_seconds)
    t = np.random.weibull(shape) * scale
    return max(0.5, t)


def sample_cycle_start_time(user: User, date: datetime,
                             building: BuildingConfig) -> Optional[float]:
    """Sample the wall-clock time of a user's next access attempt.
    Returns seconds since midnight, or None if no activity today.
    """
    # Arrival/departure with Gaussian jitter (±30 min)
    arrival   = user.usual_arrival   * 3600 + np.random.normal(0, 1800)
    departure = user.usual_departure * 3600 + np.random.normal(0, 1800)
    arrival   = max(0, min(arrival,   86400))
    departure = max(0, min(departure, 86400))

    if arrival >= departure:
        return None

    # Sample uniformly between arrival and departure
    return np.random.uniform(arrival, departure)


# ── Normal cycle generator ────────────────────────────────────────────────────

def generate_normal_cycle(user: User, door: Door,
                           base_timestamp: float) -> List[Dict]:
    """Generate one normal access cycle: badge → grant → open → close.
    Returns list of event dicts with timestamps in seconds.
    """
    events = []
    t = base_timestamp

    # Is the user allowed at this door?
    role_ok = user.role in door.allowed_roles
    hour    = int((t % 86400) / 3600)
    time_ok = door.allowed_hours[0] <= hour < door.allowed_hours[1]

    if role_ok and time_ok:
        # ACCESS_GRANTED → DOOR_OPENED → DOOR_CLOSED
        events.append(_event(t, user, door, EventType.ACCESS_GRANTED, success=True))
        t += sample_inter_arrival(2.0)   # ~2s: badge read → door opens
        events.append(_event(t, user, door, EventType.DOOR_OPENED, success=True))
        t += sample_inter_arrival(8.0)   # ~8s: person walks through
        events.append(_event(t, user, door, EventType.DOOR_CLOSED, success=True))
    else:
        # ACCESS_DENIED
        events.append(_event(t, user, door, EventType.ACCESS_DENIED, success=False))
        # Occasionally a second attempt
        if random.random() < 0.3:
            t += sample_inter_arrival(5.0)
            events.append(_event(t, user, door, EventType.FAILED_ATTEMPT, success=False))

    return events


# ── Anomaly cycle generators ──────────────────────────────────────────────────

def generate_brute_force(user: User, door: Door,
                          base_timestamp: float) -> List[Dict]:
    """Multiple rapid failed attempts followed by success or alarm."""
    events = []
    t = base_timestamp
    n_fails = random.randint(3, 8)

    for _ in range(n_fails):
        events.append(_event(t, user, door, EventType.FAILED_ATTEMPT, success=False))
        t += sample_inter_arrival(3.0, shape=5.0)   # rapid, tight timing

    if random.random() < 0.4:
        # Eventual success — suspicious
        events.append(_event(t, user, door, EventType.ACCESS_GRANTED, success=True))
        t += sample_inter_arrival(2.0)
        events.append(_event(t, user, door, EventType.DOOR_OPENED, success=True))
        t += sample_inter_arrival(8.0)
        events.append(_event(t, user, door, EventType.DOOR_CLOSED, success=True))
    else:
        # Alarm triggered
        events.append(_event(t, user, door, EventType.SYSTEM_ALARM, success=False))

    return events


def generate_after_hours(user: User, door: Door,
                          base_timestamp: float) -> List[Dict]:
    """Access attempt at unusual hour — 1am to 4am."""
    t = base_timestamp
    # Override to after-hours time
    day_start = (t // 86400) * 86400
    t = day_start + random.randint(1, 4) * 3600 + random.randint(0, 3599)

    events = [_event(t, user, door, EventType.ACCESS_AFTER_HOURS, success=True)]
    t += sample_inter_arrival(2.0)
    events.append(_event(t, user, door, EventType.DOOR_OPENED, success=True))
    t += sample_inter_arrival(300.0, shape=1.5)   # long dwell — suspicious
    events.append(_event(t, user, door, EventType.DOOR_CLOSED, success=True))
    return events


def generate_tailgating(user: User, door: Door,
                         base_timestamp: float) -> List[Dict]:
    """Badge used twice within 5 seconds — classic tailgating pattern."""
    t = base_timestamp
    events = []
    events.append(_event(t, user, door, EventType.ACCESS_GRANTED, success=True))
    t += sample_inter_arrival(2.0)
    events.append(_event(t, user, door, EventType.DOOR_OPENED, success=True))
    # Second badge swipe within 5s
    t += random.uniform(1.0, 4.5)
    events.append(_event(t, user, door, EventType.MULTI_BADGE, success=False))
    t += sample_inter_arrival(8.0)
    events.append(_event(t, user, door, EventType.DOOR_CLOSED, success=True))
    return events


def generate_door_forced(user: User, door: Door,
                          base_timestamp: float) -> List[Dict]:
    """Door opened without any badge event."""
    t = base_timestamp
    return [
        _event(t, user, door, EventType.DOOR_FORCED, success=False),
        _event(t + sample_inter_arrival(15.0, shape=1.2),
               user, door, EventType.SYSTEM_ALARM, success=False),
    ]


def generate_privilege_escalation(user: User, door: Door,
                                   base_timestamp: float,
                                   restricted_door: Door) -> List[Dict]:
    """Low-privilege user accessing a restricted zone."""
    t = base_timestamp
    events = [_event(t, user, restricted_door,
                     EventType.ACCESS_GRANTED, success=True)]
    t += sample_inter_arrival(2.0)
    events.append(_event(t, user, restricted_door, EventType.DOOR_OPENED, success=True))
    t += sample_inter_arrival(120.0, shape=1.5)
    events.append(_event(t, user, restricted_door, EventType.DOOR_CLOSED, success=True))
    return events


# ── Helper ────────────────────────────────────────────────────────────────────

def _event(timestamp: float, user: User, door: Door,
           event_type: EventType, success: bool) -> Dict:
    return {
        "timestamp":  timestamp,
        "user_id":    user.user_id,
        "user_role":  user.role,
        "door_id":    door.door_id,
        "zone":       door.zone,
        "event_type": int(event_type),
        "event_name": event_type.name,
        "success":    success,
        "is_anomaly": event_type in {
            EventType.DOOR_FORCED,
            EventType.DOOR_HELD_OPEN,
            EventType.MULTI_BADGE,
            EventType.ACCESS_AFTER_HOURS,
            EventType.SYSTEM_ALARM,
        },
    }


# ── Main stream generator ─────────────────────────────────────────────────────

def generate_event_stream(
    n_days: int = 30,
    start_date: Optional[datetime] = None,
    building: Optional[BuildingConfig] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a full synthetic access control event stream.

    Args:
        n_days:     Number of days to simulate.
        start_date: Simulation start date (default: today).
        building:   Building configuration (default: default_building()).
        seed:       Random seed for reproducibility.

    Returns:
        DataFrame sorted by timestamp with columns:
        timestamp, user_id, user_role, door_id, zone,
        event_type, event_name, success, is_anomaly
    """
    np.random.seed(seed)
    random.seed(seed)

    if start_date is None:
        start_date = datetime(2025, 1, 1)
    if building is None:
        building = default_building()

    door_map = {d.door_id: d for d in building.doors}
    restricted = [d for d in building.doors if d.zone == "server_room"]

    all_events: List[Dict] = []

    for day_idx in range(n_days):
        date = start_date + timedelta(days=day_idx)
        day_start_sec = day_idx * 86400   # seconds from epoch start

        # Each user generates several access cycles per day
        for user in building.users:
            # How many cycles today?
            n_cycles = max(1, int(np.random.poisson(3)))

            for _ in range(n_cycles):
                # Pick a door the user normally visits
                if user.usual_doors:
                    door_id = random.choice(user.usual_doors)
                else:
                    door_id = random.choice([d.door_id for d in building.doors])

                door = door_map[door_id]

                # Sample cycle start time
                t_sec = sample_cycle_start_time(user, date, building)
                if t_sec is None:
                    continue
                base_ts = day_start_sec + t_sec

                # Inject anomaly?
                is_anomaly_cycle = random.random() < building.anomaly_rate

                if is_anomaly_cycle:
                    anomaly_type = random.choices(
                        ["brute_force", "after_hours", "tailgating",
                         "door_forced", "privilege_escalation"],
                        weights=[0.25, 0.25, 0.20, 0.15, 0.15]
                    )[0]

                    if anomaly_type == "brute_force":
                        events = generate_brute_force(user, door, base_ts)
                    elif anomaly_type == "after_hours":
                        events = generate_after_hours(user, door, base_ts)
                    elif anomaly_type == "tailgating":
                        events = generate_tailgating(user, door, base_ts)
                    elif anomaly_type == "door_forced":
                        events = generate_door_forced(user, door, base_ts)
                    elif anomaly_type == "privilege_escalation" and restricted:
                        r_door = random.choice(restricted)
                        events = generate_privilege_escalation(
                            user, door, base_ts, r_door)
                    else:
                        events = generate_normal_cycle(user, door, base_ts)
                else:
                    events = generate_normal_cycle(user, door, base_ts)

                all_events.extend(events)

    df = pd.DataFrame(all_events)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Add human-readable timestamp column
    df["datetime"] = pd.to_datetime(
        start_date) + pd.to_timedelta(df["timestamp"], unit="s")

    return df


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating 30-day synthetic access control event stream...")
    df = generate_event_stream(n_days=30, seed=42)

    print(f"\nTotal events      : {len(df):,}")
    print(f"Unique users      : {df['user_id'].nunique()}")
    print(f"Unique doors      : {df['door_id'].nunique()}")
    print(f"Anomalous events  : {df['is_anomaly'].sum():,} "
          f"({df['is_anomaly'].mean()*100:.1f}%)")
    print(f"\nEvent type distribution:")
    print(df['event_name'].value_counts().to_string())
    print(f"\nDate range: {df['datetime'].min()} → {df['datetime'].max()}")
    print(f"\nSample (first 5 rows):")
    print(df[['datetime','user_id','door_id','event_name','success','is_anomaly']].head())

    # Save to data/processed for use in notebooks
    import os
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/events_30d.csv", index=False)
    print("\nSaved to data/processed/events_30d.csv")