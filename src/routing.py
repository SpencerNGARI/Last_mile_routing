"""Reusable routing utilities for the Amazon Last Mile Routing project.

Conventions
-----------
* Every route is solved and scored as a CLOSED tour: station -> all stops -> station.
* Times come from the dataset: transit times are historical averages between stops,
  service times are planned per package and summed per stop.
* Time windows and departure times are UTC. A stop's window is the intersection of
  its packages' windows (latest start, earliest end). Packages without a window
  (NaN) add no constraint.
"""

import json
import math
from pathlib import Path

import ijson
import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

MODES = ("none", "hard", "soft")


class Data:
    """Holds the three small files in memory; streams the large travel-time file."""

    def __init__(self, raw_dir):
        self.raw = Path(raw_dir)
        with open(self.raw / "route_data.json") as f:
            self.routes = json.load(f)
        with open(self.raw / "package_data.json") as f:
            self.packages = json.load(f)
        with open(self.raw / "actual_sequences.json") as f:
            self.sequences = json.load(f)

    def travel_times(self, route_ids=None):
        """Yield (route_id, matrix) for the requested routes (all if None),
        reading travel_times.json as a stream. Stops early once all are found."""
        wanted = set(route_ids) if route_ids is not None else None
        found = 0
        with open(self.raw / "travel_times.json", "rb") as f:
            for rid, tt in ijson.kvitems(f, "", use_float=True):
                if wanted is None or rid in wanted:
                    yield rid, tt
                    found += 1
                    if wanted is not None and found == len(wanted):
                        return


def get_tt(data, rid):
    """Travel-time matrix (dict of dicts, seconds) for one route."""
    for _, tt in data.travel_times([rid]):
        return tt
    raise KeyError(rid)


def observed_order(data, rid):
    seq = data.sequences[rid]["actual"]
    return sorted(seq, key=seq.get)


def tour_cost(tt, order):
    """Closed-tour travel time in seconds (includes the return to order[0])."""
    legs = sum(tt[a][b] for a, b in zip(order[:-1], order[1:]))
    return legs + tt[order[-1]][order[0]]

def stop_service_s(data, rid):
    return {s: sum(p["planned_service_time_seconds"] for p in pkgs.values())
            for s, pkgs in data.packages[rid].items()}


def _has_window(p):
    return isinstance(p["time_window"]["start_time_utc"], str)


def stop_windows(data, rid):
    """{stop_id: [(start, end), ...]} for stops with at least one windowed package."""
    out = {}
    for s, pkgs in data.packages[rid].items():
        ws = [(pd.Timestamp(p["time_window"]["start_time_utc"]),
               pd.Timestamp(p["time_window"]["end_time_utc"]))
              for p in pkgs.values() if _has_window(p)]
        if ws:
            out[s] = ws
    return out


def depart_time(data, rid):
    r = data.routes[rid]
    return pd.Timestamp(f"{r['date_YYYY_MM_DD']} {r['departure_time_utc']}")


def window_bounds(data, rid):
    """Per-stop (lo, hi) in seconds since departure, plus a count of stops whose
    package windows do not overlap (those stops get no window in the solver)."""
    depart, out, conflicts = depart_time(data, rid), {}, 0
    for s, ws in stop_windows(data, rid).items():
        lo = max(max((a - depart).total_seconds() for a, b in ws), 0)
        hi = min((b - depart).total_seconds() for a, b in ws)
        if hi < lo:
            conflicts += 1
            continue
        out[s] = (math.ceil(lo), int(hi))
    return out, conflicts


def check_feasibility(data, rid, tt, order):
    """Independent timeline check of any sequence (order[0] must be the station).

    Arriving before the latest window start means waiting; arriving after a
    window's end counts as a violation. Returns duration (closed tour, including
    service and waiting), waiting, and window-violation counts.
    """
    depart = depart_time(data, rid)
    svc, win = stop_service_s(data, rid), stop_windows(data, rid)
    t, wait_s, late_s = depart, 0.0, 0.0
    n_viol, stops_viol = 0, 0
    prev = order[0]
    for cur in order[1:]:
        t += pd.Timedelta(seconds=tt[prev][cur])
        if cur in win:
            latest_start = max(s for s, e in win[cur])
            if t < latest_start:
                wait_s += (latest_start - t).total_seconds()
                t = latest_start
            bad = [(s, e) for s, e in win[cur] if t > e]
            if bad:
                stops_viol += 1
                n_viol += len(bad)
                late_s += max((t - e).total_seconds() for s, e in bad)
        t += pd.Timedelta(seconds=svc.get(cur, 0))
        prev = cur
    t += pd.Timedelta(seconds=tt[prev][order[0]])
    return {"duration_min": (t - depart).total_seconds() / 60,
            "wait_min": wait_s / 60,
            "stops_violated": stops_viol,
            "windows_violated": n_viol,
            "total_late_min": late_s / 60}

def solve_route(tt, stop_ids, depot_id, bounds=None, service=None, mode="none",
                warm_order=None, time_limit_s=8, horizon=86400, penalty=10):
    """Minimise closed-tour travel time with OR-Tools.

    mode "none": ignore time windows.
    mode "hard": arrival at each windowed stop must fall inside its window.
    mode "soft": lateness allowed but costs `penalty` objective seconds per second late.
    warm_order: a starting sequence (e.g. the observed one). If it is infeasible under
    the model the solver starts from scratch and warm_ok is False.
    Returns (order or None, warm_ok).
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    bounds, service = bounds or {}, service or {}
    idx = {s: i for i, s in enumerate(stop_ids)}
    travel = [[int(round(tt[a][b])) for b in stop_ids] for a in stop_ids]
    svc = [int(round(service.get(s, 0))) for s in stop_ids]

    manager = pywrapcp.RoutingIndexManager(len(stop_ids), 1, idx[depot_id])
    routing = pywrapcp.RoutingModel(manager)
    arc = routing.RegisterTransitCallback(
        lambda i, j: travel[manager.IndexToNode(i)][manager.IndexToNode(j)])
    routing.SetArcCostEvaluatorOfAllVehicles(arc)

    time_cb = routing.RegisterTransitCallback(
        lambda i, j: travel[manager.IndexToNode(i)][manager.IndexToNode(j)]
        + svc[manager.IndexToNode(i)])
    routing.AddDimension(time_cb, horizon, horizon, True, "Time")
    tdim = routing.GetDimensionOrDie("Time")
    if mode != "none":
        for s, (lo, hi) in bounds.items():
            node = manager.NodeToIndex(idx[s])
            if mode == "hard":
                tdim.CumulVar(node).SetRange(lo, hi)
            else:
                tdim.CumulVar(node).SetRange(lo, horizon)
                tdim.SetCumulVarSoftUpperBound(node, hi, penalty)

    p = pywrapcp.DefaultRoutingSearchParameters()
    p.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    p.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    p.time_limit.seconds = time_limit_s

    sol, warm_ok = None, False
    if warm_order:
        routing.CloseModelWithParameters(p)
        init = routing.ReadAssignmentFromRoutes([[idx[s] for s in warm_order[1:]]], True)
        if init:
            warm_ok = True
            sol = routing.SolveFromAssignmentWithParameters(init, p)
    if sol is None:
        sol = routing.SolveWithParameters(p)
    if sol is None:
        return None, warm_ok

    order, i = [], routing.Start(0)
    while not routing.IsEnd(i):
        order.append(stop_ids[manager.IndexToNode(i)])
        i = sol.Value(routing.NextVar(i))
    return order, warm_ok


def valid(order, obs, depot):
    """A proposed sequence must start at the station and visit exactly the same stops."""
    return order is not None and order[0] == depot and sorted(order) == sorted(obs)


def evaluate(data, rid, tt, mode="hard", time_limit_s=8, return_order=False):
    """Solve one route from its observed sequence and score the result against it."""
    obs = observed_order(data, rid)
    depot = obs[0]
    if mode == "none":
        bounds, conflicts = {}, 0
    else:
        bounds, conflicts = window_bounds(data, rid)
    order, warm_ok = solve_route(tt, obs, depot, bounds, stop_service_s(data, rid),
                                 mode, warm_order=obs, time_limit_s=time_limit_s)
    if not valid(order, obs, depot):
        return {"route_id": rid, "mode": mode, "solved": False, "warm_started": warm_ok}
    co, cs = tour_cost(tt, obs), tour_cost(tt, order)
    fo = check_feasibility(data, rid, tt, obs)
    fs = check_feasibility(data, rid, tt, order)
    res = {"route_id": rid, "mode": mode, "solved": True, "warm_started": warm_ok,
           "window_conflicts": conflicts,
           "obs_travel_min": co / 60, "opt_travel_min": cs / 60,
           "travel_saving_min": (co - cs) / 60,
           "travel_saving_pct": 100 * (co - cs) / co,
           "duration_saving_min": fo["duration_min"] - fs["duration_min"],
           "obs_stops_violated": fo["stops_violated"],
           "opt_stops_violated": fs["stops_violated"],
           "obs_late_min": fo["total_late_min"],
           "opt_late_min": fs["total_late_min"],
           "opt_wait_min": fs["wait_min"]}
    if return_order:
        res["opt_order"] = order
    return res
