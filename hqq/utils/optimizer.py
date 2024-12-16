from functools import partial
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.optimize import Bounds, LinearConstraint, OptimizeResult
from scipy.optimize import milp as scipy_milp


def find_optimal_configs(
    model_metric_fp: str,
    budget: float,
    time_limit=10,
    torelance_pct: int = 0,
    verbose: bool = False,
    **kwargs,
) -> Dict:
    weight_algo = kwargs.get("weight_algo", None)
    boost_layers = kwargs.get("boost_layers", None)
    decline_layers = kwargs.get("decline_layers", None)
    boost_layers = boost_layers if boost_layers else []
    decline_layers = decline_layers if decline_layers else []
    boost_stop = kwargs.get("boost_stop", 1)
    decline_stop = kwargs.get("decline_stop", -1)
    if weight_algo == "sensi-directive":
        cfgs = _allocate_boost_decline_configs(
            model_metric_fp,
            budget,
            boost_layers,
            decline_layers,
            boost_stop,
            decline_stop,
        )
        if cfgs:
            return cfgs

    return _find_optimal_configs_milp(
        model_metric_fp,
        budget,
        time_limit,
        torelance_pct,
        verbose,
        **kwargs,
    )


def _allocate_boost_decline_configs(
    fp: str,
    budget,
    boost_layers,
    decline_layers,
    boost_stop=1,
    decline_stop=-1,
):
    budget_map = {
        8.51: (8, 32),
        8.25: (8, 64),
        8.13: (8, 128),
        4.51: (4, 32),
        4.25: (4, 64),
        4.13: (4, 128),
        3.51: (3, 32),
        3.25: (3, 64),
        3.13: (3, 128),
        2.51: (2, 32),
        2.25: (2, 64),
        2.13: (2, 128),
    }
    if budget not in budget_map:
        return None

    def boost_cfg(budget, stop):
        sort_budgets = sorted(budget_map.keys())
        idx = sort_budgets.index(budget)
        idx = idx + stop
        if idx < 0:
            idx = 0
        elif idx >= len(sort_budgets):
            idx = len(sort_budgets)
        return budget_map[sort_budgets[idx]]

    df = pd.read_csv(fp)
    max_layer = df["layer"].unique().max()
    modules = df["module"].unique()
    b1, g1 = budget_map[budget]
    b2, g2 = 8, 128
    boost_stop = 1 if not boost_stop else boost_stop
    decline_stop = 1 if not decline_stop else decline_stop
    b1_boost, g1_boost = boost_cfg(budget, boost_stop)
    b1_decline, g1_decline = boost_cfg(budget, decline_stop)
    cfgs = {}
    for layer in range(0, max_layer + 1):
        for module in modules:
            if layer in boost_layers:
                cfgs[f"{layer}.{module}"] = (b1_boost, g1_boost, b2, g2)
            elif layer in decline_layers:
                cfgs[f"{layer}.{module}"] = (b1_decline, g1_decline, b2, g2)
            else:
                cfgs[f"{layer}.{module}"] = (b1, g1, b2, g2)
    return cfgs, 0


def _find_optimal_configs_milp(
    model_metric_fp: str,
    budget: float,
    time_limit=10,
    torelance_pct: int = 0,
    verbose: bool = False,
    **kwargs,
) -> Dict:
    total_params, costs, mems, row_mapper, column_mapper = load_precomputed_metrics(
        model_metric_fp, **kwargs
    )
    # convert bit per paramter to total mega bytes
    budget_mb = (100 + torelance_pct) / 100 * total_params * budget / 8 / 1024**2
    optimizer_opts = {"disp": verbose, "time_limit": time_limit}
    result = mip_solve(budget_mb, costs, mems, optimizer_opts)
    if result.success:
        assigments = torch.from_numpy(result.x.reshape(costs.shape)).nonzero().tolist()
        configs = {}
        for assigment in assigments:
            layer, module = row_mapper[assigment[0]]
            b1, g1, b2, g2 = column_mapper[assigment[1]]
            configs[f"{layer}.{module}"] = (b1, g1, b2, g2)
        return configs, result.fun
    else:
        raise ValueError(f"milp failed: {result.message}")


def load_precomputed_metrics(
    fp: str, **kwargs
) -> Tuple[int, np.ndarray, np.ndarray, pd.MultiIndex, pd.MultiIndex]:
    df = pd.read_csv(fp)
    weight_algo = kwargs.get("weight_algo", None)
    if weight_algo == "kurt-scaled":
        # apply local(module-scoped) kurtosis weight to fnorm cost
        df["cost"] = df["fnorm"] * (1 + df["kurtosis_scaled"])
    elif weight_algo == "head-prioritized":
        # apply additional weight to the start layers
        factor = kwargs.get("factor", 1.1)
        first_layer_prioritized = partial(prioritize, layer=0, factor=factor)
        df = df.apply(first_layer_prioritized, axis=1)
        df_fnorm = df.pivot_table(
            values="cost",
            index=["layer", "module"],
            columns=["nbit1", "gsize1", "nbit2", "gsize2"],
        )
    elif weight_algo == "tail-prioritized":
        # apply additional weight to the start layers
        factor = kwargs.get("factor", 1.1)
        last_layer = df["layer"].max()
        last_layer_prioritized = partial(prioritize, layer=last_layer, factor=factor)
        df = df.apply(last_layer_prioritized, axis=1)
    elif weight_algo == "sensi-milp":
        factor = kwargs.get("factor", 2)
        func = gen_cost_factor_func(df, factor, "sensitivity", "cost_sensi")
        df = df.apply(func, axis=1)
        df["cost"] = (
            df["cost_sensi"]
            / (
                df["nbit1"]
                + 2 * df["nbit2"] / df["gsize1"]
                + 32 / df["gsize1"] / df["gsize2"]
            )
            * 100
            * 12
            * df["params"]
            / df["params"].sum()
        )
    elif weight_algo == "kurtosis-milp":
        factor = kwargs.get("factor", 2)
        # min-max scaling
        df_kurt_agg = df.groupby("module").agg(
            kurt_max=pd.NamedAgg(column="kurtosis", aggfunc="max"),
            kurt_min=pd.NamedAgg(column="kurtosis", aggfunc="min"),
        )
        df = df.merge(df_kurt_agg, how="left", on="module")
        df["kurtosis_scaled"] = (df["kurtosis"] - df["kurt_min"]) / (
            df["kurt_max"] - df["kurt_min"]
        )
        # use zscore to isolate kurtosis outliers
        func = gen_cost_factor_func(df, factor, "kurtosis_scaled", "cost_kurt")
        df = df.apply(func, axis=1)
        df["cost"] = (
            df["cost_kurt"]
            / (
                df["nbit1"]
                + 2 * df["nbit2"] / df["gsize1"]
                + 32 / df["gsize1"] / df["gsize2"]
            )
            * 100
            * 12
            * df["params"]
            / df["params"].sum()
        )
    else:
        df_fnorm["cost"] = df_fnorm["fnorm"]

    df_fnorm = df.pivot_table(
        values="cost",
        index=["layer", "module"],
        columns=["nbit1", "gsize1", "nbit2", "gsize2"],
    )

    df_memgb = df.pivot_table(
        values="memmb",
        index=["layer", "module"],
        columns=["nbit1", "gsize1", "nbit2", "gsize2"],
    )
    df_params = df.pivot_table(
        values="params",
        index=["layer", "module"],
        columns=["nbit1", "gsize1", "nbit2", "gsize2"],
    )
    total_params = df_params[df_params.columns[0]].sum()
    return (
        total_params,
        df_fnorm.to_numpy(),
        df_memgb.to_numpy(),
        df_fnorm.index,
        df_fnorm.columns,
    )


def mip_solve(
    budget: float, costs: np.ndarray, mems: np.ndarray, opts: Optional[Dict]
) -> OptimizeResult:
    N = costs.shape[0]
    coefficients = costs.reshape(-1)
    A_memory_requirements = mems.reshape(1, -1)
    A_equality = np.zeros_like(mems, shape=(N,) + mems.shape)
    A_equality[np.arange(N), np.arange(N), :] = 1.0
    A_equality = A_equality.reshape(N, -1)
    constraints = [
        LinearConstraint(A=A_memory_requirements, lb=0, ub=budget),
        LinearConstraint(A=A_equality, lb=1, ub=1),
    ]
    return scipy_milp(
        c=coefficients,
        constraints=constraints,
        integrality=np.ones_like(coefficients),
        bounds=Bounds(0, 1),
        options=opts,
    )


def prioritize(row, layer=0, factor=1.1):
    row["cost"] = row["fnorm"] * (1 if row["layer"] != layer else factor)
    return row


def gen_cost_factor_func(df, factor, src, dest):
    # use zscore to isolate kurtosis outliers
    sigma = df[src].std()
    mu = df[src].mean()

    def _set_cost_factor(row):
        row[dest] = factor if abs(row[src] - mu) / sigma > 3 else 1
        return row

    return _set_cost_factor
