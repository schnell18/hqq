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
) -> Dict:
    total_params, costs, mems, row_mapper, column_mapper = load_precomputed_metrics(
        model_metric_fp
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
    fp: str,
) -> Tuple[int, np.ndarray, np.ndarray, pd.MultiIndex, pd.MultiIndex]:
    df = pd.read_csv(fp)
    # apply global kurtosis weight to fnorm cost
    # df["weighted_fnorm"] = df["fnorm"] * df["kurtosis"] / 3
    # apply local(module-scoped) kurtosis weight to fnorm cost
    df["weighted_fnorm"] = df["fnorm"] * (1 + df["kurtosis_scaled"])
    df_fnorm = df.pivot_table(
        values="weighted_fnorm",
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
