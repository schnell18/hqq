import random
from functools import partial
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.optimize import Bounds, LinearConstraint, OptimizeResult
from scipy.optimize import milp as scipy_milp
from scipy.stats.mstats import trimmed_mean, trimmed_std


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
        df = pd.read_csv(model_metric_fp)
        modules = df["module"].unique()
        boost_mods = {mod: boost_layers for mod in modules}
        decline_mods = {mod: decline_layers for mod in modules}
        return _allocate_boost_decline_configs(
            df,
            budget,
            boost_mods,
            decline_mods,
            boost_stop,
            decline_stop,
        )
    elif weight_algo == "sensi-boost" or weight_algo == "kurt-boost":
        top_m_layer = kwargs.get("top_m_layer", 1)
        ablation = kwargs.get("ablation", False)
        if weight_algo == "sensi-boost":
            src = "sensitivity"
            dif_method = "divide"
        else:
            src = "kurtosis"
            dif_method = "subtract"
        df = pd.read_csv(model_metric_fp)
        max_layer = df["layer"].unique().max()
        modules = df["module"].unique()
        boost_mods = {mod: boost_layers for mod in modules}
        if not ablation:
            module_outliers = identify_sensitive_modules(
                df,
                src,
                top_m=top_m_layer,
                diff_method=dif_method,
            )
        else:
            module_outliers = identify_sensitive_modules_ablation(
                df,
                max_layer,
                src,
                weight_algo,
                top_m=top_m_layer,
                diff_method=dif_method,
            )
        boost_mods.update(module_outliers)
        return _allocate_boost_decline_configs(df, budget, boost_mods, None, boost_stop)
    else:
        return _find_optimal_configs_milp(
            model_metric_fp,
            budget,
            time_limit,
            torelance_pct,
            verbose,
            **kwargs,
        )


def _allocate_boost_decline_configs(
    df,
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

    b1, g1 = budget_map[budget]
    b2, g2 = 8, 128
    boost_stop = 1 if not boost_stop else boost_stop
    decline_stop = 1 if not decline_stop else decline_stop
    b1_boost, g1_boost = boost_cfg(budget, boost_stop)
    b1_decline, g1_decline = boost_cfg(budget, decline_stop)
    cfgs = {}

    modules = df.groupby(["module"]).layer.nunique().to_dict()
    for module in modules:
        layers = modules[module]
        for layer in range(0, layers):
            if (
                boost_layers
                and module in boost_layers
                and layer in boost_layers[module]
            ):
                cfgs[f"{layer}.{module}"] = (b1_boost, g1_boost, b2, g2)
            elif (
                decline_layers
                and module in decline_layers
                and layer in decline_layers[module]
            ):
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
    elif weight_algo == "sensi-milp" or weight_algo == "kurt-milp":
        ablation = kwargs.get("ablation", False)
        top_m_layer = kwargs.get("top_m_layer", 1)
        if weight_algo == "sensi-milp":
            src = "sensitivity"
            diff_method = "divide"
        else:
            src = "kurtosis"
            diff_method = "subtract"
        factor = kwargs.get("factor", 2)
        func = gen_cost_factor_func(
            df,
            factor,
            src,
            "cost",
            top_m=top_m_layer,
            diff_method=diff_method,
            ablation=ablation,
        )
        df = df.apply(func, axis=1)
    else:
        df["cost"] = df["fnorm"]

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


def gen_cost_factor_func(
    df, factor, src, dest, top_m=1, diff_method="divide", ablation=False
):
    tot_params = df["params"].sum() / 12
    module_outliers = {}
    if not ablation:
        module_outliers = identify_sensitive_modules(
            df, src, top_m=top_m, diff_method=diff_method
        )

    def _set_cost_factor(row):
        b1, g1 = row["nbit1"], row["gsize1"]
        b2, g2 = row["nbit2"], row["gsize2"]
        p, mod, layer = row["params"], row["module"], row["layer"]

        bpp = b1 + 2 * b2 / g1 + 32 / g1 / g2
        cost_factor = (
            factor if mod in module_outliers and layer in module_outliers[mod] else 1
        )
        low_bit_penalty = 4 if b1 < 3 else 2 if b1 < 4 else 1
        row[dest] = low_bit_penalty * cost_factor / bpp * 100 * (p / tot_params)
        return row

    return _set_cost_factor


def identify_sensitive_modules_ablation(
    df,
    layers,
    src,
    weight_algo,
    top_m=1,
    diff_method="divide",
):
    module_outliers_sensi = identify_sensitive_modules(
        df, "sensitivity", top_m=top_m, diff_method="divide"
    )
    module_outliers_kurt = identify_sensitive_modules(
        df, "kurtosis", top_m=top_m, diff_method="subtract"
    )

    module_outliers = {}
    for module in module_outliers_sensi:
        ls = module_outliers_sensi.get(module, [])
        lk = module_outliers_kurt.get(module, [])
        top_layers = len(ls) if weight_algo == "sensi-boost" else len(lk)
        layer_outliers = list(set(ls + lk))
        if top_layers > 0:
            module_outliers[module] = random.sample(
                list(set(range(layers)) - set(layer_outliers)), top_layers
            )
        else:
            module_outliers[module] = []

    return module_outliers


def identify_sensitive_modules(df, src, top_m=1, diff_method="divide"):
    module_outliers = {}
    modules = df["module"].unique()
    for module in modules:
        # choose b8g64 as example since sensitivity metrics
        # are identical across bit-groups
        df_s = df.query(f"nbit1 == 4 and gsize1 == 64 and module == '{module}'")
        sensi = df_s[src].to_numpy()
        if diff_method == "divide":
            a = []
            for i in range(len(sensi) - 1):
                a.append(sensi[i + 1] / sensi[i])
            diff = np.array(a)
        else:
            diff = np.diff(sensi)
        # trimm 10% value to exclude outliers
        mu = trimmed_mean(diff, limits=(0.05, 0.05))
        sigma = trimmed_std(diff, limits=(0.05, 0.05), ddof=1)
        # use zscore to isolate outliers
        zscore = np.abs(diff - mu) / sigma
        outliers = sorted(
            zip(zscore[zscore > 3], np.where(zscore > 3)[0]), key=lambda ot: -ot[0]
        )
        # keep top n outliers
        if top_m != 0 and len(outliers) > top_m:
            outliers = outliers[0:top_m]
        module_outliers[module] = [ot[1] + 1 for ot in outliers]

    return module_outliers
