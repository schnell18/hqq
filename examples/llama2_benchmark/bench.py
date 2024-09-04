import copy
import gc
import glob
import logging
import os

# from adapter.awq import create_awq_model
# from adapter.awq import quantize_awq_model
from datetime import datetime

import pandas as pd
import torch
import transformers
from adapter.autoawq import create_autoawq_model, quantize_autoawq_model
from adapter.autogptq import create_autogptq_model, quantize_autogptq_model
from adapter.hqq import create_hqq_model, quantize_hqq_model
from auto_gptq import BaseQuantizeConfig as GPTQQuantConfig
from eval_model import eval_c4, eval_wikitext2
from hqq.core.quantize import BaseQuantizeConfig as HQQQuantConfig
from leaderboard import eval_llm_leaderboard
from transformers import AutoModelForCausalLM

ALL_MODELS = [
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Meta-Llama-3-8B",
]

QUANT_METRICS_FILE_MAP = {
    "meta-llama/Llama-2-7b-hf": "data/fnorm-Llama-2-7b-hf.csv",
    "meta-llama/Llama-2-13b-hf": "data/fnorm-Llama-2-13b-hf.csv",
    "meta-llama/Meta-Llama-3-8B": "data/fnorm-Llama-3-8B.csv",
}

HHQ_CONFIGS = [
    ("b4g32", HQQQuantConfig(nbits=4, group_size=32)),
    ("b4g64", HQQQuantConfig(nbits=4, group_size=64)),
    ("b4g128", HQQQuantConfig(nbits=4, group_size=128)),
    ("b3g32", HQQQuantConfig(nbits=3, group_size=32)),
    ("b3g64", HQQQuantConfig(nbits=3, group_size=64)),
    ("b3g128", HQQQuantConfig(nbits=3, group_size=128)),
    ("mxq-3_00", HQQQuantConfig(mixed=True, budget=3.00, quant_scale=True)),
    ("mxq-4_01", HQQQuantConfig(mixed=True, budget=4.01, quant_scale=True)),
    ("mxq-3_76", HQQQuantConfig(mixed=True, budget=3.76, quant_scale=True)),
    ("mxq-3_50", HQQQuantConfig(mixed=True, budget=3.50, quant_scale=True)),
    ("mxq-2_75", HQQQuantConfig(mixed=True, budget=2.75, quant_scale=True)),
    ("mxq-2_48", HQQQuantConfig(mixed=True, budget=2.48, quant_scale=True)),
    ("mxq-4_25", HQQQuantConfig(mixed=True, budget=4.25, quant_scale=True)),
    ("mxq-4_50", HQQQuantConfig(mixed=True, budget=4.50, quant_scale=True)),
    ("mxq-4_75", HQQQuantConfig(mixed=True, budget=4.75, quant_scale=True)),
    ("mxq-5_00", HQQQuantConfig(mixed=True, budget=5.00, quant_scale=True)),
    ("b2g16", HQQQuantConfig(nbits=2, group_size=16)),
    ("b2g32", HQQQuantConfig(nbits=2, group_size=32)),
    ("b2g64", HQQQuantConfig(nbits=2, group_size=64)),
    ("mxq-3_00", HQQQuantConfig(mixed=True, budget=3.00, quant_scale=True)),
]

AUTOAWQ_CONFIGS = [
    ("b4g32", {"w_bit": 4, "q_group_size": 32, "zero_point": True, "version": "GEMM"}),
    ("b4g64", {"w_bit": 4, "q_group_size": 64, "zero_point": True, "version": "GEMM"}),
    (
        "b4g128",
        {"w_bit": 4, "q_group_size": 128, "zero_point": True, "version": "GEMM"},
    ),
    # 3-bit not supported by AutoAWQ right now
    # ("b3g64", {"w_bit": 3, "q_group_size": 64, "zero_point": True, 'version':'gemv_fast'}),
    # ("b3g128", {"w_bit": 3, "q_group_size": 128, "zero_point": True, 'version':'gemv_fast'}),
]

AWQ_CONFIGS = [
    ("b4g32", {"w_bit": 4, "q_group_size": 32, "zero_point": True}),
    ("b4g64", {"w_bit": 4, "q_group_size": 64, "zero_point": True}),
    ("b4g128", {"w_bit": 4, "q_group_size": 128, "zero_point": True}),
    ("b3g32", {"w_bit": 3, "q_group_size": 32, "zero_point": True}),
    ("b3g64", {"w_bit": 3, "q_group_size": 64, "zero_point": True}),
    ("b3g128", {"w_bit": 3, "q_group_size": 128, "zero_point": True}),
]

GPTQ_CONFIGS = [
    (
        "b4g32",
        GPTQQuantConfig(bits=4, group_size=32, damp_percent=0.01, desc_act=False),
    ),
    (
        "b4g64",
        GPTQQuantConfig(bits=4, group_size=64, damp_percent=0.01, desc_act=False),
    ),
    (
        "b4g128",
        GPTQQuantConfig(bits=4, group_size=128, damp_percent=0.01, desc_act=False),
    ),
    (
        "b3g32",
        GPTQQuantConfig(bits=3, group_size=32, damp_percent=0.01, desc_act=False),
    ),
    (
        "b3g64",
        GPTQQuantConfig(bits=3, group_size=64, damp_percent=0.01, desc_act=False),
    ),
    (
        "b3g128",
        GPTQQuantConfig(bits=3, group_size=128, damp_percent=0.01, desc_act=False),
    ),
]


def experiment_debug():
    models = [
        "meta-llama/Llama-2-7b-hf",
    ]
    tasks = {
        "HQQ": {
            "create_fn": create_hqq_model,
            "quantize_fn": quantize_hqq_model,
            "configs": HHQ_CONFIGS[-1:],
        },
    }
    do_expermient("debug_hqq_auto", models, tasks, save_dir="snapshots-hqq")


def experiment_quant_awq():
    models = ALL_MODELS
    tasks = {
        "awq": {
            "create_fn": create_autoawq_model,
            "quantize_fn": quantize_autoawq_model,
            "configs": AWQ_CONFIGS,
        },
    }
    do_expermient(
        "quant_awq",
        models,
        tasks,
        quantize_only=True,
    )


def experiment_redo_autogptq_benchmark():
    models = ALL_MODELS
    tasks = {
        "gptq": {
            "create_fn": create_autogptq_model,
            "quantize_fn": quantize_autogptq_model,
            "configs": GPTQ_CONFIGS,
        },
    }
    do_expermient(
        "eval_redo_autogptq_benchmark",
        models,
        tasks,
    )


def experiment_quant_autogptq():
    models = ALL_MODELS
    tasks = {
        "gptq": {
            "create_fn": create_autogptq_model,
            "quantize_fn": quantize_autogptq_model,
            "configs": GPTQ_CONFIGS[3:],
        },
    }
    do_expermient(
        "eval_autogptq_redo_b3",
        models,
        tasks,
    )


def experiment_eval_autogptq():
    models = ALL_MODELS
    tasks = {
        "gptq": {
            "create_fn": create_autogptq_model,
            "quantize_fn": quantize_autogptq_model,
            "configs": [GPTQ_CONFIGS[1]],
        },
    }
    do_expermient(
        "eval_autogptq",
        models,
        tasks,
    )


# def experiment_eval_autoawq_g32():
#     models = ALL_MODELS[2:]
#     tasks = {
#         'AWQ': {
#             "create_fn": create_autoawq_model,
#             "quantize_fn": quantize_autoawq_model,
#             "configs": AUTOAWQ_CONFIGS[0:1],
#         },
#     }
#     do_expermient(
#         "eval_autoawq_g32",
#         models,
#         tasks,
#     )
#


def experiment_eval_g32():
    models = ALL_MODELS
    tasks = {
        "HQQ": {
            "create_fn": create_hqq_model,
            "quantize_fn": quantize_hqq_model,
            "configs": [HHQ_CONFIGS[0], HHQ_CONFIGS[3]],
        },
        "AWQ": {
            "create_fn": create_autoawq_model,
            "quantize_fn": quantize_autoawq_model,
            "configs": AUTOAWQ_CONFIGS[0:1],
        },
        # 'GPTQ': {
        #     "create_fn": create_autogptq_model,
        #     "quantize_fn": quantize_autogptq_model,
        #     "configs": [GPTQ_CONFIGS[0], GPTQ_CONFIGS[3]],
        # },
    }
    do_expermient(
        "eval_all_g32",
        models,
        tasks,
    )


def experiment_eval_mix():
    models = ALL_MODELS
    tasks = {
        "HQQ": {
            "create_fn": create_hqq_model,
            "quantize_fn": quantize_hqq_model,
            "configs": HHQ_CONFIGS[-4:],
        },
    }
    do_expermient("eval_hqq_mix3", models, tasks)


def experiment_quant_mxq_boost():
    models = ALL_MODELS
    tasks = {
        "HQQ": {
            "create_fn": create_hqq_model,
            "quantize_fn": quantize_hqq_model,
            "configs": HHQ_CONFIGS[-3:],
        },
    }
    do_expermient(
        "quant_mxq_boost",
        models,
        tasks,
        quantize_only=True,
    )


def experiment_eval_mxq_boost():
    models = ALL_MODELS
    tasks = {
        "HQQ": {
            "create_fn": create_hqq_model,
            "quantize_fn": quantize_hqq_model,
            "configs": HHQ_CONFIGS[-3:],
        },
    }
    do_expermient(
        "eval_mxq_boost",
        models,
        tasks,
    )


def experiment_quantize_mxq_extra():
    models = ALL_MODELS
    tasks = {
        "HQQ": {
            "create_fn": create_hqq_model,
            "quantize_fn": quantize_hqq_model,
            "configs": HHQ_CONFIGS[-4:],
        },
    }
    do_expermient(
        "quant_mxq_extra",
        models,
        tasks,
        quantize_only=True,
    )


def calc_bits(b1, g1, b2, g2):
    return b1 + 2 * b2 / g1 + 32 / g1 / g2


def experiment_quant_eval_mxq_comprise():
    models = ALL_MODELS
    equiv_mxq_configs = []
    nbits = [4.06, 4.10, 4.15, 4.19, 4.24, 4.28, 4.33]
    for bits in nbits:
        cfg_name = f"mxq-{str(bits).replace('.', '_')}"
        equiv_mxq_configs.append(
            (cfg_name, HQQQuantConfig(mixed=True, budget=bits, quant_scale=True))
        )
    tasks = {
        "HQQ": {
            "create_fn": create_hqq_model,
            "quantize_fn": quantize_hqq_model,
            "configs": equiv_mxq_configs,
        },
    }
    do_expermient(
        "eval_mxq_compromise",
        models,
        tasks,
    )


# It seems the PPL degrades as we apply mem 1% memory torelance
# def experiment_quant_eval_mxq_torelance():
#     models = ALL_MODELS[0:1]
#     equiv_mxq_configs = [
#         ("mxq-2_25", HQQQuantConfig(mixed=True, budget=2.25, quant_scale=True))
#     ]
#     tasks = {
#         'HQQ': {
#             "create_fn": create_hqq_model,
#             "quantize_fn": quantize_hqq_model,
#             "configs": equiv_mxq_configs,
#         },
#     }
#     do_expermient(
#         "eval_mxq_torelance",
#         models,
#         tasks,
#     )


def experiment_quant_eval_mxq_equiv():
    models = ALL_MODELS
    equiv_mxq_configs = []
    for cfg in HHQ_CONFIGS:
        if cfg[0].startswith("b"):
            bits = calc_bits(
                cfg[1]["weight_quant_params"]["nbits"],
                cfg[1]["weight_quant_params"]["group_size"],
                8,
                128,
            )
            bits = round(bits, 2)
            cfg_name = f"mxq-{str(bits).replace('.', '_')}"
            equiv_mxq_configs.append(
                (cfg_name, HQQQuantConfig(mixed=True, budget=bits, quant_scale=True))
            )

    tasks = {
        "HQQ": {
            "create_fn": create_hqq_model,
            "quantize_fn": quantize_hqq_model,
            "configs": equiv_mxq_configs,
        },
    }
    do_expermient(
        "eval_mxq_extra",
        models,
        tasks,
    )


def experiment_eval_mxq_extra():
    models = ALL_MODELS
    tasks = {
        "HQQ": {
            "create_fn": create_hqq_model,
            "quantize_fn": quantize_hqq_model,
            "configs": HHQ_CONFIGS[-4:],
        },
    }
    do_expermient(
        "eval_mxq_extra",
        models,
        tasks,
    )


def experiment_quantize_405B():
    models = [
        "meta-llama/Meta-Llama-3.1-405B-Instruct",
    ]

    tasks = {
        "HQQ": {
            "create_fn": create_hqq_model,
            "quantize_fn": quantize_hqq_model,
            "configs": HHQ_CONFIGS[1:2],
        },
    }
    do_expermient(
        "quant_hqq_405B",
        models,
        tasks,
        quantize_only=True,
        save_dir="/data/gqq-eval/snapshots/",
    )


def experiment_quantize_mxq():
    models = ALL_MODELS
    tasks = {
        "HQQ": {
            "create_fn": create_hqq_model,
            "quantize_fn": quantize_hqq_model,
            "configs": HHQ_CONFIGS[6:],
        },
    }
    do_expermient(
        "quant_mxq_fix_storage_error",
        models,
        tasks,
        quantize_only=True,
    )


def experiment_eval_mxq():
    models = ALL_MODELS
    tasks = {
        "HQQ": {
            "create_fn": create_hqq_model,
            "quantize_fn": quantize_hqq_model,
            "configs": HHQ_CONFIGS[6:],
        },
    }
    do_expermient(
        "eval_mxq_fix_storage_error",
        models,
        tasks,
    )


def experiment_llm_leaderboard_autogptq():
    models = ALL_MODELS[:-1]
    tasks = {
        "GPTQ": {
            "create_fn": create_autogptq_model,
            "quantize_fn": quantize_autogptq_model,
            "configs": GPTQ_CONFIGS,
        },
    }
    do_expermient(
        "gptq_leaderboard",
        models,
        tasks,
        task_type="eval_leaderboard",
        save_dir="snapshots",
    )


# def experiment_autoawq():
#     models = ALL_MODELS
#     tasks = {
#         'AWQ': {
#             "create_fn": create_autoawq_model,
#             "quantize_fn": quantize_autoawq_model,
#             "configs": AUTOAWQ_CONFIGS,
#         }
#     }
#     do_expermient(
#         "awq_benchmark",
#         models,
#         tasks,
#         save_dir="snapshots"
#     )
#


def experiment_hqq():
    models = ALL_MODELS
    tasks = {
        "HQQ": {
            "create_fn": create_hqq_model,
            "quantize_fn": quantize_hqq_model,
            "configs": HHQ_CONFIGS,
        },
    }
    do_expermient("hqq_benchmark", models, tasks, save_dir="snapshots")


def experiment_hqq_mix():
    models = ALL_MODELS
    tasks = {
        "HQQ": {
            "create_fn": create_hqq_model,
            "quantize_fn": quantize_hqq_model,
            "configs": HHQ_CONFIGS[4:],
        },
    }
    do_expermient("hqq_benchmark_mix2", models, tasks, save_dir="snapshots")


def experiment_fp16_baseline():
    models = ALL_MODELS
    tasks = {
        "FP16": {
            "create_fn": create_fp16_model,
            "quantize_fn": None,
            "configs": [
                ("base", {}),
            ],
        },
    }
    do_expermient(
        "fp16_baseline",
        models,
        tasks,
    )


def _init_metrics(model_id, kind, config):
    return {
        "model": model_id.split("/")[1],
        "method": kind,
        "config": config[0],
        "config_detail": config[1],
        "quant_duration": 0,
        "quant_mem_allot": 0,
        "quant_mem_reserved": 0,
        "fp_mem_allot": 0,
        "fp_mem_reserved": 0,
        "quant_duration": 0,
        "ppl_wikitext": 0,
        "ppl_c4": 0,
        "duration_wikitext": 0,
        "duration_c4": 0,
        "ifeval": 0,
        "bbh": 0,
        "mathlevel5": 0,
        "gpqa": 0,
        "musr": 0,
        "mmlupro": 0,
    }


def do_expermient(
    experiment_name, models, tasks, task_type="quantize_only", save_dir="snapshots"
):
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    exp_result_name = experiment_name
    for kind, spec in tasks.items():
        exp_result_name += "-" + kind
        configs = spec["configs"]
        for config in configs:
            exp_result_name += "_" + config[0]
            for model_id in models:
                metric = _init_metrics(model_id, kind, config)
                print("*" * 72)
                if task_type == "quantize_only":
                    print(f"Quantizing {kind} on {model_id} w/ config: {config[0]}...")
                elif task_type == "eval_ppl":
                    print(
                        f"Evaluating {kind} PPL on {model_id} w/ config: {config[0]}..."
                    )
                else:
                    print(
                        f"Evaluating {kind} LLM Leaderboard benchmarks on {model_id} w/ config: {config[0]}..."
                    )
                print("*" * 72)

                if task_type != "eval_leaderboard":
                    create_fn = spec["create_fn"]
                    quant_fn = spec["quantize_fn"]
                    model, tokenizer, quantized = create_fn(
                        model_id, config[1], config[0], quant_fn is not None, save_dir
                    )
                    if quantized:
                        metric["quant_mem_allot"], metric["quant_mem_reserved"] = (
                            get_memory_metrics()
                        )
                    else:
                        metric["fp_mem_allot"], metric["fp_mem_reserved"] = (
                            get_memory_metrics()
                        )

                    if not quantized and quant_fn:
                        metric["fp_mem_allot"], metric["fp_mem_reserved"] = (
                            get_memory_metrics()
                        )
                        # avoid interventions between models
                        quant_config = copy.deepcopy(config[1])
                        if (
                            config[0].startswith("mxq-")
                            and model_id in QUANT_METRICS_FILE_MAP
                        ):
                            quant_config["quant_metrics_file"] = QUANT_METRICS_FILE_MAP[
                                model_id
                            ]
                        model, duration = quant_fn(
                            model,
                            tokenizer,
                            quant_config,
                            model_id,
                            config[0],
                            save_dir,
                        )
                        # persistent the quantized model
                        os.sync()
                        metric["quant_mem_allot"], metric["quant_mem_reserved"] = (
                            get_memory_metrics()
                        )
                        metric["quant_duration"] = duration
                    # Evaluate the quantized model
                    if task_type == "eval_ppl":
                        metric = eval_ppls(model, tokenizer, metric)
                    cleanup(model)
                else:
                    metric = eval_llm_leaderboard(
                        experiment_name, model_id, kind, config[0], save_dir, metric
                    )
                save_partial_metric(experiment_name, kind, model_id, config[0], metric)

    # combine metrics
    combine_metrics(experiment_name, exp_result_name)


def save_partial_metric(experiment_name, kind, model_id, config, metric):
    metrics = [metric]
    df = pd.DataFrame(metrics)
    result_dir = f"results/{experiment_name}"
    os.makedirs(result_dir, exist_ok=True)
    model_short_id = model_id.split("/")[1]
    file_name = f"{result_dir}/partial-{kind}-{model_short_id}-{config}.csv"
    df.to_csv(
        file_name,
        columns=[
            "method",
            "model",
            "config",
            "quant_duration",
            "ppl_wikitext",
            "ppl_c4",
            "duration_wikitext",
            "duration_c4",
            "quant_mem_allot",
            "quant_mem_reserved",
            "fp_mem_allot",
            "fp_mem_reserved",
            "config_detail",
            "ifeval",
            "bbh",
            "mathlevel5",
            "gpqa",
            "musr",
            "mmlupro",
        ],
        index=False,
    )


def combine_metrics(experiment_name, exp_result_name):
    dfs = []
    iters = glob.iglob(f"./results/{experiment_name}/partial-*.csv")
    for it in iters:
        df = pd.read_csv(it)
        dfs.append(df)
    combined = pd.concat(dfs)
    ts_str = datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = f"results/result-{exp_result_name}-{ts_str}.xlsx"
    combined.to_excel(file_name, index=False)


def eval_ppls(model, tokenizer, metric):
    ppl_wikitext, duration_wikitext = eval_wikitext2(model, tokenizer, verbose=True)
    ppl_c4, duration_c4 = eval_c4(model, tokenizer, verbose=True)
    metric["ppl_wikitext"] = ppl_wikitext
    metric["duration_wikitext"] = duration_wikitext
    metric["duration_c4"] = duration_c4
    return metric


def create_fp16_model(model_id, quant_config, config_id, load_quantized, save_dir):
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.float16
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer, False


def cleanup(model):
    del model
    torch.cuda.empty_cache()
    gc.collect()


def get_memory_metrics():
    return torch.cuda.memory_allocated(), torch.cuda.memory_reserved()


def main():
    # experiment_eval_all()
    # experiment_quantize_all()
    # experiment_debug()
    # experiment_autoawq()
    # experiment_autogptq()
    # experiment_hqq()
    # experiment_debug()
    # experiment_fp16_baseline()
    # experiment_quantize_mix()
    # experiment_eval_mix()
    # experiment_eval_g32()
    # experiment_eval_autoawq_g32()
    # experiment_eval_autogptq()
    # experiment_quant_autogptq()
    # experiment_quant_awq()
    # experiment_quantize_mxq()
    # experiment_eval_mxq()
    # experiment_quantize_mxq_extra()
    # experiment_eval_mxq_extra()
    # experiment_quant_eval_mxq_equiv()
    # experiment_quant_eval_mxq_torelance()
    # experiment_quant_eval_mxq_comprise()
    # experiment_quant_autogptq()
    # experiment_redo_autogptq_benchmark()
    # experiment_quantize_405B()
    experiment_llm_leaderboard_autogptq()


if __name__ == "__main__":
    # os.environ['HF_DATASETS_OFFLINE'] = '1'

    max_threads = str(min(8, os.cpu_count()))
    os.environ["OMP_NUM_THREADS"] = max_threads
    os.environ["OPENBLAS_NUM_THREADS"] = max_threads
    os.environ["MKL_NUM_THREADS"] = max_threads
    os.environ["VECLIB_MAXIMUM_THREADS"] = max_threads
    os.environ["NUMEXPR_NUM_THREADS"] = max_threads
    os.environ["NUMEXPR_MAX_THREADS"] = max_threads
    os.environ["HF_HOME"] = "/data/hugginface/"

    main()
