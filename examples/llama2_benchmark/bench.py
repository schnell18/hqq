import gc
import glob
import logging
import os
import pandas as pd
import time
import torch
import transformers

from auto_gptq import AutoGPTQForCausalLM
from auto_gptq import BaseQuantizeConfig as GPTQQuantConfig
from awq import AutoAWQForCausalLM
from datetime import datetime
from eval_model import eval_wikitext2, eval_c4, eval_ptb
from hqq.core.quantize import BaseQuantizeConfig as HQQQuantConfig
from hqq.engine.hf import AutoTokenizer as hggAutoTokenizer
from hqq.engine.hf import HQQModelForCausalLM
from tqdm import tqdm
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
   ("b4g64",    HQQQuantConfig(nbits=4, group_size=64)),
   ("b4g128",   HQQQuantConfig(nbits=4, group_size=128)),
   ("b3g64",    HQQQuantConfig(nbits=3, group_size=64)),
   ("b3g128",   HQQQuantConfig(nbits=3, group_size=128)),
   ("mix-3_62", HQQQuantConfig(mixed=True, budget=3.62, quant_scale=True)),
   ("mix-3_42", HQQQuantConfig(mixed=True, budget=3.42, quant_scale=True)),
   ("mix-3_15", HQQQuantConfig(mixed=True, budget=3.15, quant_scale=True)),
   ("mix-2_75", HQQQuantConfig(mixed=True, budget=2.75, quant_scale=True)),
]

AWQ_CONFIGS = [
    ("b4g64", {"w_bit": 4, "q_group_size": 64, "zero_point": True, 'version':'GEMM'}),
    ("b4g128", {"w_bit": 4, "q_group_size": 128, "zero_point": True, 'version':'GEMM'}),
    # 3-bit not supported by AutoAWQ right now
    #("b3g64", {"w_bit": 3, "q_group_size": 64, "zero_point": True, 'version':'gemv_fast'}),
    #("b3g128", {"w_bit": 3, "q_group_size": 128, "zero_point": True, 'version':'gemv_fast'}),
]

GPTQ_CONFIGS = [
    ("b4g64",  GPTQQuantConfig(bits=4, group_size=64, damp_percent=0.01, desc_act=False)),
    ("b4g128", GPTQQuantConfig(bits=4, group_size=64, damp_percent=0.01, desc_act=False)),
    ("b3g64",  GPTQQuantConfig(bits=3, group_size=64, damp_percent=0.01, desc_act=False)),
]

def experiment_debug():
    models = [
        "meta-llama/Llama-2-7b-hf",
    ]
    tasks = {
        'HQQ': {
            "create_fn": create_hqq_model,
            "quantize_fn": quantize_hqq_model,
            "configs": HHQ_CONFIGS[-1:]
        },
    }
    do_expermient(
        "debug_hqq_auto",
        models,
        tasks,
        save_dir = "snapshots-hqq"
    )

def experiment_eval_all():
    models = ALL_MODELS
    tasks = {
        'HQQ': {
           "create_fn": create_hqq_model,
           "quantize_fn": quantize_hqq_model,
           "configs": HHQ_CONFIGS,
        },
        'AWQ': {
            "create_fn": create_awq_model,
            "quantize_fn": quantize_awq_model,
            "configs": AWQ_CONFIGS,
        },
        'GPTQ': {
            "create_fn": create_gptq_model,
            "quantize_fn": quantize_gptq_model,
            "configs": GPTQ_CONFIGS,
        },
    }
    do_expermient(
        "eval_all_benchmark",
        models,
        tasks,
    )

def experiment_eval_mix():
    models = ALL_MODELS[:1]
    tasks = {
        'HQQ': {
            "create_fn": create_hqq_model,
            "quantize_fn": quantize_hqq_model,
            "configs": HHQ_CONFIGS[-3:],
        },
    }
    do_expermient(
        "eval_mix",
        models,
        tasks
    )

def experiment_quantize_mix():
    models = ALL_MODELS[:1]
    tasks = {
        'HQQ': {
            "create_fn": create_hqq_model,
            "quantize_fn": quantize_hqq_model,
            "configs": HHQ_CONFIGS[-3:],
        },
    }
    do_expermient(
        "quantize_mix",
        models,
        tasks,
        quantize_only = True,
    )

def experiment_quantize_all():
    models = ALL_MODELS
    tasks = {
        'HQQ': {
           "create_fn": create_hqq_model,
           "quantize_fn": quantize_hqq_model,
           "configs": HHQ_CONFIGS,
        },
        'AWQ': {
            "create_fn": create_awq_model,
            "quantize_fn": quantize_awq_model,
            "configs": AWQ_CONFIGS,
        },
        'GPTQ': {
            "create_fn": create_gptq_model,
            "quantize_fn": quantize_gptq_model,
            "configs": GPTQ_CONFIGS,
        },
    }
    do_expermient(
        "quant_all_benchmark",
        models,
        tasks,
    )

def experiment_gptq():
    models = ALL_MODELS
    tasks = {
        'GPTQ': {
            "create_fn": create_gptq_model,
            "quantize_fn": quantize_gptq_model,
            "configs": GPTQ_CONFIGS,
        },
    }
    do_expermient(
        "gptq_benchmark",
        models,
        tasks,
        save_dir = "snapshots"
    )

def experiment_awq():
    models = ALL_MODELS
    tasks = {
        'AWQ': {
            "create_fn": create_awq_model,
            "quantize_fn": quantize_awq_model,
            "configs": AWQ_CONFIGS,
        }
    }
    do_expermient(
        "awq_benchmark",
        models,
        tasks,
        save_dir = "snapshots"
    )

def experiment_hqq():
    models = ALL_MODELS
    tasks = {
        'HQQ': {
           "create_fn": create_hqq_model,
           "quantize_fn": quantize_hqq_model,
            "configs": HHQ_CONFIGS[4:],
        },
    }
    do_expermient(
        "hqq_benchmark",
        models,
        tasks,
        save_dir = "snapshots"
    )

def experiment_fp16_baseline():
    models = ALL_MODELS
    tasks = {
        'FP16': {
           "create_fn": create_fp16_model,
           "quantize_fn": None,
           "configs": [
               ("base", {}),
           ]
        },
    }
    do_expermient(
        "fp16_baseline",
        models,
        tasks,
    )

def do_expermient(
        experiment_name,
        models,
        tasks,
        quantize_only = False,
        save_dir = "snapshots"
    ):

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    metrics = []
    for kind, spec in tasks.items():
        configs = spec["configs"]
        for config in configs:
            for model_id in models:
                metric = {
                    'model': model_id.split('/')[1],
                    'method': kind,
                    'config': config[0],
                    'config_detail': config[1],
                    'quant_duration': 0,
                    'quant_mem_allot': 0,
                    'quant_mem_reserved': 0,
                    'fp_mem_allot': 0,
                    'fp_mem_reserved': 0,
                    'quant_duration': 0,
                    'ppl_wikitext': 0,
                    'ppl_ptb': 0,
                    'ppl_c4': 0,
                    'duration_wikitext': 0,
                    'duration_ptb': 0,
                    'duration_c4': 0,
                }
                print("*" * 72)
                if quantize_only:
                    print(f"Quantizing {kind} on {model_id} w/ config: {config[0]}...")
                else:
                    print(f"Benckmarking {kind} on {model_id} w/ config: {config[0]}...")
                print("*" * 72)
                create_fn = spec["create_fn"]
                quant_fn = spec["quantize_fn"]
                model, tokenizer, quantized = create_fn(
                    model_id,
                    config[1],
                    config[0],
                    quant_fn is not None,
                    save_dir
                )
                if quantized:
                    metric['quant_mem_allot'], metric['quant_mem_reserved'] = get_memory_metrics()
                else:
                    metric['fp_mem_allot'], metric['fp_mem_reserved'] = get_memory_metrics()

                if not quantized and quant_fn:
                    metric['fp_mem_allot'], metric['fp_mem_reserved'] = get_memory_metrics()
                    if config[0].startswith('mix-') and model_id in QUANT_METRICS_FILE_MAP:
                        config[1]['quant_metrics_file'] = QUANT_METRICS_FILE_MAP[model_id]
                    model, duration = quant_fn(model, tokenizer, config[1], model_id, config[0], save_dir)
                    metric['quant_mem_allot'], metric['quant_mem_reserved'] = get_memory_metrics()
                    metric['quant_duration'] = duration
                #Evaluate the quantized model
                if not quantize_only:
                    metric = eval_ppls(model, tokenizer, metric)
                save_partial_metric(experiment_name, kind, model_id, config[0], metric)
                cleanup(model)

    # combine metrics
    combine_metrics(experiment_name)

def save_partial_metric(experiment_name, kind, model_id, config, metric):
    metrics = [metric]
    df = pd.DataFrame(metrics)
    result_dir = f"results/{experiment_name}"
    os.makedirs(result_dir, exist_ok=True)
    model_short_id = model_id.split('/')[1]
    file_name = f"{result_dir}/partial-{kind}-{model_short_id}-{config}.csv"
    df.to_csv(
        file_name,
        columns=[
            "method", "model", "config", "quant_duration",
            "ppl_wikitext", "ppl_ptb", "ppl_c4",
            "duration_wikitext", "duration_ptb", "duration_c4",
            "quant_mem_allot", "quant_mem_reserved",
            "fp_mem_allot", "fp_mem_reserved",
            "config_detail",
        ],
        index=False
    )

def combine_metrics(experiment_name):
    dfs = []
    iters = glob.iglob(f"./results/{experiment_name}/partial-*.csv")
    for it in iters:
        df = pd.read_csv(it)
        dfs.append(df)
    combined = pd.concat(dfs)
    ts_str = datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = f"results/result-{experiment_name}-{ts_str}.xlsx"
    combined.to_excel(file_name, index=False)

def eval_ppls(model, tokenizer, metric):
    ppl_wikitext, duration_wikitext = eval_wikitext2(model, tokenizer, verbose=True)
    ppl_c4, duration_c4 = eval_c4(model, tokenizer, verbose=True)
    ppl_ptb, duration_ptb = eval_ptb(model, tokenizer, verbose=True)
    metric['ppl_wikitext'] = ppl_wikitext
    metric['ppl_ptb'] = ppl_ptb
    metric['ppl_c4'] = ppl_c4
    metric['duration_wikitext'] = duration_wikitext
    metric['duration_ptb'] = duration_ptb
    metric['duration_c4'] = duration_c4
    return metric

def create_fp16_model(model_id, quant_config, config_id, load_quantized, save_dir):
    model     = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer, False

def create_awq_model(model_id, quant_config, config_id, load_quantized, save_dir):
    quantized = False
    quant_path = f"{save_dir}/{model_id}-{config_id}-awq"
    if load_quantized and os.path.exists(quant_path):
        model = AutoAWQForCausalLM.from_quantized(quant_path, "", fuse_layers=False)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        quantized = True
        model = model.cuda()
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        model     = AutoAWQForCausalLM.from_pretrained(model_id)
    return model, tokenizer, quantized

def quantize_awq_model(model, tokenizer, quant_config, model_id, config_id, save_dir):
    t1 = time.time()
    model.quantize(tokenizer, quant_config=quant_config)
    t2 = time.time()
    print('Took ' + str(t2-t1) + ' seconds to quantize the model with AWQ')
    quant_path = f"{save_dir}/{model_id}-{config_id}-awq"
    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)
    return model, t2-t1

#Adapted from: https://towardsdatascience.com/4-bit-quantization-with-gptq-36b0f4f02c34
def prepare_model(model, tokenizer, n_samples=1024, max_tokens=512, use_triton=False):
	# Load data and tokenize examples
	from datasets import load_dataset
	import random
	data           = load_dataset("allenai/c4", data_files="en/c4-train.00001-of-01024.json.gz", split=f"train[:{n_samples}]")
	tokenized_data = torch.cat([tokenizer(data[i]['text'], return_tensors='pt').input_ids for i in tqdm(range(len(data)))], axis=-1) #~536K tokens

	# Format tokenized examples
	random.seed(1)
	examples_ids = []
	for _ in range(n_samples):
		i              = random.randint(0, tokenized_data.shape[1] - max_tokens - 1)
		j              = i + max_tokens
		input_ids      = tokenized_data[:, i:j]
		attention_mask = torch.ones_like(input_ids)
		examples_ids.append({'input_ids': input_ids, 'attention_mask': attention_mask})

	print('Using ' + str(len(examples_ids)) + ' samples for calibration.')
	model.quantize(examples_ids, batch_size=1, use_triton=use_triton)
	model = model.cuda();
	with torch.no_grad(): x = model(input_ids.to('cuda'));
	del examples_ids, x
	torch.cuda.empty_cache()
	gc.collect()
	return model

def create_gptq_model(model_id, quant_config, config_id, load_quantized, save_dir):
    quantized = False
    quant_path = f"{save_dir}/{model_id}-{config_id}-gptq"
    if load_quantized and os.path.exists(quant_path):
        model = AutoGPTQForCausalLM.from_quantized(quant_path, device="cuda:0")
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        quantized = True
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        model     = AutoGPTQForCausalLM.from_pretrained(model_id, quant_config)
    return model, tokenizer, quantized

def quantize_gptq_model(model, tokenizer, quant_config, model_id, config_id, save_dir):
    t1 = time.time()
    model = prepare_model(model, tokenizer)
    t2 = time.time()
    print('Took ' + str(t2-t1) + ' seconds to quantize the model with GPTQ')
    quant_path = f"{save_dir}/{model_id}-{config_id}-gptq"
    model.save_quantized(quant_path, use_safetensors=True)
    return model, t2-t1

def create_hqq_model(model_id, quant_config, config_id, load_quantized, save_dir):
    quantized = False
    quant_path = f"{save_dir}/{model_id}-{config_id}-hqq"
    if load_quantized and os.path.exists(quant_path):
        model     = HQQModelForCausalLM.from_quantized(quant_path)
        tokenizer = hggAutoTokenizer.from_pretrained(model_id)
        quantized = True
    else:
        model     = HQQModelForCausalLM.from_pretrained(model_id)
        tokenizer = hggAutoTokenizer.from_pretrained(model_id)
    return model, tokenizer, quantized

def quantize_hqq_model(model, tokenizer, quant_config, model_id, config_id, save_dir):
    t1 = time.time()
    model.quantize_model(quant_config=quant_config)
    t2 = time.time()
    print('Took ' + str(t2-t1) + ' seconds to quantize the model with HQQ')
    quant_path = f"{save_dir}/{model_id}-{config_id}-hqq"
    model.save_quantized(quant_path)
    return model, t2-t1

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
    # experiment_awq()
    # experiment_gptq()
    # experiment_hqq()
    # experiment_debug()
    # experiment_fp16_baseline()
    # experiment_quantize_mix()
    experiment_eval_mix()


if __name__ == "__main__":
    # os.environ['HF_DATASETS_OFFLINE'] = '1'
    main()
