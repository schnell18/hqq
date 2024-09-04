import torch


def is_linear_module(key):
    self_attns = ['q_proj', 'v_proj', 'k_proj', 'o_proj']
    mlps = ['gate_proj', 'up_proj', 'down_proj']
    modules = self_attns + mlps
    for module in modules:
        if module in key:
            return True
    return False


def extract_quant_config(base_dir, model_id, config):
    file_path = f"{base_dir}/{model_id}-{config}-hqq/qmodel.pt"
    dikt = torch.load(file_path, map_location='cpu')
    quant_configs = {}
    mem_fp16_all_total = 0
    mem_all_total = 0
    mem_quant_total = 0
    # search quantized linear module with meta
    for key in dikt.keys():
        m_dikt = dikt[key]
        if is_linear_module(key):
            if 'meta' in m_dikt:
                meta_dict = m_dikt['meta']
                meta_scale_dict = meta_dict.get('meta_scale', None)
                shape = meta_dict['shape']
                b1 = meta_dict['nbits']
                g1 = meta_dict['group_size']
                b2 = meta_scale_dict['nbits'] if meta_scale_dict else 8
                g2 = meta_scale_dict['group_size'] if meta_scale_dict else 128
                memmb = (b1 + 2 * b2 / (g1 * g2)) * shape[0] * shape[1] / 8 / 1024 / 1024
                mem_fp16_all_total += shape[0]*shape[1]*2/1024/1024
                mem_quant_total += memmb
                mem_all_total += memmb
                quant_configs[key] = {
                    'b1': b1,
                    'g1': g1,
                    'b2': b2,
                    'g2': g2,
                    'memmb': memmb,
                }
        else:
            w = m_dikt['weight']
            mem_all_total += w.numel()*2/1024/1024
            mem_fp16_all_total += w.numel()*2/1024/1024
    return quant_configs, mem_quant_total, mem_all_total, mem_fp16_all_total


if __name__ == "__main__":
    base_dir = '/data/gqq-eval/quant-snapshots'
    configs, _, _, _ = extract_quant_config(base_dir, 'Llama-2-7b-hf', 'mix-3_62')
