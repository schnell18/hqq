import torch

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from eval_model import eval_wikitext2

#Settings
######################################################################################
hf_auth    = None #HuggingFace token
cache_path = ''   #cache directory to store data

#Chose a model
# model_id  = "meta-llama/Llama-2-7b-hf"
# model_id  = "meta-llama/Meta-Llama-3-8B"
model_id  = "meta-llama/Llama-2-13b-hf"
#model_id  = "meta-llama/Llama-2-70b-hf"

#Load model on the CPU
######################################################################################
model     = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_id)

#Evaluate the quantized model
######################################################################################
eval_wikitext2(model, tokenizer, verbose=True)

