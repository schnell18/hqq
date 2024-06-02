#Settings
######################################################################################
hf_auth    = None #HuggingFace token
cache_path = ''   #cache directory to store data

#Chose a model
# model_id  = "meta-llama/Meta-Llama-3-8B"
model_id  = "meta-llama/Llama-2-7b-hf"
# model_id  = "meta-llama/Llama-2-13b-hf"
#model_id  = "meta-llama/Llama-2-70b-hf"

#Load model on the CPU
######################################################################################
from hqq.engine.hf import HQQModelForCausalLM, AutoTokenizer
model     = HQQModelForCausalLM.from_pretrained(model_id, use_auth_token=hf_auth)
tokenizer = AutoTokenizer.from_pretrained(model_id,       use_auth_token=hf_auth)

#Quantize the model
######################################################################################
from hqq.core.quantize import *
import time

#quant_config = BaseQuantizeConfig(nbits=8, group_size=128)
#quant_config = BaseQuantizeConfig(nbits=4, group_size=64)
quant_config = BaseQuantizeConfig(nbits=4, group_size=64)
#quant_config = BaseQuantizeConfig(nbits=2, group_size=16)
#quant_config = BaseQuantizeConfig(nbits=2, group_size=16, quant_scale=True) #scale is quantized to 8-bit/g=128

t1 = time.time()
model.quantize_model(quant_config=quant_config)
t2 = time.time()
print('Took ' + str(t2-t1) + ' seconds to quantize the model with HQQ')

#Evaluate the quantized model
######################################################################################
from eval_model import eval_wikitext2, eval_c4, eval_ptb
eval_wikitext2(model, tokenizer, verbose=True)
# eval_c4(model, tokenizer, verbose=True)
# eval_ptb(model, tokenizer, verbose=True)

