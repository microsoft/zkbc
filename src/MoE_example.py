from transformers import AutoTokenizer, SwitchTransformersForConditionalGeneration
import torch.nn as nn
import torch

# So the original switch transformer will have an astronomical number of parameters and FLOPs, so we use a small toy example.
tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")
input_text = "A <extra_id_0> walks into a bar a orders a <extra_id_1> with <extra_id_2> pinch of <extra_id_3>."
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

class SwitchWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        return self.model.generate(input_ids = input_ids)


# This is an example of a large pretrained MoE based on T5, which is far to large for timely execution (2^9.6 FLOPs).

google_MoE = SwitchTransformersForConditionalGeneration.from_pretrained("google/switch-base-8")

model = SwitchWrapper(google_MoE)
outputs = model(input_ids)
print(tokenizer.decode(outputs[0]))

# Calculate params
from thop import profile
macs, params = profile(model, inputs=(input_ids, ))
print(f"Model has {macs} FLOPs and {params} parameters")

# A alternative approach would be to use smaller experts.
from transformers import SwitchTransformersConfig, SwitchTransformersForConditionalGeneration

switch_config = SwitchTransformersConfig(vocab_size = 32128 // 2, d_model = 128, d_kv = 8, d_ff = 128, expert_capacity = 64, num_layers = 2, num_sparse_encoder_layers = 2, num_decoder_layers = 2, num_sparse_decoder_layers = 2, num_heads = 3, num_experts = 8, is_encoder_decoder=True, decoder_start_token_id=0)
small_switch_model = SwitchTransformersForConditionalGeneration(switch_config)

model = SwitchWrapper(small_switch_model)
max_token = 32128 // 2
input_ids[input_ids > max_token] = max_token-1
outputs = model(input_ids)

# Calculate params
from thop import profile
macs, params = profile(model, inputs=(input_ids, ))
print(f"Model has {macs} FLOPs and {params} parameters")

# %% 3.1 Export model to onnx

import os, json, tqdm
import glob
from utils.export import export

os.makedirs('MoE/logs', exist_ok=True)
SRS_PATH = '../kzgs/kzg%d.srs' # You may need to generate this
LOGGING = False
pipstd = lambda fname: f" >> MoE/logs/{fname}.log" if LOGGING else ""

export(model,input_array = input_ids, onnx_filename = "MoE/moe.onnx", input_filename = "MoE/input.json", reshape_input=False, opset_version=13)
# Needs opset_version=13 to work with the switch transformer as,
# "UnsupportedOperatorError: Exporting the operator 'aten::tile' to ONNX opset version 12 is not supported. Support for this operator was added in version 13, try exporting with this version."


# %% Ezkl setup
import ezkl
ezkl.gen_settings("MoE/moe.onnx", "MoE/settings.json")
os.system(f"ezkl gen-settings -M MoE/moe.onnx -O MoE/settings.json")


# [E] [0s, ezkl] - failed: Failed analyse for node #629 "/model/decoder/Tile" Tile
# Error: Failed analyse for node #629 "/model/decoder/Tile" Tile

# Caused by:
#     0: Infering facts
#     1: Applying rule GivenRule { inputs[1] }
#     2: Undetermined symbol in expression: <Sym0>