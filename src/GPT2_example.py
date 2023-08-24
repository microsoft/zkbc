import torch, json
from transformers import AutoTokenizer, GPT2LMHeadModel
import torch.nn as nn

# Load in the pretrained weights for GPT2 (or you could train your own)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
modelGPT2LMHead = GPT2LMHeadModel.from_pretrained("gpt2")

# Wrapping the huggingface model in a class so we can export it
class GPT2Wrapper(nn.Module):
    def __init__(self):
        super(GPT2Wrapper, self).__init__()
        self.model = modelGPT2LMHead

    def forward(self, input_ids):
        # For the sake of the proof, we just want the logits
        return self.model(input_ids).logits

model = GPT2Wrapper()

input_text = "Hello, my name is"
inputs = tokenizer(input_text, return_tensors="pt")
x = inputs["input_ids"]
logits = model(x)
next_token = tokenizer.decode(torch.argmax(logits[0, -1, :]).item())
print('input: ', input_text, 'input shape: ', x.shape)
print('next_token: ', next_token, 'next_token shape: ', logits.shape)

# Profile the model for the number of parameters and MACs (good to know)
from thop import profile
macs, params = profile(model, inputs=(x, ))
print(f"Total model params: {params}\nTotal model MACs (FLOPs): {macs}")

from utils.export import export

export(model, input_array=x, onnx_filename="GPT2/gpt2.onnx", input_filename="GPT2/input.json")

# Now we run the setup + calibration + witness generation
import ezkl, os
SRS_PATH = "../kzgs/kzg25.srs"

os.system("ezkl gen-settings -M GPT2/gpt2.onnx --settings-path=GPT2/settings.json --input-visibility='public'" )
os.system("ezkl calibrate-settings -M GPT2/gpt2.onnx -D GPT2/input.json --settings-path=GPT2/settings.json --target=resources")
os.system("ezkl compile-model -M GPT2/gpt2.onnx -S GPT2/settings.json --compiled-model GPT2/gpt2.ezkl")
os.system("ezkl gen-witness -M GPT2/gpt2.ezkl -D GPT2/input.json --output GPT2/witnesstokens.json --settings-path GPT2/settings.json")
os.system("ezkl mock -M GPT2/gpt2.ezkl --witness GPT2/witnesstokens.json --settings-path GPT2/settings.json") 
os.system(f"ezkl setup -M network.ezkl --srs-path={SRS_PATH} --vk-path=GPT2/vk.key --pk-path=GPT2/pk.key --settings-path=GPT2/settings.json")
os.system(f"ezkl prove -M network.ezkl --srs-path={SRS_PATH} --vk-path=GPT2/vk.key --pk-path=GPT2/pk.key --settings-path=GPT2/settings.json")