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


inputs = tokenizer("Hello, my name is", return_tensors="pt")
x = inputs["input_ids"]
logits = model(x)
next_token = tokenizer.decode(torch.argmax(logits[0, -1, :]).item())

from thop import profile
macs, params = profile(model, inputs=(x, ))

from utils.export import export

export(model, input_array=x, onnx_filename="GPT2/gpt2.onnx", input_filename="GPT2/GPT2/input.json")



# # Now we export the model
# torch.onnx.export(model,               # model being run
#                     x,                   # model input (or a tuple for multiple inputs)
#                     "gpt2.onnx",            # where to save the model (can be a file or file-like object)
#                     export_params=True,        # store the trained parameter weights inside the model file
#                     opset_version=10,          # the ONNX version to export the model to
#                     do_constant_folding=True,  # whether to execute constant folding for optimization
#                     input_names = ['input'],   # the model's input names
#                     output_names = ['output'], # the model's output names
#                     dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
#                                 'output' : {0 : 'batch_size'}})

# data_array = ((x).detach().numpy()).reshape([-1]).tolist()

# data = dict(input_data = [data_array],)
# json.dump( data, open("GPT2/input.json", 'w' ) )


# Run onnx file through runtime

# import onnxruntime as rt
# import numpy as np

# sess = rt.InferenceSession("gpt2.onnx")
# input_name = sess.get_inputs()[0].name
# label_name = sess.get_outputs()[0].name

# sess.run([label_name], {input_name: data_array})



# Now we run the setup + calibration + witness generation
import ezkl, os
SRS_PATH = "../kzgs/kzg25.srs"

os.system("ezkl gen-settings -M GPT2/gpt2.onnx --settings-path=GPT2/settings.json --input-visibility='public'" )
os.system("ezkl calibrate-settings -M GPT2/gpt2.onnx -D GPT2/input.json --settings-path=GPT2/settings.json --target=resources")
os.system("ezkl compile-model -M GPT2/gpt2.onnx -S GPT2/settings.json --compiled-model GPT2/gpt2.ezkl")
os.system("ezkl gen-witness -M GPT2/gpt2.ezkl -D GPT2/input.json --output GPT2/witnesstokens.json --settings-path GPT2/settings.json")
os.system("ezkl mock -M GPT2/gpt2.ezkl --witness GPT2/witnesstokens.json --settings-path GPT2/settings.json") 

os.system(f"ezkl setup -M network.ezkl --srs-path={SRS_PATH} --vk-path=GPT2/vk.key --pk-path=GPT2/pk.key --settings-path=GPT2/settings.json")

ezkl calibrate-settings -M GPT2/gpt2.onnx -D GPT2/input.json --settings-path=GPT2/settings.json --target=resources
ezkl compile-model -M GPT2/gpt2.onnx -S GPT2/settings.json --compiled-model GPT2/gpt2.ezkl
ezkl gen-witness -M GPT2/gpt2.ezkl -D GPT2/input.json --output GPT2/witnesstokens.json --settings-path GPT2/settings.json
ezkl mock -M GPT2/gpt2.ezkl --witness GPT2/witnesstokens.json --settings-path GPT2/settings.json
