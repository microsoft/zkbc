import torch, json
import torch.nn as nn

# %% 1. Choose your model (we're going through iterations on this)
# Current GPTNeoX models have issues.

# Load in the pretrained weights for LLM (or you could train your own)
# from transformers import AutoTokenizer, GPT2LMHeadModel
# tokenizer = AutoTokenizer.from_pretrained('gpt2')
# pretrained_model = GPT2LMHeadModel.from_pretrained('gpt2')

# from transformers import AutoModelForCausalLM, AutoTokenizer
# pretrained_model = AutoModelForCausalLM.from_pretrained('harborwater/open-llama-3b-claude-30k')
# tokenizer = AutoTokenizer.from_pretrained("harborwater/open-llama-3b-claude-30k")

# from transformers import AutoModelForCausalLM, AutoTokenizer
# pretrained_model = AutoModelForCausalLM.from_pretrained('JackFram/llama-160m')
# tokenizer = AutoTokenizer.from_pretrained("JackFram/llama-160m")


# from transformers import GPTNeoXForCausalLM, AutoTokenizer
# pretrained_model = GPTNeoXForCausalLM.from_pretrained(
#   "EleutherAI/pythia-70m-deduped",
#   revision="step3000",
#   cache_dir="./pythia-70m-deduped/step3000",
# )

# tokenizer = AutoTokenizer.from_pretrained(
#   "EleutherAI/pythia-70m-deduped",
#   revision="step3000",
#   cache_dir="./pythia-70m-deduped/step3000",
# )

# from transformers import AutoModelForCausalLM
# pretrained_model = AutoModelForCausalLM.from_pretrained('roneneldan/TinyStories-1M')
# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")


from transformers import GPTNeoXForCausalLM, AutoTokenizer
pretrained_model = GPTNeoXForCausalLM.from_pretrained(
  "EleutherAI/pythia-14m"
)
tokenizer = AutoTokenizer.from_pretrained(
  "EleutherAI/pythia-14m"
)

# from transformers import GPTNeoForCausalLM, AutoTokenizer
# pretrained_model = GPTNeoForCausalLM.from_pretrained(
#   "EleutherAI/gpt-neo-125m"
# )
# tokenizer = AutoTokenizer.from_pretrained(
#   "EleutherAI/gpt-neo-125m"
# )

# %% 1.X This is the simplest model for demonstration purposes
from transformers import AutoModelForCausalLM, AutoTokenizer
pretrained_model = AutoModelForCausalLM.from_pretrained('roneneldan/TinyStories-1M')
tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-1M")


# %% 1.2 Wrapping the huggingface model in a class so we can export it
class GPTWrapper(nn.Module):
    def __init__(self):
        super(GPTWrapper, self).__init__()
        self.model = pretrained_model

    def forward(self, input_ids):
        # For the sake of the proof, we just want the logits
        logits = self.model(input_ids).logits
        return logits[:,-1,:].argmax(dim=1)

model = GPTWrapper()

# %% 1.3 Test everything is working okay
input_text = "The goal of life is "
# input_text = "The goal of life is [MASK]."
inputs = tokenizer(input_text, return_tensors="pt")
x = inputs["input_ids"]
output = model(x)
next_token = tokenizer.decode(output)
print('input: ', input_text, 'input shape: ', x.shape)
print('next_token: ', next_token)

# Profile the model for the number of parameters and MACs (good to know)
from thop import profile
macs, params = profile(pretrained_model, inputs=(x, ))
print(f"Total model params: {params}\nTotal model MACs (FLOPs): {macs}")

# %% 2. Export the model and data for ezkl to use
from utils.export import export
export(model, input_array=x, onnx_filename="GPT2/gpt2.onnx", input_filename="GPT2/input.json", reshape_input=False)

# %% 3. Now we run the setup + calibration + witness generation
import ezkl, os, json
os.system("ezkl gen-settings -M GPT2/gpt2.onnx --settings-path=GPT2/settings.json" )
ezkl.gen_settings("GPT2/gpt2.onnx", "GPT2/settings.json")

settings = json.load(open('GPT2/settings.json', 'r'))
logrows = settings['run_args']['logrows']

os.system("ezkl calibrate-settings -M GPT2/gpt2.onnx -D GPT2/input.json --settings-path=GPT2/settings.json --target=resources")
os.system("ezkl compile-model -M GPT2/gpt2.onnx -S GPT2/settings.json --compiled-model GPT2/gpt2.ezkl")
os.system("ezkl gen-witness -M GPT2/gpt2.ezkl -D GPT2/input.json --output GPT2/witnesstokens.json --settings-path GPT2/settings.json")
os.system("ezkl mock -M GPT2/gpt2.ezkl --witness GPT2/witnesstokens.json --settings-path GPT2/settings.json") 
os.system(f"ezkl setup -M network.ezkl --srs-path={SRS_PATH % logrows} --vk-path=GPT2/vk.key --pk-path=GPT2/pk.key --settings-path=GPT2/settings.json")
os.system(f"ezkl prove -M network.ezkl --srs-path={SRS_PATH % logrows} --vk-path=GPT2/vk.key --pk-path=GPT2/pk.key --settings-path=GPT2/settings.json")