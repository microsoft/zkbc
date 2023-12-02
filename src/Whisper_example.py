from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
from tqdm import tqdm
import torch
import torch.nn as nn
import os, json
import ezkl

os.makedirs('Whisper/logs', exist_ok=True)
SRS_PATH = '../kzgs/kzg%d.srs' # You may need to generate this
LOGGING = False
pipstd = lambda fname: f" >> Whisper/logs/{fname}.log" if LOGGING else ""


# %% 1.1 Load the model and data
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
pretrained_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")


# %% 1.2 Wrapping the huggingface model in a class so we can export it
class WhisperWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = pretrained_model

    def forward(self, input_features):
        return self.model.generate(input_features)
        
model = WhisperWrapper()


# %% 1.3 load the data
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
sample = ds[0]["audio"]
input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features 


# %% 1.4 Test everything is working okay
predicted_ids = model(input_features)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

# %% 1.5 check how many flops we're working with
from thop import profile
macs, params = profile(model, inputs=(input_features, ))
print(f"Model has {macs} FLOPs and {params} parameters")


# %% 2.1 Export the model and data for ezkl to use

# opset_version = 10 will not work for the whisper models
torch.onnx.export(model, input_features, "Whisper/whisper.onnx", export_params=True, do_constant_folding=True, input_names = ['input'],output_names = ['output'],dynamic_axes={'input' : {0 : 'batch_size'},'output' : {0 : 'batch_size'}})

data_array = ((input_features).detach().numpy()).reshape([-1]).tolist()
data = dict(input_data = [data_array],
            output_data = [((o).detach().numpy()).reshape([-1]).tolist() for o in predicted_ids])

# Serialize data into file:
json.dump( data, open( "Whisper/input.json", 'w' ) )


# %% 2.2 Setup and calibrate the model for proving using ezkl
os.system("ezkl table -M Whisper/whisper.onnx" + pipstd('setup'))
os.system("ezkl gen-settings -M Whisper/whisper.onnx --settings-path=Whisper/settings.json --input-visibility='public'" + pipstd('setup') )
os.system("ezkl calibrate-settings -M Whisper/whisper.onnx -D Whisper/input.json --settings-path=Whisper/settings.json" + pipstd('setup'))

settings = json.load(open('Whisper/settings.json', 'r'))
logrows = settings['run_args']['logrows']
ezkl.get_srs(SRS_PATH % logrows, "Whisper/settings.json")

os.system("ezkl compile-circuit -M Whisper/whisper.onnx -S Whisper/settings.json --compiled-circuit Whisper/whisper.ezkl" + pipstd('setup'))
os.system("ezkl gen-witness -M Whisper/whisper.ezkl -D Whisper/input.json --output Whisper/witnessRandom.json" + pipstd('setup'))
os.system("ezkl mock -M Whisper/whisper.ezkl --witness Whisper/witnessRandom.json" + pipstd('setup'))
os.system(f"ezkl setup -M Whisper/whisper.ezkl --srs-path={SRS_PATH % logrows} --vk-path=Whisper/vk.key --pk-path=Whisper/pk.key" + pipstd('setup'))

