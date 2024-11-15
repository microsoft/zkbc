from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, AutoTokenizer, AutoFeatureExtractor
from datasets import load_dataset
import datasets
from tqdm import tqdm
import torch
import torch.nn as nn
import os, json
import ezkl

os.makedirs('ASR/logs', exist_ok=True)
SRS_PATH = '../kzgs/kzg%d.srs' # You may need to generate this
LOGGING = False
pipstd = lambda fname: f" >> ASR/logs/{fname}.log" if LOGGING else ""


# %% 1.1 Load the model and data
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
pretrained_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base")
tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base")
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")


# %% 1.2 Wrapping the huggingface model in a class so we can export it
class ASRWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = pretrained_model

    def forward(self, input_values):
        logits = self.model(input_values).logits[0]
        return torch.argmax(logits, dim=-1)
        
model = ASRWrapper()

# %% 1.3 load the data
dataset = load_dataset("mozilla-foundation/common_voice_11_0", "en", split="train")
dataset = dataset.cast_column("audio", datasets.Audio(sampling_rate=16_000))
dataset_iter = iter(dataset)
sample = next(dataset_iter)
input_values = feature_extractor(sample["audio"]["array"], return_tensors="pt", sampling_rate=16_000).input_values


# ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
# sample = ds[1]["audio"]
# input_values = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_values

# %% 1.4 Test everything is working okay
predicted_ids = model(input_values)
outputs = tokenizer.decode(predicted_ids, output_word_offsets=True)
transcription = processor.batch_decode(predicted_ids)

# %% 1.5 check how many flops we're working with
from thop import profile
macs, params = profile(model, inputs=(input_values, ))
print(f"Model has {macs} FLOPs and {params} parameters")


# %% 2.1 Export the model and data for ezkl to use

# opset_version = 10 will not work for the ASR models
torch.onnx.export(model, input_values, "ASR/ASR.onnx", export_params=True, do_constant_folding=True, input_names = ['input'],output_names = ['output'],dynamic_axes={'input' : {0 : 'batch_size'},'output' : {0 : 'batch_size'}}, opset_version=16)

data_array = ((input_values).detach().numpy()).reshape([-1]).tolist()
data = dict(input_data = [data_array],
            output_data = [((o).detach().numpy()).reshape([-1]).tolist() for o in predicted_ids])

# Serialize data into file:
json.dump( data, open( "ASR/input.json", 'w' ) )


# %% 2.2 Setup and calibrate the model for proving using ezkl
os.system("ezkl table -M ASR/ASR.onnx" + pipstd('setup'))
os.system("ezkl gen-settings -M ASR/ASR.onnx --settings-path=ASR/settings.json" + pipstd('setup') )
os.system("ezkl calibrate-settings -M ASR/ASR.onnx -D ASR/input.json --settings-path=ASR/settings.json" + pipstd('setup'))

settings = json.load(open('ASR/settings.json', 'r'))
logrows = settings['run_args']['logrows']
ezkl.get_srs(SRS_PATH % logrows, "ASR/settings.json")

os.system("ezkl compile-circuit -M ASR/ASR.onnx -S ASR/settings.json --compiled-circuit ASR/ASR.ezkl" + pipstd('setup'))
os.system("ezkl gen-witness -M ASR/ASR.ezkl -D ASR/input.json --output ASR/witnessRandom.json" + pipstd('setup'))
os.system(f"ezkl setup -M ASR/ASR.ezkl --srs-path={SRS_PATH % logrows} --vk-path=ASR/vk.key --pk-path=ASR/pk.key" + pipstd('setup'))
os.system(f"ezkl prove -M ASR/ASR.ezkl --srs-path={SRS_PATH % logrows} --pk-path=ASR/pk.key --witness ASR/witnessRandom.json" + pipstd('setup'))

