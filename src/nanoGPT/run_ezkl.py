import ezkl, os, json


# DEBUG = True
SRS_PATH = '../../kzgs/kzg%d.srs'
# LOGGING = True
# os.makedirs('GPT2/logs', exist_ok=True)
# pipstd = lambda fname: f" >> GPT2/logs/{fname}.log" if LOGGING else ""


os.system("ezkl gen-settings -M network.onnx --settings-path=settings.json" )
ezkl.gen_settings("network.onnx", "settings.json")
ezkl.calibrate_settings("input.json", "network.onnx", "settings.json", "resources")

settings = json.load(open('settings.json', 'r'))
logrows = settings['run_args']['logrows']

os.system("ezkl calibrate-settings -M network.onnx -D input.json --settings-path=settings.json --target=resources")
os.system("ezkl compile-circuit -M network.onnx -S settings.json --compiled-circuit gpt2.ezkl")
os.system("ezkl gen-witness -M gpt2.ezkl -D nano.json --output witnesstokens.json")
os.system("ezkl mock -M gpt2.ezkl --witness witnesstokens.json") 
os.system(f"ezkl setup -M gpt2.ezkl --srs-path={SRS_PATH % logrows} --vk-path=vk.key --pk-path=pk.key")
os.system(f"ezkl prove -M gpt2.ezkl --srs-path={SRS_PATH % logrows} --pk-path=pk.key --witness witnesstokens.json")