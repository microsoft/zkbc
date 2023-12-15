# This file contains the code for examples with random forests, linear regressions, and SVMs.

import json
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import sk2torch
import torch
import ezkl
import os
from torch import nn
from utils.export import export
import ezkl


SRS_PATH = '../kzgs/kzg%d.srs' # You may need to generate this
LOGGING = True
os.makedirs('sklearn/logs', exist_ok=True)
os.makedirs('sklearn/data', exist_ok=True)
pipstd = lambda fname, mname: f" >> sklearn/logs/{mname}_{fname}.log" if LOGGING else ""


cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
X = X.astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y)


# %% Linear Regression
print("Running Linear Regression")

# using hummingbird-ml to convert to pytorch
from hummingbird.ml import convert

from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X_train, y_train)
circuit = convert(reg, "torch", X[:1]).model

# Export to ONNX
circuit.eval()
export(circuit, X_test[0], 'sklearn/LinearReg.onnx', 'sklearn/lr_input.json')

res = ezkl.gen_settings('sklearn/LinearReg.onnx', 'sklearn/lr_settings.json')
res = ezkl.calibrate_settings('sklearn/lr_input.json', 'sklearn/LinearReg.onnx', 'sklearn/lr_settings.json', "resources")

os.system(f"ezkl compile-circuit -M sklearn/LinearReg.onnx -S sklearn/lr_settings.json --compiled-circuit  sklearn/LinearReg.ezkl")
os.system("ezkl gen-witness -M sklearn/LinearReg.ezkl -D sklearn/lr_input.json --output sklearn/witnessRandom.json")

settings = json.load(open('sklearn/lr_settings.json', 'r'))
logrows = settings['run_args']['logrows']
ezkl.get_srs(SRS_PATH % logrows, 'sklearn/lr_settings.json')

os.system(f"ezkl setup -M sklearn/LinearReg.ezkl --srs-path={SRS_PATH % logrows} --vk-path=sklearn/lr_vk.key --pk-path=sklearn/lr_pk.key" + pipstd('setup', 'lr'))


for i, line in enumerate(X_test):
    output = circuit(torch.tensor(line.reshape(1, -1)))
    data = dict(input_data = [(line).reshape([-1]).tolist()],
                output_data = [((o).detach().numpy()).reshape([-1]).tolist() for o in output])
    input_filename = f'sklearn/data/lr_input_{i}.json'
    json.dump( data, open( input_filename, 'w' ) )
    witness_path = f"sklearn/data/lr_witness_{i}.json"

    os.system(f"ezkl gen-witness -M sklearn/LinearReg.ezkl --data {input_filename} --output {witness_path}" + pipstd('prove', 'lr'))
    proof_path = f"sklearn/data/lr_proof_{i}.proof"
    os.system(f"ezkl prove -M sklearn/LinearReg.ezkl --witness {witness_path} --pk-path=sklearn/lr_pk.key --proof-path={proof_path} --srs-path={SRS_PATH % logrows}" + pipstd('prove', 'lr'))


# %% Random Forest
print("Running Random Forest")

clr = RandomForestClassifier(max_depth=3, n_estimators=10)
clr.fit(X_train, y_train)

trees = []
for tree in clr.estimators_:
    trees.append(sk2torch.wrap(tree))

class RandomForest(nn.Module):
    def __init__(self, trees):
        super(RandomForest, self).__init__()
        self.trees = nn.ModuleList(trees)

    def forward(self, x):
        out = self.trees[0](x)
        for tree in self.trees[1:]:
            out += tree(x)
        return out / len(self.trees)
    

torch_rf = RandomForest(trees)
# assert predictions from torch are = to sklearn 
diffs = []
for i in range(len(X_test)):
    torch_pred = torch_rf(torch.tensor(X_test[i].reshape(1, -1)))
    sk_pred = clr.predict(X_test[i].reshape(1, -1))
    diffs.append(torch_pred[0].round() - sk_pred[0])

print("num diffs", sum(diffs))

# Export to ONNX
torch_rf.eval()
export(torch_rf, X_test[0], 'sklearn/RandomForest.onnx', 'sklearn/RF_input.json')

res = ezkl.gen_settings('sklearn/RandomForest.onnx', 'sklearn/RF_settings.json')
res = ezkl.calibrate_settings('sklearn/RF_input.json', 'sklearn/RandomForest.onnx', 'sklearn/RF_settings.json', "resources")

os.system(f"ezkl compile-circuit -M sklearn/RandomForest.onnx -S sklearn/RF_settings.json --compiled-circuit  sklearn/RandomForest.ezkl")
os.system("ezkl gen-witness -M sklearn/RandomForest.ezkl -D sklearn/RF_input.json --output sklearn/witnessRandom.json")

settings = json.load(open('sklearn/RF_settings.json', 'r'))
logrows = settings['run_args']['logrows']
ezkl.get_srs(SRS_PATH % logrows, 'sklearn/RF_settings.json')

os.system(f"ezkl setup -M sklearn/RandomForest.ezkl --srs-path={SRS_PATH % logrows} --vk-path=sklearn/rf_vk.key --pk-path=sklearn/rf_pk.key" + pipstd('setup', 'rf'))

for i, line in enumerate(X_test):
    output = torch_rf(torch.tensor(line.reshape(1, -1)))
    data = dict(input_data = [(line).reshape([-1]).tolist()],
                output_data = [((o).detach().numpy()).reshape([-1]).tolist() for o in output])
    input_filename = f'sklearn/data/rf_input_{i}.json'
    json.dump( data, open( input_filename, 'w' ) )
    witness_path = f"sklearn/data/rf_witness_{i}.json"

    os.system(f"ezkl gen-witness -M sklearn/RandomForest.ezkl --data {input_filename} --output {witness_path}" + pipstd('prove', 'rf'))
    proof_path = f"sklearn/data/rf_proof_{i}.proof"
    os.system(f"ezkl prove -M sklearn/RandomForest.ezkl --witness {witness_path} --pk-path=sklearn/rf_pk.key --proof-path={proof_path} --srs-path={SRS_PATH % logrows}" + pipstd('prove', 'rf'))

# # all at once
output = torch_rf(torch.tensor(X_test))
# data = dict(input_data = [(X_test).reshape([-1]).tolist()],
#             output_data = [((o).detach().numpy()).reshape([-1]).tolist() for o in output])
data = dict(input_data = [X_test.tolist()])
            # output_data = [((o).detach().numpy()).reshape([-1]).tolist() for o in output])
input_filename = f'sklearn/data/rf_input_{i}.json'
json.dump( data, open( input_filename, 'w' ) )
witness_path = f"sklearn/data/rf_witness_{i}.json"
os.system(f"ezkl gen-witness -M sklearn/RandomForest.ezkl --data {input_filename} --output {witness_path}" + pipstd('prove', 'rf'))
proof_path = f"sklearn/data/rf_proof_{i}.proof"
os.system(f"ezkl prove -M sklearn/RandomForest.ezkl --witness {witness_path} --pk-path=sklearn/rf_pk.key --proof-path={proof_path} --srs-path={SRS_PATH % logrows}" + pipstd('prove', 'rf'))


# %% SVM
print("Running SVM")

from sklearn.svm import SVC, LinearSVC

# sk_model = SVC(probability=True, kernel='poly') # both work
sk_model = LinearSVC()
X = X.astype(np.double)
X_train, X_test, y_train, y_test = train_test_split(X, y)
sk_model.fit(X_train, y_train)
model = sk2torch.wrap(sk_model)
sk_model.predict(X_test[0].reshape(1, -1))

# Export to ONNX
model.eval()
model(torch.tensor(X_test[0].reshape(1, -1)))
export(model, X_test[0], 'sklearn/SVM.onnx', 'sklearn/svm_input.json')

res = ezkl.gen_settings('sklearn/SVM.onnx', 'sklearn/svm_settings.json')
res = ezkl.calibrate_settings('sklearn/svm_input.json', 'sklearn/SVM.onnx', 'sklearn/svm_settings.json', "resources")

os.system(f"ezkl compile-circuit -M sklearn/SVM.onnx -S sklearn/svm_settings.json --compiled-circuit  sklearn/SVM.ezkl")
os.system("ezkl gen-witness -M sklearn/SVM.ezkl -D sklearn/svm_input.json --output sklearn/witnessRandom.json")

settings = json.load(open('sklearn/svm_settings.json', 'r'))
logrows = settings['run_args']['logrows']
ezkl.get_srs(SRS_PATH % logrows, 'sklearn/svm_settings.json')

os.system(f"ezkl setup -M sklearn/SVM.ezkl --srs-path={SRS_PATH % logrows} --vk-path=sklearn/svm_vk.key --pk-path=sklearn/svm_pk.key" + pipstd('setup', 'svm'))


for i, line in enumerate(X_test):
    output = model(torch.tensor(line.reshape(1, -1)))
    data = dict(input_data = [(line).reshape([-1]).tolist()],
                output_data = [((o).detach().numpy()).reshape([-1]).tolist() for o in output])
    input_filename = f'sklearn/data/svm_input_{i}.json'
    json.dump( data, open( input_filename, 'w' ) )
    witness_path = f"sklearn/data/svm_witness_{i}.json"

    os.system(f"ezkl gen-witness -M sklearn/SVM.ezkl --data {input_filename} --output {witness_path}" + pipstd('prove', 'svm'))
    proof_path = f"sklearn/data/svm_proof_{i}.proof"
    os.system(f"ezkl prove -M sklearn/SVM.ezkl --witness {witness_path} --pk-path=sklearn/svm_pk.key --proof-path={proof_path} --srs-path={SRS_PATH % logrows}" + pipstd('prove', 'svm'))