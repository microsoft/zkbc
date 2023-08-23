# Zero Knowledge Benchmarking Checking (zkbc)

This is an example codebase for the summer internship project by Tobin South at Microsoft Research in Redmond. This codebase is not a complete collection of work on the topic and more code will be made available after publication of an academic paper on the topic.

## Abstract

In a world of increasing closed-source commercial machine learning models, statements of benchmark performance from model providers must be taken at face value. These benchmarks results – whether over task accuracies, fairness and bias evaluations, or safety checks – are traditionally impossible to verify by a model end-user without the costly or infeasible process of re-performing the benchmark on black-box model outputs. This work presents a method of verifiable model benchmarking using model inference through zkSNARKs. The resulting zero-knowledge computational proofs of model outputs over datasets can be packaged into verifiable attestations showing that models with fixed private weights achieved stated performance or fairness metrics over public inputs. These verifiable attestations can be performed on any standard neural network model with varying compute requirements. This presents a new transparency paradigm in verifiable benchmarking of private models. 

## Build and run

1. Clone & build the rust version of _ezkl_ and then pyezkl. Follow the installation instructions in the README. `git clone https://github.com/zkonduit/ezkl.git`
2. Get an onnx version of a model you want to run.  For example:
* Use `save_onnx.py` on the `https://github.com/AmrElsersy/Emotions-Recognition.git` repo.


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

## Thanks and contribution

Most of the work included in this repo was based on prior work in other repositories by Tobin South. Thanks goes to Shayla Nguyen for early idea incubation, dante & Jason Morton for support and collaboration, and Christian Paquin for mentorship.