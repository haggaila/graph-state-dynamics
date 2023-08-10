# graph-state-dynamics

This repository contains the data of exeperiments and numerical simulations presented in the paper:

_Dissipative Dynamics of Graph-State Stabilizers with Superconducting Qubits_, Liran Shirizly, Grégoire Misguich, Haggai Landa, [arXiv:2308.01860](https://arxiv.org/abs/2308.01860)

Together with the [data](./output), the complete source code allowing to run the [experiments](./project_experiments) and [simulations](./project_simulations) is included.

The [lindbladmpo](https://github.com/qiskit-community/lindbladmpo) repository is required (together with [ITensor](https://github.com/ITensor/ITensor)), and assumed to be cloned to a directory side by side with current repo's root folder. See the [installation guide](https://github.com/qiskit-community/lindbladmpo/blob/main/INSTALL.md) of _lindbladmpo_ for instructions on compiling the C++ binary necessary for running the simulations.

The `requirements.txt` file allows for pip installing the dependencies of the code.

A simple code example allowing to run charge-parity characterization is given [here](./project_experiments/run-parity-characterization.py).
