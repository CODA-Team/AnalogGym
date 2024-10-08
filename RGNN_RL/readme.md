# **Examples of using the AnalogGym with the [relational graph neural network and reinforcement learning algorithm](https://github.com/ChrisZonghaoLi/sky130_ldo_rl)**



## Initialization
To normalize the transistor attributes used in the observation matrix, you'll need the mean and standard deviation. 
These values can be obtained by running initial experiments with the class method `_init_random_sim`. 

For demonstration, we have already provided this data in the file `AMP_NMCF_op_mean_std.json`. 
When switching to a different circuit, you will need to generate this file yourself. 
To do so, simply set the parameter `'run_initial=True'` at the indicated location in [`main_AMP.py`](https://github.com/CODA-Team/AnalogGym/blob/main/RGNN_RL/main_AMP.py#L32).


## Simulation

For the simulation setup and execution, you can check the following scripts:

- For the **Amplifier** simulation, refer to [main_AMP.py](https://github.com/CODA-Team/AnalogGym/blob/main/RGNN_RL/main_AMP.py).


- For the **LDO** simulation, refer to [main_LDO.py](https://github.com/CODA-Team/AnalogGym/blob/main/RGNN_RL/main_LDO.py).


These scripts provide detailed examples of how the simulations are run, including the optimization process for each circuit.


### Extraction of Device Parameters for `OP` Analysis

To automate the extraction of all relevant device parameters during the `OP` analysis, we provide a script called `dev_params.py`. 

Running this script will generate a file named `AMP_NMCF_dev_params.spice`. 

Once the `OP` analysis is completed, all device parameters will be saved in a file called `AMP_NMCF_op`.

## File Descriptions

Here is a breakdown of key files and directories:

- `/saved_agents`, `/saved_memories`, `/saved_weights`: Stores data generated during the simulation process.
- `/mosfet_model`: Contains the SKY130 PDK (Process Design Kit).
- `/simulations`: Contains SPICE files and is where Ngspice runs.
- `environment.yml`: Defines the environmental requirements for the simulation.
- `Dockerfile`: Describes how to build the Docker images.
- `ckt_graphs.py`: Defines graph info and specifications for AMP and LDO circuits.
- `dev_params.py`: Extracts device parameters (e.g., threshold voltage, transconductance) used as RL observations.
- `AMP_NMCF.py`: Defines the AMP environment (Gymnasium compatible).
- `LDO_TB.py`: Defines the LDO environment (Gymnasium compatible).
- `ddpg.py`: Implements the DDPG algorithm.
- `models.py`: Contains various Graph Neural Network (GNN) models.
- `utils.py`: Provides utility functions (e.g., extracting simulation performance).
- `main_AMP.py`: Runs optimization for AMP circuits.
- `main_LDO.py`: Runs optimization for LDO circuits.
- `torch-1.13.1+cpu-cp310-cp310-linux_x86_64.whl`, `torch_cluster-1.6.1+pt113cpu-cp310-cp310-linux_x86_64.whl`, `torch_scatter-2.1.1+pt113cpu-cp310-cp310-linux_x86_64.whl`, `torch_sparse-0.6.17+pt113cpu-cp310-cp310-linux_x86_64.whl`, `torch_spline_conv-1.2.2+pt113cpu-cp310-cp310-linux_x86_64.whl`: Pre-downloaded PyTorch files for setup.

## Supplementary Information

Please note that the command `conda install source-forge ngspice` can only install up to version `ngspice-41`, which doesn’t support DC scanning for temperature and current in circuit simulations. To use these features, manually download and replace with `ngspice-42` or `ngspice-43` from [ngspice on SourceForge](https://sourceforge.net/projects/ng-spice-rework/files/ng-spice-rework/43/).

To replace the older version, navigate to the following directory (assuming the `analoggym-env` conda environment is already created):

```bash
/Anaconda/envs/analoggym-env/Library/bin
```
Overwrite the old `ngspice.exe` with the newer version. Ensure that the `analoggym-env` environment meets all requirements in `environment.yml`.
