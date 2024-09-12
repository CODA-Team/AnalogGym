# **AnalogGym**

## **About AnalogGym**

This repository is the analog circuit synthesis testing suite, **AnalogGym**.

AnalogGym encompasses 30 circuit topologies in five key categories: sensing front ends, voltage references, AMPs, low dropout regulators (LDOs), and phase-locked loops (PLLs). 
Among these, the LDOs and AMPs support the open-source [Ngspice](https://ngspice.sourceforge.io/) simulator and the [SkyWater](https://github.com/google/skywater-pdk)  process design kit (PDK), allowing for greater accessibility and reproducibility. 

## **Table of Contents**

- [Getting Started](#Getting_Started)
- [AnalogGym Contents](#AnalogGym_Contents)
- [Usage](#Usage)
- [Usage](#Usage)
- [Citation](#Citation)
- [Contact](#Contact)

<h2 id="Getting_Started">**Getting Started**</h2>

Examples of using the AnalogGym with the relational graph neural network and reinforcement learning algorithm[^1], referencing [this repository](https://github.com/ChrisZonghaoLi/sky130_ldo_rl). A [Docker version](https://github.com/CODA-Team/AnalogGym/tree/main/RGNN_RL_Docker) and a [downloadable code](https://github.com/CODA-Team/AnalogGym/tree/main/RGNN_RL) package that can be run locally are provided.

[^1]: Z. Li and A. C. Carusone, "Design and Optimization of Low-Dropout Voltage Regulator Using Relational Graph Neural Network and Reinforcement Learning in Open-Source SKY130 Process," 2023 IEEE/ACM International Conference on Computer-Aided Design (ICCAD), San Francisco, CA, USA, 2023, pp. 01-09, doi: 10.1109/ICCAD57390.2023.10323720.

<h2 id="AnalogGym_Contents">**AnalogGym Contents**</h2>

The test circuits provided in AnalogGym include:

- `netlist` folder: Contains pre-packaged circuit files that require no modification.
- `testbench` folder: Includes testbench files for running simulations with the simulator.
- `design variables` folder: Stores the input parameters for each circuit separately.
- `schematic` folder: Provides circuit diagrams for reference and visualization.

## **Usage**

Note that in the sky130 PDK, transistors have a drain-source breakdown voltage of 1.8V and a threshold voltage of 1V. Consequently, the supply voltage is maintained at 1.8V, rather than being reduced to 1.2V, to meet the required reliability and operational standards.

<h3 id="Workflow_in_AnalogGym">Workflow in AnalogGym</h3>

<img width="879" alt="AnalogGym_Flow" src="https://github.com/user-attachments/assets/2e06e4cc-7042-42c1-a395-9157f3677d56">

The design flow decouples circuit configuration from the optimization process, allowing for flexible parameter tuning. 
The circuit parameters are maintained in independent configuration files in the `design variables` folder.
Different netlists can be switched in the testbench, with each netlist representing an encapsulated circuit.

<h3 id="Testbench">Testbench</h3>

| Line | Ngspice Testbench Description |
|------|------------------------------------------------------------|
| 1    | `.include ./path_to_spice_netlist/circuit_name`  — *Include the SPICE netlist* |
| 2    | `.include ./path_to_decision_variable/circuit_name` — *Include the circuit parameters (decision variables)* |
| 3    | `.include ./mosfet_model/sky130_pdk/libs.tech/ngspice/corners/tt.spice` — *Include PDK, modify Process in PVT* |
| 4    | `.PARAM supply_voltage = 1.3` — *Specify supply voltage for PVT* |
| 5    | `.temp 27` — *Specify temperature for PVT* |
| 6    | `.PARAM PARAM_CLOAD = 10p` — *Specify load capacitance* |
| ...  | *Simulation commands; no modifications required.* |

<h3 id="Simulation">Simulation</h3>

For the simulation setup and execution, you can check the following scripts:

- For the **Amplifier** simulation, refer to [main_AMP.py](https://github.com/CODA-Team/AnalogGym/blob/main/RGNN_RL/main_AMP.py).
- For the **LDO** simulation, refer to [main_LDO.py](https://github.com/CODA-Team/AnalogGym/blob/main/RGNN_RL/main_LDO.py).

<h3 id="Performance_Extraction">Performance Extraction</h3>

When extracting performance metrics for the included AMP and LDO circuits, the following points should be noted:

- **Amplifier (AMP)**:
  - For AMPs, **Slew Rate (SR)** and **Settling Time** are not directly measurable from simulations and must be **derived** from transient response analysis.
  
- **Low-Dropout Regulator (LDO)**:
  - LDO performance varies under **light load** (5mA) and **heavy load** (55mA) conditions, as load current affects efficiency, stability, and response time. Light load (minload) may reduce power consumption but can slow response, while heavy load (maxload) demands a higher current supply while maintaining stable output voltage.

Two performance extraction scripts are provided for reference: [AMP](https://github.com/CODA-Team/AnalogGym/blob/main/AnalogGym/Amplifier/perf_extraction_amp.py) and [LDO](https://github.com/CODA-Team/AnalogGym/blob/main/AnalogGym/Low%20Dropout%20Regulator/perf_extraction_LDO.py).

<h3 id="Additional_Resources">Additional Resources</h3>

- For a detailed tutorial on using Ngspice, please refer to [this link](https://ngspice.sourceforge.io/tutorials.html).
- Detailed documentation can be found in [doc](https://coda-team.github.io/AnalogGym/)


## **Citation**

Please cite us if you find AnalogGym useful.

- AnalogGym: An Open and Practical Testing Suite for Analog Circuit Synthesis, Jintao Li, Haochang Zhi, Ruiyu Lyu, Wangzhen Li, Zhaori Bi<sup>\*</sup>, Keren Zhu<sup>\*</sup>, Yanhan Zhen, Weiwei Shan, Changhao Yan, Fan Yang, Yun Li<sup>\*</sup>, and Xuan Zeng<sup>\*</sup> IEEE/ACM International Conference on Computer-Aided Design (ICCAD '24), October 27--31, 2024, New York, NY, USA  (To appear)

## **Contact**

If you have any questions, are seeking collaboration, or would like to contribute circuit designs, please contact us at [j.t.li@i4ai.org](mailto:j.t.li@i4ai.org).

<img src="./docs/images/logos/4school.png" alt="school_logo" width="90%"/>
