# **AnalogGym**

(Under Construction)

## About AnalogGym

This repository is the analog circuit synthesis testing suite, **AnalogGym**.

AnalogGym encompasses 30 circuit topologies in five key categories: sensing front ends, voltage references, AMPs, low dropout regulators (LDOs), and phase-locked loops (PLLs). 
Among these, the LDOs and AMPs support the open-source [Ngspice](https://ngspice.sourceforge.io/) simulator and the [SkyWater](https://github.com/google/skywater-pdk)  process design kit (PDK), allowing for greater accessibility and reproducibility. 

## Table of Contents

- [Getting Started](#Getting_Started)
- [AnalogGym Contents](#AnalogGym_Contents)
- [Usage](#Usage)
- [Citation](#Citation)

<h2 id="Getting_Started">Getting Started</h2>

Examples of using the AnalogGym with the Relational Graph Neural Network and Reinforcement Learning algorithm[^1], referencing [this repository](https://github.com/ChrisZonghaoLi/sky130_ldo_rl).

[^1]: Z. Li and A. C. Carusone, "Design and Optimization of Low-Dropout Voltage Regulator Using Relational Graph Neural Network and Reinforcement Learning in Open-Source SKY130 Process," 2023 IEEE/ACM International Conference on Computer Aided Design (ICCAD), San Francisco, CA, USA, 2023, pp. 01-09, doi: 10.1109/ICCAD57390.2023.10323720.

<h2 id="AnalogGym_Contents">AnalogGym Contents</h2>

The test circuits provided in AnalogGym include:

- `netlist` folder: Contains pre-packaged circuit files that require no modification.
- `testbench` folder: Includes testbench files for running simulations with the simulator.
- `design variables` folder: Stores the input parameters for each circuit separately.
- `schematic` folder: Provides circuit diagrams for reference and visualization.

## Usage

Note that in the sky130 process library, transistors have a drain-source breakdown voltage of 1.8V and a threshold voltage of 1V. Consequently, the supply voltage is maintained at 1.8V, rather than being reduced to 1.2V, to meet the required reliability and operational standards.

### Workflow in AnalogGym

<img width="879" alt="AnalogGym_Flow" src="https://github.com/user-attachments/assets/2e06e4cc-7042-42c1-a395-9157f3677d56">

The design flow decouples circuit configuration from the optimization process, allowing for flexible parameter tuning. 
The circuit parameters are maintained in independent configuration files in the `design variables` folder.
Different netlists can be switched in the testbench, with each netlist representing an encapsulated circuit.

### Testbench

| Line | Ngspice Testbench Description |
|------|------------------------------------------------------------|
| 1    | `.include ./path_to_spice_netlist/circuit_name`  — *Include the SPICE netlist* |
| 2    | `.include ./path_to_decision_variable/circuit_name` — *Include the circuit parameters (decision variables)* |
| 3    | `.include ./mosfet_model/sky130_pdk/libs.tech/ngspice/corners/tt.spice` — *Include PDK, modify Process in PVT* |
| 4    | `.PARAM supply_voltage = 1.3` — *Specify supply voltage for PVT* |
| 5    | `.temp 27` — *Specify temperature for PVT* |
| 6    | `.PARAM PARAM_CLOAD = 10p` — *Specify load capacitance* |
| ...  | *Simulation commands; no modifications required.* |


Overview of the components required for using AnalogGym:

- [How to configure and run different circuits via testbench](https://coda-team.github.io/AnalogGym/)
- [How to extract performance metrics from the simulation output files](https://coda-team.github.io/AnalogGym/)
- [How to invoke the simulator and key tips for running simulations](https://coda-team.github.io/AnalogGym/)


- For a detailed tutorial on using Ngspice, please refer to [this link](https://ngspice.sourceforge.io/tutorials.html).
- Detailed documentations can be found in [doc](https://coda-team.github.io/AnalogGym/)




## **Citation**

Please cite us if you find AnalogGym useful.

- AnalogGym: An Open and Practical Testing Suite for Analog Circuit Synthesis, Jintao Li, Haochang Zhi, Ruiyu Lyu, Wangzhen Li, Zhaori Bi<sup>\*</sup>, Keren Zhu<sup>\*</sup>, Yanhan Zhen， Weiwei Shan, Changhao Yan, Fan Yang, Yun Li<sup>\*</sup>, and Xuan Zeng<sup>\*</sup> IEEE/ACM International Conference on Computer-Aided Design (ICCAD '24), October 27--31, 2024, New York, NY, USA  (To appear)




<img src="./docs/images/logos/4school.png" alt="school_logo" width="90%"/>
