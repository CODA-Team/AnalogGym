# -*- coding: utf-8 -*-
"""
    this module is written by users
    this module defines Design Varas,circuit,and all settings...
"""
import re

# Project
prj_name = "pll_vco"  
## prj_name = Project name

# Design Variables
DX = [
('c_cbank_half_cell', 8.33e-07, 9.996e-06, 5e-09, 4.165e-06, 'NO'),
('c_cvar', 9.26e-07, 1.1112e-05, 5e-09, 4.63e-06, 'NO'),
('c_load', 9.26e-07, 1.1112e-05, 5e-09, 4.63e-06, 'NO'),
('l_bias1', 5e-08, 2.4e-07, 5e-09, 1e-07, 'NO'),
('l_bias2', 5e-08, 2.4e-07, 5e-09, 1e-07, 'NO'),
('l_cbank_half_cell', 5e-08, 2.4e-07, 5e-09, 1e-07, 'NO'),
('l_cvar', 8e-08, 9.6e-07, 5e-09, 4e-07, 'NO'),
('l_load', 5e-08, 2.4e-07, 5e-09, 1e-07, 'NO'),
('r_cvar', 1000.0, 12000.0, 1, 5000.0, 'NO'),
('w_bias1', 2e-06, 1e-05, 5e-09, 1e-05, 'NO'),
('w_bias2', 2e-06, 1e-05, 5e-09, 1e-05, 'NO'),
('w_cbank_half_cell', 2e-07, 2.4e-06, 5e-09, 1e-06, 'NO'),
('w_cvar', 9e-07, 1e-05, 5e-09, 4.5e-06, 'NO'),
('w_load', 2e-07, 2.4e-06, 5e-09, 1e-06, 'NO'),
     ]
## DX = [('name',L,U,step,init,[discrete list]),....] if there is no discrete, do not write

# Setting
setting_1 = [
'run_sim.sh', 'pll_vco.log',
[
("Frequency_Max", None, None, ">", "5e9", "extract_perf(file,'Frequency_Max')", "10"),
("Frequency_Min", "<", "4.8e9", None, None, "extract_perf(file,'Frequency_Min')", "10"),
("PN_1K", "<", "-25", None, None, "extract_perf(file,'PN_1K')", "10"),
("PN_10K", "<", "-50", None, None, "extract_perf(file,'PN_10K')", "10"),
("PN_100K", "<", "-80", None, None, "extract_perf(file,'PN_100K')", "10"),
("PN_1M", "<", "-100", None, None, "extract_perf(file,'PN_1M')", "10"),
("PN_10M", "<", "-120", None, None, "extract_perf(file,'PN_10M')", "10"),
("Kvco_11", "<", "1e8", ">", "2.5e7", "extract_perf(file,'Kvco_11')", "10"),
("Kvco_02", "<", "1e8", ">", "2.5e7", "extract_perf(file,'Kvco_02')", "10"),
("IDC", None, None, None, None, "extract_perf(file,'IDC')", None)]
]
##setting_1 = ['test1.sp', 'test1.lis', [("pm",None, None, ">","90","extract_pm(file)", "5"),]]
##setting_x = ['test name', 'result file', [(per1, </empty/<=,num1,>/empty/>=,num2, "extract function", weight),...]]
setting = [setting_1]
##setting = [setting_1, setting_2, ...]
FOM = ['IDC', "0.015", None]
##FOM = ['-pm', "-90", 10]
##FOM = ['-gain', "-95", None]
##FOM = ['fuc', num1/None, weight/None] :minimize FOM, if FOM has constraint, the form must be "FOM < num"

# extract performance from result file
def extract_perf(file, perf):
    pattern_str = '\s*'+perf+'\s*([\d.eE+\-]+)'
    pattern = re.compile(pattern_str)
    with open(file, 'r', encoding='iso-8859-1') as f:
        lines = f.readlines()
        for line in lines:
            result = pattern.search(line)
            if result and (result.group(1)!="e"):
                return result.group(1)
        return "0"

# Control prrameter
DEL_OUT_FOLDER = True
Mode = "Ocean"
CPU_CORES = 2
##for DE, GA, PSO, SA, turbo, this parameter can be more according to the Init_num

# set the optimization algorithm
'''
# if using src/scripts/bak
def get_totNum(Init_num, Max_eval, Algorithm):
    if Algorithm in ("Bayes", "BOc", "weibo_py", "bobyqa_py"):
        totNum = Init_num + Max_eval
    elif Algorithm in ("PSO", "GA"):
        totNum = Init_num * (Max_eval+2)
    elif Algorithm == "SA":
        totNum = Init_num * Max_eval+1
    elif Algorithm == "pycma":
        totNum = Init_num * Max_eval+2
    elif Algorithm == "DE":
        totNum = Init_num * Max_eval * 2
    elif Algorithm in ("SQP", "bobyqa"):
        totNum = 600
    elif Algorithm in ("random"):
        totNum = Init_num * Max_eval
    elif Algorithm in ("turbo"):
        totNum = Init_num*Tr_num + Max_eval
    return totNum
'''

# Algorithm
Algorithm = "weibo_py" 
##"Bayes", "BOc", "weibo_py", "DE", "GA", "PSO", "SQP", "bobyqa", "SA", "random", "turbo", "pycma", "bobyqa_py"
Init_num = 4
Max_eval = 3
Tr_num = 2 # for trust region related algorithms
##"Bayes": (20, 80),
##"BOc": (20, 80),
##"weibo_py": (20, 80),
##"DE": (20, 20),
##"GA": (20, 20),
##"PSO": (20, 20),
##"SQP": (200, 400),
##"bobyqa": (200, 400),
##"SA": (20, 20),
##"random": (100, 100),
##"turbo": (2*dim, 100)
##"pycma": (4+3*np.log(dim), 100+150*(dim+3)**2//init_num**0.5)
##"bobyqa_py": (2*dim+1, 100)
'''
# if using src/scripts/bak
TOT_NUM = get_totNum(Init_num, Max_eval, Algorithm)
'''
WITH_CONS = False
##for python algos this parameter can be True 
##for cxx algos, this parameter is useless
##but note that BOc always carries out constrained optimization
##and note that turbo can only carry out cost optimization
WITH_INIT = True
##for python algos this parameter can be True
##for cxx algos, this parameter is useless (always False)
