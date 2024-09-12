# -*- coding: utf-8 -*-
"""
    this module is written by users
    this module defines Design Varas,circuit,and all settings...
"""
import re

# Project
''' absolute path and dir of the whole circuit
'''
prj_path = "./"  # prj_path doesn't matter fro now, see control.py
prj_name = "chargepump"  # prj_name = Project name

# Design Variables
DX = [('q_llower', 3e-6, 9e-6, 1.0e-8, 6.9e-6, 'NO'),
      ('q_wlower', 1e-6, 4e-6, 1.0e-8, 3e-6, 'NO'),
      ('q_lupper', 1e-6, 2e-6, 1.0e-8, 1.5e-6, 'NO'),
      ('q_wupper', 5e-6, 20e-6, 1.0e-8, 10e-6, 'NO'),
      ('q_lc', 1e-6, 4e-6, 1.0e-8, 3e-6, 'NO'),
      ('q_wc', 5e-6, 20e-6, 1.0e-8, 10e-6, 'NO'),
      ('q_lref', 1e-6, 4e-6, 1.0e-8, 3e-6, 'NO'),
      ('q_wref', 5e-6, 20e-6, 1.0e-8, 6e-6, 'NO'),
      ('q_lq', 1e-6, 4e-6, 1.0e-8, 3e-6, 'NO'),
      ('q_wq', 5e-6, 20e-6, 1.0e-8, 6e-6, 'NO'),
      ('lpdbin', 0.55e-6, 2e-6, 1.0e-8, 1e-6, 'NO'),
      ('wpdbin', 4e-6, 15e-6, 1.0e-8, 10e-6, 'NO'),
      ('lpdin', 0.55e-6, 2e-6, 1.0e-8, 1e-6, 'NO'),
      ('wpdin', 2e-6, 6e-6, 1.0e-8, 3e-6, 'NO'),
      ('luumid', 0.6e-6, 1.2e-6, 1.0e-8, 1e-6, 'NO'),
      ('wuumid', 5e-6, 20e-6, 1.0e-8, 10e-6, 'NO'),
      ('lumid', 0.55e-6, 2e-6, 1.0e-8, 1e-6, 'NO'),
      ('wumid', 8e-6, 25e-6, 1.0e-8, 10e-6, 'NO'),
      ('lp4', 0.5e-6, 2e-6, 1.0e-8, 1e-6, 'NO'),
      ('wp4', 1e-6, 4e-6, 1.0e-8, 2e-6, 'NO'),
      ('ln4', 0.55e-6, 2e-6, 1.0e-8, 1e-6, 'NO'),
      ('wn4', 1e-6, 4e-6, 1.0e-8, 2e-6, 'NO'),
      ('lnsupp', 0.5e-6, 2e-6, 1.0e-8, 1e-6, 'NO'),
      ('wnsupp', 1e-6, 4e-6, 1.0e-8, 2e-6, 'NO'),
      ('lnsupp2', 0.8e-6, 2.4e-6, 1.0e-8, 1e-6, 'NO'),
      ('wnsupp2', 1e-6, 4e-6, 1.0e-8, 2e-6, 'NO'),
      ('li10', 2e-6, 10e-6, 1.0e-8, 3e-6, 'NO'),
      ('wi10', 0.8e-6, 4e-6, 1.0e-8, 2e-6, 'NO'),
      ('lb1', 2e-6, 10e-6, 1.0e-8, 3e-6, 'NO'),
      ('wb1', 5e-6, 25e-6, 1.0e-8, 10e-6, 'NO'),
      ('lb2', 2e-6, 8e-6, 1.0e-8, 3e-6, 'NO'),
      ('wb2', 1e-6, 3e-6, 1.0e-8, 2e-6, 'NO'),
      ('lb3', 0.5e-6, 2e-6, 1.0e-8, 1e-6, 'NO'),
      ('wb3', 1e-6, 6e-6, 1.0e-8, 2e-6, 'NO'),
      ('lb4', 0.55e-6, 3e-6, 1.0e-8, 1e-6, 'NO'),
      ('wb4', 4e-6, 16e-6, 1.0e-8, 8e-6, 'NO'),
      #('m',1,5,1,2,'NO'),
      #('m2',1,5,1,2,'NO')
     ]

# DX = [('name',L,U,step,init,[discrete list]),....] if there is no discrete, do not write

# setting
setting_1 = ['sim.sh', 'de_result.po',
[("diff1", "<", "20", None, None, "extract_diff1(file)", "10"),
 ("diff2", "<", "20", None, None, "extract_diff2(file)", "10"),
 ("diff3", "<", "5", None, None, "extract_diff3(file)", "10"),
 ("diff4", "<", "5", None, None, "extract_diff4(file)", "10"),
 ("deviation", "<", "5", None, None, "extract_deviation(file)", "10"),
 ("obj", None, None, None, None, "extract_obj(file)", None)
]
]
#setting_1 = ['test1.sp', [("pm",None, None, ">","90","extract_pm(file)", "5"),]]
# setting_x = ['test name',[(per1, </empty/<=,num1,>/empty/>=,num2, "extract function", weight),...]]
setting = [setting_1]
FOM = ['obj', "100", None]
#FOM = ['-pm', "-90", 10]
#FOM = ['-gain', "-95", None]
# FOM = ['fuc', num1/None, weight/None] :minimize FOM, if FOM has constraint, the form must be "FOM < num"

# extract performance from .lis
def extract_obj(file):
    pattern = re.compile(r'obj= *([\d.eE+\-]+)')
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            result = pattern.search(line)
            if result:
                return result.group(1)
        return "1e3"

def extract_diff1(file):
    pattern = re.compile(r'diff1= *([\d.eE+\-]+)')
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            result = pattern.search(line)
            if result:
                return result.group(1)
        return "1e3"

def extract_diff2(file):
    pattern = re.compile(r'diff2= *([\d.eE+\-]+)')
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            result = pattern.search(line)
            if result:
                return result.group(1)
        return "1e3"

def extract_diff3(file):
    pattern = re.compile(r'diff3= *([\d.eE+\-]+)')
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            result = pattern.search(line)
            if result:
                return result.group(1)
        return "1e3"

def extract_diff4(file):
    pattern = re.compile(r'diff4= *([\d.eE+\-]+)')
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            result = pattern.search(line)
            if result:
                return result.group(1)
        return "1e3"

def extract_deviation(file):
    pattern = re.compile(r'dev= *([\d.eE+\-]+)')
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            result = pattern.search(line)
            if result:
                return result.group(1)
        return "1e3"


# control prrameter
DEL_OUT_FOLDER = True
Mode = "Spice"
CPU_CORES = 1
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
    elif Algorithm in ("pyVTS"):
        totNum = Init_num + Max_eval
    return totNum
'''

# Algorithm
Algorithm = "turbo" 
##"Bayes", "BOc", "weibo_py", "DE", "GA", "PSO", "SQP", "bobyqa", "SA", "random", "turbo", "pycma", "bobyqa_py", "pyVTS"
Init_num = 20
Max_eval = 10
Tr_num = 1 # for trust region related algorithms
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
##"pyVTS": (2*dim, 100)
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
