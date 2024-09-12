import torch
from torch import Tensor
from tabulate import tabulate
import numpy as np

import os

PWD = os.getcwd()
SPICE_TESTBENCH_DIR = f'{PWD}/LDO/ldo_spice_testbench'
def ac(file_name):
    try:
        LDO_testbench_ac = open(f'{SPICE_TESTBENCH_DIR}/{file_name}', 'r')  
        lines_ac = LDO_testbench_ac.readlines()   
        freq = []  
        PSRR = []                     
        dcgain = []
        for line in lines_ac:
            Vac = line.split(' ')                
            Vac = [i for i in Vac if i != '']     
            freq.append(float(Vac[0]))            
            PSRR.append(float(Vac[1]))
            dcgain.append(float(Vac[3]))
        return freq, PSRR, dcgain
    except Exception as e:
        print(f"Simulation errors in file {file_name}: {e}")
        return [], [], []
            
def dc(file_name):
    try:
        LDO_testbench_dc = open(f'{SPICE_TESTBENCH_DIR}/{file_name}', 'r')
        lines_dc = LDO_testbench_dc.readlines()
        Vin_dc = []                    
        Vout_dc = []
        for line in lines_dc:
            Vdc = line.split(' ')
            Vdc = [i for i in Vdc if i != '']
            Vin_dc.append(float(Vdc[0]))
            Vout_dc.append(float(Vdc[1])) 
    
        dx = Vin_dc[1] - Vin_dc[0]
        dydx = np.gradient(Vout_dc, dx)      
            
        return Vin_dc, Vout_dc
    except Exception as e:
        print(f"Simulation errors in file {file_name}: {e}")
        return [], []

def LR_Power_vos(file_name):
    try:
        LDO_testbench_LR_Power_vos = open(f'{SPICE_TESTBENCH_DIR}/{file_name}', 'r')
        lines_dc = LDO_testbench_LR_Power_vos.readlines()
        IL = []                    
        LDR = []
        Power_maxload = []
        Power_minload = []
        vos_maxload = []
        vos_minload = []
        for line in lines_dc:
            Vdc = line.split(' ')
            Vdc = [i for i in Vdc if i != '']
            IL.append(float(Vdc[0]))
            LDR.append(float(Vdc[1])) 
            Power_maxload.append(float(Vdc[3])) 
            Power_minload.append(float(Vdc[5])) 
            vos_maxload.append(float(Vdc[7]))
            vos_minload.append(float(Vdc[9]))    
    
        return IL, LDR, Power_maxload, Power_minload, vos_maxload, vos_minload
    except Exception as e:
        print(f"Simulation errors in file {file_name}: {e}")
        return [], [], [], [], [], []
      
def tran(file_name):
    try:
        LDO_testbench_tran = open(f'{SPICE_TESTBENCH_DIR}/{file_name}', 'r')
        lines_tran = LDO_testbench_tran.readlines()
        time = []                     
        v_undershoot = []
        v_overshoot = []
        for line in lines_tran:
            line = line.split(' ')
            line = [i for i in line if i != '']
            time.append(float(line[0]))
            v_undershoot.append(float(line[1])) 
            v_overshoot.append(float(line[3])) 
            
        return time, v_undershoot, v_overshoot
    except Exception as e:
        print(f"Simulation errors in file {file_name}: {e}")
        return [], [], []

def get_info():
    '''Evaluate the performance'''
    ''' LNR at maxload '''
    dc_LNR_maxload = dc(file_name='LDO_TB_ACDC_LNR_maxload')
    LNR_maxload = dc_LNR_maxload[1][1]

    ''' LNR at minload '''
    dc_LNR_minload = dc(file_name='LDO_TB_ACDC_LNR_minload')
    LNR_minload = dc_LNR_minload[1][1]     

    ''' LDR '''
    dc_LR_Power_vos = LR_Power_vos(file_name='LDO_TB_ACDC_LR_Power_vos')
    LDR = dc_LR_Power_vos[1][1]
    Power_maxload = dc_LR_Power_vos[2][1]
    Power_minload = dc_LR_Power_vos[3][1]
    vos_maxload_1 = dc_LR_Power_vos[4][1]
    vos_minload_1 = dc_LR_Power_vos[5][1]
    vos_maxload = abs(vos_maxload_1)
    vos_minload = abs(vos_minload_1)
    
    ''' AC & Loop at maxload'''
    PSRR_dcgain_maxload = ac(file_name='LDO_TB_ACDC_PSRR_dcgain_maxload')
    PSRR_maxload = PSRR_dcgain_maxload[1][1]

    dcgain_maxload = PSRR_dcgain_maxload[2][1]
    if dcgain_maxload > 0 :
        try:
            GBW_PM_maxload = ac(file_name='LDO_TB_ACDC_GBW_PM_maxload')
            GBW_maxload = GBW_PM_maxload[1][1]
            phase_margin_maxload = GBW_PM_maxload[2][1]
        except: 
            if phase_margin_maxload > 180 or phase_margin_maxload < 0:
                phase_margin_maxload = 0
            else:
                phase_margin_maxload = phase_margin_maxload
    else :
        GBW_maxload = 0 
        phase_margin_maxload = 0
                

    ''' AC & Loop at minload'''
    PSRR_dcgain_minload = ac(file_name='LDO_TB_ACDC_PSRR_dcgain_minload')
    PSRR_minload = PSRR_dcgain_minload[1][1]

    dcgain_minload = PSRR_dcgain_minload[2][1]
    if dcgain_minload > 0 :
        try:
            GBW_PM_minload = ac(file_name='LDO_TB_ACDC_GBW_PM_minload')
            GBW_minload = GBW_PM_minload[1][1]
            phase_margin_minload = GBW_PM_minload[2][1]
        except: 
            if phase_margin_minload > 180 or phase_margin_minload < 0:
                phase_margin_minload = 0
            else:
                phase_margin_minload = phase_margin_minload
    else :
        GBW_minload = 0
        phase_margin_minload = 0

    ''' Tran test '''       
    tran_result = tran(file_name='LDO_TB_Tran_meas')
    v_undershoot = tran_result[1][1]
    v_overshoot = abs(tran_result[2][1])      

    print(tabulate(
        [
            ['LDR', LDR],
            ['LNR_maxload', LNR_maxload],
            ['LNR_minload', LNR_minload],
            ['Power_maxload', Power_maxload],
            ['Power_minload', Power_minload],
            ['vos_maxload', vos_maxload],
            ['vos_minload', vos_minload],

            ['PSRR_maxload (dB)', PSRR_maxload], 
            ['PSRR_minload (dB)', PSRR_minload],
            ['GBW_maxload', GBW_maxload],
            ['GBW_minload', GBW_minload],
            ['phase_margin_maxload (deg) ', phase_margin_maxload],
            ['phase_margin_minload (deg) ', phase_margin_minload],
        
            ['v_undershoot', v_undershoot], 
            ['v_overshoot', v_overshoot], 
            ],
        headers=['param', 'num', 'target'], tablefmt='orgtbl', numalign='right', floatfmt=".8f"
        ))
                    
    return {
            'LDR': LDR, 
            'LNR_maxload': LNR_maxload,
            'LNR_minload': LNR_minload, 
            'Power_maxload': Power_maxload,
            'Power_minload': Power_minload,
            'vos_maxload': vos_maxload,
            'vos_minload': vos_minload,

            'PSRR_maxload (dB)': PSRR_maxload,  
            'PSRR_minload (dB)': PSRR_minload, 
            'GBW_maxload': GBW_maxload, 
            'GBW_minload': GBW_minload, 
            'phase_margin_maxload (deg) ': phase_margin_maxload, 
            'phase_margin_minload (deg) ': phase_margin_minload, 
        
            'v_undershoot': v_undershoot, 
            'v_overshoot': v_overshoot, 
            }
if __name__ == '__main__':
    get_info()  

            
