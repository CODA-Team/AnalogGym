import torch
import numpy as np
import os
import math
import json
from tabulate import tabulate
import gymnasium as gym
from gymnasium import spaces

from ckt_graphs import GraphAMPNMCF
from dev_params import DeviceParams
from utils import ActionNormalizer, OutputParser2
from datetime import datetime

date = datetime.today().strftime('%Y-%m-%d')
PWD = os.getcwd()
SPICE_NETLIST_DIR = f'{PWD}/simulations'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

CktGraph = GraphAMPNMCF
            
class AMPNMCFEnv(gym.Env, CktGraph, DeviceParams):

    def __init__(self):
        gym.Env.__init__(self)
        CktGraph.__init__(self)
        DeviceParams.__init__(self, self.ckt_hierarchy)

        self.CktGraph = CktGraph()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.obs_shape, dtype=np.float64)
        self.action_space = spaces.Box(low=-1, high=1, shape=self.action_shape, dtype=np.float64)
        
    def _initialize_simulation(self):
        self.W_M0, self.L_M0, self.M_M0, \
            self.W_M8, self.L_M8, self.M_M8,\
            self.W_M10, self.L_M10, self.M_M10,\
            self.W_M11, self.L_M11, self.M_M11, \
            self.W_M17, self.L_M17, self.M_M17,\
            self.W_M21, self.L_M21, self.M_M21,\
            self.W_M23, self.L_M23, self.M_M23, \
            self.Ib,  \
            self.M_C0, \
            self.M_C1 = \
        np.array([3.3362978670411754 ,4.742308604047522, 15,          
                  4.248077067647905, 0.502195331105472, 11,
                  3.1794641651469213, 3.0358051647305397, 12,
                  1.6729651500310327, 2.5908412055522803, 336,
                  0.5878148470263471, 1.390856319070185, 14,
                  6.092117946109758, 0.5002226836950241, 4,
                  0.5019434139098478, 4.213849479423464, 44,
                  4.660878181441053e-06, 
                  10,
                  10])

        """Run the initial simulations."""  
        action = np.array([self.W_M0, self.L_M0, self.M_M0, \
                           self.W_M8, self.L_M8, self.M_M8,\
                           self.W_M10, self.L_M10, self.M_M10,\
                           self.W_M11, self.L_M11, self.M_M11,\
                           self.W_M17, self.L_M17, self.M_M17,\
                           self.W_M21, self.L_M21, self.M_M21,\
                           self.W_M23, self.L_M23, self.M_M23,\
                           self.Ib,  \
                           self.M_C0, \
                           self.M_C1])
        self.do_simulation(action)
        
    def _do_simulation(self, action: np.array):
        W_M0, L_M0, M_M0,\
        W_M8, L_M8, M_M8,\
        W_M10, L_M10, M_M10,\
        W_M11, L_M11, M_M11,\
        W_M17, L_M17, M_M17,\
        W_M21, L_M21, M_M21,\
        W_M23, L_M23, M_M23, \
        Ib, \
        M_C0, \
        M_C1 = action 
        
        M_M0 = int(M_M0)
        M_M8 = int(M_M8)
        M_M10 = int(M_M10)
        M_M11 = int(M_M11)
        M_M17 = int(M_M17)
        M_M21 = int(M_M21)
        M_M23 = int(M_M23)
        M_C0 = int(M_C0)
        M_C1 = int(M_C1)
        
        # update netlist
        try:
            # open the netlist of the testbench
            AMP_NMCF_vars = open(f'{SPICE_NETLIST_DIR}/AMP_NMCF_vars.spice', 'r')
            lines = AMP_NMCF_vars.readlines()
            lines[0] = f'.param MOSFET_0_8_W_BIASCM_PMOS={W_M0} MOSFET_0_8_L_BIASCM_PMOS={L_M0} MOSFET_0_8_M_BIASCM_PMOS={M_M0}\n'
            lines[1] = f'.param MOSFET_8_2_W_gm1_PMOS={W_M8} MOSFET_8_2_L_gm1_PMOS={L_M8} MOSFET_8_2_M_gm1_PMOS={M_M8}\n'
            lines[2] = f'.param MOSFET_10_1_W_gm2_PMOS={W_M10} MOSFET_10_1_L_gm2_PMOS={L_M10} MOSFET_10_1_M_gm2_PMOS={M_M10}\n'
            lines[3] = f'.param MOSFET_11_1_W_gmf2_PMOS={W_M11} MOSFET_11_1_L_gmf2_PMOS={L_M11} MOSFET_11_1_M_gmf2_PMOS={M_M11}\n'
            lines[4] = f'.param MOSFET_17_7_W_BIASCM_NMOS={W_M17} MOSFET_17_7_L_BIASCM_NMOS={L_M17} MOSFET_17_7_M_BIASCM_NMOS={M_M17}\n'
            lines[5] = f'.param MOSFET_21_2_W_LOAD2_NMOS={W_M21} MOSFET_21_2_L_LOAD2_NMOS={L_M21} MOSFET_21_2_M_LOAD2_NMOS={M_M21}\n'
            lines[6] = f'.param MOSFET_23_1_W_gm3_NMOS={W_M23} MOSFET_23_1_L_gm3_NMOS={L_M23} MOSFET_23_1_M_gm3_NMOS={M_M23}\n'
            lines[7] = f'.param CURRENT_0_BIAS={Ib}\n'
            lines[8] = f'.param M_C0={M_C0}\n'
            lines[9] = f'.param M_C1={M_C1}\n'
            
            AMP_NMCF_vars = open(f'{SPICE_NETLIST_DIR}/AMP_NMCF_vars.spice', 'w')
            AMP_NMCF_vars.writelines(lines)
            AMP_NMCF_vars.close()
            os.system(f'cd {SPICE_NETLIST_DIR}&& ngspice -b -o AMP_NMCF_ACDC.log AMP_NMCF_ACDC.cir')
            os.system(f'cd {SPICE_NETLIST_DIR}&& ngspice -b -o AMP_NMCF_Tran.log AMP_NMCF_Tran.cir')
            print('*** Simulations Done! ***')
        except:
            print('ERROR')

    def do_simulation(self, action):
        self._do_simulation(action)
        self.sim_results = OutputParser2(self.CktGraph)
        self.op_results = self.sim_results.dcop(file_name='AMP_NMCF_op')

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._initialize_simulation()
        observation = self._get_obs()
        info = self._get_info()
        return observation, info
    
    def close(self):
        return None
    
    def step(self, action):
        action = ActionNormalizer(action_space_low=self.action_space_low, action_space_high = \
                                       self.action_space_high).action(action) # convert [-1.1] range back to normal range
        action = action.astype(object)
        
        print(f"action: {action}")
        
        self.W_M0, self.L_M0, self.M_M0,\
        self.W_M8, self.L_M8, self.M_M8,\
        self.W_M10, self.L_M10, self.M_M10,\
        self.W_M11, self.L_M11, self.M_M11,\
        self.W_M17, self.L_M17, self.M_M17,\
        self.W_M21, self.L_M21, self.M_M21,\
        self.W_M23, self.L_M23, self.M_M23, \
        self.Ib,  \
        self.M_C0, \
        self.M_C1 = action        
        
        ''' run simulations '''
        self.do_simulation(action)
        
        '''get observation'''
        observation = self._get_obs()
        info = self._get_info()

        reward = self.reward
        
        if reward >= 0:
            terminated = True
        else:
            terminated = False
          
        print(tabulate(
            [
                ['TC', self.TC, self.TC_target],
                ['Power', self.Power, self.Power_target],
                ['vos', self.vos, self.vos_target],
                ['cmrrdc', self.cmrrdc, self.cmrrdc_target],
                ['dcgain', self.dcgain, self.dcgain_target],

                ['GBW', self.GBW, self.GBW_target],
                ['phase_margin (deg)', self.phase_margin, self.phase_margin_target],
                ['PSRP', self.PSRP, self.PSRP_target],
                ['PSRN', self.PSRN, self.PSRN_target],

                ['sr', self.sr, self.sr_target],
                ['settlingTime', self.settlingTime, self.settlingTime_target], 

                ['TC score', self.TC_score, ''],
                ['Power score', self.Power_score, ''],
                ['vos score', self.vos_score, ''],
                ['cmrrdc score', self.cmrrdc_score, ''],
                ['dcgain score', self.dcgain_score, ''],

                ['GBW score', self.GBW_score, ''],
                ['phase_margin (deg) score', self.phase_margin_score, ''],
                ['PSRP score', self.PSRP_score, ''],
                ['PSRN score', self.PSRN_score, ''],

                ['sr score', self.sr_score, ''],
                ['settlingTime score', self.settlingTime_score,''],
 
                ['Reward', reward, '']
                ],
            headers=['param', 'num', 'target'], tablefmt='orgtbl', numalign='right', floatfmt=".8f"
            ))

        return observation, reward, terminated, False, info
        
    def _get_obs(self):
        # pick some .OP params from the dict:
        try:
            f = open(f'{SPICE_NETLIST_DIR}/AMP_NMCF_op_mean_std.json')
            self.op_mean_std = json.load(f)
            self.op_mean = self.op_mean_std['OP_M_mean']
            self.op_std = self.op_mean_std['OP_M_std']
            self.op_mean = np.array([self.op_mean['id'], self.op_mean['gm'], self.op_mean['gds'], self.op_mean['vth'], self.op_mean['vdsat'], self.op_mean['vds'], self.op_mean['vgs']])
            self.op_std = np.array([self.op_std['id'], self.op_std['gm'], self.op_std['gds'], self.op_std['vth'], self.op_std['vdsat'], self.op_std['vds'], self.op_std['vgs']])
        except:
            print('You need to run <_random_op_sims> to generate mean and std for transistor .OP parameters')
        
        self.OP_M0 = self.op_results['M0']
        self.OP_M0_norm = (np.array([self.OP_M0['id'],
                                self.OP_M0['gm'],
                                self.OP_M0['gds'],
                                self.OP_M0['vth'],
                                self.OP_M0['vdsat'],
                                self.OP_M0['vds'],
                                self.OP_M0['vgs']
                                ]) - self.op_mean)/self.op_std
        self.OP_M1 = self.op_results['M1']
        self.OP_M1_norm = (np.array([self.OP_M1['id'],
                                self.OP_M1['gm'],
                                self.OP_M1['gds'],
                                self.OP_M1['vth'],
                                self.OP_M1['vdsat'],
                                self.OP_M1['vds'],
                                self.OP_M1['vgs']
                                ]) - self.op_mean)/self.op_std
        self.OP_M2 = self.op_results['M2']
        self.OP_M2_norm = (np.array([self.OP_M2['id'],
                                self.OP_M2['gm'],
                                self.OP_M2['gds'],
                                self.OP_M2['vth'],
                                self.OP_M2['vdsat'],
                                self.OP_M2['vds'],
                                self.OP_M2['vgs']
                                ]) - self.op_mean)/self.op_std
        self.OP_M3 = self.op_results['M3']
        self.OP_M3_norm = (np.abs([self.OP_M3['id'],
                                self.OP_M3['gm'],
                                self.OP_M3['gds'],
                                self.OP_M3['vth'],
                                self.OP_M3['vdsat'],
                                self.OP_M3['vds'],
                                self.OP_M3['vgs']
                                ]) - self.op_mean)/self.op_std
        self.OP_M4 = self.op_results['M4']
        self.OP_M4_norm = (np.abs([self.OP_M4['id'],
                                self.OP_M4['gm'],
                                self.OP_M4['gds'],
                                self.OP_M4['vth'],
                                self.OP_M4['vdsat'],
                                self.OP_M4['vds'],
                                self.OP_M4['vgs']
                                ]) - self.op_mean)/self.op_std
        self.OP_M5 = self.op_results['M5']
        self.OP_M5_norm = (np.abs([self.OP_M5['id'],
                                self.OP_M5['gm'],
                                self.OP_M5['gds'],
                                self.OP_M5['vth'],
                                self.OP_M5['vdsat'],
                                self.OP_M5['vds'],
                                self.OP_M5['vgs']
                                ]) - self.op_mean)/self.op_std
        self.OP_M6 = self.op_results['M6']
        self.OP_M6_norm = (np.array([self.OP_M6['id'],
                                self.OP_M6['gm'],
                                self.OP_M6['gds'],
                                self.OP_M6['vth'],
                                self.OP_M6['vdsat'],
                                self.OP_M6['vds'],
                                self.OP_M6['vgs']
                                ]) - self.op_mean)/self.op_std
        self.OP_M7 = self.op_results['M7']
        self.OP_M7_norm = (np.array([self.OP_M7['id'],
                                self.OP_M7['gm'],
                                self.OP_M7['gds'],
                                self.OP_M7['vth'],
                                self.OP_M7['vdsat'],
                                self.OP_M7['vds'],
                                self.OP_M7['vgs']
                                ]) - self.op_mean)/self.op_std
        self.OP_M8 = self.op_results['M8']
        self.OP_M8_norm = (np.array([self.OP_M8['id'],
                                self.OP_M8['gm'],
                                self.OP_M8['gds'],
                                self.OP_M8['vth'],
                                self.OP_M8['vdsat'],
                                self.OP_M8['vds'],
                                self.OP_M8['vgs']
                                ]) - self.op_mean)/self.op_std
        self.OP_M9 = self.op_results['M9']
        self.OP_M9_norm = (np.array([self.OP_M9['id'],
                                self.OP_M9['gm'],
                                self.OP_M9['gds'],
                                self.OP_M9['vth'],
                                self.OP_M9['vdsat'],
                                self.OP_M9['vds'],
                                self.OP_M9['vgs']
                                ]) - self.op_mean)/self.op_std
        self.OP_M10 = self.op_results['M10']
        self.OP_M10_norm = (np.array([self.OP_M10['id'],
                                self.OP_M10['gm'],
                                self.OP_M10['gds'],
                                self.OP_M10['vth'],
                                self.OP_M10['vdsat'],
                                self.OP_M10['vds'],
                                self.OP_M10['vgs']
                                ]) - self.op_mean)/self.op_std
        self.OP_M11 = self.op_results['M11']
        self.OP_M11_norm = (np.array([self.OP_M11['id'],
                                self.OP_M11['gm'],
                                self.OP_M11['gds'],
                                self.OP_M11['vth'],
                                self.OP_M11['vdsat'],
                                self.OP_M11['vds'],
                                self.OP_M11['vgs']
                                ]) - self.op_mean)/self.op_std
        self.OP_M12 = self.op_results['M12']
        self.OP_M12_norm = (np.array([self.OP_M12['id'],
                                self.OP_M12['gm'],
                                self.OP_M12['gds'],
                                self.OP_M12['vth'],
                                self.OP_M12['vdsat'],
                                self.OP_M12['vds'],
                                self.OP_M12['vgs']
                                ]) - self.op_mean)/self.op_std
        self.OP_M13 = self.op_results['M13']
        self.OP_M13_norm = (np.array([self.OP_M13['id'],
                                self.OP_M13['gm'],
                                self.OP_M13['gds'],
                                self.OP_M13['vth'],
                                self.OP_M13['vdsat'],
                                self.OP_M13['vds'],
                                self.OP_M13['vgs']
                                ]) - self.op_mean)/self.op_std
        self.OP_M14 = self.op_results['M14']
        self.OP_M14_norm = (np.array([self.OP_M14['id'],
                                self.OP_M14['gm'],
                                self.OP_M14['gds'],
                                self.OP_M14['vth'],
                                self.OP_M14['vdsat'],
                                self.OP_M14['vds'],
                                self.OP_M14['vgs']
                                ]) - self.op_mean)/self.op_std
        self.OP_M15 = self.op_results['M15']
        self.OP_M15_norm = (np.array([self.OP_M15['id'],
                                self.OP_M15['gm'],
                                self.OP_M15['gds'],
                                self.OP_M15['vth'],
                                self.OP_M15['vdsat'],
                                self.OP_M15['vds'],
                                self.OP_M15['vgs']
                                ]) - self.op_mean)/self.op_std
        self.OP_M16 = self.op_results['M16']
        self.OP_M16_norm = (np.array([self.OP_M16['id'],
                                self.OP_M16['gm'],
                                self.OP_M16['gds'],
                                self.OP_M16['vth'],
                                self.OP_M16['vdsat'],
                                self.OP_M16['vds'],
                                self.OP_M16['vgs']
                                ]) - self.op_mean)/self.op_std
        self.OP_M17 = self.op_results['M17']
        self.OP_M17_norm = (np.array([self.OP_M17['id'],
                                self.OP_M17['gm'],
                                self.OP_M17['gds'],
                                self.OP_M17['vth'],
                                self.OP_M17['vdsat'],
                                self.OP_M17['vds'],
                                self.OP_M17['vgs']
                                ]) - self.op_mean)/self.op_std
        self.OP_M18 = self.op_results['M18']
        self.OP_M18_norm = (np.array([self.OP_M18['id'],
                                self.OP_M18['gm'],
                                self.OP_M18['gds'],
                                self.OP_M18['vth'],
                                self.OP_M18['vdsat'],
                                self.OP_M18['vds'],
                                self.OP_M18['vgs']
                                ]) - self.op_mean)/self.op_std
        self.OP_M19 = self.op_results['M19']
        self.OP_M19_norm = (np.array([self.OP_M19['id'],
                                self.OP_M19['gm'],
                                self.OP_M19['gds'],
                                self.OP_M19['vth'],
                                self.OP_M19['vdsat'],
                                self.OP_M19['vds'],
                                self.OP_M19['vgs']
                                ]) - self.op_mean)/self.op_std
        self.OP_M20 = self.op_results['M20']
        self.OP_M20_norm = (np.array([self.OP_M20['id'],
                                self.OP_M20['gm'],
                                self.OP_M20['gds'],
                                self.OP_M20['vth'],
                                self.OP_M20['vdsat'],
                                self.OP_M20['vds'],
                                self.OP_M20['vgs']
                                ]) - self.op_mean)/self.op_std
        self.OP_M21 = self.op_results['M21']
        self.OP_M21_norm = (np.array([self.OP_M21['id'],
                                self.OP_M21['gm'],
                                self.OP_M21['gds'],
                                self.OP_M21['vth'],
                                self.OP_M21['vdsat'],
                                self.OP_M21['vds'],
                                self.OP_M21['vgs']
                                ]) - self.op_mean)/self.op_std
        self.OP_M22 = self.op_results['M22']
        self.OP_M22_norm = (np.array([self.OP_M22['id'],
                                self.OP_M22['gm'],
                                self.OP_M22['gds'],
                                self.OP_M22['vth'],
                                self.OP_M22['vdsat'],
                                self.OP_M22['vds'],
                                self.OP_M22['vgs']
                                ]) - self.op_mean)/self.op_std
        self.OP_M23 = self.op_results['M23']
        self.OP_M23_norm = (np.array([self.OP_M23['id'],
                                self.OP_M23['gm'],
                                self.OP_M23['gds'],
                                self.OP_M23['vth'],
                                self.OP_M23['vdsat'],
                                self.OP_M23['vds'],
                                self.OP_M23['vgs']
                                ]) - self.op_mean)/self.op_std
        # it is not straightforward to extract resistance info from sky130 resistor, using the following approximation instead
        # normalize all passive components
        self.OP_C0_norm = ActionNormalizer(action_space_low=self.C0_low, action_space_high=self.C0_high).reverse_action(self.op_results['C0']['c']) # convert to (-1, 1)
        self.OP_C1_norm = ActionNormalizer(action_space_low=self.C1_low, action_space_high=self.C1_high).reverse_action(self.op_results['C1']['c']) # convert to (-1, 1)
        
        # state shall be in the order of node (node0, node1, ...)
        observation = np.array([
                               [0,0,0,0,      0,      self.OP_M0_norm[0],self.OP_M0_norm[1],self.OP_M0_norm[2],self.OP_M0_norm[3],self.OP_M0_norm[4],self.OP_M0_norm[5],self.OP_M0_norm[6]],
                               [0,0,0,0,      0,     self.OP_M1_norm[0],self.OP_M1_norm[1],self.OP_M1_norm[2],self.OP_M1_norm[3],self.OP_M1_norm[4],self.OP_M1_norm[5],self.OP_M1_norm[6]],
                               [0,0,0,0,      0,      self.OP_M2_norm[0],self.OP_M2_norm[1],self.OP_M2_norm[2],self.OP_M2_norm[3],self.OP_M2_norm[4],self.OP_M2_norm[5],self.OP_M2_norm[6]],
                               [0,0,0,0,      0,      self.OP_M3_norm[0],self.OP_M3_norm[1],self.OP_M3_norm[2],self.OP_M3_norm[3],self.OP_M3_norm[4],self.OP_M3_norm[5],self.OP_M3_norm[6]],
                               [0,0,0,0,      0,      self.OP_M4_norm[0],self.OP_M4_norm[1],self.OP_M4_norm[2],self.OP_M4_norm[3],self.OP_M4_norm[4],self.OP_M4_norm[5],self.OP_M4_norm[6]],
                               [0,0,0,0,      0,      self.OP_M5_norm[0],self.OP_M5_norm[1],self.OP_M5_norm[2],self.OP_M5_norm[3],self.OP_M5_norm[4],self.OP_M5_norm[5],self.OP_M5_norm[6]],
                               [0,0,0,0,      0,      self.OP_M6_norm[0],self.OP_M6_norm[1],self.OP_M6_norm[2],self.OP_M6_norm[3],self.OP_M6_norm[4],self.OP_M6_norm[5],self.OP_M6_norm[6]],
                               [0,0,0,0,      0,      self.OP_M7_norm[0],self.OP_M7_norm[1],self.OP_M7_norm[2],self.OP_M7_norm[3],self.OP_M7_norm[4],self.OP_M7_norm[5],self.OP_M7_norm[6]],
                               [0,0,0,0,      0,      self.OP_M8_norm[0],self.OP_M8_norm[1],self.OP_M8_norm[2],self.OP_M8_norm[3],self.OP_M8_norm[4],self.OP_M8_norm[5],self.OP_M8_norm[6]],
                               [0,0,0,0,      0,      self.OP_M9_norm[0],self.OP_M9_norm[1],self.OP_M9_norm[2],self.OP_M9_norm[3],self.OP_M9_norm[4],self.OP_M9_norm[5],self.OP_M9_norm[6]],
                               [0,0,0,0,      0,      self.OP_M10_norm[0],self.OP_M10_norm[1],self.OP_M10_norm[2],self.OP_M10_norm[3],self.OP_M10_norm[4],self.OP_M10_norm[5],self.OP_M10_norm[6]],
                               [0,0,0,0,      0,      self.OP_M11_norm[0],self.OP_M11_norm[1],self.OP_M11_norm[2],self.OP_M11_norm[3],self.OP_M11_norm[4],self.OP_M11_norm[5],self.OP_M11_norm[6]],
                               [0,0,0,0,      0,      self.OP_M12_norm[0],self.OP_M12_norm[1],self.OP_M12_norm[2],self.OP_M12_norm[3],self.OP_M12_norm[4],self.OP_M12_norm[5],self.OP_M12_norm[6]],
                               [0,0,0,0,      0,      self.OP_M13_norm[0],self.OP_M13_norm[1],self.OP_M13_norm[2],self.OP_M13_norm[3],self.OP_M13_norm[4],self.OP_M13_norm[5],self.OP_M13_norm[6]],
                               [0,0,0,0,      0,      self.OP_M14_norm[0],self.OP_M14_norm[1],self.OP_M14_norm[2],self.OP_M14_norm[3],self.OP_M14_norm[4],self.OP_M14_norm[5],self.OP_M14_norm[6]],
                               [0,0,0,0,      0,      self.OP_M15_norm[0],self.OP_M15_norm[1],self.OP_M15_norm[2],self.OP_M15_norm[3],self.OP_M15_norm[4],self.OP_M15_norm[5],self.OP_M15_norm[6]],
                               [0,0,0,0,      0,      self.OP_M16_norm[0],self.OP_M16_norm[1],self.OP_M16_norm[2],self.OP_M16_norm[3],self.OP_M16_norm[4],self.OP_M16_norm[5],self.OP_M16_norm[6]],
                               [0,0,0,0,      0,      self.OP_M17_norm[0],self.OP_M17_norm[1],self.OP_M17_norm[2],self.OP_M17_norm[3],self.OP_M17_norm[4],self.OP_M17_norm[5],self.OP_M17_norm[6]],     
                               [0,0,0,0,      0,      self.OP_M18_norm[0],self.OP_M18_norm[1],self.OP_M18_norm[2],self.OP_M18_norm[3],self.OP_M18_norm[4],self.OP_M18_norm[5],self.OP_M18_norm[6]],
                               [0,0,0,0,      0,      self.OP_M19_norm[0],self.OP_M19_norm[1],self.OP_M19_norm[2],self.OP_M19_norm[3],self.OP_M19_norm[4],self.OP_M19_norm[5],self.OP_M19_norm[6]],
                               [0,0,0,0,      0,      self.OP_M20_norm[0],self.OP_M20_norm[1],self.OP_M20_norm[2],self.OP_M20_norm[3],self.OP_M20_norm[4],self.OP_M20_norm[5],self.OP_M20_norm[6]],
                               [0,0,0,0,      0,      self.OP_M21_norm[0],self.OP_M21_norm[1],self.OP_M21_norm[2],self.OP_M21_norm[3],self.OP_M21_norm[4],self.OP_M21_norm[5],self.OP_M21_norm[6]],
                               [0,0,0,0,      0,      self.OP_M22_norm[0],self.OP_M22_norm[1],self.OP_M22_norm[2],self.OP_M22_norm[3],self.OP_M22_norm[4],self.OP_M22_norm[5],self.OP_M22_norm[6]],
                               [0,0,0,0,      0,      self.OP_M23_norm[0],self.OP_M23_norm[1],self.OP_M23_norm[2],self.OP_M23_norm[3],self.OP_M23_norm[4],self.OP_M23_norm[5],self.OP_M23_norm[6]],
                               
                               [self.Vdd,0,0,0,0,      0,0,0,0,0,0,0],
                               [0,self.GND,0,0,0,      0,0,0,0,0,0,0],
                               [0,0,self.Ib,0,0,       0,0,0,0,0,0,0],
                               [0,0,0,self.OP_C0_norm,0,    0,0,0,0,0,0,0],        
                               [0,0,0,0,self.OP_C1_norm,       0,0,0,0,0,0,0],                              
                               ])
        # clip the obs for better regularization
        observation = np.clip(observation, -5, 5)
        
        return observation 

    def _get_info(self):
        '''Evaluate the performance'''
        ''' DC '''
        self.dc_results = self.sim_results.dc(file_name='AMP_NMCF_ACDC_DC')
        self.TC = self.dc_results[1][1]
        self.Power = self.dc_results[2][1]
        self.vos_1 = self.dc_results[3][1]
        self.vos = abs(self.vos_1)
             
        self.TC_score = np.min([(self.TC_target - self.TC) / (self.TC_target + self.TC), 0])
        self.Power_score = np.min([(self.Power_target - self.Power) / (self.Power_target + self.Power), 0])
        self.vos_score = np.min([(self.vos_target - self.vos) / (self.vos_target + self.vos), 0])

        ''' AC '''
        self.ac_results = self.sim_results.ac(file_name='AMP_NMCF_ACDC_AC')
        self.cmrrdc = self.ac_results[1][1]
        if self.cmrrdc > 0 :
            self.cmrrdc_score = -1
        else : 
            self.cmrrdc_score = np.min([(self.cmrrdc - self.cmrrdc_target) / (self.cmrrdc + self.cmrrdc_target), 0])
            if self.cmrrdc < self.cmrrdc_target:
                self.cmrrdc_score = 0

        self.PSRP = self.ac_results[2][1]
        if self.PSRP > 0 :
            self.PSRP_score = -1
        else : 
            self.PSRP_score = np.min([(self.PSRP - self.PSRP_target) / (self.PSRP + self.PSRP_target), 0])
            if self.PSRP < self.PSRP_target:
                self.PSRP_score = 0

        self.PSRN = self.ac_results[3][1]
        if self.PSRN > 0 :
            self.PSRN_score = -1
        else : 
            self.PSRN_score = np.min([(self.PSRN - self.PSRN_target) / (self.PSRN + self.PSRN_target), 0])
            if self.PSRN < self.PSRN_target:
                self.PSRN_score = 0

        self.dcgain = self.ac_results[4][1]
        if self.dcgain > 0 :
            try:
                self.dcgain_score = np.min([(self.dcgain - self.dcgain_target) / (self.dcgain + self.dcgain_target), 0])
                self.GBW_PM_results = self.sim_results.GBW_PM(file_name='AMP_NMCF_ACDC_GBW_PM')
                self.GBW = self.GBW_PM_results[1][1]
                self.GBW_score = np.min([(self.GBW - self.GBW_target) / (self.GBW + self.GBW_target), 0])
                self.phase_margin = self.GBW_PM_results[2][1]
                self.phase_margin_score = np.min([(self.phase_margin - self.phase_margin_target) / (self.phase_margin + self.phase_margin_target), 0])
            except: 
                if self.phase_margin > 180 or self.phase_margin < 0:
                    self.phase_margin = 0
                else:
                    self.phase_margin = self.phase_margin
        else :
            self.dcgain_score = -1
            self.GBW = 0
            self.GBW_score = -1
            self.phase_margin = 0
            self.phase_margin_score = -1
      
        """ Tran """
        self.tran_results = self.sim_results.tran(file_name='AMP_NMCF_Tran')
        self.sr_rise = self.tran_results[1][1]
        self.sr_fall = self.tran_results[2][1]
        self.sr = (self.sr_rise + self.sr_fall) / 2 
        self.sr_score = np.min([(self.sr - self.sr_target) / (self.sr + self.sr_target), 0])

        """ setting_time """
        self.meas = {}
        self.d0 = 0.01
        # path = './benchmarks/TB_Amplifier_ACDC/'
        self.time_data, self.vin_data, self.vout_data = self.sim_results.extract_tran_data(file_name='tran.dat')
        if self.time_data is None:
            return None
        self.d0_settle, self.d1_settle, self.d2_settle, self.stable, self.SR_p, self.settling_time_p, self.SR_n, self.settling_time_n = self.sim_results.analyze_amplifier_performance(self.vin_data, self.vout_data, self.time_data, self.d0)
        self.d0_settle = abs(self.d0_settle)
        self.d1_settle = abs(self.d1_settle)
        self.d2_settle = abs(self.d2_settle)
        self.SR_n = abs(self.SR_n)
        self.SR_p = abs(self.SR_p)
        self.settlingTime_p = abs(self.settling_time_p)
        self.settlingTime_n = abs(self.settling_time_n)
    
        if math.isnan(self.d0_settle):
            self.d0_settle = 10
    
        if math.isnan(self.d1_settle) or math.isnan(self.d2_settle) :
            if math.isnan(self.d1_settle):
                self.d0_settle += 10
            if math.isnan(self.d2_settle):
                self.d0_settle += 10
            self.d_settle = self.d0_settle
        else:
            self.d_settle = max(self.d0_settle, self.d1_settle, self.d2_settle)
    
        if math.isnan(self.SR_p) or math.isnan(self.SR_n) :
            self.SR = -self.d_settle
        else:
            self.SR = min(self.SR_p,self.SR_n)
    
        if math.isnan(self.settlingTime_p) or math.isnan(self.settlingTime_n) :
            self.settlingTime = self.d_settle
        else:
            self.settlingTime = max(self.settlingTime_p, self.settlingTime_n)
        
        self.meas['d_settle'] = self.d_settle
        self.meas['SR'] = self.SR
        self.meas['settlingTime'] = self.settlingTime
        self.settlingTime_score = np.min([(self.settlingTime_target - self.settlingTime) / (self.settlingTime_target + self.settlingTime), 0])

    
        """ Total reward """
        self.reward = self.TC_score + self.Power_score + self.vos_score + self.cmrrdc_score + \
                      self.dcgain_score + self.GBW_score + self.phase_margin_score + self.PSRP_score + \
                      self.PSRN_score + self.sr_score + self.settlingTime_score 
                                        
        return {
                'TC': self.TC, 
                'Power': self.Power, 
                'vos': self.vos, 
                'cmrrdc': self.cmrrdc, 
                'dcgain': self.dcgain, 

                'GBW': self.GBW, 
                'phase_margin (deg)': self.phase_margin, 
                'PSRP': self.PSRP, 
                'PSRN': self.PSRN, 

                'sr': self.sr, 
                'setting_time':self.settlingTime
            }


    def _init_random_sim(self, max_sims=100):
        '''
        
        This is NOT the same as the random step in the agent, here is basically 
        doing some completely random design variables selection for generating
        some device parameters for calculating the mean and variance for each
        .OP device parameters (getting a statistical idea of, how each ckt parameter's range is like'), 
        so that you can do the normalization for the state representations later.
    
        '''
        random_op_count = 0
        OP_M_lists = []
        OP_R_lists = []
        OP_C_lists = []
        OP_V_lists = []
        OP_I_lists = []
        
        while random_op_count <= max_sims :
            print(f'* simulation #{random_op_count} *')
            action = np.random.uniform(self.action_space_low, self.action_space_high, self.action_dim) 
            print(f'action: {action}')
            self._do_simulation(action)
    
            sim_results = OutputParser2(self.CktGraph)
            op_results = sim_results.dcop(file_name='AMP_NMCF_op')
            
            OP_M_list = []
            OP_R_list = []
            OP_C_list = []
            OP_V_list = []
            OP_I_list = []

            for key in list(op_results):
                if key[0] == 'M' or key[0] == 'm':
                    OP_M = np.array([op_results[key][f'{item}'] for item in list(op_results[key])])    
                    OP_M_list.append(OP_M)
                elif key[0] == 'R' or key[0] == 'r':
                    OP_R = np.array([op_results[key][f'{item}'] for item in list(op_results[key])])    
                    OP_R_list.append(OP_R)
                elif key[0] == 'C' or key[0] == 'c':
                    OP_C = np.array([op_results[key][f'{item}'] for item in list(op_results[key])])    
                    OP_C_list.append(OP_C)   
                elif key[0] == 'V' or key[0] == 'v':
                    OP_V = np.array([op_results[key][f'{item}'] for item in list(op_results[key])])    
                    OP_V_list.append(OP_V)
                elif key[0] == 'I' or key[0] == 'i':
                    OP_I = np.array([op_results[key][f'{item}'] for item in list(op_results[key])])    
                    OP_I_list.append(OP_I)   
                else:
                    None
                    
            OP_M_list = np.array(OP_M_list)
            OP_R_list = np.array(OP_R_list)
            OP_C_list = np.array(OP_C_list)
            OP_V_list = np.array(OP_V_list)
            OP_I_list = np.array(OP_I_list)
                        
            OP_M_lists.append(OP_M_list)
            OP_R_lists.append(OP_R_list)
            OP_C_lists.append(OP_C_list)
            OP_V_lists.append(OP_V_list)
            OP_I_lists.append(OP_I_list)
            
            random_op_count = random_op_count + 1

        OP_M_lists = np.array(OP_M_lists)
        OP_R_lists = np.array(OP_R_lists)
        OP_C_lists = np.array(OP_C_lists)
        OP_V_lists = np.array(OP_V_lists)
        OP_I_lists = np.array(OP_I_lists)
        
        if OP_M_lists.size != 0:
            OP_M_mean = np.mean(OP_M_lists.reshape(-1, OP_M_lists.shape[-1]), axis=0)
            OP_M_std = np.std(OP_M_lists.reshape(-1, OP_M_lists.shape[-1]),axis=0)
            OP_M_mean_dict = {}
            OP_M_std_dict = {}
            for idx, key in enumerate(self.params_mos):
                OP_M_mean_dict[key] = OP_M_mean[idx]
                OP_M_std_dict[key] = OP_M_std[idx]
        
        if OP_R_lists.size != 0:
            OP_R_mean = np.mean(OP_R_lists.reshape(-1, OP_R_lists.shape[-1]), axis=0)
            OP_R_std = np.std(OP_R_lists.reshape(-1, OP_R_lists.shape[-1]),axis=0)
            OP_R_mean_dict = {}
            OP_R_std_dict = {}
            for idx, key in enumerate(self. params_r):
                OP_R_mean_dict[key] = OP_R_mean[idx]
                OP_R_std_dict[key] = OP_R_std[idx]
                
        if OP_C_lists.size != 0:
            OP_C_mean = np.mean(OP_C_lists.reshape(-1, OP_C_lists.shape[-1]), axis=0)
            OP_C_std = np.std(OP_C_lists.reshape(-1, OP_C_lists.shape[-1]),axis=0)
            OP_C_mean_dict = {}
            OP_C_std_dict = {}
            for idx, key in enumerate(self.params_c):
                OP_C_mean_dict[key] = OP_C_mean[idx]
                OP_C_std_dict[key] = OP_C_std[idx]     
                
        if OP_V_lists.size != 0:
            OP_V_mean = np.mean(OP_V_lists.reshape(-1, OP_V_lists.shape[-1]), axis=0)
            OP_V_std = np.std(OP_V_lists.reshape(-1, OP_V_lists.shape[-1]),axis=0)
            OP_V_mean_dict = {}
            OP_V_std_dict = {}
            for idx, key in enumerate(self.params_v):
                OP_V_mean_dict[key] = OP_V_mean[idx]
                OP_V_std_dict[key] = OP_V_std[idx]
        
        if OP_I_lists.size != 0:
            OP_I_mean = np.mean(OP_I_lists.reshape(-1, OP_I_lists.shape[-1]), axis=0)
            OP_I_std = np.std(OP_I_lists.reshape(-1, OP_I_lists.shape[-1]),axis=0)
            OP_I_mean_dict = {}
            OP_I_std_dict = {}
            for idx, key in enumerate(self.params_i):
                OP_I_mean_dict[key] = OP_I_mean[idx]
                OP_I_std_dict[key] = OP_I_std[idx]

        self.OP_M_mean_std = {
            'OP_M_mean': OP_M_mean_dict,         
            'OP_M_std': OP_M_std_dict
            }

        with open(f'{SPICE_NETLIST_DIR}/AMP_NMCF_op_mean_std.json','w') as file:
            json.dump(self.OP_M_mean_std, file)