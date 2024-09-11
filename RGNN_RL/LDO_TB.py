import torch
import numpy as np
import os
import json
from tabulate import tabulate
import gymnasium as gym
from gymnasium import spaces

from ckt_graphs import GraphLDOtestbench
from dev_params import DeviceParams
from utils import ActionNormalizer, OutputParser
from datetime import datetime

date = datetime.today().strftime('%Y-%m-%d')
PWD = os.getcwd()
SPICE_NETLIST_DIR = f'{PWD}/simulations'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

CktGraph = GraphLDOtestbench
            
class LDOtestbenchEnv(gym.Env, CktGraph, DeviceParams):

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
        self.W_power, self.L_power, self.M_power, \
        self.W_M17, self.L_M17, self.M_M17,\
        self.W_M21, self.L_M21, self.M_M21,\
        self.Ib,  \
        self.M_C0, \
        self.M_CL = \
        np.array([4.7731718389796995 ,1.0036600464756384, 3,          
                  5.479611261286143, 3.6591292755218054, 8,
                  8.366790237867656, 1.0854081932306854, 7,
                  8.967108977628818, 0.15324273949718603, 928,
                  1.4406027822407284, 4.139558701381334, 10,
                  6.085375847752627, 1.88747423370982, 8,
                  3e-6, 
                  34,
                  253])

        """Run the initial simulations."""  
        action = np.array([self.W_M0, self.L_M0, self.M_M0, \
                           self.W_M8, self.L_M8, self.M_M8,\
                           self.W_M10, self.L_M10, self.M_M10,\
                           self.W_power, self.L_power, self.M_power, \
                           self.W_M17, self.L_M17, self.M_M17,\
                           self.W_M21, self.L_M21, self.M_M21,\
                           self.Ib,  \
                           self.M_C0, \
                           self.M_CL])
        self.do_simulation(action)
        
    def _do_simulation(self, action: np.array):
        W_M0, L_M0, M_M0,\
        W_M8, L_M8, M_M8,\
        W_M10, L_M10, M_M10,\
        W_power, L_power, M_power, \
        W_M17, L_M17, M_M17,\
        W_M21, L_M21, M_M21,\
        Ib, \
        M_C0, \
        M_CL = action 
        
        M_M0 = int(M_M0)
        M_M8 = int(M_M8)
        M_M10 = int(M_M10)
        M_power = int(M_power)
        M_M17 = int(M_M17)
        M_M21 = int(M_M21)
        M_C0 = int(M_C0)
        M_CL = int(M_CL)
        
        # update netlist
        try:
            # open the netlist of the testbench
            LDO_testbench_vars = open(f'{SPICE_NETLIST_DIR}/LDO_TB_vars.spice', 'r')
            lines = LDO_testbench_vars.readlines()
            
            lines[0] = f'.param mosfet_0_8_w_biascm_pmos={W_M0} mosfet_0_8_l_biascm_pmos={L_M0} mosfet_0_8_m_biascm_pmos={M_M0}\n'
            lines[1] = f'.param mosfet_8_2_w_gm1_pmos={W_M8} mosfet_8_2_l_gm1_pmos={L_M8} mosfet_8_2_m_gm1_pmos={M_M8}\n'
            lines[2] = f'.param mosfet_10_1_w_gm2_pmos={W_M10} mosfet_10_1_l_gm2_pmos={L_M10} mosfet_10_1_m_gm2_pmos={M_M10}\n'
            lines[3] = f'.param mosfet_11_1_w_power_pmos={W_power} mosfet_11_1_l_power_pmos={L_power} mosfet_11_1_m_power_pmos={M_power}\n'
            lines[4] = f'.param mosfet_17_7_w_biascm_nmos={W_M17} mosfet_17_7_l_biascm_nmos={L_M17} mosfet_17_7_m_biascm_nmos={M_M17}\n'
            lines[5] = f'.param mosfet_21_2_w_load2_nmos={W_M21} mosfet_21_2_l_load2_nmos={L_M21} mosfet_21_2_m_load2_nmos={M_M21}\n'
            lines[6] = f'.param current_0_bias={Ib}\n'
            lines[7] = f'.param M_C0={M_C0}\n'
            lines[8] = f'.param M_CL={M_CL}\n'
            
            LDO_testbench_vars = open(f'{SPICE_NETLIST_DIR}/LDO_TB_vars.spice', 'w')
            LDO_testbench_vars.writelines(lines)
            LDO_testbench_vars.close()
            os.system(f'cd {SPICE_NETLIST_DIR}&& ngspice -b -o LDO_TB_ACDC.log LDO_TB_ACDC.cir')
            os.system(f'cd {SPICE_NETLIST_DIR}&& ngspice -b -o LDO_TB_Tran.log LDO_TB_Tran.cir')
            print('*** Simulations Done! ***')
        except:
            print('ERROR')

    def do_simulation(self, action):
        self._do_simulation(action)
        self.sim_results = OutputParser(self.CktGraph)
        self.op_results = self.sim_results.dcop(file_name='LDO_TB_op')

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
        self.W_power, self.L_power, self.M_power, \
        self.W_M17, self.L_M17, self.M_M17,\
        self.W_M21, self.L_M21, self.M_M21,\
        self.Ib,  \
        self.M_C0, \
        self.M_CL = action        
        
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
                ['LDR', self.LDR, self.LDR_target],
                ['LNR_maxload', self.LNR_maxload, self.LNR_target],
                ['LNR_minload', self.LNR_minload, self.LNR_target],
                ['Power_maxload', self.Power_maxload, self.Power_maxload_target],
                ['Power_minload', self.Power_minload, self.Power_minload_target],
                ['vos_maxload', self.vos_maxload, self.vos_target],
                ['vos_minload', self.vos_minload, self.vos_target],

                ['PSRR_maxload (dB)', self.PSRR_maxload, self.PSRR_target], 
                ['PSRR_minload (dB)', self.PSRR_minload, self.PSRR_target],
                ['GBW_maxload', self.GBW_maxload, self.GBW_target],
                ['GBW_minload', self.GBW_minload, self.GBW_target],
                ['phase_margin_maxload (deg) ', self.phase_margin_maxload, self.phase_margin_target],
                ['phase_margin_minload (deg) ', self.phase_margin_minload, self.phase_margin_target],

                ['v_undershoot', self.v_undershoot, self.v_undershoot_target], 
                ['v_overshoot', self.v_overshoot, self.v_overshoot_target], 
                
                ['CL (pF)', self.op_results['CL']['c']*1e12, ''],
                
                ['LDR score', self.LDR_score, ''],
                ['LNR_maxload score', self.LNR_maxload_score, ''],
                ['LNR_minload score', self.LNR_minload_score, ''],
                ['Power_maxload score', self.Power_maxload_score, ''],
                ['Power_minload score', self.Power_minload_score, ''],
                ['vos_maxload score', self.vos_maxload_score, ''],
                ['vos_minload score', self.vos_minload_score, ''],
                
                ['PSRR_maxload (dB) score', self.PSRR_maxload_score, ''],
                ['PSRR_minload (dB) score', self.PSRR_minload_score, ''],
                ['GBW_maxload score', self.GBW_maxload_score, ''],
                ['GBW_minload score', self.GBW_minload_score, ''],
                ['PM score maxload', self.phase_margin_maxload_score, ''],
                ['PM score minload', self.phase_margin_minload_score, ''],

                ['v_undershoot score', self.v_undershoot_score, ''], 
                ['v_overshoot score', self.v_overshoot_score, ''],                

                ['CL area score', self.CL_area_score, ''],
                ['Reward', reward, '']
                ],
            headers=['param', 'num', 'target'], tablefmt='orgtbl', numalign='right', floatfmt=".8f"
            ))

        return observation, reward, terminated, False, info
        
    def _get_obs(self):
        # pick some .OP params from the dict:
        try:
            f = open(f'{SPICE_NETLIST_DIR}/LDO_TB_op_mean_std.json')
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
        self.OP_M24 = self.op_results['M24']
        self.OP_M24_norm = (np.array([self.OP_M24['id'],
                                self.OP_M24['gm'],
                                self.OP_M24['gds'],
                                self.OP_M24['vth'],
                                self.OP_M24['vdsat'],
                                self.OP_M24['vds'],
                                self.OP_M24['vgs']
                                ]) - self.op_mean)/self.op_std
        # it is not straightforward to extract resistance info from sky130 resistor, using the following approximation instead
        # normalize all passive components
        self.OP_C0_norm = ActionNormalizer(action_space_low=self.C0_low, action_space_high=self.C0_high).reverse_action(self.op_results['C0']['c']) # convert to (-1, 1)
        self.OP_CL_norm = ActionNormalizer(action_space_low=self.CL_low, action_space_high=self.CL_high).reverse_action(self.op_results['CL']['c']) # convert to (-1, 1)
        
        self.r1_norm=self.r1/200000
        self.r0_norm=self.r0/200000
        # state shall be in the order of node (node0, node1, ...)
        observation = np.array([
                               [0,0,0,0,      0,0,0,      self.OP_M0_norm[0],self.OP_M0_norm[1],self.OP_M0_norm[2],self.OP_M0_norm[3],self.OP_M0_norm[4],self.OP_M0_norm[5],self.OP_M0_norm[6]],
                               [0,0,0,0,      0,0,0,      self.OP_M1_norm[0],self.OP_M1_norm[1],self.OP_M1_norm[2],self.OP_M1_norm[3],self.OP_M1_norm[4],self.OP_M1_norm[5],self.OP_M1_norm[6]],
                               [0,0,0,0,      0,0,0,      self.OP_M2_norm[0],self.OP_M2_norm[1],self.OP_M2_norm[2],self.OP_M2_norm[3],self.OP_M2_norm[4],self.OP_M2_norm[5],self.OP_M2_norm[6]],
                               [0,0,0,0,      0,0,0,      self.OP_M3_norm[0],self.OP_M3_norm[1],self.OP_M3_norm[2],self.OP_M3_norm[3],self.OP_M3_norm[4],self.OP_M3_norm[5],self.OP_M3_norm[6]],
                               [0,0,0,0,      0,0,0,      self.OP_M4_norm[0],self.OP_M4_norm[1],self.OP_M4_norm[2],self.OP_M4_norm[3],self.OP_M4_norm[4],self.OP_M4_norm[5],self.OP_M4_norm[6]],
                               [0,0,0,0,      0,0,0,      self.OP_M5_norm[0],self.OP_M5_norm[1],self.OP_M5_norm[2],self.OP_M5_norm[3],self.OP_M5_norm[4],self.OP_M5_norm[5],self.OP_M5_norm[6]],
                               [0,0,0,0,      0,0,0,      self.OP_M6_norm[0],self.OP_M6_norm[1],self.OP_M6_norm[2],self.OP_M6_norm[3],self.OP_M6_norm[4],self.OP_M6_norm[5],self.OP_M6_norm[6]],
                               [0,0,0,0,      0,0,0,      self.OP_M7_norm[0],self.OP_M7_norm[1],self.OP_M7_norm[2],self.OP_M7_norm[3],self.OP_M7_norm[4],self.OP_M7_norm[5],self.OP_M7_norm[6]],
                               [0,0,0,0,      0,0,0,      self.OP_M8_norm[0],self.OP_M8_norm[1],self.OP_M8_norm[2],self.OP_M8_norm[3],self.OP_M8_norm[4],self.OP_M8_norm[5],self.OP_M8_norm[6]],
                               [0,0,0,0,      0,0,0,      self.OP_M9_norm[0],self.OP_M9_norm[1],self.OP_M9_norm[2],self.OP_M9_norm[3],self.OP_M9_norm[4],self.OP_M9_norm[5],self.OP_M9_norm[6]],
                               [0,0,0,0,      0,0,0,      self.OP_M10_norm[0],self.OP_M10_norm[1],self.OP_M10_norm[2],self.OP_M10_norm[3],self.OP_M10_norm[4],self.OP_M10_norm[5],self.OP_M10_norm[6]],
                               [0,0,0,0,      0,0,0,      self.OP_M11_norm[0],self.OP_M11_norm[1],self.OP_M11_norm[2],self.OP_M11_norm[3],self.OP_M11_norm[4],self.OP_M11_norm[5],self.OP_M11_norm[6]],
                               [0,0,0,0,      0,0,0,      self.OP_M12_norm[0],self.OP_M12_norm[1],self.OP_M12_norm[2],self.OP_M12_norm[3],self.OP_M12_norm[4],self.OP_M12_norm[5],self.OP_M12_norm[6]],
                               [0,0,0,0,      0,0,0,      self.OP_M13_norm[0],self.OP_M13_norm[1],self.OP_M13_norm[2],self.OP_M13_norm[3],self.OP_M13_norm[4],self.OP_M13_norm[5],self.OP_M13_norm[6]],
                               [0,0,0,0,      0,0,0,      self.OP_M14_norm[0],self.OP_M14_norm[1],self.OP_M14_norm[2],self.OP_M14_norm[3],self.OP_M14_norm[4],self.OP_M14_norm[5],self.OP_M14_norm[6]],
                               [0,0,0,0,      0,0,0,      self.OP_M15_norm[0],self.OP_M15_norm[1],self.OP_M15_norm[2],self.OP_M15_norm[3],self.OP_M15_norm[4],self.OP_M15_norm[5],self.OP_M15_norm[6]],
                               [0,0,0,0,      0,0,0,      self.OP_M16_norm[0],self.OP_M16_norm[1],self.OP_M16_norm[2],self.OP_M16_norm[3],self.OP_M16_norm[4],self.OP_M16_norm[5],self.OP_M16_norm[6]],
                               [0,0,0,0,      0,0,0,      self.OP_M17_norm[0],self.OP_M17_norm[1],self.OP_M17_norm[2],self.OP_M17_norm[3],self.OP_M17_norm[4],self.OP_M17_norm[5],self.OP_M17_norm[6]],     
                               [0,0,0,0,      0,0,0,      self.OP_M18_norm[0],self.OP_M18_norm[1],self.OP_M18_norm[2],self.OP_M18_norm[3],self.OP_M18_norm[4],self.OP_M18_norm[5],self.OP_M18_norm[6]],
                               [0,0,0,0,      0,0,0,      self.OP_M19_norm[0],self.OP_M19_norm[1],self.OP_M19_norm[2],self.OP_M19_norm[3],self.OP_M19_norm[4],self.OP_M19_norm[5],self.OP_M19_norm[6]],
                               [0,0,0,0,      0,0,0,      self.OP_M20_norm[0],self.OP_M20_norm[1],self.OP_M20_norm[2],self.OP_M20_norm[3],self.OP_M20_norm[4],self.OP_M20_norm[5],self.OP_M20_norm[6]],
                               [0,0,0,0,      0,0,0,      self.OP_M21_norm[0],self.OP_M21_norm[1],self.OP_M21_norm[2],self.OP_M21_norm[3],self.OP_M21_norm[4],self.OP_M21_norm[5],self.OP_M21_norm[6]],
                               [0,0,0,0,      0,0,0,      self.OP_M22_norm[0],self.OP_M22_norm[1],self.OP_M22_norm[2],self.OP_M22_norm[3],self.OP_M22_norm[4],self.OP_M22_norm[5],self.OP_M22_norm[6]],
                               [0,0,0,0,      0,0,0,      self.OP_M24_norm[0],self.OP_M24_norm[1],self.OP_M24_norm[2],self.OP_M24_norm[3],self.OP_M24_norm[4],self.OP_M24_norm[5],self.OP_M24_norm[6]],
                               
                               [self.Vdd,0,0,0,       0,0,0,      0,0,0,0,0,0,0],
                               [0,self.GND,0,0,       0,0,0,      0,0,0,0,0,0,0],
                               [0,0,self.Ib,0,   0,0,0,    0,0,0,0,0,0,0],
                               [0,0,0,self.OP_C0_norm,   0,0,0,    0,0,0,0,0,0,0],        
                               [0,0,0,0,      self.OP_CL_norm,0,0,       0,0,0,0,0,0,0],
                               [0,0,0,0,      0,self.r1_norm,0,       0,0,0,0,0,0,0],
                               [0,0,0,0,      0,0,self.r0_norm,       0,0,0,0,0,0,0],
                               
                               ])
        # clip the obs for better regularization
        observation = np.clip(observation, -5, 5)
        
        return observation
        
    def _get_info(self):
        '''Evaluate the performance'''
        ''' LNR at maxload '''
        self.dc_LNR_maxload = self.sim_results.dc(file_name='LDO_TB_ACDC_LNR_maxload')
        self.LNR_maxload = self.dc_LNR_maxload[1][1]
        if self.LNR_maxload < 0:
            self.LNR_maxload_score = -1
        else :      
            self.LNR_maxload_score = np.min([(self.LNR_target - self.LNR_maxload) / (self.LNR_target + self.LNR_maxload), 0])

        ''' LNR at minload '''
        self.dc_LNR_minload = self.sim_results.dc(file_name='LDO_TB_ACDC_LNR_minload')
        self.LNR_minload = self.dc_LNR_minload[1][1]     
        if self.LNR_minload < 0:
            self.LNR_minload_score = -1
        else :           
            self.LNR_minload_score = np.min([(self.LNR_target - self.LNR_minload) / (self.LNR_target + self.LNR_minload), 0])

        ''' LDR '''
        self.dc_LR_Power_vos = self.sim_results.LR_Power_vos(file_name='LDO_TB_ACDC_LR_Power_vos')
        self.LDR = self.dc_LR_Power_vos[1][1]
        self.Power_maxload = self.dc_LR_Power_vos[2][1]
        self.Power_minload = self.dc_LR_Power_vos[3][1]
        self.vos_maxload_1 = self.dc_LR_Power_vos[4][1]
        self.vos_minload_1 = self.dc_LR_Power_vos[5][1]
        self.vos_maxload = abs(self.vos_maxload_1)
        self.vos_minload = abs(self.vos_minload_1)

        if self.LDR < 0:
            self.LDR_score = -1
        else :
            self.LDR_score = np.min([(self.LDR_target - self.LDR) / (self.LDR_target + self.LDR), 0])

        self.Power_maxload_score = np.min([(self.Power_maxload_target - self.Power_maxload) / (self.Power_maxload_target + self.Power_maxload), 0])
        self.Power_minload_score = np.min([(self.Power_minload_target - self.Power_minload) / (self.Power_minload_target + self.Power_minload), 0])
        self.vos_maxload_score = np.min([(self.vos_target - self.vos_maxload) / (self.vos_target + self.vos_maxload), 0])
        self.vos_minload_score = np.min([(self.vos_target - self.vos_minload) / (self.vos_target + self.vos_minload), 0])
    
        ''' AC & Loop at maxload'''
        self.PSRR_dcgain_maxload = self.sim_results.ac(file_name='LDO_TB_ACDC_PSRR_dcgain_maxload')
        self.PSRR_maxload = self.PSRR_dcgain_maxload[1][1]
        if self.PSRR_maxload > 0 :
            self.PSRR_maxload_score = -1
        else :
            self.PSRR_maxload_score = np.min([(self.PSRR_maxload - self.PSRR_target) / (self.PSRR_maxload + self.PSRR_target), 0])
            if self.PSRR_maxload < self.PSRR_target :
                self.PSRR_maxload_score = 0

        self.dcgain_maxload = self.PSRR_dcgain_maxload[2][1]
        if self.dcgain_maxload > 0 :
            try:
                self.GBW_PM_maxload = self.sim_results.ac(file_name='LDO_TB_ACDC_GBW_PM_maxload')
                self.GBW_maxload = self.GBW_PM_maxload[1][1]
                self.GBW_maxload_score = np.min([(self.GBW_maxload - self.GBW_target) / (self.GBW_maxload + self.GBW_target), 0])
                self.phase_margin_maxload = self.GBW_PM_maxload[2][1]
                self.phase_margin_maxload_score = np.min([(self.phase_margin_maxload - self.phase_margin_target) / (self.phase_margin_maxload + self.phase_margin_target), 0])
            except: 
                if self.phase_margin_maxload > 180 or self.phase_margin_maxload < 0:
                    self.phase_margin_maxload = 0
                else:
                    self.phase_margin_maxload = self.phase_margin_maxload
        else :
            self.GBW_maxload = 0
            self.GBW_maxload_score = -1
            self.phase_margin_maxload = 0
            self.phase_margin_maxload_score = -1          

        ''' AC & Loop at minload'''
        self.PSRR_dcgain_minload = self.sim_results.ac(file_name='LDO_TB_ACDC_PSRR_dcgain_minload')
        self.PSRR_minload = self.PSRR_dcgain_minload[1][1]
        if self.PSRR_minload > 0 :
            self.PSRR_minload_score = -1
        else :
            self.PSRR_minload_score = np.min([(self.PSRR_minload - self.PSRR_target) / (self.PSRR_minload + self.PSRR_target), 0])
            if self.PSRR_minload < self.PSRR_target :
                self.PSRR_minload_score = 0

        self.dcgain_minload = self.PSRR_dcgain_minload[2][1]
        if self.dcgain_minload > 0 :
            try:
                self.GBW_PM_minload = self.sim_results.ac(file_name='LDO_TB_ACDC_GBW_PM_minload')
                self.GBW_minload = self.GBW_PM_minload[1][1]
                self.GBW_minload_score = np.min([(self.GBW_minload - self.GBW_target) / (self.GBW_minload + self.GBW_target), 0])
                self.phase_margin_minload = self.GBW_PM_minload[2][1]
                self.phase_margin_minload_score = np.min([(self.phase_margin_minload - self.phase_margin_target) / (self.phase_margin_minload + self.phase_margin_target), 0])
            except: 
                if self.phase_margin_minload > 180 or self.phase_margin_minload < 0:
                    self.phase_margin_minload = 0
                else:
                    self.phase_margin_minload = self.phase_margin_minload
        else :
            self.GBW_minload = 0
            self.GBW_minload_score = -1
            self.phase_margin_minload = 0
            self.phase_margin_minload_score = -1   

        ''' Tran test '''       
        self.tran_result = self.sim_results.tran(file_name='LDO_TB_Tran_meas')
        self.v_undershoot = self.tran_result[1][1]
        self.v_overshoot = abs(self.tran_result[2][1])      
        self.v_undershoot_score = np.min([(self.v_undershoot_target - self.v_undershoot) / (self.v_undershoot_target + self.v_undershoot), 0])
        self.v_overshoot_score = np.min([(self.v_overshoot_target - self.v_overshoot) / (self.v_overshoot_target + self.v_overshoot), 0])

        """ Decap score """
        self.CL_area_score = (self.CL_low - self.op_results['CL']['c']) / (self.CL_low + self.op_results['CL']['c'])
    
        """ Total reward """
        self.reward = self.LDR_score + self.LNR_maxload_score + self.LNR_minload_score + \
                      self.Power_maxload_score + self.Power_minload_score +self.vos_maxload_score + self.vos_minload_score + \
                      self.PSRR_maxload_score + self.PSRR_minload_score + self.GBW_maxload_score + self.GBW_minload_score + \
                      self.phase_margin_maxload_score + self.phase_margin_minload_score + \
                      self.v_overshoot_score + self.v_undershoot_score  
                        
        if self.reward >= 0:
            self.reward = self.reward + self.CL_area_score + 10
                    
        return {
                'LDR': self.LDR, 
                'LNR_maxload': self.LNR_maxload,
                'LNR_minload': self.LNR_minload, 
                'Power_maxload': self.Power_maxload,
                'Power_minload': self.Power_minload,
                'vos_maxload': self.vos_maxload,
                'vos_minload': self.vos_minload,

                'PSRR_maxload (dB)': self.PSRR_maxload,  
                'PSRR_minload (dB)': self.PSRR_minload, 
                'GBW_maxload': self.GBW_maxload, 
                'GBW_minload': self.GBW_minload, 
                'phase_margin_maxload (deg) ': self.phase_margin_maxload, 
                'phase_margin_minload (deg) ': self.phase_margin_minload, 
        
                'v_undershoot': self.v_undershoot, 
                'v_overshoot': self.v_overshoot, 
                
                'CL (pF)': self.op_results['CL']['c']*1e12
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
    
            sim_results = OutputParser(self.CktGraph)
            op_results = sim_results.dcop(file_name='LDO_TB_op')
            
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

        with open(f'{SPICE_NETLIST_DIR}/LDO_TB_op_mean_std.json','w') as file:
            json.dump(self.OP_M_mean_std, file)