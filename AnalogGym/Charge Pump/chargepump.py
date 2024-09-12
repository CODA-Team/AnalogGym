import os
import shutil
import subprocess
# import pickle
# import time
import math
import numpy as np
import re

class ChargePump:
    def __init__(self, index=0):
        self.name = "ChargePump"
        self.suffix = ""
        self.index = index
        self.dir = os.path.dirname(__file__)
        assert os.path.exists(os.path.join(self.dir, "circuit"))
        # self.database = "ChargePump.pkl"
        self.mode = "spice" # or "ocean"
        self.del_folders = True

        # Design Variables
        ## DX = [('name',L,U,step,init,[discrete list]),....] if there is no discrete, do not write
        self.DX = [
              ('q_llower', 3e-6, 9e-6, 1.0e-8, 6.9e-6, 'NO'),
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
             ]

        self.in_dim = len(self.DX)
        self.real_init = np.array([dx[4] for dx in self.DX])
        self.real_lb = np.array([dx[1] for dx in self.DX])
        self.real_ub = np.array([dx[2] for dx in self.DX])
        self.init = (self.real_init-self.real_lb)/(self.real_ub - self.real_lb)

        self.run_file = "sim.sh"
        self.result_file = "de_result.po"

        self.perform_setting = {
         "diff1": ("<", 20, None, None, "diff1", 200, 10),
         "diff2": ("<", 20, None, None, "diff2", 200, 10),
         "diff3": ("<", 5, None, None, "diff3", 50, 10),
         "diff4": ("<", 5, None, None, "diff4", 50, 10),
         "deviation": ("<", 5, None, None, "deviation", 50, 10),
         "obj": (None, None, None, None, "obj", 1000, None)
        }

        self.fom_setting = (100, None)

    def cal_fom(self, meas_dict):
        fom = meas_dict["obj"]
        return fom

    def write_param(self, dx_real_dict):
        if self.mode == "spice":
            with open("param", "w") as handler:
                for dx_name, dx_real in dx_real_dict.items():
                    handler.write(".param {}={}\n".format(dx_name, dx_real))
        elif self.mode == "ocean":
            handler.write('ocnxlSweepVar(\"' + str(dx_name) + '\" ' + '\"' + str(dx_real) + '\")\n')
        else:
            raise Exception("unknown self.mode")

    def extract_perf(self, file_name, perf):
        pattern_str = perf+'\s*=\s*([\d.eE+\-]+)'
        pattern = re.compile(pattern_str)
        with open(file_name, 'r') as f:
            lines = f.readlines()
            for line in lines:
                result = pattern.search(line)
                if result:
                    val = result.group(1)
                    return float(val)
            return False

    ##################################################################

    def set_name_suffix(self, suffix):
        self.suffix = suffix
        return self

    def __call__(self, x, realx=False, index=None):
        # while(os.path.exists("index_lock")):
        #     print("index is locked, wait for 1s")
        #     time.sleep(1)
        # open("index_lock", "a").close()
        # if os.path.exists("index.pkl"):
        #     with open("index.pkl", "rb") as fr:
        #         old_index = pickle.load(fr)
        #         tmp_index = old_index + 1
        # else:
        #     tmp_index = 0
        # with open("index.pkl", "wb") as fw:
        #     pickle.dump(tmp_index, fw)
        # os.remove("index_lock")
        if index is None: # sequentially
            tmp_index = self.index
            self.index += 1
        else: # parallel with index updated in global
            tmp_index = index
        tmp_dir = "{}_{}_{}".format(self.name, self.suffix, tmp_index)
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        cwd = os.getcwd()
        shutil.copytree(
            os.path.join(self.dir, "circuit"), 
            os.path.join(cwd, tmp_dir)
        )
        print("{} is created, waiting for simulation".format(tmp_dir))
        os.chdir(tmp_dir)
        if not realx:
            x_01 = x
            dx_real_dict = self.dx_map(x_01)
        else:
            x_real = x
            x_name = [dx[0] for dx in self.DX]
            dx_real_dict = dict(zip(x_name,x_real))
        self.write_param(dx_real_dict)
        subprocess.Popen([self.run_file]).wait()
        print("{} simulation done".format(tmp_dir))
        meas_dict = self.read_meas(self.result_file)
        fom = self.cal_fom(meas_dict)
        cost = self.cal_cost(meas_dict, fom)
        print("{} get cost {}".format(tmp_dir, cost))
        os.chdir(cwd)
        # self.update_database(tmp_index, dx_real_dict, meas_dict, fom, cost)
        if self.del_folders:
            shutil.rmtree(tmp_dir)
        return cost

    def dx_map(self, x_01):
        dx_real_dict = {}
        for dx_tup, dx_01 in zip(self.DX, x_01):
            dx_name = dx_tup[0]
            dx_lb = dx_tup[1]
            dx_ub = dx_tup[2]
            dx_step = dx_tup[3]
            dx_real_range = dx_01*(dx_ub-dx_lb)
            plus = 1 if (dx_real_range%dx_step)/dx_step >= 0.5 else 0
            round_range = dx_real_range//dx_step*dx_step + plus*dx_step
            dx_real = round_range + dx_lb
            if dx_real > dx_ub:
                dx_real = dx_ub
            if dx_real < dx_lb:
                dx_real = dx_lb
            dx_real_dict[dx_name] = dx_real
        return dx_real_dict

    def read_meas(self, file_name):
        meas_dict = {}
        for perform_name, perform_tup in self.perform_setting.items():
            perform_value = self.extract_perf(file_name, perform_tup[4])
            if not perform_value:
                perform_value = perform_tup[5]
            meas_dict[perform_name] = perform_value
        return meas_dict

    def cal_cost(self, meas_dict, fom):
        cons_list = []
        for perform_name, perform_value in meas_dict.items():
            tup = self.perform_setting[perform_name]
            spec_weight= tup[-1] if tup[-1] else 1
            if "<" in tup:
                if tup[1] != 0:
                    cons_list.append(
                        (perform_value - tup[1])/abs(tup[1])*spec_weight
                    )
                else:
                    cons_list.append(
                        (2/(1+math.exp(-1*perform_value))-1)*spec_weight
                    )
            if ">" in tup:
                if tup[3] != 0:
                    cons_list.append(
                        -(perform_value - tup[3])/abs(tup[3])*spec_weight
                    )
                else:
                    cons_list.append(
                        -(2/(1+math.exp(-1*perform_value))+1)*spec_weight
                    )
            if ("<" not in tup) and (">" not in tup):
                continue
        cons_cost = sum([x if x>0 else 0 for x in cons_list])
        fom_weight = self.fom_setting[-1] if self.fom_setting[-1] else 1
        fom_cost = (fom - self.fom_setting[0])/abs(self.fom_setting[0])*fom_weight
        cost = cons_cost + fom_cost
        return cost

    # def update_database(self, index, dx_real_dict, meas_dict, fom, cost):
    #     # Prepare datas
    #     x_real = [dx_real for dx_real in dx_real_dict.values()]
    #     while(os.path.exists("database_lock")):
    #         print("database is locked, wait for 1s")
    #         time.sleep(1)
    #     open("database_lock", "a").close()
    #     if os.path.exists(self.database):
    #         with open(self.database, "rb") as fr:
    #             datas = pickle.load(fr)
    #             datas.append(dict(index=index, x_real=x_real, meas=meas_dict, fom=fom, cost=cost, time=time.time()))
    #     else:
    #         datas = [dict(index=index, x_real=x_real, meas=meas_dict, fom=fom, cost=cost, time=time.time())]
    #     with open(self.database, "wb") as fw:
    #         pickle.dump(datas, fw)
    #     os.remove("database_lock")

if __name__ == '__main__':
    accia = ACCIA()
    accia(accia.real_init, realx=True)
    accia([0.5]*accia.in_dim)
