import numpy as np
import re
import sys


def get_simple_mt(mtName):
    '''
    read simulation result from one .mt* file
    '''
    pattern = re.compile(r'[\s\t]*([\de.-]+)[\s\t]+([\de.-]+)[\s\t]+([\de.-]+)[\s\t]+([\de.-]+).+')
    
    with open(mtName) as f:
        lines = f.readlines()
        result0 = pattern.match(lines[4])
        result1 = pattern.match(lines[5])
        
    up_imin = float(result0.group(1))*1e6
    up_iavg = float(result0.group(2))*1e6
    up_imax = float(result0.group(3))*1e6
    lo_imin = float(result0.group(4))*1e6
    lo_iavg = float(result1.group(1))*1e6
    lo_imax = float(result1.group(2))*1e6

    const = 40
    diff1 = up_imax - up_iavg
    diff2 = up_iavg - up_imin
    diff3 = lo_imax - lo_iavg
    diff4 = lo_iavg - lo_imin
    diff = diff1+diff2+diff3+diff4
    deviation = abs(up_iavg-const) + abs(lo_iavg-const) 
    fom = 0.3*diff+0.5*deviation
    return fom,diff1,diff2,diff3,diff4,deviation
   
    
def get_result_mt(mtNameBas, num):
    results = np.zeros((num,5))
    for i in range(num):
        mtName = mtNameBas + str(i)
        fom,diff1,diff2,diff3,diff4,deviation = get_simple_mt(mtName)

        results[i,0] = diff1
        results[i,1] = diff2
        results[i,2] = diff3
        results[i,3] = diff4
        results[i,4] = deviation
          
    diff1 = max(results[:,0])
    diff2 = max(results[:,1])
    diff3 = max(results[:,2])
    diff4 = max(results[:,3])
    diff = diff1+diff2+diff3+diff4
    deviation = max(results[:,4])
    fom = 0.3*diff+0.5*deviation
    return fom,diff1,diff2,diff3,diff4,deviation


def write_de_result(fom, diff1, diff2, diff3, diff4, deviation):
    with open('de_result.po','w') as f:
        f.write('obj= '+str(fom)+'\n')
        f.write('diff1= '+str(diff1)+'\n')
        f.write('diff2= '+str(diff2)+'\n')
        f.write('diff3= '+str(diff3)+'\n')
        f.write('diff4= '+str(diff4)+'\n')
        f.write('dev= '+str(deviation)+'\n')

def write_resultpo(fom, diff1, diff2, diff3, diff4, deviation):
    fom_cons = fom
    diff1_cons = diff1 - 20
    diff2_cons = diff2 - 20
    diff3_cons = diff3 - 5
    diff4_cons = diff4 - 5
    dev_cons = deviation - 5
    content = str(fom_cons) +' '+ str(diff1_cons) +' '+ str(diff2_cons) +' '+ str(diff3_cons) +' '+ str(diff4_cons) +' '+ str(dev_cons)
    with open('result.po','w') as f:
        f.write(content)


def get_result(cornerNum, resultForm='de'):
    if cornerNum==1:
        fom,diff1,diff2,diff3,diff4,deviation = get_simple_mt('chgp.mt0')
    else:
        fom,diff1,diff2,diff3,diff4,deviation = get_result_mt('chgp.mt', cornerNum)
    if resultForm=='de':
        write_de_result(fom, diff1, diff2, diff3, diff4, deviation)
    elif resultForm=='weibo':
        write_resultpo(fom, diff1, diff2, diff3, diff4, deviation)


argv = sys.argv[1:]
cornerNum = argv[0]
resultForm = argv[1]
get_result(int(cornerNum), resultForm)