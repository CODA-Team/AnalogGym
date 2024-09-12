import os
import psutil

proc = psutil.Popen(["sim.sh"])
try:
    proc.wait(timeout=1200)
except psutil.TimeoutExpired:
    # proc.terminate()
    for subproc in proc.children(recursive=True):
        subproc.kill()
    proc.kill()
    with open("pll_vco.log","w") as handler:
        handler.write("no result in 20 mins")

if not os.path.exists("pll_vco.log"):
    with open("pll_vco.log","w") as handler:
        handler.write("fatal error")
    
