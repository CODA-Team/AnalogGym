.subckt SMCNR_SE_2st_AMP vdda gnda vin vip vout  
xm1 outp outp gnda gnda nch_mac W=1.5u L=10.0u m=1.0 nf=1.0 
xm3 outn outp gnda gnda nch_mac W=1.5u L=10.0u m=1.0 nf=1.0 
xm7 ibias ibias vdda vdda pch_mac W=0.22u L=10.0u m=1.0 nf=1.0 
xm6 net53 ibias vdda vdda pch_mac W=0.22u L=10.0u m=2.0 nf=1.0 
xm5 vout ibias vdda vdda pch_mac W=0.22u L=10.0u m=10.0 nf=1.0 
xm2 outn vip net53 vdda pch_mac W=7.52u L=8.24u m=1.0 nf=1.0 
xm0 outp vin net53 vdda pch_mac W=7.52u L=8.24u m=1.0 nf=1.0  
xm4 vout outn gnda gnda nch_mac W=1.48u L=10.0u m=10.0 nf=1.0   
ibias0 ibias 0 100n 
r0 net027 vout 100000 
c0 outn net027 2p 
.ENDS SMCNR_SE_2st_AMP 