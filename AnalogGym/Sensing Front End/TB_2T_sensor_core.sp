*###############
*#    Title    #
*###############
*HSPICE GMID METHOD v2.0*
*Harrychang*

*###############
*#    Model    #
*###############
.lib "..\Simulation_model\TSMC180HV\c018bcd_gen2_v1d6_usage.l" tt_lib
.PARAM supply_voltage = 1
.TEMP 27

*################
*#    Option    #
*################
.option post dccap probe node print
.option post_version = 9601  
.option nomod

*###############
*#   SubCKt    #    
*###############
.SUBCKT PTAT_SENSOR VDDA vsense GNDA
xm0 vdda vsense vsense gnda nch_mac l=200e-9 w=8e-6 multi=8 nf=8 
xm1 vsense vsense gnda gnda nch_mac l=1e-6 w=4e-6 multi=8 nf=4 
.ENDS

*################
*# GroundSupply #
*################
VGNDA GNDA 0 0

*###############
*#   SVT  TB   #
*###############
VVDDA VDDA_SVT 0 DC=supply_voltage AC=1
x0 VDDA_SVT vsense_svt gnda PTAT_SENSOR
c0 vsense_svt 0 0.2p

*################
*#  OP analys.  #
*################
.OP

*################
*#    TC.meas   #
*################
.DC temp -20 120 0.01

.measure dc vout0 FIND V(vsense_svt) AT 0
.measure dc vout25 FIND V(vsense_svt) AT 25
.measure dc vout50 FIND V(vsense_svt) AT 50
.measure dc vout75 FIND V(vsense_svt) AT 75
.measure dc vout100 FIND V(vsense_svt) AT 100
.measure dc lsb_25_75C param = '(vout75-vout25)/50'

.measure dc VTC FIND V(vsense_svt) AT 25
.measure dc maxval MAX V(vsense_svt) from=0 to=120
.measure dc minval MIN V(vsense_svt) from=0 to=120
.measure dc avgval AVG V(vsense_svt) from=0 to=120
.measure dc ppavl  PP V(vsense_svt) from=0 to=120
.measure dc TC param='ppavl/avgval/120'


*################
*#  Ivdd.meas   #
*################
.measure dc Ivdd0 FIND I(VVDDA) AT 0
.measure dc Ivdd25 FIND I(VVDDA) AT 25
.measure dc Ivdd100 FIND I(VVDDA) AT 100

*################
*#    LS.meas   #
*################
.DC VVDDA 'supply_voltage *0.5' 'supply_voltage *1.5' 0.01 
.measure dc VLS FIND V(vsense_svt) AT supply_voltage
.measure dc maxval1 MAX V(vsense_svt) from='supply_voltage *0.8'  to='supply_voltage *1.2'
.measure dc minval1 MIN V(vsense_svt) from='supply_voltage *0.8'  to='supply_voltage *1.2'
.measure dc avgval1 AVG V(vsense_svt) from='supply_voltage *0.8'  to='supply_voltage *1.2'
.measure dc ppavl1  PP V(vsense_svt) from='supply_voltage *0.8'  to='supply_voltage *1.2'
.measure dc vdd_sweep_range param = '0.4*supply_voltage'
.measure dc LS param='ppavl1/avgval1/vdd_sweep_range'

*################
*#SmallSig meas #
*################
.ac dec 100 10u 10meg
.noise v(vsense_svt) VVDDA listckt=0 listfreq=1 10 100 1000 10000 listcount=1 listsources=1 
.measure ac DCPSRp min vdb(vsense_svt)

.end