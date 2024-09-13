**TestBench for single-end Amplifier

*###############
*#    PARAM    #
*###############
.PARAM supply_voltage = 1.8
.PARAM PARAM_CLOAD = 10p
.TEMP 27

*###############
*#   Modify   #
.PARAM GBW_ideal = 1e3
.PARAM STEP_TIME = '3.2/GBW_ideal'
.PARAM TRAN_SIM_TIME = '9.6/GBW_ideal'
.PARAM RUN_TRAN = 0
*#   Modify   #
*###############


*###############
*#    Option   #
*###############
.option post print probe measure
.option post_version = 9601  
.option runval=1
.option nomod
*###############
*#   Sub.ckt   #
*###############
.include "..\Amp_schematic\SMCNR_SE_2st_AMP.sp"

*###############
*#    Model    #
*###############
.lib "..\Simulation_model\TSMC180HV\c018bcd_gen2_v1d6_usage.l" pre_simu
.lib "..\Simulation_model\TSMC180HV\c018bcd_gen2_v1d6_usage.l" tt_lib

*################
*# Power Supply #
*################
VVDDA VDDA 0 supply_voltage 
VGNDA GNDA 0 0

*#######################################################
**OP_AMP port definition:
**vdda gnda vin vip vout1 SMCNR_SE_2st_AMP**
*#######################################################

*###############
*#   Diff TB   #
*###############
VVIP VIP 0 'supply_voltage *0.5' AC=1
xi1 vdda gnda vin vip vout1 SMCNR_SE_2st_AMP    *ADM
vlstb vout1 vin dc=0  *iprobe 

*###############
*#   Comm TB   #
*###############
xi2 vdda gnda cm2 cm1 cm3 SMCNR_SE_2st_AMP    *ACM
vcmdc cm0 0 'supply_voltage*0.5'
vcmac1 cm1 cm0 0 ac=1
vcmac2 cm2 cm3 0 ac=1

*###############
*#  Slew  TB   #
*###############

*###############
*#   Modify   #
VVISR visr 0 pulse('supply_voltage*0.3' 'supply_voltage*0.7' 100u 1n 1n '1.5*STEP_TIME' '4*STEP_TIME')
*#   Modify   #
*###############

*lv hv tdelay trise tfall tpw tperiod
xi3 vdda gnda vout3 visr vout3 SMCNR_SE_2st_AMP    *SR

*###############
*# PSRR   TB   #
*###############
VVindc Vindc 0 'supply_voltage *0.5' 
VGNDApsrr gndpsrr 0 0 AC=1
VVDDApsrr vddpsrr 0 supply_voltage  AC=1
xi4 vddpsrr gnda ppsr1 vindc ppsr1 SMCNR_SE_2st_AMP    *pPSRR
xi5 vdda gndpsrr npsr1 vindc npsr1 SMCNR_SE_2st_AMP    *nPSRR

*###############
*#   DC  TB    #
*###############
VVDDdc VDDdc 0 supply_voltage 
xi6 vdddc gnda vout6 vindc vout6 SMCNR_SE_2st_AMP    *DCsweep

*###############
*#     Load    #
*###############
Cload1 VOUT1 0 PARAM_CLOAD
Cload2 VOUT2 0 PARAM_CLOAD
Cload3 VOUT3 0 PARAM_CLOAD
Cload4 VOUT4 0 PARAM_CLOAD
Cload5 VOUT5 0 PARAM_CLOAD
Cload6 VOUT6 0 PARAM_CLOAD

*################
*#  OP analys.  #
*################
.OP

*################
*#    TC.meas   #
*################
.DC temp -40 125 0.01
.measure dc VTC FIND V(vout6) AT 25
.measure dc maxval MAX V(vout6) from=-40 to=125
.measure dc minval MIN V(vout6) from=-40 to=125
.measure dc avgval AVG V(vout6) from=-40 to=125
.measure dc ppavl  PP V(vout6) from=-40 to=125
.measure dc TC param='ppavl/avgval/165'

*################
*#  Ivdd.meas   #
*################
.measure dc Ivdd0 FIND I(VVDDDC) AT 0
.measure dc Ivdd25 FIND I(VVDDDC) AT 25
.measure dc IvddN40 FIND I(VVDDDC) AT -40
.measure dc Ivdd125 FIND I(VVDDDC) AT 125

*################
*#   Vos.meas   #
*################
.meas dc voutN40 FIND V(vout6) AT -40
.meas dc vout125 FIND V(vout6) AT 125
.meas dc vosN40 param = 'voutN40-supply_voltage *0.5'
.meas dc vos125 param = 'vout125-supply_voltage *0.5'

*################
*#    LS.meas   #
*################
.DC VVDDDC 'supply_voltage *0.5' 'supply_voltage *1.5' 0.01 
.measure dc VLS FIND V(vout6) AT supply_voltage
.measure dc maxval1 MAX V(vout6) from='supply_voltage *0.8'  to='supply_voltage *1.2'
.measure dc minval1 MIN V(vout6) from='supply_voltage *0.8'  to='supply_voltage *1.2'
.measure dc avgval1 AVG V(vout6) from='supply_voltage *0.8'  to='supply_voltage *1.2'
.measure dc ppavl1  PP V(vout6) from='supply_voltage *0.8'  to='supply_voltage *1.2'
.measure dc vdd_sweep_range param = '0.4*supply_voltage'
.measure dc LS param='ppavl1/avgval1/vdd_sweep_range'

*################
*#SmallSig meas #
*################
.lstb mode=single vsource=vlstb
*.probe ac lstb(db) lstb(p)
.ac dec 100 10u 200meg
.meas lstb PM phase_margin
.param stable = 'PM>0'
.meas lstb GBW unity_gain_freq
.meas lstb DCGAIN loop_gain_minifreq
.measure ac fdpole when lstb(db) = 'DCGAIN-3'
.noise v(vout1) vvip listckt=0 listfreq=10 100 1000 10000 listcount=1 listsources=1 
*################
*#  CMRR meas   #
*################
*.print ac vdb(cm3)
.meas ac cmrrdc min vdb(cm3) 

*################
*#  PSRR meas   #
*################
*.print ac vdb(npsr1)
*.print ac vdb(ppsr1)
.measure ac DCPSRp min vdb(ppsr1) 
.measure ac DCPSRn min vdb(npsr1) 

*################
*#   SR meas    #
*################
*.tran 1n 2m
*.check slew (500n 1.5m) vout3 ('supply_voltage*0.7' 'supply_voltage*0.3' 'supply_voltage*0.6' 'supply_voltage*0.4')


*###############
*#   Modify   #
.IF(RUN_TRAN==1)
.tran 1n TRAN_SIM_TIME
.check slew (100u '100u+STEP_TIME') vout3 ('supply_voltage*0.7' 'supply_voltage*0.3' 'supply_voltage*0.6' 'supply_voltage*0.4')
.check slew ('100u+1.5*STEP_TIME' '100u+2.5*STEP_TIME') vout3 ('supply_voltage*0.7' 'supply_voltage*0.3' 'supply_voltage*0.6' 'supply_voltage*0.4')
.ENDIF
*#   Modify   #
*###############


.alter
.TEMP 55
.PARAM supply_voltage = 1.5

.alter
.TEMP 105
.PARAM supply_voltage = 2

.end
