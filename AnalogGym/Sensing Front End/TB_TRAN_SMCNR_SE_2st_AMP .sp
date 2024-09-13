**TestBench for single-end Amplifier

*###############
*#    PARAM    #
*###############
.PARAM supply_voltage = 1.8
.PARAM PARAM_CLOAD = 10p
.TEMP 27

*###############
*#   Modify   #
.PARAM GBW_ideal = 160e3
.PARAM STEP_TIME = '3.2/GBW_ideal'
.PARAM TRAN_SIM_TIME = '100u + 20/GBW_ideal'
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



*###############
*#   Modify   #
VVISR visr 0 pulse('supply_voltage*0.3' 'supply_voltage*0.7' 100u 1n 1n '1.5*STEP_TIME' '4*STEP_TIME')
*#   Modify   #
*###############

*lv hv tdelay trise tfall tpw tperiod
xi3 vdda gnda vout3 visr vout3 SMCNR_SE_2st_AMP    *SR

*###############
*#     Load    #
*###############
Cload3 VOUT3 0 PARAM_CLOAD


*################
*#  OP analys.  #
*################
.OP

*################
*#   SR meas    #
*################

.tran 1n TRAN_SIM_TIME
.check slew (100u '100u+STEP_TIME') vout3 ('supply_voltage*0.7' 'supply_voltage*0.3' 'supply_voltage*0.6' 'supply_voltage*0.4')
.check slew ('100u+1.5*STEP_TIME' '100u+2.5*STEP_TIME') vout3 ('supply_voltage*0.7' 'supply_voltage*0.3' 'supply_voltage*0.6' 'supply_voltage*0.4')

.alter
.TEMP 55
.PARAM supply_voltage = 1.5

.alter
.TEMP 105
.PARAM supply_voltage = 2

.end
