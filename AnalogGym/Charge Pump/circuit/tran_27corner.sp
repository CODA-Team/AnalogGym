***simulation file***
*.option
*+ fast
*+ post node list
*+ method=gear
*+ runlvl=6
*+ probe=1
*+ accurate=6
*+ dcon=-1
*+ modmonte=1

.inc 'netlist'

.ic v(vcp_net)=vcp

.inc 'param'


.lib './smic.40/hspice/v1p4/l0040ll_v1p4_1r.lib' TT
.lib './smic.40/hspice/v1p4/l0040ll_v1p4_1r.lib' RES_TT
.param vdd33 = 2.97
.temp -40

.param vcp   = 2.2
.param lc1 = 7e-6 ;
.param wc1 = 7e-6 ;
.param lc2 = 7e-6 ;
.param wc2 = 10e-6;
.param lc3 = 5e-6 ;
.param wc3 = 10e-6;

.op
.tran 2p 200n $ sweep monte=100

*.probe v(OUT_0) V(LOCK) v(XI0.CK_REF) V(XI0.net0134) v(xi0.net68)  v(xi0.DIV10)  v(xi0.UP)  v(xi0.UPB) v(xi0.DNB) v(xi0.DN)  v(xi0.DN12)  v(xi0.UP12) V(xi0.CP_OUT) V(xi0.vctr) V(xi0.xi66.quench) v(xi0.pd_clk_ready) v(xi0.LOCK_REF) v(xi0.xi60.ph_3) v(xi0.xi60.ph_7) v(xi0.xi60.vcovdd)  v(xi0.xi60.reset) v(xi0.xi60.resetn) 

*.print i(xi0.vupper) i(xi0.vlower)

.measure tran up_imin min i(xi0.vupper) from=20e-9 to=100e-9
.measure tran up_iavg avg i(xi0.vupper) from=20e-9 to=180e-9
.measure tran up_imax max i(xi0.vupper) from=20e-9 to=100e-9

.measure tran lo_imin min i(xi0.vlower) from=20e-9 to=100e-9
.measure tran lo_iavg avg i(xi0.vlower) from=20e-9 to=180e-9
.measure tran lo_imax max i(xi0.vlower) from=20e-9 to=100e-9




******2***
.alter
.param vdd33 = 3.3
.temp -40

******3***
.alter
.param vdd33 = 3.63
.temp -40

******4***
.alter
.param vdd33 = 2.97
.temp 125


******5***
.alter
.param vdd33 = 3.3
.temp 125


******6***
.alter
.param vdd33 = 3.63
.temp 125


******7***
.alter
.param vdd33 = 2.97
.temp 25


******8***
.alter
.param vdd33 = 3.3
.temp 25


******9***
.alter
.param vdd33 = 3.63
.temp 25


******10***
.alter
.lib './smic.40/hspice/v1p4/l0040ll_v1p4_1r.lib' SS
.lib './smic.40/hspice/v1p4/l0040ll_v1p4_1r.lib' RES_SS
.param vdd33 = 2.97
.temp -40

******11***
.alter
.param vdd33 = 3.3
.temp -40

******12***
.alter
.param vdd33 = 3.63
.temp -40

******13***
.alter
.param vdd33 = 2.97
.temp 125


******14***
.alter
.param vdd33 = 3.3
.temp 125


******15***
.alter
.param vdd33 = 3.63
.temp 125


******16***
.alter
.param vdd33 = 2.97
.temp 25


******17***
.alter
.param vdd33 = 3.3
.temp 25


******18***
.alter
.param vdd33 = 3.63
.temp 25



******19***
.alter
.lib './smic.40/hspice/v1p4/l0040ll_v1p4_1r.lib' FF
.lib './smic.40/hspice/v1p4/l0040ll_v1p4_1r.lib' RES_FF
.param vdd33 = 2.97
.temp -40

******20***
.alter
.param vdd33 = 3.3
.temp -40

******21***
.alter
.param vdd33 = 3.63
.temp -40

******22***
.alter
.param vdd33 = 2.97
.temp 125


******23***
.alter
.param vdd33 = 3.3
.temp 125


******24***
.alter
.param vdd33 = 3.63
.temp 125

******25***
.alter
.param vdd33 = 2.97
.temp 25


******26***
.alter
.param vdd33 = 3.3
.temp 25


******27***
.alter
.param vdd33 = 3.63
.temp 25


.end
