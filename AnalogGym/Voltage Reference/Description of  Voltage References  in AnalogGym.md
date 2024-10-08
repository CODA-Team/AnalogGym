# Description of  Voltage References  in AnalogGym

## Problem model along with detailed relevant information

In the following sections, we present the detailed netlists for two circuits discussed in our paper: the 'Three-output Voltage Reference' and the 'Sub-threshold Voltage and Current Reference.' Accompanying these netlists, we also provide the "SPICE" files necessary for their simulation, which serve to facilitate a deeper understanding of their operation and characteristics.

The netlists we are sharing here represent the optimal designs generated by the algorithms we've developed and discussed in our manuscript.  By providing these netlists, our goal is to enable researchers and practitioners to not only reproduce the results demonstrated in our paper but also to provide a basis for further comparative analysis with other methodologies or algorithms in this domain.

It's important to note that, due to the confidentiality agreements surrounding the Process Design Kit (PDK) we utilized, we have omitted the PDK import paths from the shared files. While this exclusion might pose a limitation to directly replicating the exact environment we used, the netlists themselves contain sufficient detail about the transistor parameters and circuit configurations to allow for an accurate reconstruction of the circuits in a similar PDK environment. We encourage readers and fellow researchers to utilize these netlists and stimulus files to explore the effectiveness and efficiency of our proposed algorithms in analog circuit design. 

In a netlist file, which is a detailed description of an electronic circuit in terms of its components and connections, various symbols and abbreviations represent different aspects of the circuit's components, particularly transistors. In this work, `W`, `L`, and `M` are decision variables typically determined by optimization algorithms, while `sd`, `ad`, `as`, `ps`, `pd`, `nrs`, `nrd`, `sa`, and `sb` are layout-dependent parameters that vary based on the process node used in circuit fabrication. Let's break these down for clarity:

1. **W (Width):** This represents the width of the transistor channel. It's a critical parameter influencing the current-carrying capability of the transistor.

2. **L (Length):** This is the length of the transistor channel. It affects various aspects of transistor behavior, including its current drive, speed, and threshold voltage.

3. **M (Multiplicity):** Refers to the number of identical transistors used in parallel. This is often used to scale up the current-driving ability of a circuit element while maintaining other characteristics.

The other parameters (`sd`, `ad`, `as`, `ps`, `pd`, `nrs`, `nrd`, `sa`, and `sb`) are related to the physical layout of the transistor in the integrated circuit and are influenced by the specific manufacturing process:

4. **sd:** The area of the source and drain regions of the transistor.

5. **ad , as:** Separate specifications for the areas of the drain and source regions.

6. **ps , pd:** These parameters specify the perimeters of the source and drain regions, which can be important for understanding parasitic capacitances and resistances.

7. **nrs , nrd:** These are used in calculating the sheet resistance of the source and drain regions.

8. **sa , sb:** These parameters define the spacing between multiple fingers of a multi-finger transistor layout.

Each of these parameters plays a vital role in defining the electrical characteristics and performance of the transistor in the circuit. The values for `W`, `L`, and `M` are often determined through optimization algorithms to meet specific performance criteria, while the layout-dependent parameters are calculated based on the constraints and capabilities of the manufacturing process technology.

### Three-output voltage reference

- "SPICE" file

```verilog
.TEMP 25
.option post=1
.option accuracy=1
.option symb=1
.option dcon=1
.option captab
.option nomod

*File path of PDK
.lib "PDK file" pre_simu 
.lib "PDK file" ss_lib 
*File path of netlist
.include "netlist file"

VGNDA GND 0 0
VDDA VDD 0 1.8
.op

**sweep temp **
.dc temp -20 80 0.01
.measure dc VTC1 FIND V(vref1) AT 25
.measure dc maxval1 MAX V(vref1) from=-20 to=80
.measure dc minval1 MIN V(vref1) from=-20 to=80
.measure dc avgval1 AVG V(vref1) from=-20 to=80
.measure dc ppavl1  PP V(vref1) from=-20 to=80
.measure dc TC1 param='ppavl1/avgval1/100'

.measure dc VTC2 FIND V(vref2) AT 25
.measure dc maxval2 MAX V(vref2) from=-20 to=80
.measure dc minval2 MIN V(vref2) from=-20 to=80
.measure dc avgval2 AVG V(vref2) from=-20 to=80
.measure dc ppavl2  PP V(vref2) from=-20 to=80
.measure dc TC2 param='ppavl2/avgval2/100'

.measure dc VTC3 FIND V(vref3) AT 25
.measure dc maxval3 MAX V(vref3) from=-20 to=80
.measure dc minval3 MIN V(vref3) from=-20 to=80
.measure dc avgval3 AVG V(vref3) from=-20 to=80
.measure dc ppavl3  PP V(vref3) from=-20 to=80
.measure dc TC3 param='ppavl3/avgval3/100'

.DC VDDA 0 2.5 0.01
.measure dc VLS1 FIND V(vref1) AT 1.8
.measure dc maxval4 MAX V(vref1) from=1 to=2.5
.measure dc minval4 MIN V(vref1) from=1 to=2.5
.measure dc avgval4 AVG V(vref1) from=1 to=2.5
.measure dc ppavl4  PP V(vref1) from=1 to=2.5
.measure dc LS1 param='ppavl4/avgval4/1.5'

.measure dc VLS2 FIND V(vref2) AT 1.8
.measure dc maxval5 MAX V(vref2) from=1 to=2.5
.measure dc minval5 MIN V(vref2) from=1 to=2.5
.measure dc avgval5 AVG V(vref2) from=1 to=2.5
.measure dc ppavl5  PP V(vref2) from=1 to=2.5
.measure dc LS2 param='ppavl5/avgval5/1.5'

.measure dc VLS3 FIND V(vref3) AT 1.8
.measure dc maxval6 MAX V(vref3) from=1 to=2.5
.measure dc minval6 MIN V(vref3) from=1 to=2.5
.measure dc avgval6 AVG V(vref3) from=1 to=2.5
.measure dc ppavl6  PP V(vref3) from=1 to=2.5
.measure dc LS3 param='ppavl6/avgval6/1.5'

* Alter the process corners
.alter 
*File path of PDK
.del lib "PDK file" ss_lib 
.lib "PDK file" ff_lib 
.alter 
*File path of PDK
.del lib "PDK file" ff_lib 
.lib "PDK file" fs_lib 
.alter 
*File path of PDK
.del lib "PDK file" fs_lib 
.lib "PDK file" sf_lib 
.end
```

- Netlist

```verilog
mo1a vref1 vref1 net3 gnd NMOS W= 0.22u L=10.0u m=1 nf=1 sd=540e-9 ad=1.056e-13 as=1.056e-13 ps=1.4e-06 pd=1.4e-06 nrs=1.2273 nrd=1.2273 sa=4.8e-07 sb=4.8e-07  
mo1b net3 vref1 net4 gnd NMOS W=0.22u L=10.0u m=1 nf=1 sd=540e-9 ad=1.056e-13 as=1.056e-13 ps=1.4e-06 pd=1.4e-06 nrs=1.2273 nrd=1.2273 sa=4.8e-07 sb=4.8e-07 
mo1c net4 vref1 vref2 gnd NMOS W=0.22u L=10.0u m=1 nf=1 sd=540e-9 ad=1.056e-13 as=1.056e-13 ps=1.4e-06 pd=1.4e-06 nrs=1.2273 nrd=1.2273 sa=4.8e-07 sb=4.8e-07 
mo2 vref2 vref1 vref3 gnd NMOS W=0.22u L=0.22u m=7.0 nf=1.0 sd=540e-9 ad=1.056e-13 as=1.056e-13 ps=1.4e-06 pd=1.4e-06 nrs=1.2273 nrd=1.2273 sa=4.8e-07 sb=4.8e-07 
mo3a vref3 vref2 net5 gnd NMOS W= 0.22u L=8.95u m=1 nf=1 sd=540e-9 ad=1.056e-13 as=1.056e-13 ps=1.4e-06 pd=1.4e-06 nrs=1.2273 nrd=1.2273 sa=4.8e-07 sb=4.8e-07 
mo3b net5 vref2 net1 gnd NMOS W= 0.22u L=8.95u m=1 nf=1 sd=540e-9 ad=1.056e-13 as=1.056e-13 ps=1.4e-06 pd=1.4e-06 nrs=1.2273 nrd=1.2273 sa=4.8e-07 sb=4.8e-07 
mo3c net1 vref2 net2 gnd NMOS W= 0.22u L=8.95u m=1 nf=1 sd=540e-9 ad=1.056e-13 as=1.056e-13 ps=1.4e-06 pd=1.4e-06 nrs=1.2273 nrd=1.2273 sa=4.8e-07 sb=4.8e-07 
mo3d net2 vref2 gnd gnd NMOS W= 0.22u L=8.95u m=1 nf=1 sd=540e-9 ad=1.056e-13 as=1.056e-13 ps=1.4e-06 pd=1.4e-06 nrs=1.2273 nrd=1.2273 sa=4.8e-07 sb=4.8e-07 
mp1 a a b gnd NMOS W=16.34u L=0.43u m=1.0 nf=1.0 sd=540e-9 ad=7.8432e-12 as=7.8432e-12 ps=3.364e-05 pd=3.364e-05 nrs=0.0165 nrd=0.0165 sa=4.8e-07 sb=4.8e-07
mp2 b a gnd gnd NMOS W=0.32u L=4.5u m=1.0 nf=1.0 sd=540e-9 ad=1.536e-13 as=1.536e-13 ps=1.6e-06 pd=1.6e-06 nrs=0.8438 nrd=0.8438 sa=4.8e-07 sb=4.8e-07 
mp3 vn vn b gnd NMOS W=5.98u L=8.84u m=1.0 nf=1.0 sd=540e-9 ad=2.8704e-12 as=2.8704e-12 ps=1.292e-05 pd=1.292e-05 nrs=0.0452 nrd=0.0452 sa=4.8e-07 sb=4.8e-07
mp4 vp vp gnd gnd NMOS W=0.22u L=10.0u m=1.0 nf=1.0 sd=540e-9 ad=1.056e-13 as=1.056e-13 ps=1.4e-06 pd=1.4e-06 nrs=1.2273 nrd=1.2273 sa=4.8e-07 sb=4.8e-07
ma1 av vn trail gnd NMOS W=0.25u L=10.0u m=1.0 nf=1.0 sd=540e-9 ad=1.2e-13 as=1.2e-13 ps=1.46e-06 pd=1.46e-06 nrs=1.08 nrd=1.08 sa=4.8e-07 sb=4.8e-07 
ma2 a1 vp trail gnd NMOS W=0.25u L=10.0u m=1.0 nf=1.0 sd=540e-9 ad=1.2e-13 as=1.2e-13 ps=1.46e-06 pd=1.46e-06 nrs=1.08 nrd=1.08 sa=4.8e-07 sb=4.8e-07  

ma3a trail vref1 net6 gnd NMOS W=0.22u L=10.0u m=1.0 nf=1.0 sd=540e-9 ad=1.056e-13 as=1.056e-13 ps=1.4e-06 pd=1.4e-06 nrs=1.2273 nrd=1.2273 sa=4.8e-07 sb=4.8e-07 
ma3b net6 vref1 net7 gnd NMOS W=0.22u L=10.0u m=1.0 nf=1.0 sd=540e-9 ad=1.056e-13 as=1.056e-13 ps=1.4e-06 pd=1.4e-06 nrs=1.2273 nrd=1.2273 sa=4.8e-07 sb=4.8e-07 
ma3c net7 vref1 gnd gnd NMOS W=0.22u L=10.0u m=1.0 nf=1.0 sd=540e-9 ad=1.056e-13 as=1.056e-13 ps=1.4e-06 pd=1.4e-06 nrs=1.2273 nrd=1.2273 sa=4.8e-07 sb=4.8e-07 
ma4 av a1 vdd vdd PMOS W=20.0u L=8.71u m=1.0 nf=1.0 sd=540e-9 ad=9.6e-12 as=9.6e-12 ps=4.096e-05 pd=4.096e-05 nrs=0.0135 nrd=0.0135 sa=4.8e-07 sb=4.8e-07 
ma5 a1 a1 vdd vdd PMOS W=20.0u L=8.71u m=1.0 nf=1.0 sd=540e-9 ad=9.6e-12 as=9.6e-12 ps=4.096e-05 pd=4.096e-05 nrs=0.0135 nrd=0.0135 sa=4.8e-07 sb=4.8e-07

ms1 vdd s1 vdd vdd PMOS l=2e-6 w=2e-6 m=1 nf=1 sd=540e-9 ad=9.6e-13 as=9.6e-13 ps=4.96e-06 pd=4.96e-06 nrs=0.135 nrd=0.135 sa=4.8e-07 sb=4.8e-07
ms2 av s1 gnd gnd NMOS l=2e-6 w=440e-9 m=1 nf=1 sd=540e-9 ad=2.112e-13 as=2.112e-13 ps=1.84e-06 pd=1.84e-06 nrs=0.6136 nrd=0.6136 sa=4.8e-07 sb=4.8e-07
ms3 s1 vref3 gnd gnd NMOS l=4e-6 w=220e-9 m=1 nf=1  sd=540e-9 ad=1.056e-13 as=1.056e-13 ps=1.4e-06 pd=1.4e-06 nrs=1.2273 nrd=1.2273 sa=4.8e-07 sb=4.8e-07

mm1 a av vdd vdd PMOS W=20.0u L=2.66u m=2.0 nf=1.0 sd=540e-9 ad=9.6e-12 as=9.6e-12 ps=4.096e-05 pd=4.096e-05 nrs=0.0135 nrd=0.0135 sa=4.8e-07 sb=4.8e-07  
mm2 vn av vdd vdd PMOS W=20.0u L=2.66u m=2.0 nf=1.0 sd=540e-9 ad=9.6e-12 as=9.6e-12 ps=4.096e-05 pd=4.096e-05 nrs=0.0135 nrd=0.0135 sa=4.8e-07 sb=4.8e-07
mm3 vp av vdd vdd PMOS W=20.0u L=2.66u m=8.0 nf=1.0 sd=540e-9 ad=9.6e-12 as=9.6e-12 ps=4.096e-05 pd=4.096e-05 nrs=0.0135 nrd=0.0135 sa=4.8e-07 sb=4.8e-07 
mm4 vref1 av vdd vdd PMOS W=20.0u L=2.66u m=1.0 nf=1.0 sd=540e-9 ad=9.6e-12 as=9.6e-12 ps=4.096e-05 pd=4.096e-05 nrs=0.0135 nrd=0.0135 sa=4.8e-07 sb=4.8e-07
```

### Sub-threshold voltage and current reference

- "SPICE" file

```verilog
.TEMP 25
.option post=2
.option accuracy=1
.option symb=1
.option dcon=1
.option captab
.option nomod

*File path of PDK
.lib "PDK file" pre_simu 
.lib "PDK file" ss_lib 
*File path of netlist
.include "netlist file"

VGNDA GND 0 0
*VDDA VDD 0 0.9
VDDA VDDA 0 0.9
.op

**sweep temp **
.DC temp -40 125 0.01
.measure dc VTC FIND V(VREF) AT 25
.measure dc maxval MAX V(VREF) from=-40 to=125
.measure dc minval MIN V(VREF) from=-40 to=125
.measure dc avgval AVG V(VREF) from=-40 to=125
.measure dc ppavl  PP V(VREF) from=-40 to=125
.measure dc TC param='ppavl/avgval/165'

*.DC VDDA 0 2.1 0.01
*.measure dc VLS FIND V(VREF) AT 1.8
*.measure dc maxval1 MAX V(VREF) from=0.5 to=2
*.measure dc minval1 MIN V(VREF) from=0.5 to=2
*.measure dc avgval1 AVG V(VREF) from=0.5 to=2
*.measure dc ppavl1  PP V(VREF) from=0.5 to=2
*.measure dc LS param='ppavl1/avgval1/1.5'


.dc VDDA 0 2 0.01 
.measure dc VLS FIND V(VREF) AT 1.8
.measure dc maxval1 MAX V(VREF) from=0.8 to=2
.measure dc minval1 MIN V(VREF) from=0.8 to=2
.measure dc avgval1 AVG V(VREF) from=0.8 to=2
.measure dc ppavl1  PP V(vref)  from=0.8 to=2
.measure dc LS param='ppavl1/avgval1/1.2'

* Alter the process corners
.alter 
*File path of PDK
.del lib "PDK file" ss_lib 
.lib "PDK file" ff_lib 
.alter 
*File path of PDK
.del lib "PDK file" ff_lib 
.lib "PDK file" fs_lib 
.alter 
*File path of PDK
.del lib "PDK file" fs_lib 
.lib "PDK file" sf_lib 
.end

```



- Netlist

```verilog
.subckt OPA ib out va vb vcc vdd 
mb1 ib ib vcc vcc NMOS W=7.23u L=2.85u m=1.0 nf=1.0 sd=540e-9 ad=3.4704e-12 as=3.4704e-12 ps=1.542e-05 pd=1.542e-05 nrs=0.0373 nrd=0.0373 sa=4.8e-07 sb=4.8e-07
mb2 net1 ib vcc vcc NMOS W=7.23u L=2.85u m=1.0 nf=1.0 sd=540e-9 ad=3.4704e-12 as=3.4704e-12 ps=1.542e-05 pd=1.542e-05 nrs=0.0373 nrd=0.0373 sa=4.8e-07 sb=4.8e-07
mb3 net1 net1 vdd vdd PMOS W=2.67u L=9.59u m=1.0 nf=1.0 sd=540e-9 ad=1.2816e-12 as=1.2816e-12 ps=6.3e-06 pd=6.3e-06 nrs=0.1011 nrd=0.1011 sa=4.8e-07 sb=4.8e-07
mb4 vbn net1 vdd vdd PMOS W=2.67u L=9.59u m=1.0 nf=1.0 sd=540e-9 ad=1.2816e-12 as=1.2816e-12 ps=6.3e-06 pd=6.3e-06 nrs=0.1011 nrd=0.1011 sa=4.8e-07 sb=4.8e-07
mb5 vbn vbn net7 vcc NMOS W=20.0u L=10.0u m=1.0 nf=1.0 sd=540e-9 ad=9.6e-12 as=9.6e-12 ps=4.096e-05 pd=4.096e-05 nrs=0.0135 nrd=0.0135 sa=4.8e-07 sb=4.8e-07  
mb6 net7 net7 vcc vcc NMOS W=20.0u L=10.0u m=1.0 nf=1.0 sd=540e-9 ad=9.6e-12 as=9.6e-12 ps=4.096e-05 pd=4.096e-05 nrs=0.0135 nrd=0.0135 sa=4.8e-07 sb=4.8e-07  
ma1 net9 net1 vdd vdd PMOS W=2.67u L=9.59u m=1.0 nf=1.0 sd=540e-9 ad=1.2816e-12 as=1.2816e-12 ps=6.3e-06 pd=6.3e-06 nrs=0.1011 nrd=0.1011 sa=4.8e-07 sb=4.8e-07
ma2 net25 va net9 vdd PMOS W=16.6u L=7.52u m=1.0 nf=1.0 sd=540e-9 ad=7.968e-12 as=7.968e-12 ps=3.416e-05 pd=3.416e-05 nrs=0.0163 nrd=0.0163 sa=4.8e-07 sb=4.8e-07 
ma3 net29 vb net9 vdd PMOS W=16.6u L=7.52u m=1.0 nf=1.0 sd=540e-9 ad=7.968e-12 as=7.968e-12 ps=3.416e-05 pd=3.416e-05 nrs=0.0163 nrd=0.0163 sa=4.8e-07 sb=4.8e-07 
ma4 net25 ib vcc vcc NMOS W=0.71u L=2.07u m=9.0 nf=1.0 sd=540e-9 ad=3.408e-13 as=3.408e-13 ps=2.38e-06 pd=2.38e-06 nrs=0.3803 nrd=0.3803 sa=4.8e-07 sb=4.8e-07
ma5 net29 ib vcc vcc NMOS W=0.71u L=2.07u m=9.0 nf=1.0 sd=540e-9 ad=3.408e-13 as=3.408e-13 ps=2.38e-06 pd=2.38e-06 nrs=0.3803 nrd=0.3803 sa=4.8e-07 sb=4.8e-07
ma6 net10 vbn net25 vcc NMOS W=20.0u L=10.0u m=9.0 nf=1.0 sd=540e-9 ad=9.6e-12 as=9.6e-12 ps=4.096e-05 pd=4.096e-05 nrs=0.0135 nrd=0.0135 sa=4.8e-07 sb=4.8e-07
ma7 out vbn net29 vcc NMOS W=20.0u L=10.0u m=9.0 nf=1.0 sd=540e-9 ad=9.6e-12 as=9.6e-12 ps=4.096e-05 pd=4.096e-05 nrs=0.0135 nrd=0.0135 sa=4.8e-07 sb=4.8e-07
ma8 net10 net10 vdd vdd PMOS W=3.0u L=10.0u m=4.0 nf=1.0 sd=540e-9 ad=1.44e-12 as=1.44e-12 ps=6.96e-06 pd=6.96e-06 nrs=0.09 nrd=0.09 sa=4.8e-07 sb=4.8e-07
ma9 out net10 vdd vdd PMOS W=3.0u L=10.0u m=4.0 nf=1.0 sd=540e-9 ad=1.44e-12 as=1.44e-12 ps=6.96e-06 pd=6.96e-06 nrs=0.09 nrd=0.09 sa=4.8e-07 sb=4.8e-07
.ends OPA 

xi1 ib net3 net4 net2 gnd vdda OPA 
c1 vref gnd 1e-12 
c0 vdda net3 1e-12 
q0 gnd gnd net2 pbhvnwpsub2_ga m=1 
m1 net2 net3 vdda vdda PMOS W=18.56u L=10.0u m=8.0 nf=1.0 sd=540e-9 ad=8.9088e-12 as=8.9088e-12 ps=3.808e-05 pd=3.808e-05 nrs=0.0145 nrd=0.0145 sa=4.8e-07 sb=4.8e-07
m2 net4 net3 vdda vdda PMOS W=37.12u L=10.0u m=8.0 nf=1.0 sd=540e-9 ad=1.7818e-11 as=1.7818e-11 ps=7.52e-05 pd=7.52e-05 nrs=0.0073 nrd=0.0073 sa=4.8e-07 sb=4.8e-07
m3 net8 net3 vdda vdda PMOS W=18.56u L=10.0u m=8.0 nf=1.0 sd=540e-9 ad=8.9088e-12 as=8.9088e-12 ps=3.808e-05 pd=3.808e-05 nrs=0.0145 nrd=0.0145 sa=4.8e-07 sb=4.8e-07
m4 net9 net3 vdda vdda PMOS W=18.56u L=10.0u m=8.0 nf=1.0 sd=540e-9 ad=8.9088e-12 as=8.9088e-12 ps=3.808e-05 pd=3.808e-05 nrs=0.0145 nrd=0.0145 sa=4.8e-07 sb=4.8e-07
m5 ib net3 vdda vdda PMOS W=2.94u L=8.73u m=1.0 nf=1.0 sd=540e-9 ad=1.4112e-12 as=1.4112e-12 ps=6.84e-06 pd=6.84e-06 nrs=0.0918 nrd=0.0918 sa=4.8e-07 sb=4.8e-07

m6 net8 net8 net10 gnd NMOS W=12.76u L=5.74u m=6.0 nf=1.0 sd=540e-9 ad=6.1248e-12 as=6.1248e-12 ps=2.648e-05 pd=2.648e-05 nrs=0.0212 nrd=0.0212 sa=4.8e-07 sb=4.8e-07
m7 net10 net8 net5 gnd NMOS W=1.13u L=3.79u m=1.0 nf=1.0 sd=540e-9 ad=5.424e-13 as=5.424e-13 ps=3.22e-06 pd=3.22e-06 nrs=0.2389 nrd=0.2389 sa=4.8e-07 sb=4.8e-07 
m8 net9 net9 vref gnd NMOS W=15.29u L=1.11u m=8.0 nf=1.0 sd=540e-9 ad=7.3392e-12 as=7.3392e-12 ps=3.154e-05 pd=3.154e-05 nrs=0.0177 nrd=0.0177 sa=4.8e-07 sb=4.8e-07 
m9 vref net9 net10 gnd NMOS W=2.64u L=10.0u m=2.0 nf=1.0 sd=540e-9 ad=1.2672e-12 as=1.2672e-12 ps=6.24e-06 pd=6.24e-06 nrs=0.1023 nrd=0.1023 sa=4.8e-07 sb=4.8e-07 
ms1 vdda net1 vdda vdda PMOS W=7.06u L=0.18u m=10.0 nf=1.0 sd=540e-9 ad=3.3888e-12 as=3.3888e-12 ps=1.508e-05 pd=1.508e-05 nrs=0.0382 nrd=0.0382 sa=4.8e-07 sb=4.8e-07
ms2 net3 net1 gnd gnd NMOS W=1.98u L=0.85u m=1.0 nf=1.0 sd=540e-9 ad=9.504e-13 as=9.504e-13 ps=4.92e-06 pd=4.92e-06 nrs=0.1364 nrd=0.1364 sa=4.8e-07 sb=4.8e-07 
ms3 net1 vref gnd gnd NMOS W=6.27u L=0.18u m=1.0 nf=1.0 sd=540e-9 ad=3.0096e-12 as=3.0096e-12 ps=1.35e-05 pd=1.35e-05 nrs=0.0431 nrd=0.0431 sa=4.8e-07 sb=4.8e-07 
xr2  net5 gnd rppolyhri3k W=0.73u L=1000.0u m=1.0 s=1.0 mf=1 
xr1_1__dmy0  net4 xr1_1__dmy0  rppolyhri3k W=0.42u L=1000.0u m=1.0 s=1.0 mf=1 
xr1_2__dmy0  xr1_1__dmy0 net5  rppolyhri3k W=0.42u L=1000.0u m=1.0 s=1.0 mf=1 
```

