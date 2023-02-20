import random
from pathlib import Path

def gen_rou_varying59(flow_ns,flow_ew):
    prob = []
    random.seed(42)
    path = str(Path("nets/single-intersection/two_way/single-intersection_dynamic_{}_{}.rou.xml".format(flow_ns,flow_ew)).absolute())
    with open(path, "w") as routes:
        print("""<routes>
        <vType id="typeWE" accel="2.6" decel="4.5" sigma="0" length="5" minGap="2.5" maxSpeed="15.65" \
guiShape="passenger"/>
        <vType id="typeEW" accel="2.6" decel="4.5" sigma="0" length="5" minGap="2.5" maxSpeed="15.65" \
guiShape="passenger"/>
        <vType id="typeNS" accel="2.6" decel="4.5" sigma="0" length="5" minGap="2.5" maxSpeed="15.65" \
        guiShape="passenger"/>
        <vType id="typeSN" accel="2.6" decel="4.5" sigma="0" length="5" minGap="2.5" maxSpeed="15.65" \
        guiShape="passenger"/>
        <route id="eastbound" edges="1i 2i" />
        <route id="westbound" edges="-2i -1i" />
        <route id="northbound" edges="-3i 4i" />
        <route id="southbound" edges="-4i 3i" />
        """, file=routes)
        for rou in ["northbound","southbound"]:
            print('     <flow id="{}_{}" route="{}" begin="0" end="600" probability="{}" type="typeWE" \
                departSpeed="max" departPos="base" departLane="best"/>'.format(rou,0,rou,400/3600),file=routes)
  
        for rou in ["westbound","eastbound"]:
            print('    <flow id="{}_{}" route="{}" begin="0" end="600" probability="{}" type="typeWE" \
                departSpeed="max" departPos="base" departLane="best"/>'.format(rou,0,rou,700/3600),file=routes)

        for rou in ["northbound","southbound"]:
            print('     <flow id="{}_{}" route="{}" begin="600" end="1200" probability="{}" type="typeWE" \
                departSpeed="max" departPos="base" departLane="best"/>'.format(rou,1,rou,600/3600),file=routes)

        for rou in ["westbound","eastbound"]:
            print('    <flow id="{}_{}" route="{}" begin="600" end="1200" probability="{}" type="typeWE" \
                departSpeed="max" departPos="base" departLane="best"/>'.format(rou,1,rou,1100/3600),file=routes)

        for rou in ["northbound","southbound"]:
            print('     <flow id="{}_{}" route="{}" begin="1200" end="1800" probability="{}" type="typeWE" \
                departSpeed="max" departPos="base" departLane="best"/>'.format(rou,2,rou,500/3600),file=routes)

        for rou in ["westbound","eastbound"]:
            print('    <flow id="{}_{}" route="{}" begin="1200" end="1800" probability="{}" type="typeWE" \
                departSpeed="max" departPos="base" departLane="best"/>'.format(rou,2,rou,900/3600),file=routes)
        print("</routes>", file=routes)   
    return 

def gen_rou_varying77(flow_ns,flow_ew):
    prob = []
    random.seed(42)
    path = str(Path("nets/single-intersection/two_way/single-intersection_dynamic_{}_{}.rou.xml".format(flow_ns,flow_ew)).absolute())
    with open(path, "w") as routes:
        print("""<routes>
        <vType id="typeWE" accel="2.6" decel="4.5" sigma="0" length="5" minGap="2.5" maxSpeed="15.65" \
guiShape="passenger"/>
        <vType id="typeEW" accel="2.6" decel="4.5" sigma="0" length="5" minGap="2.5" maxSpeed="15.65" \
guiShape="passenger"/>
        <vType id="typeNS" accel="2.6" decel="4.5" sigma="0" length="5" minGap="2.5" maxSpeed="15.65" \
        guiShape="passenger"/>
        <vType id="typeSN" accel="2.6" decel="4.5" sigma="0" length="5" minGap="2.5" maxSpeed="15.65" \
        guiShape="passenger"/>
        <route id="eastbound" edges="1i 2i" />
        <route id="westbound" edges="-2i -1i" />
        <route id="northbound" edges="-3i 4i" />
        <route id="southbound" edges="-4i 3i" />
        """, file=routes)
        for rou in ["northbound","southbound"]:
            print('     <flow id="{}_{}" route="{}" begin="0" end="600" probability="{}" type="typeWE" \
                departSpeed="max" departPos="base" departLane="best"/>'.format(rou,0,rou,600/3600),file=routes)
  
        for rou in ["westbound","eastbound"]:
            print('    <flow id="{}_{}" route="{}" begin="0" end="600" probability="{}" type="typeWE" \
                departSpeed="max" departPos="base" departLane="best"/>'.format(rou,0,rou,600/3600),file=routes)

        for rou in ["northbound","southbound"]:
            print('     <flow id="{}_{}" route="{}" begin="600" end="1200" probability="{}" type="typeWE" \
                departSpeed="max" departPos="base" departLane="best"/>'.format(rou,1,rou,800/3600),file=routes)

        for rou in ["westbound","eastbound"]:
            print('    <flow id="{}_{}" route="{}" begin="600" end="1200" probability="{}" type="typeWE" \
                departSpeed="max" departPos="base" departLane="best"/>'.format(rou,1,rou,900/3600),file=routes)

        for rou in ["northbound","southbound"]:
            print('     <flow id="{}_{}" route="{}" begin="1200" end="1800" probability="{}" type="typeWE" \
                departSpeed="max" departPos="base" departLane="best"/>'.format(rou,2,rou,700/3600),file=routes)

        for rou in ["westbound","eastbound"]:
            print('    <flow id="{}_{}" route="{}" begin="1200" end="1800" probability="{}" type="typeWE" \
                departSpeed="max" departPos="base" departLane="best"/>'.format(rou,2,rou,600/3600),file=routes)
        print("</routes>", file=routes)   
    return 


def gen_rou_varying37(flow_ns,flow_ew):
    prob = []
    random.seed(42)
    path = str(Path("nets/single-intersection/two_way/single-intersection_dynamic_{}_{}.rou.xml".format(flow_ns,flow_ew)).absolute())
    with open(path, "w") as routes:
        print("""<routes>
        <vType id="typeWE" accel="2.6" decel="4.5" sigma="0" length="5" minGap="2.5" maxSpeed="15.65" \
guiShape="passenger"/>
        <vType id="typeEW" accel="2.6" decel="4.5" sigma="0" length="5" minGap="2.5" maxSpeed="15.65" \
guiShape="passenger"/>
        <vType id="typeNS" accel="2.6" decel="4.5" sigma="0" length="5" minGap="2.5" maxSpeed="15.65" \
        guiShape="passenger"/>
        <vType id="typeSN" accel="2.6" decel="4.5" sigma="0" length="5" minGap="2.5" maxSpeed="15.65" \
        guiShape="passenger"/>
        <route id="eastbound" edges="1i 2i" />
        <route id="westbound" edges="-2i -1i" />
        <route id="northbound" edges="-3i 4i" />
        <route id="southbound" edges="-4i 3i" />
        """, file=routes)
        for rou in ["northbound","southbound"]:
            print('     <flow id="{}_{}" route="{}" begin="0" end="600" probability="{}" type="typeWE" \
                departSpeed="max" departPos="base" departLane="best"/>'.format(rou,0,rou,200/3600),file=routes)
  
        for rou in ["westbound","eastbound"]:
            print('    <flow id="{}_{}" route="{}" begin="0" end="600" probability="{}" type="typeWE" \
                departSpeed="max" departPos="base" departLane="best"/>'.format(rou,0,rou,500/3600),file=routes)

        for rou in ["northbound","southbound"]:
            print('     <flow id="{}_{}" route="{}" begin="600" end="1200" probability="{}" type="typeWE" \
                departSpeed="max" departPos="base" departLane="best"/>'.format(rou,1,rou,400/3600),file=routes)

        for rou in ["westbound","eastbound"]:
            print('    <flow id="{}_{}" route="{}" begin="600" end="1200" probability="{}" type="typeWE" \
                departSpeed="max" departPos="base" departLane="best"/>'.format(rou,1,rou,900/3600),file=routes)

        for rou in ["northbound","southbound"]:
            print('     <flow id="{}_{}" route="{}" begin="1200" end="1800" probability="{}" type="typeWE" \
                departSpeed="max" departPos="base" departLane="best"/>'.format(rou,2,rou,300/3600),file=routes)

        for rou in ["westbound","eastbound"]:
            print('    <flow id="{}_{}" route="{}" begin="1200" end="1800" probability="{}" type="typeWE" \
                departSpeed="max" departPos="base" departLane="best"/>'.format(rou,2,rou,700/3600),file=routes)
        print("</routes>", file=routes)   
    return

gen_rou_varying59("500.0","900.0")
gen_rou_varying77("700.0","700.0")
gen_rou_varying37("300.0","700.0")