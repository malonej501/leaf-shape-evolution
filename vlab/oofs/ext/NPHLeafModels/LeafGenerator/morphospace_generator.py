"""This is for generating the morphospace plot seen in figure 2."""
import pwriter
from pdict import *
import subprocess
import os

wd = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
wid = 0
step = 0
leafid = "pc1_alt"
leafid_index = leafids.index(leafid)
plist = [ele[leafid_index] for ele in list(pdict.values())]
print(leafid, plist)
plist[14] = 1
plist[87] = "M_PI*0.2"
plist[88] = "M_PI*0.3"
plist[115] = 0  # no. morphogens visible
plist[48] = 0.09  # Fairing2
plist[50] = 0.09  # Stretch2
plist[61]  # Fairing3
plist[74]  # Fairing4]
plist[4]  # FinalFrame


# Define the point in parameter space that you want to generate the images for
fairings = [0.001, 35, 70, 105, 140]
final_frames = [12000, 24000, 36000, 48000, 60000]

for ff in final_frames:
    for f in fairings:
        print(f"{leafid}_FF{ff}_F{f}")
        plist[4] = ff
        # plist[61] = f
        plist[61] = 0.08
        plist[74] = f
        # plist[74] = 0.009

        input_parameters = dict(zip(pdict.keys(), plist))

        pwriter.Input(
            input_parameters, wd + f"/bin{wid}", wid
        )  # + f"/parameters{wid}")

        process = subprocess.Popen(
            wd +
            f"/LeafGenerator/Start.sh bin{wid} plant{wid}.l FF{ff}_F{f}.png",
            shell=True,
            stdout=subprocess.PIPE,
            bufsize=1,
            universal_newlines=True,
            preexec_fn=os.setpgrp,
        )

        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(f"Error in subprocess: {stderr}")
        else:
            print(f"Process completed successfully: {stdout}")
