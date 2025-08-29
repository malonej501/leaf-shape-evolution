"""For checking the completeness of evolutionary simulation outputs and merging
simulations generated over multiple executions."""
import os
import shutil
import sys
from pdict import leafids

ROOT = "../test"  # root directory containing separate leaffinder runs
M_DIR = ROOT + "_merged"  # new directory to store merged leaf directories
WALK_LEN = None  # expected walk_length for check_steps
MODE = 1  # specify whether to copy leaf directories with WALK_LEN pngs or
# steps in each walk directory 0 - pngs, 1 - steps
LOGGING = False  # log console output to file


def initialise():
    """Create the new directory to store merged leaf directories"""

    if not os.path.exists(M_DIR):
        os.mkdir(M_DIR)
    else:
        print(f"Error: Directory {M_DIR} already exists.")
        sys.exit(1)


def walk_merge():
    """
    Iterates through separate leaffinder runs in a root directory and copies 
    all leaf directories that contain WALK_LEN .pngs or steps in each walk 
    directory to a new run_dir
    """

    for run_dir in os.listdir(ROOT):  # loop through runs
        run_path = os.path.join(ROOT, run_dir)
        if not os.path.isdir(run_path):
            continue
        for leaf_dir in os.listdir(run_path):  # loop through leaves
            leaf_path = os.path.join(run_path, leaf_dir)
            if not os.path.isdir(leaf_path):
                continue
            # assume leafid has all complete walks until we find one incomplete
            all_present = True
            for walk_dir in os.listdir(leaf_path):  # loop through walks
                walk_path = os.path.join(leaf_path, walk_dir)
                if not os.path.isdir(walk_path):
                    continue
                if MODE == 0:
                    png_files = [f for f in os.listdir(
                        walk_path) if f.endswith('.png')]
                    # stop looping through walks if fewer than 80 .pngs present
                    if len(png_files) < WALK_LEN:
                        print(f"No. pngs < {WALK_LEN} in {walk_path}")
                        all_present = False
                        break
                if MODE == 1:
                    # get step numbers from .png file names
                    steps = set([int(file.split('_')[-2]) for file in
                                 os.listdir(walk_path) if
                                 file.endswith('.png')])
                    if WALK_LEN - 1 not in steps:  # stop loop if no last step
                        print(f"No. steps < {WALK_LEN} in {walk_path}")
                        all_present = False
                        break

            if not all_present:
                continue
            # If all walks are present, copy the leaf directory
            if MODE == 0:
                print(f"All {WALK_LEN} pngs in {leaf_path}")
            if MODE == 1:
                print(f"All {WALK_LEN} steps in {leaf_path}")
            dest_path = os.path.join(M_DIR, leaf_dir)
            shutil.copytree(leaf_path, dest_path)
            print(f"Copied {leaf_path} to {dest_path}")


def check_complete():
    """Check that all leaf directories have been copied to the new directory"""

    all_present = True

    for leafid in leafids:
        leaf_path = os.path.join(M_DIR, leafid)
        if not os.path.exists(leaf_path):
            print(f"Error: {leaf_path} not present in {M_DIR}")
            all_present = False

    if all_present:
        print(f"All {len(leafids)} leaf directories have been copied "
              f"successfully into {M_DIR}")


def check_steps(root):
    """Check that there is a png for each step in each walk directory of a 
    multi-run and return the run/walk/step for which png is missing"""

    expected_steps = set(range(0, WALK_LEN))
    all_present = True

    for run_dir in os.listdir(root):  # loop through runs
        run_path = os.path.join(root, run_dir)
        if not os.path.isdir(run_path):
            continue
        for leaf_dir in os.listdir(run_path):  # loop through leaves
            leaf_path = os.path.join(run_path, leaf_dir)
            if not os.path.isdir(leaf_path):
                continue
            for walk_dir in os.listdir(leaf_path):  # loop through walks
                walk_path = os.path.join(leaf_path, walk_dir)
                if os.path.isdir(walk_path):
                    continue
                steps = set(
                    [int(file.split('_')[-2]) for file in
                     os.listdir(walk_path) if file.endswith('.png')])
                # if last step present, return steps with missing pngs
                if WALK_LEN - 1 in steps:
                    missing = expected_steps.symmetric_difference(
                        steps)
                    if missing:
                        all_present = False
                        print(
                            f"Missing .png for {walk_path}: {missing}")

    if all_present:
        print("All steps are associated with a .png!")
    else:
        print("Some steps are not associated with a .png! This will result "
              "in not all leaf directories being copied or missing steps in "
              "the final merged dataset.")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)


def check_empty(dr):
    """Check which directories contain leaf images."""

    no_empty_walks = 0
    grand_total_pngs = 0
    # get no. walks in a leaf dir
    n_walks = len(os.listdir(f"{dr}/{leafids[0]}"))
    exp_grand_total_pngs = WALK_LEN * len(leafids) * n_walks
    incomplete_leafids = []
    complete_leafids = []
    for leaf_dir in os.listdir(dr):
        complete = True  # assume leafid complete until we find one empty walk
        leaf_path = os.path.join(dr, leaf_dir)
        if os.path.isdir(leaf_path):
            for walk_dir in os.listdir(leaf_path):
                walk_path = os.path.join(leaf_path, walk_dir)
                if os.path.isdir(walk_path):
                    steps = set([int(file.split('_')[-2])
                                for file in os.listdir(walk_path)
                                if file.endswith('.png')])
                    total_pngs = len(steps)
                    grand_total_pngs += total_pngs
                    if not steps:  # check if completely empty
                        # print(f"{walk_path}")
                        complete = False
                        no_empty_walks += 1
                    elif WALK_LEN - 1 not in steps:  # check if partially empty
                        # print(f"Incomplete directory: {walk_path}")
                        # print partially full walk directories
                        print(f"{walk_path} {total_pngs} #")
                        complete = False
                    else:
                        # print full walk directories
                        print(f"{walk_path} {total_pngs} ###")
            if complete:
                complete_leafids.append(leaf_dir)
            else:
                incomplete_leafids.append(leaf_dir)

    print(f"Expected no. walks per leafid: {n_walks}")
    print(f"Total pngs found: {grand_total_pngs} of {exp_grand_total_pngs} "
          f"expected ({grand_total_pngs/exp_grand_total_pngs*100:.2f}%)")
    print(f"Complete leaf directories found in {dr}: {complete_leafids}")
    print(f"Empty walks found in {dr}: {incomplete_leafids}")
    print(f"Total incomplete leaf directories: {len(incomplete_leafids)} "
          f"of {len(leafids)} "
          f"expected ({len(incomplete_leafids)/len(leafids)*100:.2f}%)")
    print(f"Total empty walks: {no_empty_walks} of {len(leafids) * n_walks} "
          f"expected ({no_empty_walks/(len(leafids) * n_walks)*100:.2f}%)")


def print_help():
    """Print help message for the script"""
    help_message = """
    Usage: python3 walk_merge.py [options]

    Options:
        -h              Show this help message and exit.
        -id [string]    Specify the root directory containing separate 
                        leaffinder runs, or separate walks if using -f 1.
        -l              Enable logging to file.
        -wl [int]       Specify the expected walk length to check if walks are 
                        complete.
        -f  [function]  Pass function you want to perform:
                        0   ...merge leaf directories
                        1   ...print empty walk directories
    """
    print(help_message)


if __name__ == "__main__":
    args = sys.argv[1:]
    if "-h" in args:
        print_help()
    else:
        if "-id" in args:
            ROOT = str(args[args.index("-id") + 1])
            M_DIR = ROOT + "_merged"
        else:
            print(f"WARNING: No run_id specified, defaulting to {ROOT}")
        if "-l" in args:
            LOGGING = True
        if "-wl" in args:
            WALK_LEN = int(args[args.index("-wl") + 1])
        if "-f" in args:
            func = int(args[args.index("-f") + 1])
            if func == 0:
                print("Merge parameters:")
                print(f"ROOT = {ROOT}")
                print(f"M_DIR = {M_DIR}")
                print(f"WALK_LEN = {WALK_LEN}")
                print(f"MODE = {MODE}")
                check_steps(ROOT)
                initialise()
                walk_merge()
                check_complete()
            if func == 1:
                if LOGGING:  # save console output to log file in ROOT
                    log_file = open(f"{ROOT}/leaf_counts_wl{WALK_LEN}.log",
                                    "w", encoding="utf-8")
                    sys.stdout = log_file
                print("Checking for empty walk directories in:")
                print(f"ROOT = {ROOT}")
                print(f"WALK_LEN = {WALK_LEN}")
                check_empty(ROOT)
                if LOGGING:
                    log_file.close()
                    sys.stdout = sys.__stdout__
