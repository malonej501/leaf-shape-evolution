"""For fitting continuous time Markov chain to phylogenies using BayesTraits
and plotting the trace for MCMC."""
import os
import shutil
import sys
import multiprocessing
import subprocess
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

# set default parameter values
RUN_ID = "test"
METHOD = "MCMC"  # or "MLE"
PRIOR = 0  # 0 - uniform(0,0.1), 1 - uniform(0,100)
ITERATIONS = 1100000  # length of chain default 1100000
ML_DATA = "ML_9_species_genus_30-07-25"  # "ML_8_species_genus"  # ML_6_genus
BURNIN = 100000  # to discard from posteriors - default 100000
DATASETS = [
    # "geeta_phylo_geeta_class",
    # "jan_genus_phylo_nat_26-09-24_class",  # these are the genus trees
    # "zun_genus_phylo_nat_26-09-24_class",
    "jan_nat_species_11-07-25",
    "zun_nat_species_11-07-25",
]
LOG_EXCL = [  # columns to exclude from the log file
    # "Iteration",
    "Tree No",
    "Model string",
    "Unnamed: 19",
    "Unnamed: 22",
]

RATE_MAP = {
    "q01": "ul",
    "q02": "ud",
    "q03": "uc",
    "q10": "lu",
    "q12": "ld",
    "q13": "lc",
    "q20": "du",
    "q21": "dl",
    "q23": "dc",
    "q30": "cu",
    "q31": "cl",
    "q32": "cd",
}
# plt.rcParams["font.family"] = "CMU Serif"
# plt.rcParams["font.size"] = 12


def import_data():
    """Import alltrees and labels from the data directory."""
    trees = []
    for file in os.listdir("data"):
        if file.endswith(("class.tre", "class.txt")):
            file_path = os.path.join("data", file)
            base = file_path.split(".")[0]
            if base not in trees:
                trees.append(base)

    files = sorted(tuple((f"{tree}.tre", f"{tree}.txt") for tree in trees))

    return files


def get_log(file):
    """Read the log file, prune the header and return a DataFrame the data."""
    log = None
    with open(file, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

        # find the start of the log table
        start_index = None
        for i, line in enumerate(lines):
            if "Tree No" in line:
                start_index = i
                break
        fh.seek(0)
        log = pd.read_csv(fh, sep="\t", skiprows=start_index)

    return log


def get_ml_rates(directory):
    """Retrieve maximum likelihood rates from all log files in the specified
    directory, combine and save to csv."""
    logs = []
    for file in os.listdir(directory):
        if file.endswith(".Log.txt"):
            data_name = file.rsplit(".")[0]
            filepath = os.path.join(directory, file)
            log = get_log(filepath)
            log_mean = log.mean()
            log_mean_df = pd.DataFrame([log_mean], columns=log.columns)
            log_mean_df.insert(0, "dataset", data_name)
            logs.append(log_mean_df)

    ml = pd.concat(logs, axis=0, ignore_index=True)
    ml.sort_values(by="dataset", inplace=True)
    if "Tree No" in ml.columns:
        ml.drop("Tree No", axis=1, inplace=True)
    if "Unnamed: 18" in ml.columns:
        ml.drop("Unnamed: 18", axis=1, inplace=True)
    ml.to_csv(directory + "/mean_rates_all.csv", index=False)

    return ml


def get_marginal_likelihood():
    """Get marginal likelihoods from all .Stones.txt files in the data 
    directory and save to .csv."""

    marglhs = []
    for file in os.listdir("data"):
        if file.endswith(".Stones.txt"):
            data_name = file.rsplit(".")[0]
            filepath = os.path.join("data", file)
            with open(filepath, "r", encoding="utf-8") as fh:
                lines = fh.readlines()
                # find marginal likelihood value
                for line in lines:
                    if "Log marginal likelihood" in line:
                        marglh_val = float(line.rsplit("\t")[1])
                        marglh = pd.DataFrame(
                            {
                                "dataset": [data_name],
                                "log_marginal_likelihood": [marglh_val],
                            }
                        )
                        marglhs.append(marglh)
    marglhs_df = pd.concat(marglhs, axis=0, ignore_index=True)
    marglhs_df.sort_values(by="dataset", inplace=True)
    marglhs_df.reset_index(inplace=True, drop=True)

    marglhs_df.to_csv(
        f"data/{RUN_ID}_log_marginal_likelihoods_all.csv", index=False)
    return marglhs_df


def plot_trace_full(file, export):
    """Plot the trace of the log file, excluding burn-in iterations."""
    save_fig_path = file.rsplit("/", 1)[0]
    log_file_name = file.rsplit("/", 1)[1]
    data_name = log_file_name.rsplit(".")[0]

    # Get ML data for comparison
    ml = pd.read_csv(f"data/{ML_DATA}/mean_rates_all.csv")
    # ML_DATA = pd.read_csv("data/ML_scaletrees0.001_1/mean_rates_all.csv")
    ml = ml[ml["dataset"] == data_name].reset_index()

    log = get_log(file)
    # summary statistics
    print(log.describe())

    # export just the cleaned posteriors from the log file
    log_no_burnin = log.loc[log["Iteration"] >= BURNIN]

    rates = log_no_burnin.filter(like="q", axis=1)
    rates.to_csv(save_fig_path + f"/{data_name}_{RUN_ID}.csv", index=False)

    print(log)
    num_vars = len([var for var in log.columns if var not in LOG_EXCL])

    fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(15, 12), sharex=True)

    # If only one subplot, `axes` is not a list, so we handle that case
    if num_vars == 1:
        axes = [axes]
    axes = axes.flatten()

    counter = 0
    for var in log.columns:
        if var not in LOG_EXCL:
            axes[counter].plot(log["Iteration"], log[var])
            axes[counter].set_ylabel(var)
            axes[counter].grid(True)
            # Add ML values
            if var in ml.columns:
                axes[counter].axhline(
                    y=ml.loc[0, var], linestyle="--", color="C1", label="ML"
                )
            counter += 1

    fig.text(0.5, 0.01, "Iteration", ha="center")
    fig.suptitle(data_name)
    fig.tight_layout()
    # Adding a legend

    if export:
        plt.savefig(save_fig_path +
                    f"/{RUN_ID}_{data_name}_trace_single.pdf", format="pdf")

    # plt.show()
    return fig


def plot_trace(file, export):
    """Plot the trace of the log file, excluding burn-in iterations."""
    save_fig_path = file.rsplit("/", 1)[0]
    log_file_name = file.rsplit("/", 1)[1]
    data_name = log_file_name.rsplit(".")[0]

    # Get ML data for comparison
    ml = pd.read_csv(f"data/{ML_DATA}/mean_rates_all.csv")
    # ML_DATA = pd.read_csv("data/ML_scaletrees0.001_1/mean_rates_all.csv")
    ml = ml[ml["dataset"] == data_name].reset_index()

    log = get_log(file)
    # additional log columns to exclude
    LOG_EXCL.extend(
        ["Lh", "Root P(0)", "Root P(1)", "Root P(2)", "Root P(3)"])
    # ignore if not all columns in log_exclude are present
    log_filt = log.drop(columns=LOG_EXCL, errors="ignore")
    fig, axes = plt.subplots(
        nrows=4,
        ncols=4,
        figsize=(10, 7),
        sharex=True,
        sharey=True
    )

    idx = 1  # start from 1 to skip the iteration column
    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            if i == j:
                ax.axis("off")
                continue
            ax.plot(log_filt["Iteration"], log_filt.iloc[:, idx], c="C0")
            ax.set_ylabel(RATE_MAP.get(log_filt.columns[idx]))
            ax.tick_params(axis="y", which="both", labelleft=True)
            ax.axhline(ml.loc[0, log_filt.columns[idx]],
                       linestyle="--", color="C1", label="ML")
            # ax.set_ylim(0, 1)
            idx += 1

    fig.text(0.5, 0.01, "Iteration", ha="center")
    fig.tight_layout()
    # Adding a legend

    if export:
        plt.savefig(save_fig_path +
                    f"/{RUN_ID}_{data_name}_trace_single.pdf", format="pdf")

    return fig


def plot_marglhs():
    """Plot the marginal likelihoods for each dataset."""
    marglhs = get_marginal_likelihood()

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(
        marglhs["dataset"], marglhs["log_marginal_likelihood"], color="skyblue"
    )

    # Adding labels on top of the bars
    for br, label in zip(bars, marglhs["log_marginal_likelihood"]):
        height = br.get_height()
        ax.text(
            br.get_x() + br.get_width() / 2,
            height + 0.1,  # Adjusted for better visibility
            round(label, 2),
            ha="center",
            va="bottom",
        )
    ax.set_xticklabels(marglhs["dataset"], rotation=45)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Log Marginal Likelihood")
    plt.tight_layout()

    return fig


def run_bayestraits(tree, labels):
    """Run BayesTraits on the given tree and labels."""
    print("Running BayesTraits on:", tree, labels)
    result = subprocess.run(
        f"./BayesTraitsV4 {tree} {labels} < input.txt", shell=True,
        capture_output=True, text=True, check=True
    )
    return result.stdout


def write_input_file():
    """Write the input file for BayesTraits using specified 
    hyperparameters."""
    with open("input.txt", "w", encoding="utf-8") as f:
        if METHOD == "MLE":
            f.write(
                "1\n"  # Multistate
                "1\n"  # mle method
                "run\n"
            )
        if METHOD == "MCMC":
            pr = "0 0.1" if PRIOR == 0 else "0 100"
            f.write(
                "1\n"  # Multistate
                "2\n"  # mcmc method
                f"priorall uniform {pr}\n"
                "burnin 0\n"  # keep full log
                f"iterations {ITERATIONS}\n"
                "run\n"
            )


def run_select_trees(data):
    """Run BayesTraits on selected trees and labels, using multiprocessing."""
    run_dir = os.path.join("data", RUN_ID)
    os.makedirs(run_dir, exist_ok=True)
    write_input_file()

    processes = []
    for tree, labels in data:
        process = multiprocessing.Process(target=run_bayestraits,
                                          args=(tree, labels))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    if METHOD == "MCMC":
        with PdfPages(f"data/{RUN_ID}_trace.pdf") as pdf:
            for _, label in data:
                logfilepath = label + ".Log.txt"
                fig = plot_trace_full(logfilepath, export=False)
                pdf.savefig(fig)
            if os.path.exists(DATASETS[0] + ".txt.Stones.txt"):
                fig = plot_marglhs()
                pdf.savefig(fig)

    for file in os.listdir("data"):
        if file.endswith((".Log.txt", ".pdf", ".Stones.txt", ".csv")):
            source_file = os.path.join("data", file)
            destination_file = os.path.join(run_dir, file)
            if os.path.isfile(source_file):
                shutil.move(source_file, destination_file)
                print(f"Moved: {source_file} to {destination_file}")

    if METHOD == "MLE":
        get_ml_rates(run_dir)


def print_help():
    """Print the help message for the script."""
    help_message = """
    Usage: python3 inference.py [options]

    Options:
        -h              Show this help message and exit.
        -id [run id]    The name given to the mcmc/mle run.
        -d  [datasets]  Pass datasets you want to run inference on, 
                        separated by ",":
                        e.g. "ALL" or "jan_phylo_nat_class-geeta_
                        phylo_geeta_class"
        -ml [data]      Pass the ML dataset you want to compare the mcmc 
                        posteriors to.
                        e.g. "ML_7_species_genus"
        -f  [function]  Pass function you want to perform:
                        0   ...run mcmc inference
                        1   ...run mle inference
                        2   ...plot individual mcmc trace from file
    """
    print(help_message)


if __name__ == "__main__":
    args = sys.argv[1:]
    if "-h" in args:
        print_help()
    else:
        if "-id" in args:
            RUN_ID = str(args[args.index("-id") + 1])
        if "-d" in args:
            DATASETS = args[args.index("-d") + 1].split(",")
        if "-ml" in args:
            ML_DATA = str(args[args.index("-ml") + 1])
        if "-f" in args:
            func = int(args[args.index("-f") + 1])
            if DATASETS == ["ALL"]:  # load file names
                tree_data = import_data()
            else:
                tree_data = sorted(
                    tuple(
                        (f"data/{d}.tre", f"data/{d}.txt")
                        for d in DATASETS
                    )
                )
            # for i in tree_data:
            #     print(i)
            # exit()
            if func == 0:
                METHOD = "MCMC"
                run_select_trees(tree_data)
            if func == 1:
                METHOD = "MLE"
                run_select_trees(tree_data)
            if func == 2:
                for _, lab in tree_data:
                    # This assumes the log file is now in the run_id directory,
                    # rather than /data
                    logfilename = lab.replace("data/", "") + ".Log.txt"
                    LOGFILEPATH = f"data/{RUN_ID}/{logfilename}"
                    plot_trace(LOGFILEPATH, export=True)
                # plot_marglhs(run_id)
