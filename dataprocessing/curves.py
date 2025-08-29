"""For generating figures 3 and 8"""
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.offsetbox import (
    OffsetImage, AnnotationBbox, TextArea, VPacker)
from PIL import Image
from scipy import linalg, integrate


PLOT = 4  # type of plot to produce
# 1-two rows with mean model separated by initial state,
# 2-proportion of shapes over simulation time against model predictions
# 4-same as 2 but with sim and phylo data on separate rows
# 5-just the timeseries data
PHYLORATES1 = "zun_genus_phylo_nat_26-09-24_class_uniform0-0.1_genus_1"
PHYLORATES2 = "jan_genus_phylo_nat_26-09-24_class_uniform0-0.1_genus_1"
PHYLORATES_ML1 = "zun_genus_phylo_nat_26-09-24_class"
PHYLORATES_ML2 = "jan_genus_phylo_nat_26-09-24_class"
# PHYLORATES1 = "zun_nat_species_11-07-25_uniform0-0.1_species_11-07-25_1"
# PHYLORATES2 = "jan_nat_species_11-07-25_uniform0-0.1_species_11-07-25_1"
# PHYLORATES_ML1 = "zun_nat_species_11-07-25_class"
# PHYLORATES_ML2 = "jan_nat_species_11-07-25_class"
# ML_RATES = "ML6_genus_mean_rates_all.csv"
ML_RATES = "ML8_mean_rates_all.csv"
# SIMRATES = "MUT2_mcmc_05-02-25"  # best fit so far
# SIMRATES = "test"
# SIMRATES = "MUT2_320_mle_20-03-25" # best fit so far 24-04-25
SIMRATES = "MUT2_320_mcmc_2_24-04-25"
# SIMRATES = "MUT5_320_mcmc_23-07-25_1"
# simdata = "MUT2.2_trajectories_shape.csv"
# simdata = "MUT5_mcmc_10-02-25.csv"
SIMDATA = "MUT2_320_mle_23-04-25.csv"
# SIMDATA = "MUT5_320_mcmc_23-07-25_1.csv"
# SIMDATA = "pwalks_10_160_leaves_full_13-03-25_MUT2_CLEAN.csv"
# SIMDATA = "MUT2_mle_11-02-25.csv"
MC_ERR_SAMP = 500  # no. monte carlo samples for mcmc error propagation
T_STAT = 0  # 0=mean prop, 1=prop - stat used for the timeseries plot
EQ_INIT = False  # plot timeseries from equal numbers of each initial shape
LB = 2.5  # 5 #5 # credible interval for phylo and sim mcmc rates
UB = 97.5  # 95 #95
PHYLO_XLIM = 200
SIM_XLIM = 320  # 60
AUTO_SIM_XLIM = True  # set sim xlim to where prop ~= phy prop at phy xlim
VARIABLE_SIM_XLIM = False  # False = fix sim xlim to mean auto_sim_xlim
SIM_XON = 0  # begin x axis at this value
RESET_FIRST_CAT = False  # redefine first_cat to shape at step SIM_XON
YSCALE = "linear"  # log or linear
# show phylogeny CTMC fit ontop as well as sim data and CTMC fit in plot 2
SHOW_PHYLO = True
SHOW_STAT_DIST = False  # show stationary values for phylo and sim CTMC
LEAF_ICONS = True  # show leaf icons in plot
V = False  # verbose for debugging
AL = 1  # alpha for line plots
AF = 0.2  # alpha for fill_between

# sns.set_palette("colorblind")
ORDER = ["u", "l", "d", "c"]
VAR = "mean_prop" if T_STAT == 0 else "prop"

# exclude these in concatenator to infer from equal numbers of each initshape
EXCL = ["p0_121", "p1_82", "p2_195", "p9_129", "p5_249", "pu3", "p2_78_alt",
        "p3_60",  # unlobed
        "p8_1235", "p1_35", "p12b",  # dissected
        "pc4", "p12de", "p7_437", "p2_346_alt", "p6_1155"]  # compound
G_PARAMS = {name: val for name, val in globals().items()if name.isupper()}


def plot_data_from_probcurves(curves, t_plot, t_calc):
    """Takes a list of probability matrices and returns a dataframe with the
    transition probabilities in a format that can be plotted"""

    # attach plot time to data, along with time used for calculation
    plot_data = {"t": [], "t_calc": [], "first_cat": [],
                 "shape": [], "P": [], "lb": [], "ub": []}

    for i, li in enumerate(curves):
        for j, matrix in enumerate(li):
            for row in range(matrix.shape[0]):
                for column in range(matrix.shape[1]):
                    if j == 0:
                        plot_data["t_calc"].append(t_calc[i])
                        plot_data["t"].append(t_plot[i])
                        plot_data["first_cat"].append(row)
                        plot_data["shape"].append(column)
                        plot_data["P"].append(matrix[row, column])
                    elif j == 1:
                        plot_data["lb"].append(matrix[row, column])
                    elif j == 2:
                        plot_data["ub"].append(matrix[row, column])

    plot_data = pd.DataFrame(plot_data)
    mapping = {0: "u", 1: "l", 2: "d", 3: "c"}
    plot_data["first_cat"] = plot_data["first_cat"].replace(mapping)
    plot_data["shape"] = plot_data["shape"].replace(mapping)
    # replace any upper bound value greater than 1 with 1 (because prob)
    plot_data["ub"] = plot_data["ub"].clip(upper=1)
    return plot_data


def get_phylo_rates(phylo=PHYLORATES1, q_mthd="mcmc"):
    """Get the mean and confidence intervals for the posterior distributions
    of all transition rates from phylogenetic mcmc inference specified by
    phylo"""

    if q_mthd == "mcmc":  # read mcmc posteriors
        phylo_dir = "../phylogeny/rates/uniform_1010000steps"
        phylo_rates_list = []

        for filename in os.listdir(phylo_dir):
            if filename.endswith(".csv"):
                path = os.path.join(phylo_dir, filename)
                df = pd.read_csv(path)
                df["phylo-class"] = filename[:-4]
                phylo_rates_list.append(df)

        p_rate = pd.concat(phylo_rates_list, ignore_index=True)
        # Choose phylogeny for curves
        p_rate = p_rate[p_rate["phylo-class"] == phylo]
        p_rate.drop(columns="phylo-class", inplace=True)
        p_rate.reset_index(drop=True, inplace=True)
    elif q_mthd == "mle":  # read mle values
        p_rate = pd.read_csv(
            f"../phylogeny/rates/ML/{ML_RATES}")
        p_rate.loc[~p_rate["dataset"].str.contains("_class", na=False),
                   "dataset"] += "_class"  # ensure each row contains _class
        p_rate = p_rate[p_rate["dataset"] == phylo]
        p_rate = p_rate.drop(columns=["dataset", "Lh", "Root P(0)",
                                      "Root P(1)", "Root P(2)", "Root P(3)"])
    # Insert stasis rates
    p_rate.insert(0, "q00", -p_rate["q01"] - p_rate["q02"] - p_rate["q03"],)
    p_rate.insert(5, "q11", -p_rate["q10"] - p_rate["q12"] - p_rate["q13"],)
    p_rate.insert(10, "q22", -p_rate["q20"] - p_rate["q21"] - p_rate["q23"],)
    p_rate.insert(15, "q33", -p_rate["q30"] - p_rate["q31"] - p_rate["q32"],)

    # Calculate means
    phylo_summary = p_rate.mean().reset_index()
    phylo_summary.columns = ["transition", "mean_rate"]
    # Calculate confidence intervals
    confidence_intervals = {}
    for col in p_rate.columns:
        data = p_rate[col].dropna()
        confidence_intervals[col] = (
            np.percentile(data, LB),  # credible interval
            np.percentile(data, UB),
        )

    phylo_summary["lb"] = [i[0] for i in confidence_intervals.values()]
    phylo_summary["ub"] = [i[1] for i in confidence_intervals.values()]

    if V:
        print(f"Phylogeny rates\n{phylo_summary}")
    return phylo_summary, p_rate


def get_sim_rates(q_mthd="mcmc"):
    """Get the mean and confidence intervals for the posterior distributions
    of all transition rates from simulation mcmc inference specified by
    SIMRATES"""

    sim_rates = (
        pd.read_csv(f"markov_fitter_reports/emcee/{SIMRATES}/posteriors_"
                    f"{SIMRATES}.csv" if q_mthd == "mcmc" else
                    f"markov_fitter_reports/emcee/{SIMRATES}/ML_{SIMRATES}"
                    ".csv")
    )
    name_map = {"0": "q01", "1": "q02", "2": "q03", "3": "q10", "4": "q12",
                "5": "q13", "6": "q20", "7": "q21", "8": "q23", "9": "q30",
                "10": "q31", "11": "q32"}

    # for sim_rates in avg_list:
    sim_rates = sim_rates.rename(columns=name_map)
    sim_rates.insert(
        0, "q00", -sim_rates["q01"] - sim_rates["q02"] - sim_rates["q03"])
    sim_rates.insert(
        5, "q11", -sim_rates["q10"] - sim_rates["q12"] - sim_rates["q13"])
    sim_rates.insert(
        10,
        "q22",
        -sim_rates["q20"] - sim_rates["q21"] - sim_rates["q23"],
    )
    sim_rates.insert(
        15,
        "q33",
        -sim_rates["q30"] - sim_rates["q31"] - sim_rates["q32"],
    )
    # Calculate means
    sim_summary = sim_rates.mean().reset_index()
    sim_summary.columns = ["transition", "mean_rate"]

    # Tweaks
    # sim_summary.loc[1, "mean_rate"] -= 0.003  # decrease q01
    # sim_summary.loc[0, "mean_rate"] += 0.003  # increase q00
    # sim_summary.loc[3, "mean_rate"] -= 0.002  # decrease q01
    # sim_summary.loc[0, "mean_rate"] += 0.002  # increase q00
    # sim_summary.loc[12, "mean_rate"] -= 0.015  # decrease q30
    # sim_summary.loc[15, "mean_rate"] += 0.015  # increase q33``

    # Calculate confidence intervals
    confidence_intervals = {}
    for col in sim_rates.columns:
        data = sim_rates[col].dropna()
        confidence_intervals[col] = (
            np.percentile(data, LB),
            np.percentile(data, UB),
        )
    sim_summary["lb"] = [i[0] for i in confidence_intervals.values()]
    sim_summary["ub"] = [i[1] for i in confidence_intervals.values()]
    if V:
        print(f"Simulation rates\n{sim_summary}")
    return sim_summary, sim_rates


def get_timeseries(groupby_first_cat=False):
    """
    Get the mean proportion of each shape category at each step of the walk
    from the simulation data.
    T_STAT      statistic that is used to summaries the timeseries
                0   ...mean of the proportion across leafids within first_cats
                1   ...proportion across all leafids
    groupby_first_cat: if True, group by first_cat before calculating means
    """
    concat = pd.read_csv(SIMDATA)

    if RESET_FIRST_CAT:
        # reset first_cat to the shape at step SIM_XON
        concat["first_cat"] = concat.groupby(["leafid", "walkid"])[
            "shape"].transform(lambda x: x.iloc[SIM_XON])
        # create new unique leafid by appending first_cat - ensure that
        # there are no duplicate leafid, step, shape rows
        concat["leafid"] = concat["leafid"] + "_" + concat["first_cat"]
        step_sim_xon = concat[concat["step"] == SIM_XON]
        mis = step_sim_xon[step_sim_xon["first_cat"] != step_sim_xon["shape"]]
        assert len(mis) == 0, f"Mismatch in first_cat and shape at {SIM_XON}"

    if EQ_INIT:
        concat = concat[~concat["leafid"].isin(EXCL)].reset_index(drop=True)
        if V:
            print(concat.drop_duplicates(
                subset=["leafid"]).sort_values(by="first_cat"))

    # get the number of leaves in each initial group
    shape_counts = concat.groupby(["leafid", "first_cat"]).size().reset_index()
    init_shape_counts = shape_counts["first_cat"].value_counts().reset_index()
    init_shape_counts = init_shape_counts.set_index("first_cat")[
        "count"].to_dict()

    if T_STAT == 0:
        # total of each shape per step for each leafid
        tseries = (
            concat.groupby(["leafid", "first_cat", "step", "shape"])
            .agg(shape_total=("shape", "size"))
            .reset_index()
        )
        # check for leafids with duplicate steps
        dup = tseries[tseries.duplicated(
            subset=["leafid", "step", "shape"])]
        assert len(dup) == 0, f"Duplicate leafid, step, shape rows: {dup}"

        # fill in transitions that didn't occur
        all_transitions = pd.MultiIndex.from_product(
            [tseries["leafid"].unique(), tseries["step"].unique(),
             {"u", "l", "d", "c"}], names=["leafid", "step", "shape"]
        )  # all possible combinations of leafid, step and shape, then reindex
        tseries = tseries.set_index(["leafid", "step", "shape"])
        tseries = tseries.reindex(
            all_transitions, fill_value=None).reset_index()

        # fill empty first-cat values with the first non nan first_cat value
        tseries["first_cat"] = tseries.groupby(
            ["leafid"])["first_cat"].transform("first")
        tseries = tseries.fillna(0)

        # no. active walks per step for each leafid
        tseries_total = (
            tseries.groupby(["leafid", "first_cat", "step"])
            .agg(no_active_walks=("shape_total", "sum"))
            .reset_index()
        )  # if using pseudo walks, will always be 1

        # proportion of active walks in each shape category for each leafid
        tseries = tseries.merge(
            tseries_total, on=["leafid", "first_cat", "step"])
        tseries["proportion"] = (
            tseries["shape_total"] / tseries["no_active_walks"]
        )  # for pseudo walks, will always be either 1 or 0

        # mean prop active walks per shape for all leaves in each first_cat
        tseries = (  # don't group by first cat if PLOT == 2 or 4
            tseries.groupby(
                ["first_cat", "step", "shape"] if groupby_first_cat else
                ["step", "shape"])
            .agg(mean_prop=("proportion", "mean"),
                 sterr=("proportion", "sem"),
                 n=("proportion", "size"),
                 total=("proportion", "sum"))
            .reset_index()
        )
        if not groupby_first_cat:  # PLOT 2 or 4 mean calculated across all first_cats
            assert tseries[tseries["n"] !=
                           48].empty, "n is not 48 for all steps."

    elif T_STAT == 1:
        tseries = concat.groupby(
            ["first_cat", "step", "shape"]
        ).size().reset_index(name="shape_total")
        # no. active walks per step
        tseries_total = tseries.groupby(["first_cat", "step"]).agg(
            no_active_walks=("shape_total", "sum")).reset_index()
        # proportion of active walks in each shape category
        tseries = tseries.merge(
            tseries_total, on=["first_cat", "step"])
        tseries["prop"] = tseries["shape_total"] / tseries["no_active_walks"]
        tseries["sterr"] = 0

    tseries["lb"] = tseries[VAR] - 1.96 * tseries["sterr"]
    tseries["ub"] = tseries[VAR] + 1.96 * tseries["sterr"]

    if groupby_first_cat:
        # add initial state to the tseries
        tseries["step"] = tseries["step"] + 1
        for i in ORDER:
            for j in ORDER:
                if i == j:
                    tseries.loc[-1] = {"first_cat": i, "step": 0, "shape": i,
                                       VAR: 1, "sterr": 0, "lb": 1, "ub": 1}
                else:
                    tseries.loc[-1] = {"first_cat": i, "step": 0, "shape": j,
                                       VAR: 0, "sterr": 0, "lb": 0, "ub": 0}
                tseries.index = tseries.index + 1
                tseries = tseries.sort_index()
    if V:
        print(f"Simulation data\n{tseries}")
    tseries.to_csv("tseries.csv")
    return tseries


def plot_sim_and_phylogeny_curves_sep_init():
    """Plot the MEAN proportion of each shape category at each step of the walk
    from the simulation data against the predicted proportions from the
    simulation and phylogeny CTMCs. The scale for the phylogeny is different
    to the simulation scale. The means are calculated separately for walks
    beginning in each shape category."""

    cmap = plt.get_cmap("tab10")

    # load icons
    icon_filenames = ["u.png", "l.png", "d.png", "c.png"]
    icons = [os.path.join("uldc_model_icons", path) for path in icon_filenames]
    icon_imgs = [Image.open(path) for path in icons]

    #### Get phylo-rates ####
    _, raw_zun = get_phylo_rates(PHYLORATES1, "mcmc")
    _, raw_zun_mle = get_phylo_rates(PHYLORATES_ML1, "mle")
    _, raw_jan_mle = get_phylo_rates(PHYLORATES_ML2, "mle")

    #### Get sim-rates ####
    _, raw_sim = get_sim_rates("mcmc")
    _, raw_sim_mle = get_sim_rates("mle")

    #### Get sim timeseries data ####
    tseries = get_timeseries(groupby_first_cat=True)
    tseries = tseries.rename(columns={"mean_prop": "prop"})  # line is mean

    tseries_no_firstcat = get_timeseries(groupby_first_cat=False)
    tseries_no_firstcat = tseries_no_firstcat.rename(
        columns={"mean_prop": "prop"})
    init_ratio_no_firstcat = np.array([
        tseries_no_firstcat[tseries_no_firstcat["shape"]
                            == "u"]["prop"].values[0],
        tseries_no_firstcat[tseries_no_firstcat["shape"]
                            == "l"]["prop"].values[0],
        tseries_no_firstcat[tseries_no_firstcat["shape"]
                            == "d"]["prop"].values[0],
        tseries_no_firstcat[tseries_no_firstcat["shape"]
                            == "c"]["prop"].values[0]
    ])

    ppzs = []
    ppzs_mle = []
    psims = []
    psims_mle = []
    for i, s in enumerate(ORDER):  # loop for different inits
        init_ratio = np.zeros(4)
        init_ratio[i] = 1
        # produce phylo-curves
        ppz = get_plot_vals_alt(raw_zun, init_ratio, "phylo", "mcmc")
        ppz_mle = get_plot_vals_alt(raw_zun_mle, init_ratio, "phylo", "mle")
        ppz["first_cat"] = s  # append first_cat
        ppz_mle["first_cat"] = s
        ppzs.append(ppz)
        ppzs_mle.append(ppz_mle)
        # produce sim-curves
        psim = get_plot_vals_alt(raw_sim, init_ratio, "sim", "mcmc")
        psim_mle = get_plot_vals_alt(raw_sim_mle, init_ratio, "sim", "mle")

        psim["first_cat"] = s  # append first_cat
        psim_mle["first_cat"] = s
        psims.append(psim)
        psims_mle.append(psim_mle)
    ppz = pd.concat(ppzs, ignore_index=True)
    ppz_mle = pd.concat(ppzs_mle, ignore_index=True)
    psim = pd.concat(psims, ignore_index=True)
    psim_mle = pd.concat(psims_mle, ignore_index=True)
    psim_mle_no_firstcat = get_plot_vals_alt(
        raw_sim_mle, init_ratio_no_firstcat, "sim", "mle")
    ppz_mle_no_firstcat = get_plot_vals_alt(
        raw_zun_mle, init_ratio_no_firstcat, "phylo", "mle")
    ppj_mle_no_firstcat = get_plot_vals_alt(
        raw_jan_mle, init_ratio_no_firstcat, "phylo", "mle")

    piz, pij, pis = stat_dist(  # get approx stationary distribution from mle
        raw_zun_mle, raw_jan_mle, raw_sim_mle)
    # use mle prop curves to estimate simxlims
    print(init_ratio_no_firstcat, ppz_mle_no_firstcat,
          ppj_mle_no_firstcat, psim_mle_no_firstcat)

    sim_xlims = calc_xlims_thalf(init_ratio_no_firstcat, ppz_mle_no_firstcat,
                                 ppj_mle_no_firstcat, psim_mle_no_firstcat,
                                 piz, pij, pis)

    # Create subplots
    # plt.rcParams["font.family"] = "CMU Serif"
    fig, axs = plt.subplots(nrows=4, ncols=5, figsize=(11, 6),
                            gridspec_kw={"height_ratios": [0.01, 3, 0.5, 3]},
                            sharey="row", layout="constrained")
    axs_g0 = axs[0][1:]  # for icons
    axs_g1 = axs[1][1:]  # for timeseries
    axs_g3 = axs[3][1:]  # for phylogeny
    order_full = ["Unlobed", "Lobed", "Dissected", "Compound"]

    for i, ax in enumerate(axs_g1):  # loop for first_cat
        if AUTO_SIM_XLIM:
            # if not variable sim xlim, use the mean across shapes
            s_xlim_loc = sim_xlims["xlim"][i] if VARIABLE_SIM_XLIM \
                else sim_xlims["xlim"].mean()
            print(f"s_xlim_loc: {s_xlim_loc}")

        ts = tseries[tseries["first_cat"] == ORDER[i]]  # sub for first_cat
        sim = psim[psim["first_cat"] == ORDER[i]]
        sim_mle = psim_mle[psim_mle["first_cat"] == ORDER[i]]
        for j, s in enumerate(ORDER):  # loop for shape
            ts_s = ts[ts["shape"] == s]  # add sim data
            ax.plot(ts_s["step"], ts_s["prop"],
                    label=s, color=cmap(j), ls="--")
            ax.fill_between(ts_s["step"], ts_s["lb"], ts_s["ub"],
                            alpha=AF, color=cmap(j), ec=None)
            sim_s = sim[sim["shape"] == s]  # add sim ctmc
            sim_mle_s = sim_mle[sim_mle["shape"] == s]
            ax.plot(sim_mle_s["t"], sim_mle_s["P"],
                    label=s,  color=cmap(j))
            # ax.fill_between(sim_s["t"], sim_s["lb"], sim_s["ub"],
            #                 alpha=af, color=cmap(j), ec=None)
            ax.axvline(x=s_xlim_loc, color="grey", ls="--",
                       label=f"{s} xlim")
            ax.set_title(order_full[i])
        ax.set_xlabel("Step")
        ax.set_xlim(SIM_XON, SIM_XLIM)
        ax.set_xticks(np.arange(SIM_XON, SIM_XLIM + 1, (SIM_XLIM-SIM_XON)/4))

        if i == 0:
            ax.tick_params(axis="y", labelleft=True)
            ax.set_ylabel("P")

    for i, ax in enumerate(axs_g3):  # add phylo data
        ppz_s = ppz[ppz["first_cat"] == ORDER[i]]
        ppz_mle_s = ppz_mle[ppz_mle["first_cat"] == ORDER[i]]
        for s in ORDER:
            ppz_ss = ppz_s[ppz_s["shape"] == s]
            ppz_mle_ss = ppz_mle_s[ppz_mle_s["shape"] == s]
            ax.plot(ppz_mle_ss["t"], ppz_mle_ss["P"], label=s)
            ax.fill_between(ppz_ss["t"], ppz_ss["lb"], ppz_ss["ub"], alpha=AF)
        ax.set_xlabel("Branch length (Myr)")
        ax.set_xlim(0, PHYLO_XLIM)
        ax.set_xticks(np.arange(0, PHYLO_XLIM+1, PHYLO_XLIM/4))

        if i == 0:
            ax.tick_params(axis="y", labelleft=True)
            ax.set_ylabel("P")

    for i, ax in enumerate(axs_g0):  # add icons
        imbg_box = OffsetImage(icon_imgs[i], zoom=0.12, alpha=0.5)
        aln = (0.5, 0)
        ab = AnnotationBbox(
            imbg_box,
            aln,
            xycoords="axes fraction",
            box_alignment=aln,  # upper right corner alignment
            frameon=False,
            pad=0.2,
        )
        ax.add_artist(ab)
        ax.axis("off")

    axs[1, 0].text(0.5, 0.5, f"Simulation Data\nand CTMC ({SIMRATES[0:4]})",
                   ha="center", va="center")
    axs[3, 0].text(0.5, 0.5, "Phylogeny CTMC\nZuntini et al. (2024)",
                   ha="center", va="center")

    for ax in axs.flatten():
        ax.grid(alpha=0.3)
        if ax not in axs_g1 and ax not in axs_g3:
            ax.axis("off")
    handles, _ = axs_g3[0].get_legend_handles_labels()
    fig.legend(
        handles,
        order_full,
        loc="outside right",
        title="Final shape",
        ncol=1,
    )
    plt.savefig(
        f"curves_{PLOT}_mcerr{MC_ERR_SAMP}_{SIMRATES}_{PHYLORATES1}.pdf",
        format="pdf", dpi=1200, metadata={"Keywords": str(G_PARAMS)})
    plt.show()


def get_timeseries_alt():
    """Return the proportion of each shape category at each step of the walk
    from the simulation data"""

    concat = pd.read_csv(SIMDATA)

    if EQ_INIT:
        concat = concat[~concat["leafid"].isin(EXCL)].reset_index(drop=True)
        if V:
            print(concat.drop_duplicates(subset=["leafid"]).sort_values(
                by="first_cat"))

    tseries = concat.groupby(["step", "shape"]).size().reset_index(
        name="shape_total")  # no. each shape at each step
    tseries["total"] = tseries.groupby("step")["shape_total"].transform(
        "sum")  # total no. shapes at each step
    tseries["prop"] = tseries["shape_total"] / tseries["total"]
    # proportion of each shape at each step

    return tseries


def get_plot_vals(rates, init_ratio, dtype):
    """Get the predicted proportion of each shape at each step of the walk
    from the simulation or phylogeny rates"""

    # no. time points must match no. steps in sim data, phylo time scale is
    # different
    t_vals = None
    if dtype == "phylo":
        t_vals = np.linspace(0, PHYLO_XLIM, SIM_XLIM)
    elif dtype == "sim":
        t_vals = np.linspace(0, SIM_XLIM, SIM_XLIM+1)

    q = np.array(rates["mean_rate"].values).reshape(4, 4)
    ql = np.array(rates["lb"].values).reshape(4, 4)
    qu = np.array(rates["ub"].values).reshape(4, 4)
    # multiply predicted transition probability by initial proportion of shapes
    # to get the predicted proportion of each shape at each step
    curve_dat = []  # store p for each t in t_vals
    for t in t_vals:
        pt = linalg.expm(q * t) * init_ratio[:, None]
        pllt = linalg.expm(ql * t) * init_ratio[:, None]
        puut = linalg.expm(qu * t) * init_ratio[:, None]
        curve_dat.append([pt, pllt, puut])
    # N.B. the phylo time scale is different to the sim time scale, but we
    # replace the phylo time scale with the sim time scale when plotting.
    # make pseudo time scale for phylo, so it can be plotted with sim data
    plot_dat = None
    if dtype == "phylo":
        plot_dat = pd.DataFrame(plot_data_from_probcurves(
            curve_dat, np.arange(SIM_XLIM), t_vals))
    elif dtype == "sim":  # sim time scale the same for plot and calc
        plot_dat = pd.DataFrame(plot_data_from_probcurves(
            curve_dat, t_vals, t_vals))

    # sum predicted proportions from different initial shapes
    plot_dat = plot_dat.groupby(["t", "shape"]).agg(
        P=("P", "sum"), lb=("lb", "sum"), ub=("ub", "sum")).reset_index()
    if dtype == "phylo":
        # add actual phylo time scale to phylo plot data for reference
        plot_dat["t_actual"] = [i for i in t_vals for _ in range(4)]

    return plot_dat


def get_plot_vals_alt(rates, init_ratio, dtype, q_mthd):
    """Get the predicted proportion of each shape at each step of the walk
    from the simulation or phylogeny rates with error bands using Monte Carlo
    error propagation."""

    # no. time points must match no. steps in sim data, phylo time scale is
    # different
    t_vals = None
    if dtype == "phylo":
        t_vals = np.linspace(0, PHYLO_XLIM, PHYLO_XLIM+1)
    elif dtype == "sim":
        t_vals = np.linspace(0, SIM_XLIM, SIM_XLIM+1)

    # multiply predicted transition probability by initial proportion of shapes
    # to get the predicted proportion of each shape at each step
    curves = []  # store p for each t in t_vals
    for i in range(MC_ERR_SAMP):
        print(f"monte carlo {i}/{MC_ERR_SAMP}", end="\r")
        samp = rates.apply(  # choose random rate from each column
            lambda col: col.sample(1).values[0], axis=0)
        # recalculate diagonals
        samp["q00"] = -samp["q01"] - samp["q02"] - samp["q03"]
        samp["q11"] = -samp["q10"] - samp["q12"] - samp["q13"]
        samp["q22"] = -samp["q20"] - samp["q21"] - samp["q23"]
        samp["q33"] = -samp["q30"] - samp["q31"] - samp["q32"]
        q = np.array(samp.values).reshape(4, 4)
        curve_dat = []
        for t in t_vals:
            pt = linalg.expm(q * t) * init_ratio[:, None]
            pllt = np.zeros((4, 4))
            puut = np.zeros((4, 4))
            curve_dat.append([pt, pllt, puut])
        plot_dat = pd.DataFrame(
            plot_data_from_probcurves(curve_dat, t_vals, t_vals))
        # sum predicted proportions from different initial shapes
        plot_dat = plot_dat.groupby(["t", "shape"]).agg(
            P=("P", "sum")).reset_index()
        plot_dat.insert(0, "mc_err_samp", i)
        curves.append(plot_dat)
        if q_mthd == "mle":  # if using MLE q data, no mc error sampling
            break
    if q_mthd == "mcmc":
        print(f"Done {dtype} Monte Carlo error sampling")
        assert len(curves) == MC_ERR_SAMP, "Incorrect no. MC samples"
    elif q_mthd == "mle":
        print(f"Collected {dtype} MLE rates")
        assert len(curves) == 1, "Incorrect MLE rate length"
    curves = pd.concat(curves, ignore_index=True)  # combine mc samples
    curves = curves.groupby(["t", "shape"]).agg(  # median and CI
        P=("P", "median"), lb=("P", lambda x: np.percentile(x, LB)),
        ub=("P", lambda x: np.percentile(x, UB))).reset_index()
    return curves


def stat_dist(*args):
    """Approximate the stationary distribution of the transition rates from a
    dataframe of q values, by taking a large value of t for P=exp(q*t)."""
    for rates in args:

        q = rates.iloc[0].values.reshape(4, 4)

        rowsums = np.sum(q, axis=1)  # check q is valid
        assert np.allclose(rowsums, 0), "q rowsums are not zero"

        # numerical approximation with large t
        pi = linalg.expm(q * 1e10)
        assert np.allclose(pi, pi[0]), "stationary distribution not found"

        pi = pi[0]  # take first row as stationary distribution
        yield pi


def calc_xlims(init_ratio, ppz, ppj, psim, piz, pij, pis):
    """Approximate xlims for simulation plot such that a fair comparison can
    be drawn with phylogeny."""

    # get the difference between p at stationary distribution and phylo xlim
    # for each phylogeny and shape category as a fraction of the difference
    # between p at stationary distribution and initial distribution
    difs = []
    for i, dat in enumerate([ppz, ppj]):
        pi = [piz, pij, pis][i]  # stationary distribution for dataset
        for s, shape in enumerate(ORDER):
            init_s = init_ratio[s]  # prop of shape at t=0
            dat_s = dat[dat["shape"] == shape]
            pi_s = pi[s]  # stationary dist for shape
            dif_pi_init = pi_s - init_s  # p stat - p init
            p_pxl = dat_s[dat_s["t"] ==
                          PHYLO_XLIM]["P"].values[0]  # p phylo_xlim
            dif_pi_pxl = pi_s - p_pxl  # p static - p phylo_xlim
            dif_ratio = dif_pi_pxl / dif_pi_init  # fraction of p_stat - p_init
            difs.append({"dataset": ["zun", "jan"][i], "shape": shape,
                         "pi": pi_s, "init_ratio": init_s,
                        "p_phylo_xlim": dat_s["P"].iloc[-1],
                         "dif_ratio": dif_ratio})

    difs = pd.DataFrame(difs)  # average the dif ratios for all phylos
    mean_phylo_dif_ratios = difs.groupby("shape")["dif_ratio"].mean()

    # find the closest p in sim timeseries to where the difference between
    # this value and the stationary distribution is equal to the difference
    # between the phylo xlim and the stationary distribution as a fraction of
    # the difference between the phylo xlim and the initial distribution
    # for each shape category
    xlims = []
    for s, shape in enumerate(ORDER):  # calc expected p_sim_xlim
        dat_s = psim[psim["shape"] == shape].reset_index(drop=True)
        mpdr = mean_phylo_dif_ratios.iloc[s]  # mean within shape
        init_s = init_ratio[s]  # init dist for sim
        pi_s = pis[s]  # stat sim dist
        dif_pi_init = pi_s - init_s  # diff between init and stationary dist
        dif_pi_pxl = dif_pi_init * mpdr  # diff at pxl scaled for sim
        target_p = pi_s - dif_pi_pxl
        closest_idx = (dat_s["P"] - target_p).abs().idxmin()
        xlim = dat_s["t"].iloc[closest_idx]

        xlims.append({"shape": shape, "mean_phylo_dif_ratio": mpdr,
                      "init_ratio": init_s, "pi": pi_s,
                      "dif_pi_init": dif_pi_init,  "dif_pi_pxl": dif_pi_pxl,
                      "target_p": target_p, "closest_idx": closest_idx,
                      "xlim": xlim})
    xlims = pd.DataFrame(xlims)
    return xlims


def calc_xlims_thalf(init_ratio, ppz, ppj, psim, piz, pij, pis):
    """Approximate xlims for simulation plot such that a fair comparison can
    be drawn with phylogeny, using the time to a fraction of the stationary
    distribution."""
    frac = 0.5  # fraction of stationary distribution to use

    # get the difference between p at stationary distribution and phylo xlim
    # for each phylogeny and shape category as a fraction of the difference
    # between p at stationary distribution and initial distribution
    difs = []
    for i, dat in enumerate([ppz, ppj]):
        pi = [piz, pij][i]  # stationary distribution for dataset
        for s, shape in enumerate(ORDER):
            dat_s = dat[dat["shape"] == shape]  # shape_data
            difs.append({"dataset": ["zun", "jan"][i], "shape": shape,
                         # stationary dist for shape is pi[s]
                         "init_ratio": init_ratio[s], "pi": pi[s],
                        "p_xlim": dat_s["P"].iloc[-1],
                         })
    difs = pd.DataFrame(difs)  # average the dif ratios for all phylos
    difs["pi-init"] = difs["pi"] - difs["init_ratio"]
    difs["p_xlim-init"] = difs["p_xlim"] - difs["init_ratio"]
    difs[f"(pi-init)*{frac}"] = difs["pi-init"] * frac
    # calculate the no. half lives between the initial distribution and
    # the distribution at phylo xlim
    difs[f"(p_xlim-init)/((pi-init)*{frac})"] = (
        difs["p_xlim-init"] / (difs[f"(pi-init)*{frac}"]))

    # Take the average across the two phylogenies ppj and ppz
    mean_p_half_phylo = difs.groupby("shape")[
        f"(p_xlim-init)/((pi-init)*{frac})"].mean()

    # find the closest p in sim timeseries to where the difference between
    # this value and the stationary distribution is equal to the difference
    # between the phylo xlim and the stationary distribution as a fraction of
    # the difference between the phylo xlim and the initial distribution
    # for each shape category
    xlims = []
    for s, shape in enumerate(ORDER):  # calc expected p_sim_xlim
        dat_s = psim[psim["shape"] == shape].reset_index(drop=True)
        mphf = mean_p_half_phylo.loc[shape]  # mean within shape

        xlims.append({"shape": shape,
                      f"(p_xlim-init)/((pi-init)*{frac})_phylo": mphf,
                      "init_ratio": init_ratio[s], "pi": pis[s],
                      })
    xlims = pd.DataFrame(xlims)
    xlims["pi-init"] = xlims["pi"] - xlims["init_ratio"]
    xlims[f"(pi-init)*{frac}"] = xlims["pi-init"] * frac
    # multiply no. half lives between initial distribution and phylo xlim
    # by the simulation half life to get the predicted p_xlim for sim
    xlims["(p_xlim-init)_predicted"] = (
        xlims[f"(p_xlim-init)/((pi-init)*{frac})_phylo"] *
        xlims[f"(pi-init)*{frac}"]
    )
    xlims["p_xlim_predicted"] = (
        xlims["(p_xlim-init)_predicted"] + xlims["init_ratio"]
    )

    for s, shape in enumerate(xlims["shape"]):  # calc expected p_sim_xlim
        dat_s = psim[psim["shape"] == shape].reset_index(drop=True)
        target_p = xlims["p_xlim_predicted"].iloc[s]
        closest_idx = (dat_s["P"] - target_p).abs().idxmin()
        xlim = dat_s["t"].iloc[closest_idx]
        xlims.loc[s, "xlim"] = xlim
    return xlims


def load_leaf_imgs():
    """Get leaf images for plotting"""
    icon_filenames = [
        "leaf_p7a_0_0.png",
        "leaf_p8ae_0_0.png",
        "leaf_pd1_0_0.png",
        "leaf_pc1_alt_0_0.png",
    ]
    icons = [os.path.join("uldc_model_icons", path) for path in icon_filenames]
    icon_imgs = [plt.imread(path) for path in icons]

    return icon_imgs


def plot_sim_and_phylogeny_curves_prop():
    """Plot the proportion of each shape category at each step of the
    walk from the simulation data against the predicted proportions from the
    simulation and phylogeny CTMCs. The scale for the phylogeny is different
    to the simulation scale."""

    al = 1  # alpha for lines
    af = 0.2  # alpha for fill
    icon_imgs = load_leaf_imgs()  # get leaf icons

    #### Get phylo-rates ####
    _, raw_zun = get_phylo_rates(PHYLORATES1, "mcmc")
    _, raw_jan = get_phylo_rates(PHYLORATES2, "mcmc")
    _, raw_zun_mle = get_phylo_rates(PHYLORATES_ML1, "mle")
    _, raw_jan_mle = get_phylo_rates(PHYLORATES_ML2, "mle")
    #### Get sim-rates ####
    _, raw_sim = get_sim_rates("mcmc")
    _, raw_sim_mle = get_sim_rates("mle")
    #### Get sim timeseries data ####
    # tseries = get_timeseries_alt()
    tseries = get_timeseries()  # for error bar on timeseries
    tseries = tseries.rename(columns={"mean_prop": "prop"})  # line is mean
    init_ratio = np.array([  # get proportion of each shape at step=0 in data
        tseries[tseries["shape"] == "u"]["prop"].values[0],
        tseries[tseries["shape"] == "l"]["prop"].values[0],
        tseries[tseries["shape"] == "d"]["prop"].values[0],
        tseries[tseries["shape"] == "c"]["prop"].values[0]
    ])

    piz, pij, pis = stat_dist(  # get approx stationary distribution from mle
        raw_zun_mle, raw_jan_mle, raw_sim_mle)

    # get predicted proportions from phylo and sim rate matrices
    psim = get_plot_vals_alt(raw_sim, init_ratio, "sim", "mcmc")
    psim_mle = get_plot_vals_alt(raw_sim_mle, init_ratio, "sim", "mle")
    if SHOW_PHYLO:
        ppz = get_plot_vals_alt(raw_zun, init_ratio, "phylo", "mcmc")
        ppj = get_plot_vals_alt(raw_jan, init_ratio, "phylo", "mcmc")
        ppz_mle = get_plot_vals_alt(raw_zun_mle, init_ratio, "phylo", "mle")
        ppj_mle = get_plot_vals_alt(raw_jan_mle, init_ratio, "phylo", "mle")
        # use mle prop curves to estimate simxlims
        sim_xlims = calc_xlims(init_ratio, ppz_mle, ppj_mle,
                               psim_mle, piz, pij, pis)

    # Create subplots
    # plt.rcParams["font.family"] = "CMU Serif"
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(
        9, 6), sharex=not VARIABLE_SIM_XLIM, sharey=False)
    p_xlim_loc = PHYLO_XLIM  # default xlims
    s_xlim_loc = SIM_XLIM

    for i, ax in enumerate(axs.flat):
        if SHOW_PHYLO and AUTO_SIM_XLIM:
            s_xlim_loc = sim_xlims["xlim"][i] if VARIABLE_SIM_XLIM \
                else sim_xlims["xlim"].mean()
            print(f"s_xlim_loc: {s_xlim_loc}")

        def fwd(x, s=s_xlim_loc):  # map from sim to phylo timescale
            # x is the value on sim timescale, so multiply by ratio of phylo to
            # sim timescale
            return x * (p_xlim_loc / s)

        def inv(x, s=s_xlim_loc):  # map from phylo to sim time scale
            return x * (s / p_xlim_loc)

        tseries_sub = tseries[tseries["shape"] == ORDER[i]]
        l_data, = ax.plot(tseries_sub["step"],  # plot timeseries
                          tseries_sub["prop"], c="C0", zorder=0)
        if {"lb", "ub"}.issubset(tseries_sub.columns):  # show error
            ax.fill_between(tseries_sub["step"], tseries_sub["lb"],
                            tseries_sub["ub"], alpha=af, color="C0", ec=None)
        sim_sub = psim[psim["shape"] == ORDER[i]]
        sim_sub_mle = psim_mle[psim_mle["shape"] == ORDER[i]]
        l_fit, = ax.plot(sim_sub_mle["t"], sim_sub_mle["P"],  # plot sim ctmc
                         c="C1", ls="--", alpha=al)
        ax.fill_between(sim_sub["t"], sim_sub["lb"], sim_sub["ub"],
                        alpha=af, color="C1", ec=None)
        ax.set_xlim(0, s_xlim_loc)
        if i in [2, 3]:
            ax.set_xlabel("Simulation step")
        if SHOW_PHYLO:
            secax = ax.secondary_xaxis('top', functions=(fwd, inv))

            ppz_sub = ppz[ppz["shape"] == ORDER[i]]
            ppz_sub_mle = ppz_mle[ppz_mle["shape"] == ORDER[i]]
            l_phy_z, = ax.plot(inv(ppz_sub_mle["t"]), ppz_sub_mle["P"], c="C2",
                               ls="--", alpha=al)
            ax.fill_between(inv(ppz_sub["t"]), ppz_sub["lb"], ppz_sub["ub"],
                            alpha=af, color="C2", ec=None)

            ppj_sub = ppj[ppj["shape"] == ORDER[i]]
            ppj_sub_mle = ppj_mle[ppj_mle["shape"] == ORDER[i]]
            l_phy_j, = ax.plot(inv(ppj_sub_mle["t"]), ppj_sub_mle["P"], c="C3",
                               ls="--", alpha=al)
            ax.fill_between(inv(ppj_sub["t"]), ppj_sub["lb"], ppj_sub["ub"],
                            alpha=af, color="C3", ec=None)
            if i in [0, 1]:  # avoid repeating labels
                secax.set_xlabel("Branch length (Myr)")
            else:
                secax.set_xticklabels([])

        ax.set_title(["Unlobed", "Lobed", "Dissected", "Compound"][i])

        print(f"p_xlim_loc: {p_xlim_loc}")
        ax.grid(alpha=0.3)
        if LEAF_ICONS:
            imbg_box = OffsetImage(icon_imgs[i], zoom=0.08, alpha=0.5)
            aln = (1, 1) if i != 0 else (0, 1)
            ab = AnnotationBbox(
                imbg_box,
                aln,
                xycoords="axes fraction",
                box_alignment=aln,
                frameon=False,
                pad=0.2,
            )
            ax.add_artist(ab)

        if SHOW_STAT_DIST:
            ax.axhline(y=pis[i], color="C1", ls="--", alpha=0.5)
            ax.axhline(y=piz[i], color="C2", ls="--", alpha=0.5)
            ax.axhline(y=pij[i], color="C3", ls="--", alpha=0.5)
        if YSCALE == "log":
            ax.set_yscale("log")
    fig.supylabel("Proportion")
    if SHOW_PHYLO:
        leg = fig.legend([l_data, l_fit, l_phy_z, l_phy_j],
                         ["Simulation Data", "Simulation CTMC",
                          "Phylogeny CTMC\nZuntini et al. (2024)",
                          "Phylogeny CTMC\nJanssens et al. (2021)"],
                         loc="center left", bbox_to_anchor=(0.75, 0.5))
    else:
        leg = fig.legend([l_data, l_fit],
                         ["Simulation Data", "Simulation CTMC"],
                         loc="center left", bbox_to_anchor=(0.75, 0.5))
    for lh in leg.legend_handles:
        lh.set_alpha(1)
    fig.subplots_adjust(right=0.73)  # make room for legend
    left = min(ax.get_position().x0 for ax in axs.flat)
    right = max(ax.get_position().x1 for ax in axs.flat)
    center = (left + right) / 2  # center of 2x2 plot grid for x lab
    # fig.supxlabel("Simulation step", x=center)
    plt.tight_layout()
    fig.subplots_adjust(right=0.73)  # make room for legend
    plt.savefig(
        f"curves_{PLOT}_mcerr{MC_ERR_SAMP}_{SIMRATES}.pdf", format="pdf",
        dpi=1200)
    plt.show()


def plot_sim_and_phylogeny_prop_sep():
    """Plot the proportion of each shape category at each step of the
    walk from the simulation data against the predicted proportions from the
    simulation and phylogeny CTMCs. The scale for the phylogeny is different
    to the simulation scale."""

    ylim = 0.35  # ylim for lobed, dissected, compound
    leg_inside = True  # whether to put legend inside or outside axs
    icon_imgs = load_leaf_imgs()  # get leaf icons

    #### Get phylo-rates ####
    _, raw_zun = get_phylo_rates(PHYLORATES1, "mcmc")
    _, raw_jan = get_phylo_rates(PHYLORATES2, "mcmc")
    _, raw_zun_mle = get_phylo_rates(PHYLORATES_ML1, "mle")
    _, raw_jan_mle = get_phylo_rates(PHYLORATES_ML2, "mle")
    #### Get sim-rates ####
    _, raw_sim = get_sim_rates("mcmc")
    _, raw_sim_mle = get_sim_rates("mle")
    #### Get sim timeseries data ####
    # tseries = get_timeseries_alt()
    tseries = get_timeseries()  # for error bar on timeseries
    tseries = tseries.rename(columns={"mean_prop": "prop"})  # line is mean
    init_ratio = np.array([  # get proportion of each shape at step=0 in data
        tseries[tseries["shape"] == "u"]["prop"].values[0],
        tseries[tseries["shape"] == "l"]["prop"].values[0],
        tseries[tseries["shape"] == "d"]["prop"].values[0],
        tseries[tseries["shape"] == "c"]["prop"].values[0]
    ])

    piz, pij, pis = stat_dist(  # get approx stationary distribution from mle
        raw_zun_mle, raw_jan_mle, raw_sim_mle)

    # get predicted proportions from phylo and sim rate matrices
    psim = get_plot_vals_alt(raw_sim, init_ratio, "sim", "mcmc")
    psim_mle = get_plot_vals_alt(raw_sim_mle, init_ratio, "sim", "mle")
    ppz = get_plot_vals_alt(raw_zun, init_ratio, "phylo", "mcmc")
    ppj = get_plot_vals_alt(raw_jan, init_ratio, "phylo", "mcmc")
    ppz_mle = get_plot_vals_alt(raw_zun_mle, init_ratio, "phylo", "mle")
    ppj_mle = get_plot_vals_alt(raw_jan_mle, init_ratio, "phylo", "mle")
    # use mle prop curves to estimate simxlims
    sim_xlims = calc_xlims_thalf(init_ratio, ppz_mle, ppj_mle,
                                 psim_mle, piz, pij, pis)

    # Create subplots
    # plt.rcParams["font.family"] = "CMU Serif"
    fsize = (5, 6) if leg_inside else (6.5, 6)
    fig, axs = plt.subplots(
        nrows=4, ncols=3, figsize=fsize, layout="constrained",
        sharey="row", sharex="col" if not VARIABLE_SIM_XLIM else False,
        gridspec_kw={"width_ratios": [0.01, 1, 1]})
    axs_g1 = [axs[0, 0], axs[1, 0], axs[2, 0], axs[3, 0]]  # for leaf icons
    axs_g2 = [axs[0, 1], axs[1, 1], axs[2, 1], axs[3, 1]]
    axs_g3 = [axs[0, 2], axs[1, 2], axs[2, 2], axs[3, 2]]
    p_xlim_loc = PHYLO_XLIM  # default xlims
    s_xlim_loc = SIM_XLIM

    for i, ax in enumerate(axs_g2):  # simulation
        if AUTO_SIM_XLIM:
            # if not variable sim xlim, use the mean across shapes
            s_xlim_loc = sim_xlims["xlim"][i] if VARIABLE_SIM_XLIM \
                else sim_xlims["xlim"].mean()
            print(f"s_xlim_loc: {s_xlim_loc}")

        tseries_sub = tseries[tseries["shape"] == ORDER[i]]
        print(tseries_sub)
        l_data, = ax.plot(tseries_sub["step"],  # plot timeseries
                          tseries_sub["prop"], c="C0", zorder=0)
        if {"lb", "ub"}.issubset(tseries_sub.columns):  # show error
            ax.fill_between(tseries_sub["step"], tseries_sub["lb"],
                            tseries_sub["ub"], alpha=AF, color="C0", ec=None)
        sim_sub = psim[psim["shape"] == ORDER[i]]
        sim_sub_mle = psim_mle[psim_mle["shape"] == ORDER[i]]
        l_fit, = ax.plot(sim_sub_mle["t"], sim_sub_mle["P"],  # plot sim ctmc
                         c="C1", ls="--", alpha=AL)
        ax.fill_between(sim_sub["t"], sim_sub["lb"], sim_sub["ub"],
                        alpha=AF, color="C1", ec=None)
        ax.set_xlim(0, s_xlim_loc)
        ax.grid(alpha=0.3)
        if i in [1, 2, 3]:
            ax.set_ylim(0, ylim)
        if i == 3:
            ax.set_xlabel("Time (step)")
        if i == 0:
            ax.set_title("Simulation")
        ax.tick_params(axis="y", labelleft=True)  # make sure ytick labs on
        if leg_inside and i == 0:
            ax.legend([l_data, l_fit], ["Data", "CTMC"],
                      loc="lower right", frameon=False)

    for i, ax in enumerate(axs_g3):  # phylogeny

        ppz_sub = ppz[ppz["shape"] == ORDER[i]]
        ppz_sub_mle = ppz_mle[ppz_mle["shape"] == ORDER[i]]
        l_phy_z, = ax.plot(ppz_sub_mle["t"], ppz_sub_mle["P"], c="C2",
                           ls="--", alpha=AL)
        ax.fill_between(ppz_sub["t"], ppz_sub["lb"], ppz_sub["ub"],
                        alpha=AF, color="C2", ec=None)

        ppj_sub = ppj[ppj["shape"] == ORDER[i]]
        ppj_sub_mle = ppj_mle[ppj_mle["shape"] == ORDER[i]]
        l_phy_j, = ax.plot(ppj_sub_mle["t"], ppj_sub_mle["P"], c="C3",
                           ls="--", alpha=AL)
        ax.fill_between(ppj_sub["t"], ppj_sub["lb"], ppj_sub["ub"],
                        alpha=AF, color="C3", ec=None)
        ax.set_xlim(0, p_xlim_loc)
        ax.grid(alpha=0.3)
        if i in [1, 2, 3]:
            ax.set_ylim(0, ylim)
        if i == 3:
            ax.set_xlabel("Time (Myr)")
        if i == 0:
            ax.set_title("Phylogeny")
        if leg_inside and i == 0:
            ax.legend([l_phy_z, l_phy_j],
                      ["CTMC Zun.",
                       "CTMC Jan."],
                      loc="lower right", frameon=False)

    for i, ax in enumerate(axs_g1):
        ax.axis("off")
        if LEAF_ICONS:
            img_box = OffsetImage(icon_imgs[i], zoom=0.08, alpha=0.5)
            text_box = TextArea(
                ["Unlobed", "Lobed", "Dissected", "Compound"][i],)
            a_box = VPacker(children=[img_box, text_box],
                            align="center", pad=0, sep=4)
            aln = (0.5, 0.5)  # box alignment axes frac
            ab = AnnotationBbox(
                a_box,
                aln,
                xycoords="axes fraction",
                box_alignment=aln,
                frameon=False,
                pad=0.2,
            )
            ax.add_artist(ab)

    fig.supylabel("Proportion")
    if not leg_inside:
        if SHOW_PHYLO:
            leg = fig.legend([l_data, l_fit, l_phy_z, l_phy_j],
                             ["Simulation Data", "Simulation CTMC",
                             "Phylogeny CTMC\nZuntini et al. (2024)",
                              "Phylogeny CTMC\nJanssens et al. (2021)"],
                             loc="outside center right"
                             )
        else:
            leg = fig.legend([l_data, l_fit],
                             ["Simulation Data", "Simulation CTMC"],
                             loc="outside center right")
        for lh in leg.legend_handles:
            lh.set_alpha(1)
    plt.savefig(
        f"curves_{PLOT}_mcerr{MC_ERR_SAMP}_{SIMRATES}_{PHYLORATES1}.pdf",
        format="pdf", dpi=1200, metadata={"Keywords": str(G_PARAMS)})
    plt.savefig(
        f"curves_{PLOT}_mcerr{MC_ERR_SAMP}_{SIMRATES}_{PHYLORATES1}.svg",
        format="svg", dpi=1200, metadata={"Keywords": str(G_PARAMS)})
    plt.show()


def plot_tseries(show_ctmc=True):
    """Plot just the time series data from the simulation."""
    icon_imgs = load_leaf_imgs()  # get leaf icons

    tseries = get_timeseries()
    tseries = tseries.rename(columns={"mean_prop": "prop"})  # line is mean

    #### Get sim-rates ####
    _, raw_sim = get_sim_rates("mcmc")
    _, raw_sim_mle = get_sim_rates("mle")
    #### Get sim timeseries data ####
    # tseries = get_timeseries_alt()
    tseries = get_timeseries()  # for error bar on timeseries
    tseries = tseries.rename(columns={"mean_prop": "prop"})  # line is mean
    init_ratio = np.array([  # get proportion of each shape at step=0 in data
        tseries[tseries["shape"] == "u"]["prop"].values[0],
        tseries[tseries["shape"] == "l"]["prop"].values[0],
        tseries[tseries["shape"] == "d"]["prop"].values[0],
        tseries[tseries["shape"] == "c"]["prop"].values[0]
    ])
    # get predicted proportions from phylo and sim rate matrices
    psim = get_plot_vals_alt(raw_sim, init_ratio, "sim", "mcmc")
    psim_mle = get_plot_vals_alt(raw_sim_mle, init_ratio, "sim", "mle")

    fig, axs = plt.subplots(figsize=(6, 6), nrows=2, ncols=2, sharex=True,
                            layout="constrained")
    axs = axs.flatten()
    for i, shape in enumerate(ORDER):
        tseries_sub = tseries[tseries["shape"] == shape].reset_index(drop=True)
        print(tseries_sub)
        l_tser, = axs[i].plot(tseries_sub["step"],
                              tseries_sub["prop"], c="C0", zorder=0)
        if {"lb", "ub"}.issubset(tseries_sub.columns):  # show error
            axs[i].fill_between(tseries_sub["step"], tseries_sub["lb"],
                                tseries_sub["ub"], alpha=AF, color="C0",
                                ec=None)
        if show_ctmc:  # plot sim ctmc
            sim_sub = psim[psim["shape"] == ORDER[i]]
            sim_sub_mle = psim_mle[psim_mle["shape"] == ORDER[i]]
            l_fit, = axs[i].plot(sim_sub_mle["t"], sim_sub_mle["P"],
                                 c="C1", ls="--")
            axs[i].fill_between(sim_sub["t"], sim_sub["lb"], sim_sub["ub"],
                                alpha=AF, color="C1", ec=None)

        axs[i].grid(alpha=0.3)
        axs[i].set_title(["Unlobed", "Lobed", "Dissected", "Compound"][i])
        if show_ctmc and i == 3:
            axs[i].legend([l_tser, l_fit], [
                          "Simulation Data", "Simulation CTMC"])

        if LEAF_ICONS:
            img_box = OffsetImage(icon_imgs[i], zoom=0.08, alpha=0.5)
            aln = (0, 1)
            ab = AnnotationBbox(
                img_box,
                aln,
                xycoords="axes fraction",
                box_alignment=(0, 0),  # anchor is bottom left
                frameon=False,
                pad=0.2,
            )
            axs[i].add_artist(ab)

    fig.supxlabel("Simulation step")
    fig.supylabel("Proportion")
    plt.savefig(f"tseries_{PLOT}_{SIMRATES}_mcerr{MC_ERR_SAMP}.svg",
                format="svg", dpi=600)
    plt.show()


if __name__ == "__main__":
    for name, val in G_PARAMS.items():
        print(f"{name} {val}")
    if PLOT == 1:
        plot_sim_and_phylogeny_curves_sep_init()
    if PLOT == 2:
        plot_sim_and_phylogeny_curves_prop()
    if PLOT == 4:
        plot_sim_and_phylogeny_prop_sep()
    if PLOT == 5:
        plot_tseries()
