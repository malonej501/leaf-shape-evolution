"""For generating figure 7a."""
import os
from datetime import date
import numpy as np
import pandas as pd
from textwrap import wrap
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from PIL import Image


PLOT = 0  # 0-arrow violin, 1-arrow single
P1_DSET = 5  # dataset to plot for arrow_plot in PLOT_ORDER[P1_DSET]
SF = 2  # scale factor for the thickness of the arrows - 2 for P0, 5 for P1
C_VAL = 2  # increase to increase the curviness of the arrows
DC_VAL = 5  # increas to increase diagonal arrow curviness
RAD = 0.5  # padding between leaf icon and circle where arrows join
# the credible interval for shading the arrows grey or black
# (>CI must be above or below zero to be black)
CI = 0.95

NORM_MTHD = "meanmean"
# ML_DATA = "ML6_genus_mean_rates_all"
ML_DATA = "ML8_mean_rates_all"
# ML_DATA = "ML4_mean_rates_all"
# sim1 = "MUT1_mcmc_11-12-24"
# sim2 = "MUT2_mcmc_11-12-24"
# SIM1 = "MUT1_06-02-25"
SIM1 = "MUT2_320_mcmc_2_24-04-25"
SIM2 = "MUT5_320_mcmc_23-07-25_1"
# SIM2 = "MUT2_mcmc_05-02-25"

PLOT_ORDER = [
    SIM1,
    SIM2,
    # "jan_phylo_nat_class_uniform0-0.1_5",
    # "zuntini_phylo_nat_class_10-09-24_genera_class_uniform0-0.1_5",
    # "geeta_phylo_geeta_class_uniform0-100_6",
    "jan_genus_phylo_nat_26-09-24_class_uniform0-0.1_genus_1",
    "zun_genus_phylo_nat_26-09-24_class_uniform0-0.1_genus_1",
    "jan_nat_species_11-07-25_uniform0-0.1_species_11-07-25_1",
    "zun_nat_species_11-07-25_uniform0-0.1_species_11-07-25_1",
    "geeta_phylo_geeta_class_uniform0-100_genus_1",
    # "jan_equal_genus_phylo_nat_class_uniform0-0.1_4",
    # "zuntini_genera_equal_genus_phylo_nat_class_uniform0-0.1_4",
]
TRANS_FNAME = ["luvd.png", "duvd.png", "cuvd.png",  # transition icons
               "ldvd.png", "lcvd.png", "dcvd.png"]
ICON_FNAME = ["leaf_p7a_0_0.png", "leaf_p8ae_0_0.png",  # single leaf icons
              "leaf_pd1_0_0.png", "leaf_pc1_alt_0_0.png"]
SHOW_TITLES = True  # show titles on the plots
# PLOT_TITLES = ["MUT2", "MUT5", "Janssens et al. (2020)",
#                "Zuntini et al. (2024)", "Geeta et al. (2012)"]
PLOT_TITLES = ["MUT2", "MUT5", "Janssens et al. (2020)\ngenus",
               "Zuntini et al. (2024)\ngenus",
               "Janssens et al. (2020)\nspecies",
               "Zuntini et al. (2024)\nspecies",
               "Geeta et al. (2012)\ngenus"]  # in PLOT_ORDER
G_PARAMS = {name: val for name, val in globals().items()if name.isupper()}


rates_map2 = {
    "0": "u→l",
    "1": "u→d",
    "2": "u→c",
    "3": "l→u",
    "4": "l→d",
    "5": "l→c",
    "6": "d→u",
    "7": "d→l",
    "8": "d→c",
    "9": "c→u",
    "10": "c→l",
    "11": "c→d",
}

rates_map3 = {
    "q01": "u→l",
    "q02": "u→d",
    "q03": "u→c",
    "q10": "l→u",
    "q12": "l→d",
    "q13": "l→c",
    "q20": "d→u",
    "q21": "d→l",
    "q23": "d→c",
    "q30": "c→u",
    "q31": "c→l",
    "q32": "c→d",
}


def get_rates_batch(directory):
    """Retreive all phylo rates and concatenate into a single dataframe"""
    data = []

    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            path = os.path.join(directory, filename)
            df = pd.read_csv(path)
            df["phylo-class"] = filename[:-4]
            data.append(df)

    data_concat = pd.concat(data, ignore_index=True)

    return data_concat


def calc_net_rate(rates):
    """Calculate the net rate for each transition by subtracting the reverse
    transition rate from the forward transition rate"""
    transitions = rates_map3.values()
    for fwd in transitions:
        bwd = fwd[2] + "→" + fwd[0]
        if fwd != bwd:
            col_name = f"{fwd}-{bwd}"
            rates[col_name] = rates[fwd] - rates[bwd]
    return rates


def import_phylo_and_sim_rates(calc_diff, p_order=PLOT_ORDER):
    """Import Q-matrix posteriors for phylo and sim and calculate net rates"""
    # to return diff between rates rather than raw rates, set calc_diff true
    #### import data ####
    phylo_rates = get_rates_batch(directory="rates/uniform_1010000steps")
    s1 = pd.read_csv(f"../dataprocessing/markov_fitter_reports/emcee/{SIM1}/"
                     f"posteriors_{SIM1}.csv")
    s2 = pd.read_csv(f"../dataprocessing/markov_fitter_reports/emcee/{SIM2}/"
                     f"posteriors_{SIM2}.csv")
    s1["phylo-class"] = SIM1
    s2["phylo-class"] = SIM2

    sim_rates = pd.concat([s1, s2]).reset_index(drop=True)
    sim_rates = sim_rates.rename(columns=rates_map2)
    phylo_rates = phylo_rates.rename(columns=rates_map3)
    phylo_sim = pd.concat([sim_rates, phylo_rates]).reset_index(drop=True)
    phylo_sim = phylo_sim.rename(columns={"phylo-class": "Dataset"})

    phylo_sim = calc_net_rate(phylo_sim) if calc_diff else phylo_sim

    phy_sim = pd.melt(
        phylo_sim,
        id_vars=["Dataset"],
        var_name="transition",
        value_name="rate",
    )
    phy_sim["dataname"] = phy_sim["Dataset"].apply(
        lambda x: x.split("_class", 1)[0] + "_class" if "_class" in x else x
    )

    phy_sim["dataname"] = phy_sim["dataname"].apply(
        lambda x: x.split("_uniform", 1)[0] if "_uniform" in x else x
    )

    # filter to only rows with dataset in the PLOT_ORDER list
    phy_sim = phy_sim[
        phy_sim["Dataset"].isin(p_order)
    ].reset_index(drop=True)

    return phy_sim


def import_phylo_ml_rates(calc_diff, p_order=PLOT_ORDER):
    """Import Q-matrix ML estimates for phylo"""
    qml = pd.read_csv(f"rates/ML/{ML_DATA}.csv")
    qml.drop(
        columns=[
            "Lh",
            "Root P(0)",
            "Root P(1)",
            "Root P(2)",
            "Root P(3)",
            "Unnamed: 11",
        ],
        inplace=True,
        errors="ignore",
    )
    qml = qml.rename(columns=rates_map3)
    qml["dataname"] = qml["dataset"].apply(
        lambda x: x.split("_class", 1)[0] + "_class" if "_class" in x else x
    )

    qml = calc_net_rate(qml) if calc_diff else qml

    qml_long = pd.melt(
        qml,
        id_vars=["dataname", "dataset"],
        var_name="transition",
        value_name="rate",
    )
    # filter to only rows with dataset in the PLOT_ORDER list
    ml_plot_order = [x.split("_class", 1)[0] + "_class" for x in p_order]
    qml_long = qml_long[
        qml_long["dataname"].apply(
            lambda x: any(x in y for y in ml_plot_order))
    ].reset_index(drop=True)
    return qml_long


def import_sim_ml_rates(calc_diff):
    """Import Q-matrix ML estimates for sim"""
    mut1_ml = pd.read_csv(
        f"../dataprocessing/markov_fitter_reports/emcee/{SIM1}/ML_{SIM1}.csv")
    mut2_ml = pd.read_csv(
        f"../dataprocessing/markov_fitter_reports/emcee/{SIM2}/ML_{SIM2}.csv")
    mut1_ml["Dataset"] = SIM1
    mut2_ml["Dataset"] = SIM2
    sim_ml = pd.concat([mut1_ml, mut2_ml], ignore_index=True)
    sim_ml = sim_ml.rename(columns=rates_map2)

    if calc_diff:
        transitions = rates_map3.values()
        for fwd in transitions:
            bwd = fwd[2] + "→" + fwd[0]
            if fwd != bwd:
                col_name = f"{fwd}-{bwd}"
                sim_ml[col_name] = sim_ml[fwd] - sim_ml[bwd]

    sim_ml_long = pd.melt(sim_ml, id_vars="Dataset",
                          var_name="transition", value_name="rate")
    sim_ml_long["dataname"] = sim_ml_long["Dataset"]
    return sim_ml_long


def test_rates_diff_from_zero(phy_sim):
    """Return fraction of normalised rate posteriors that are > 0"""
    phy_sim_filt = phy_sim[
        phy_sim["transition"].str.contains(
            r"^[a-zA-Z]→[a-zA-Z]-[a-zA-Z]→[a-zA-Z]$"
        )
    ]
    transitions = phy_sim_filt["transition"].unique()
    results = []
    for trans in transitions:
        mcmc_phy_pdata = phy_sim[phy_sim["transition"] == trans]
        for name, group in mcmc_phy_pdata.groupby("Dataset"):
            n = len(group)
            ng0 = len(group[group["rate_norm"] > 0])  # number of rates > 0

            mean = group["rate_norm"].mean()
            # t_stat, p_val = stats.ttest_1samp(group["rate_norm"], 0)
            std = group["rate_norm"].std()
            results.append(
                {
                    "dataset": name,
                    "transition": trans,
                    "mean_rate_norm": mean,
                    # "t_stat": t_stat,
                    # "p_val": p_val,
                    "std": std,
                    "lb": mean - std,
                    "ub": mean + std,
                    "n": n,
                    "prop_over_zero": ng0 / n,
                }
            )
    r = pd.DataFrame(results)
    return r


def arc(ax, startpoint, endpoint, curvature, rate, rate_c, rate_std,
        significant, cur):
    """Draw an arc between two points with a given curvature and thickness
     proportional to rate"""
    # increase cur to increase the curviness of the arrows
    if not rate_std:
        if significant:
            arrow = FancyArrowPatch(
                (startpoint[0], startpoint[1]),  # Start point
                (endpoint[0], endpoint[1]),  # End point
                connectionstyle=f"arc3,rad={curvature}.{cur}",  # Curvature
                mutation_scale=10,  # Arrow head size
                arrowstyle="-|>",  # Arrow style
                color=rate_c,  # Arrow color
                lw=rate,  # Line width
            )
        else:
            arrow = FancyArrowPatch(
                (startpoint[0], startpoint[1]),  # Start point
                (endpoint[0], endpoint[1]),  # End point
                connectionstyle=f"arc3,rad={curvature}.{cur}",  # Curvature
                mutation_scale=10,  # Arrow head size
                arrowstyle="-|>",  # Arrow style
                color=rate_c,  # Arrow color
                lw=rate,  # Line width
                zorder=0,
            )
        arrow.set_joinstyle("miter")
        arrow.set_capstyle("butt")
        ax.add_patch(arrow)

    if rate_std:
        arrow_ub = FancyArrowPatch(
            (startpoint[0], startpoint[1]),  # Start point
            (endpoint[0], endpoint[1]),  # End point
            connectionstyle=f"arc3,rad={curvature}.{cur}",  # Curvature
            mutation_scale=10,  # Arrow head size
            arrowstyle="-|>",  # Arrow style
            color="black",  # Arrow color
            fill=False,
            lw=rate + rate_std + 10,  # Line width
        )
        arrow_ub.set_joinstyle("miter")
        arrow = FancyArrowPatch(
            (startpoint[0], startpoint[1]),  # Start point
            (endpoint[0], endpoint[1]),  # End point
            connectionstyle=f"arc3,rad={curvature}.{cur}",  # Curvature
            mutation_scale=10,  # Arrow head size
            arrowstyle="-|>",  # Arrow style
            alpha=0.2,
            color=rate_c,  # Arrow color
            lw=rate,  # Line width
        )
        arrow.set_joinstyle("miter")
        arrow_lb = FancyArrowPatch(
            (startpoint[0], startpoint[1]),  # Start point
            (endpoint[0], endpoint[1]),  # End point
            connectionstyle=f"arc3,rad={curvature}.{cur}",  # Curvature
            mutation_scale=10,  # Arrow head size
            arrowstyle="-|>",  # Arrow style
            fill=False,
            color="white",
            lw=rate - rate_std,  # Line width
        )
        arrow_lb.set_joinstyle("miter")

        # ax.add_patch(arrow)
        ax.add_patch(arrow_ub)
        ax.add_patch(arrow_lb)


def nodes(ax, texts, icon_imgs):
    """Draw nodes with leaf icons and text"""
    # textplacement goes from top right clockwise
    c_proper = {"u": (2, 6), "l": (6, 6), "d": (6, 2), "c": (2, 2)}  # node pos
    text_placement = ((-0.1, 1), (0.1, 1), (0.1, -1), (-0.1, -1))
    for i, center in enumerate(c_proper.values()):
        x, y = center
        theta = np.linspace(0, 2 * np.pi, 100)
        w, h = icon_imgs[i].size
        scf = 0.0021

        ax.imshow(
            icon_imgs[i],
            extent=[x - (w * scf / 2), x + (w * scf / 2),
                    y - (h * scf / 2), y + (h * scf / 2)],
            rasterized=True,
        )

        ax.text(
            x + text_placement[i][0],
            y + text_placement[i][1],
            texts[i],
            horizontalalignment="left" if i in [1, 2] else "right",
            verticalalignment="center",
            color="black",
            fontsize=9,
        )


def normalise_rates(phy_sim, ml_q_phy, ml_q_sim):
    """
    Normalise the rates across datasets with the specified method
    NORM_MTHD = "meanmean", "zscore", "zscore+2.7", "zscore_global", "minmax"
    """
    # only calculate the mean rate for transitions, not transition differences
    phy_sim_filt = phy_sim[
        phy_sim["transition"].isin(rates_map3.values())
    ]
    phy_sim["mean_rate"] = phy_sim_filt.groupby(
        ["Dataset", "transition"]
    )["rate"].transform(
        "mean"
    )  # get mean rate for each transition per dataset

    if NORM_MTHD == "meanmean":
        # get the mean mean transition rate per dataset (i.e. the centre of the
        # rates for that dataset)
        phy_sim["mean_mean"] = phy_sim.groupby(["Dataset"])[
            "mean_rate"
        ].transform("mean")
        # normalise by dividing by the mean mean transition rate for each
        # dataset
        phy_sim["rate_norm"] = phy_sim["rate"] / phy_sim["mean_mean"]

        # merge mcmc mean-means with ML-rates
        ml_q_phy = pd.merge(
            ml_q_phy,
            phy_sim[["dataname", "mean_mean"]].drop_duplicates(
                subset=["dataname"]
            ),
            on="dataname",
        )
        ml_q_sim = pd.merge(
            ml_q_sim,
            phy_sim[["dataname", "mean_mean"]].drop_duplicates(
                subset=["dataname"]
            ),
            on="dataname",
        )
        # normalise ML rates
        ml_q_phy["rate_norm"] = ml_q_phy["rate"] / ml_q_phy["mean_mean"]

        ml_q_sim["rate_norm"] = ml_q_sim["rate"] / ml_q_sim["mean_mean"]
        # phy_sim["initial_shape"], phy_sim["final_shape"] = zip(
        #     *phy_sim["transition"].map(rates_map)
        # )

    elif NORM_MTHD == "zscore":
        # get the mean mean transition rate per dataset (i.e. the centre of the
        # rates for that dataset)
        phy_sim["mean_mean"] = phy_sim.groupby(["Dataset"])[
            "mean_rate"
        ].transform("mean")
        # get the stdev of the mean transition rate per dataset
        phy_sim["std_mean"] = phy_sim.groupby("Dataset")[
            "mean_rate"
        ].transform("std")
        # z-score normalisation
        phy_sim["rate_norm"] = (
            phy_sim["rate"] - phy_sim["mean_mean"]
        ) / phy_sim["std_mean"]
        # normalise ML rates
        ml_q_phy["rate_norm"] = (
            ml_q_phy["rate"] - phy_sim["mean_mean"]
        ) / phy_sim["std_mean"]

    elif NORM_MTHD == "zscore+2.7":
        # get the mean mean transition rate per dataset (i.e. the centre of the
        # rates for that dataset)
        phy_sim["mean_mean"] = phy_sim.groupby(["Dataset"])[
            "mean_rate"
        ].transform("mean")
        # get the stdev of the mean transition rate per dataset
        phy_sim["std_mean"] = phy_sim.groupby("Dataset")[
            "mean_rate"
        ].transform("std")
        # z-score normalisation
        phy_sim["rate_norm"] = (
            (phy_sim["rate"] - phy_sim["mean_mean"])
            / phy_sim["std_mean"]
        ) + 2.7  # move data up by 2.7 to get rid of negatives
        # normalise ML rates
        ml_q_phy["rate_norm"] = (
            (ml_q_phy["rate"] - phy_sim["mean_mean"])
            / phy_sim["std_mean"]
        ) + 2.7

    elif NORM_MTHD == "zscore_global":
        phy_sim["dataset_mean"] = phy_sim.groupby(["Dataset"])[
            "rate"
        ].transform(
            "mean"
        )  # get mean overall rate for each dataset
        # get the stdev of rates per dataset
        phy_sim["dataset_std"] = phy_sim.groupby("Dataset")[
            "rate"
        ].transform("std")
        # zscore normalisation
        phy_sim["rate_norm"] = (
            phy_sim["rate"] - phy_sim["dataset_mean"]
        ) / phy_sim["dataset_std"]
        # normalise ML rates
        ml_q_phy["rate_norm"] = (
            ml_q_phy["rate"] - phy_sim["dataset_mean"]
        ) / phy_sim["dataset_std"]

    elif NORM_MTHD == "minmax":
        # min max normalisation
        phy_sim["min_mean"] = phy_sim.groupby("Dataset")[
            "mean_rate"
        ].transform("min")
        phy_sim["max_mean"] = phy_sim.groupby("Dataset")[
            "mean_rate"
        ].transform("max")
        phy_sim["rate_norm"] = (
            phy_sim["rate"] - phy_sim["min_mean"]
        ) / (phy_sim["max_mean"] - phy_sim["min_mean"])
        # normalise ML rates
        ml_q_phy["rate_norm"] = (
            ml_q_phy["rate"] - phy_sim["min_mean"]
        ) / (phy_sim["max_mean"] - phy_sim["min_mean"])

    else:
        raise RuntimeError("Invalid NORM_MTHD argument.")

    return phy_sim, ml_q_phy, ml_q_sim


def arcs(plot_data, ax, c_val, dc_val):
    """Draw single arcs between each node with thickness porportional to the
    net rate."""
    c = {  # define arrow attachment points for nodes
        "uN": (2, 6 + RAD), "uE": (2 + RAD, 6), "uS": (2, 6 - RAD),
        "uW": (2 - RAD, 6), "lN": (6, 6 + RAD), "lE": (6 + RAD, 6),
        "lS": (6, 6 - RAD), "lW": (6 - RAD, 6), "dN": (6, 2 + RAD),
        "dE": (6 + RAD, 2), "dS": (6, 2 - RAD), "dW": (6 - RAD, 2),
        "cN": (2, 2 + RAD), "cE": (2 + RAD, 2), "cS": (2, 2 - RAD),
        "cW": (2 - RAD, 2),
    }
    # map transitions to their arc parameters: (start, end, curvature, is_diag)
    arc_map = {
        "ul": ("uS", "lS", "+", False),
        "lu": ("lN", "uN", "+", False),
        "ld": ("lE", "dE", "-", False),
        "dl": ("dE", "lE", "+", False),
        "dc": ("dS", "cS", "-", False),
        "cd": ("cS", "dS", "+", False),
        "cu": ("cW", "uW", "-", False),
        "uc": ("uE", "cE", "-", False),
        # diagonals
        "ud": ("uS", "dW", "+", True),
        "du": ("dW", "uS", "-", True),
        "cl": ("cE", "lS", "+", True),
        "lc": ("lS", "cE", "-", True),
    }

    for _, row in plot_data.iterrows():
        at = row["fwd"][0]
        to = row["fwd"][2]
        key = at + to
        r = row["mean_rate_norm"] * SF
        rc = "lightgrey" if row["prop_over_zero"] < CI else "black"
        sig = row["prop_over_zero"] >= CI
        rs = False  # set to false to disable multi-arrows

        if r > 0 and key in arc_map:
            start, end, curvature, is_diag = arc_map[key]
            curv = dc_val if is_diag else c_val
            arc(ax, c[start], c[end], curvature, r, rc, rs, sig, curv)


def import_imgs():
    """Import leaf icon images"""
    icons = [os.path.join("uldc_model_icons", p) for p in ICON_FNAME]
    transition_icons = [
        os.path.join("uldc_model_icons", p) for p in TRANS_FNAME
    ]
    icon_imgs = [Image.open(p) for p in icons]
    trans_imgs = [Image.open(p) for p in transition_icons]

    return icon_imgs, trans_imgs


def add_curly_brace(fig, x0, x1, y, label, height=0.02, text_offset=0.01):
    """Add a curly brace (approximate) between x0 and x1 at height y in figure
    coordinates."""
    # Draw the bracket as a series of lines (approximate curly brace)
    # You can refine this with Bezier curves for a more curly look if desired
    fig_width = fig.get_figwidth()
    fig_height = fig.get_figheight()
    # Main horizontal line
    fig.lines.append(plt.Line2D(
        [x0, x1], [y+height, y+height], transform=fig.transFigure, color='black',
        linewidth=1.5))
    # Left vertical
    fig.lines.append(plt.Line2D([x0, x0], [
                     y, y+height], transform=fig.transFigure,
        color='black', linewidth=1.5))
    # Right vertical
    fig.lines.append(plt.Line2D([x1, x1], [
                     y, y+height], transform=fig.transFigure,
        color='black', linewidth=1.5))
    # Add label
    fig.text((x0 + x1)/2, y + height + text_offset, label,
             ha='center', va='bottom', fontsize=12)


def arrow_viol_h():
    """Plot arrows and violin plots for all phylo and sim rates"""
    # plt.rcParams["font.family"] = "CMU Serif"
    icon_imgs, trans_imgs = import_imgs()  # import icon images

    # load rate data
    ml_q_phy = import_phylo_ml_rates(calc_diff=True)
    ml_q_sim = import_sim_ml_rates(calc_diff=True)
    phy_sim = import_phylo_and_sim_rates(calc_diff=True)
    phy_sim, ml_q_phy, ml_q_sim = normalise_rates(phy_sim, ml_q_phy, ml_q_sim)

    # process rate data
    q_data = test_rates_diff_from_zero(phy_sim)
    # q_data.to_csv(
    #     f"q_data_{NORM_MTHD}_{str(date.today())}.csv", index=False)
    q_data["fwd"] = q_data["transition"].str[:3]
    q_data["bwd"] = q_data["transition"].str[-3:]
    # q_data.to_csv(
    #     f"q_data_bw_fw_{NORM_MTHD}_{str(date.today())}.csv", index=False)

    cmap = plt.get_cmap("viridis")
    q_data["std_c"] = q_data["std"].map(cmap)

    # define arrow properties and plot titles
    texts = ["", "", "", ""]  # labels for the leaf icons
    transitions = ["l→u-u→l", "d→u-u→d", "c→u-u→c",
                   "l→d-d→l", "l→c-c→l", "d→c-c→d"]

    # Set up fig
    fig, axs = plt.subplots(
        2, len(PLOT_ORDER), figsize=(14, 6), sharey="row")
    ax_g1 = axs[0]  # 1st row subplots
    ax_g2 = axs[1]  # 2nd row subplots
    for i, ax in enumerate(ax_g1):  # arrow plots
        # if i == 1:
        #     ax.axis("off")
        #     continue
        ax.axis("off")
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 8)
        plot_data = q_data[q_data["dataset"] == PLOT_ORDER[i]]
        nodes(ax, texts, icon_imgs)  # draw nodes with leaf icons
        arcs(plot_data, ax, C_VAL, DC_VAL)  # draw arrow

        if SHOW_TITLES:
            ax.set_title(PLOT_TITLES[i], fontsize=9)
            # ax.set_title("\n".join(wrap(PLOT_ORDER[i], 20)), fontsize=9)

    for i, ax in enumerate(ax_g2):  # violin plots
        # if i == 1:
        #     ax.axis("off")
        #     continue
        mcmc_phy_pdata = phy_sim[phy_sim["Dataset"] == PLOT_ORDER[i]]
        ml_phy_pdata = ml_q_phy[ml_q_phy["dataname"].apply(
            lambda x, i=i: x in PLOT_ORDER[i])]
        ml_sim_pdata = ml_q_sim[
            ml_q_sim["Dataset"].apply(lambda x, i=i: x in PLOT_ORDER[i])]
        rates = []
        ml_rates = []
        for transition in transitions:
            rates.append(
                mcmc_phy_pdata["rate_norm"][
                    mcmc_phy_pdata["transition"] == transition
                ].squeeze()
            )
            if not ml_phy_pdata.empty:
                x = ml_phy_pdata[
                    ml_phy_pdata["transition"] == transition
                ].reset_index(drop=True)

                if not x.empty:
                    ml_rates.append(x.loc[0, "rate_norm"])
            if not ml_sim_pdata.empty:
                x = ml_sim_pdata[
                    ml_sim_pdata["transition"] == transition
                ].reset_index(drop=True)
                if not x.empty:
                    ml_rates.append(x.loc[0, "rate_norm"])
            if ml_phy_pdata.empty and ml_sim_pdata.empty:
                ml_rates.append(np.nan)

        ax.violinplot(rates, showextrema=False, showmeans=True)
        if ML_DATA:
            pos = list(range(1, len(transitions) + 1))
            ax.scatter(pos, ml_rates, color="black", zorder=5, s=8,
                       facecolors="white")  # , marker="D")
        # ax.axhline(0, linestyle="--", color="gray", alpha=0.5)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(-8, 8)
        ax.set_xticks(
            list(range(1, len(transitions) + 1)),
            transitions,
            fontsize=9,
        )
        xl, yl, xh, yh = np.array(ax.get_position()).ravel()
        w = xh - xl
        h = yh - yl
        size = 0.1  # size for transition icons
        for x, xtick_pos in enumerate(range(1, len(transitions) + 1)):
            xp = (
                xl
                + (w / (len(transitions)) * xtick_pos)
                - (0.5 * (w / (len(transitions))))
            )
            ax1 = fig.add_axes(
                [xp - (size * 0.5), yl - size -
                    (0.1 * size), size, size]
            )
            ax1.axison = False
            ax1.imshow(trans_imgs[x])
        ax.set_xticklabels([])
        ax.set_ylabel("Net normalised rate")
        if i > 0:
            ax.set_ylabel("")
        # if i == 2:
        #     ax.yaxis.set_tick_params(labelleft=True)
        #     ax.set_ylabel("Net normalised rate")
    # Add curly brace for the first row
    xpad = 0.005
    yoff = 0.1
    # Get the left and right edges of the first two columns
    x0_sim = ax_g1[0].get_position().x0 - xpad
    x1_sim = ax_g1[1].get_position().x1 + xpad
    # Get the left and right edges of the remaining columns
    x0_phy = ax_g1[2].get_position().x0 - xpad
    x1_phy = ax_g1[-1].get_position().x1 + xpad
    # Choose a y position above the axes (e.g., 0.98 in figure coordinates)
    y_brace = ax_g1[0].get_position().y1 + yoff

    add_curly_brace(fig, x0_sim, x1_sim, y_brace, "Simulation")
    add_curly_brace(fig, x0_phy, x1_phy, y_brace, "Phylogeny")

    # Create legend
    labels = ["Compound", "Dissected", "Lobed", "Unlobed"]
    # left_pos, bottom_pos, width, height
    axlgnd = fig.add_axes([0.88, 0.4, 0.1, 0.2])
    icon_w, icon_h = icon_imgs[0].size
    axlgnd.set_ylim(0, len(icon_imgs) * icon_h)
    for i, img in enumerate(reversed(icon_imgs)):
        axlgnd.imshow(img, extent=(0, icon_w, i*icon_h, (i*icon_h) + icon_h))
        axlgnd.text(icon_w, (i*icon_h) + (icon_h * 0.5),
                    labels[i], ha="left", va="center", fontsize=9)
        axlgnd.axis("off")

    # plt.tight_LAYOUT()
    # plt.savefig(f"arrow_violin_plot_ci{CI}.svg", format="svg", dpi=1200)
    print(f"Exported arrow_violin_plot_ci{CI}.svg")
    # plt.savefig(f"arrow_violin_plot_ci{CI}.pdf", format="pdf", dpi=1200,
    #             metadata={"Keywords": str(G_PARAMS)})
    print(f"Exported arrow_violin_plot_ci{CI}.pdf")

    plt.show()


def arrow_plot():
    """
    Plot arrow plot for single dataset given by PLOT_ORDER[dset]
    [dset]  0 - MUT1
            1 - MUT2
            2 - Janssens et al. (2020)
            3 - Zuntini et al. (2024)
            4 - Geeta et al. (2012)
    """

    icon_imgs, _ = import_imgs()  # import icon images
    texts = ["Unlobed", "Lobed",
             "Dissected", "Compound"]  # labels for the leaf icons

    # load rate data
    ml_q_phy = import_phylo_ml_rates(calc_diff=True)
    ml_q_sim = import_sim_ml_rates(calc_diff=True)
    phy_sim = import_phylo_and_sim_rates(calc_diff=True)
    phy_sim, ml_q_phy, ml_q_sim = normalise_rates(phy_sim, ml_q_phy, ml_q_sim)

    # process rate data
    q_data = test_rates_diff_from_zero(phy_sim)
    q_data["fwd"] = q_data["transition"].str[:3]
    q_data["bwd"] = q_data["transition"].str[-3:]
    # q_data.to_csv(
    #     f"q_data_bw_fw_{NORM_MTHD}_{str(date.today())}.csv", index=False)

    _, ax = plt.subplots(figsize=(3, 3))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    plot_data = q_data[q_data["dataset"] == PLOT_ORDER[P1_DSET]]  # sub dataset

    # Plotting
    ax.axis("off")
    nodes(ax, texts, icon_imgs)  # draw nodes with leaf icons
    arcs(plot_data, ax, C_VAL, DC_VAL)  # draw arrows
    if SHOW_TITLES:
        # ax.set_title(PLOT_TITLES[dset])
        ax.set_title("\n".join(wrap(PLOT_ORDER[P1_DSET], 20)), fontsize=9)
    plt.savefig(
        f"arrow_plot_ci{CI}_{PLOT_ORDER[P1_DSET]}.svg", format="svg", dpi=1200,
        metadata={"Keywords": str(G_PARAMS)})
    print(f"Exported arrow_plot_ci{CI}_{PLOT_ORDER[P1_DSET]}.svg")
    plt.show()


if __name__ == "__main__":
    for name, val in G_PARAMS.items():
        print(f"{name} {val}")
    if PLOT == 0:
        arrow_viol_h()
    elif PLOT == 1:
        arrow_plot()
