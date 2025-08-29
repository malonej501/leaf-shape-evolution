"""For generating figure 7b."""
import os
from datetime import date
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from arrow_violin import import_phylo_and_sim_rates, import_phylo_ml_rates, \
    import_sim_ml_rates, normalise_rates, rates_map2

NORM_METHOD = "meanmean"
# ML_DATA = "ML6_genus_mean_rates_all"  # ML data for the phylogeny
ML_DATA = "ML8_mean_rates_all"
LEGEND = False
# XLABS = ["MUT1", "MUT2", "Janssens", "Zuntini", "Geeta"]  # in PLOT_ORDER
XLABS = ["MUT2", "MUT5", "Jan. gen.", "Zun. gen.",
         "Jan. sp.", "Zun. sp.", "Geeta sp."]

PLOT_ORDER = [
    # "MUT1_06-02-25",
    "MUT2_320_mcmc_2_24-04-25",
    "MUT5_320_mcmc_23-07-25_1",
    # "jan_phylo_nat_class_uniform0-0.1_1",
    # "zuntini_phylo_nat_class_10-09-24_genera_class_uniform0-0.1_2",
    # "geeta_phylo_geeta_class_uniform0-100_4",
    "jan_genus_phylo_nat_26-09-24_class_uniform0-0.1_genus_1",
    "zun_genus_phylo_nat_26-09-24_class_uniform0-0.1_genus_1",
    "jan_nat_species_11-07-25_uniform0-0.1_species_11-07-25_1",
    "zun_nat_species_11-07-25_uniform0-0.1_species_11-07-25_1",
    "geeta_phylo_geeta_class_uniform0-100_genus_1",
]
G_PARAMS = {name: val for name, val in globals().items()if name.isupper()}


def plot_phylo_and_sim_rates_with_leaf_icons():
    """Violins for the Q matrix showing estimates from simulations and
    phylogeny including MCMC and ML estimates"""

    ml_ph_l = import_phylo_ml_rates(calc_diff=False, p_order=PLOT_ORDER)
    ml_sm_l = import_sim_ml_rates(calc_diff=False)
    ph_sm_l = import_phylo_and_sim_rates(calc_diff=False, p_order=PLOT_ORDER)
    print(ph_sm_l["Dataset"].unique())

    ph_sm_l, ml_ph_l, ml_sm_l = normalise_rates(ph_sm_l, ml_ph_l, ml_sm_l)
    ml_sm_l = ml_sm_l.rename(columns={"Dataset": "dataset"})
    ml_all = pd.concat([ml_ph_l, ml_sm_l], ignore_index=True)  # combine ML

    #### plotting ####
    # plt.rcParams["font.family"] = "CMU Serif"
    if LEGEND:
        fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(10, 8))
    else:
        fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(9, 9))
    counter = -1
    legend_labels = []
    for i in range(1, 5):
        for j in range(0, 4):
            ax = axes[i, j]
            ax.grid(axis="y", alpha=0.3)
            if i - 1 == j:
                ax.axis("off")
                continue  # skip diagonals
            counter += 1
            transition = list(rates_map2.values())[counter]
            plot_data = ph_sm_l[ph_sm_l["transition"]
                                == transition]
            ml_plot_data = ml_all[ml_all["transition"] == transition]
            rates = []
            ml_rates = []
            for dataset in PLOT_ORDER:
                rates.append(plot_data["rate_norm"][
                    plot_data["Dataset"] == dataset].squeeze())

                # get ml_plot_data in correct order
                x = ml_plot_data[ml_plot_data["dataset"].apply(
                    lambda x, d=dataset: x in d)].reset_index(drop=True)

                if not x.empty:
                    ml_rates.append(x.loc[0, "rate_norm"])
                elif x.empty:
                    ml_rates.append(np.nan)
                if dataset not in legend_labels:
                    legend_labels.append(dataset)

            ax.axvline(2.5, linestyle="--", color="grey", alpha=0.5)
            bp = ax.violinplot(rates, showextrema=False, showmeans=True)

            # for median in bp["medians"]:
            #     median.set_visible(False)
            # ax.set_title(transition)
            if NORM_METHOD == "zscore":
                ax.set_ylim(-2.7, 8)  # for z-score normalisation
            elif NORM_METHOD == "zscore+2.7":
                ax.set_ylim(0, 10.7)  # for z-score norm + 2.7
            elif NORM_METHOD == "zscore_global":
                ax.set_ylim(-2, 5)
            elif NORM_METHOD == "meanmean":
                ax.set_ylim(0, 10)  # for mean-mean normalisation
            elif NORM_METHOD == "minmax":
                ax.set_ylim(-0.5, 3)  # for min-max normalisation

            # plot ML values
            if ML_DATA:
                pos = list(range(1, len(PLOT_ORDER) + 1))
                ax.scatter(pos, ml_rates, color="black", zorder=5, s=8,
                           facecolors="white")

            xticklabs = ["S1", "S2"]
            xticklabs.extend([f"P{i}" for i in range(1, len(PLOT_ORDER) - 1)])
            ax.set_xticks(list(range(1, len(PLOT_ORDER) + 1)))
            if i == 4:
                ax.set_xticks(list(range(1, len(PLOT_ORDER) + 1)), XLABS,
                              fontsize=9)
                ax.set_xticklabels(ax.get_xticklabels(),
                                   rotation=45, ha="right")
            if (i, j) == (3, 3):
                ax.set_xticks(list(range(1, len(PLOT_ORDER) + 1)), XLABS,
                              fontsize=9)
                ax.set_xticklabels(ax.get_xticklabels(),
                                   rotation=45, ha="right")
            if j != 0 and (i, j) != (1, 1):
                ax.set_yticklabels([])
            if i != 4 and (i, j) != (3, 3):
                ax.set_xticklabels([])

    ### plot leaf images ####
    icon_filenames = [
        "leaf_p7a_0_0.png",
        "leaf_p8ae_0_0.png",
        "leaf_pd1_0_0.png",
        "leaf_pc1_alt_0_0.png",
    ]
    icons = [os.path.join("uldc_model_icons", path) for path in icon_filenames]
    icon_imgs = [Image.open(path) for path in icons]
    img_width, img_height = icon_imgs[1].size
    scale_factor = 0.5
    shape_cats = ["Unlobed", "Lobed", "Dissected", "Compound"]

    for j in range(0, 4):
        ax = axes[0, j]
        ax.axis("off")
        ax.imshow(icon_imgs[j])
        ax.text(img_width / 2, img_height,
                shape_cats[j], ha="center", va="top")
        ax.set_xlim(img_width / scale_factor, (-img_width / 2) / scale_factor)
        ax.set_ylim(img_height, -(img_height / scale_factor))
    for i in range(1, 5):
        ax = axes[i, 4]
        ax.axis("off")
        ax.imshow(icon_imgs[i - 1])
        ax.text(img_width / 2, img_height,
                shape_cats[i - 1], ha="center", va="top")
        ax.set_xlim(0, (img_width / scale_factor) +
                    ((img_width / 2) / scale_factor))
        ax.set_ylim(img_height / scale_factor,
                    (-img_height / 2) / scale_factor)
    axes[0, 4].axis("off")

    xlab_pos = 0.43
    ylab_pos = 0.45
    fig.supxlabel("Dataset", x=xlab_pos, ha="center")
    fig.supylabel("Normalised rate", y=ylab_pos, ha="center", va="center")
    fig.text(xlab_pos, 0.9, "Final shape",
             ha="center", va="center", fontsize=12)
    fig.text(0.9, ylab_pos, "Initial shape", ha="center",
             va="center", rotation=270, fontsize=12)
    plt.tight_layout()
    if LEGEND:
        if NORM_METHOD in ["zscore", "zscore+2.7", "zscore_global"]:
            plt.subplots_adjust(hspace=0.2, wspace=0.2,  # for z-score-norm
                                right=0.745, left=0.064)
        else:
            plt.subplots_adjust(hspace=0.2, wspace=0.2,  # for min-max-norm
                                right=0.745, left=0.044)  # or mean-mean-norm
    else:
        plt.subplots_adjust(hspace=0.2, wspace=0.2)

    plt.savefig(f"violin_{str(date.today())}.pdf", format="pdf", dpi=1200,
                metadata={"Keywords": str(G_PARAMS)})
    plt.show()


if __name__ == "__main__":
    for name, val in G_PARAMS.items():
        print(f"{name} {val}")
    plot_phylo_and_sim_rates_with_leaf_icons()
