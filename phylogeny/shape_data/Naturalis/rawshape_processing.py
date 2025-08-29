"""This script is for checking all expected herbarium images are present and
correctly labelled, as well as combining and cleaning labels split across
multiple files."""
import os
import re
import pandas as pd
from datetime import datetime

WD = "jan_zun_nat_ang_11-07-25"
LAB_DIR = WD + "/labels_separate"  # raw shape labels
SP_EXP = WD + "/jan_zun_union_nat_species_11-07-25.csv"  # expected species
SP_EXP_NAME = SP_EXP.split(".")[0]


def check_alldownloaded(shape_data, sp_full_df):
    """Check if all images from the sample have been downloaded."""

    missing_in_shape_data = set(
        sp_full_df["species"]) - set(shape_data["species"])
    print(f"Items in {SP_EXP} but not in label data:")
    print(missing_in_shape_data if missing_in_shape_data else "None")
    missing_in_sp_full_df = set(
        shape_data["species"]) - set(sp_full_df["species"])
    print(f"Items present in label data but not in {SP_EXP}:")
    print(missing_in_sp_full_df if missing_in_sp_full_df else "None")


def main():
    """Main function to process raw shape labels and expected species data."""

    sp_full_df = pd.read_csv(SP_EXP)  # expected species

    shapes = pd.DataFrame()

    # Get all shape labels from the separate files
    filelist = []
    for f in os.listdir(LAB_DIR):
        if "img_labels" in f and f.endswith(".csv") and "full" not in f:
            filelist.append(f)

    # Sort files by the numerical component in their names
    sorted_file_list = sorted(
        filelist,
        key=lambda x: int(re.search(r"\d+", x).group()),
    )

    # Read and concatentate all shape label files
    for f in sorted_file_list:
        if "img_labels" in f and f.endswith(".csv") and "full" not in f:
            print(f"Processing file: {f}")
            df = pd.read_csv(f"{LAB_DIR}/{f}")
            shapes = pd.concat([shapes, df], ignore_index=True)

    print(f"Unique shapes found: {set(shapes['shape'])}\n Cleaning up...")

    # clean up common mistakes in shape labels
    shapes.replace("y", "a", inplace=True)
    shapes.replace("f", "a", inplace=True)
    shapes.replace("s", "a", inplace=True)
    shapes.replace("v", "a", inplace=True)

    n_lbs_init = len(shapes)
    n_missing = len(sp_full_df) - n_lbs_init
    print(f"Total no. labels: {n_lbs_init} of {len(sp_full_df)} expected,\n" +
          f" {n_missing} missing")
    print(shapes["shape"].value_counts())

    # remove numerical name component
    shapes["species"] = shapes["species"].replace("\d+", "", regex=True)
    # check if all expected images have been labelled
    check_alldownloaded(shapes, sp_full_df)
    # drop any remaining duplicates
    shapes.drop_duplicates(subset="species", keep="first", inplace=True)
    n_dups = n_lbs_init - len(shapes)
    print(f"{n_dups} duplicates removed")

    # drop ambiguous species
    n_ambig = shapes[shapes["shape"] == "a"].shape[0]
    print(f"Dropping {n_ambig} ambiguous species")
    shapes_unambig = shapes[shapes["shape"] != "a"
                            ].reset_index(drop=True)
    print(f"Unambiguous species count: {len(shapes_unambig)}")

    shapes_unambig["shape"] = shapes_unambig["shape"].replace(
        {"u": 0, "l": 1, "d": 2, "c": 3})

    # drop any duplicates
    species_full_clean = sp_full_df.drop_duplicates(
        subset="species", keep="first")

    species_full_labelled = pd.merge(
        species_full_clean, shapes_unambig, on="species", how="inner"
    )
    print(species_full_labelled)

    print(f"Expected no. labels: {len(sp_full_df)}")
    print(f"No. missing labels: {n_missing}")
    print(f"No. duplicate species: {n_dups}")
    print(f"No. ambiguous labels: {n_ambig}")
    print(f"Unambiguous labels: {len(species_full_labelled)}")
    exp_total = len(sp_full_df) - n_missing - n_dups - n_ambig
    print(f"n_exp_labs - n_missing - n_dups - n_ambig = {exp_total}")
    assert len(species_full_labelled) == exp_total, \
        "Number of expected labels does not match the number of labels found!"
    print(species_full_labelled["shape"].value_counts())

    species_full_labelled.to_csv(
        f"{SP_EXP_NAME}_labelled.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
