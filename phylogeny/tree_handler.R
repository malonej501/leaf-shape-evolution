# This script is for generating an intersect between trees and herbarium data.
# The output  will include both tree and tip labels which can then be used
# for rate inference with BayesTraits.
#
# This script requires digitised herabrium data records to be present in
# shape_data/Naturalis/botnary-20240108.dwca. 
library(ape)

# setwd("phylogeny")

load_tree <- function(tree_path) {
  tree <- read.tree(paste0("phylogenies/", tree_path))
  return(tree)
}

load_trees <- function(path) {
  ## Load all tree files in specified directory
  path <- "phylo_data/raw_trees"
  tree_files <- list.files(
    path = path,
    pattern = "\\.nwk$|\\.tree$|\\.tre$|\\.txt$|\\.nex$",
    full.names = TRUE
  )
  trees <- c()
  for (file in tree_files) {
    print(file)
    if (grepl("zuntini", file)) {
      tree <- read.tree(file)
    } else {
      tree <- read.nexus(file)
    }
    trees[[basename(file)]] <- tree
  }
  tree_filenames <- basename(tree_files)
  return(trees)
}

load_shape_data_full <- function() {
  ## Load full naturalis data with associated digitised records
  path <- "shape_data/Naturalis/botany-20240108.dwca"
  leaf_data_occur <- read.csv(paste0(path, "/Occurrence.txt"))
  names(leaf_data_occur)[names(leaf_data_occur) == "id"] <- "CoreId"
  leaf_data_media <- read.csv(paste0(path, "/Multimedia.txt"))
  # only return records with an associated digitised image
  shape_data <- merge(leaf_data_occur, leaf_data_media, by = "CoreId")
  return(shape_data)
}

load_naturalis_sample_data <- function() {
  # sample <- read.csv(paste0(
  #   "shape_data/Naturalis",
  #   "/jan_zun_nat_ang_11-07-25/",
  #   "Naturalis_occurrence_ang_sppfam-1_11-07-25_multimedia.csv"
  # ))
  sample <- read.csv(paste0(
    "shape_data/Naturalis/jan_zun_nat_ang_26-09-24/",
    "Naturalis_multimedia_ang_sample_26-09-24.csv"
  ))
  return(list(janssens_nat = sample, zuntini_nat = sample))
}

unique_tips <- function(tree) {
  ## Drop all but the first occurance of any duplicated tip labels
  # Get the tip labels
  tip_labels <- tree$tip.label
  # Identify unique labels and keep the first occurrence
  unique_labels <- unique(tip_labels)
  # Create a vector to store tips to keep
  tips_to_keep <- character()
  for (label in unique_labels) {
    # Keep the first occurrence of each label
    tips_to_keep <- c(tips_to_keep, tip_labels[which(tip_labels == label)[1]])
  }
  # Drop tips that are not in tips_to_keep
  tips_to_drop <- tip_labels[!tip_labels %in% tips_to_keep]
  pruned_tree <- drop.tip(tree, tips_to_drop)
  return(pruned_tree)
}

sub_sample_nat_tree_intersect <- function(nat_tree_intersect, tax_level) {
  ## If there are more than sample_size taxa in the intersect between the
  ## tree and naturalis data, draw sample_size taxa at random and return.
  sample_size <- 3100
  if (nrow(nat_tree_intersect) > sample_size) {
    print(paste(
      "more than", sample_size, tax_level,
      "in the naturalis intersect data",
      sep = " "
    ))
    # if there are more than 3100 species, randomly sample 3100
    nat_tree_intersect <- nat_tree_intersect[sample(
      nrow(nat_tree_intersect),
      sample_size
    ), ]
    print(paste("randomly sampled", sample_size, tax_level, sep = " "))
  }
  return(nat_tree_intersect)
}

nat_tree_intersect <- function(export, sub_sample = TRUE) {
  ## Get intersect between phylogenies and naturalis data at the species and
  ## genus level, returning both tree files and label files
  trees <- load_trees()
  print("Done loading trees")
  nat_samp_data <- load_naturalis_sample_data()
  tree_names <- c("janssens", "zuntini")
  genus_trees <- list()
  species_trees <- list()
  for (name in tree_names) {
    print(name)
    tree <- trees[[grep(name, names(trees))]]
    # return genus and species of the tip_label
    tip_gen_sp <- sub("^[^_]*_[^_]*_(.*)", "\\1", tree$tip.label)
    # return just the genus
    tip_gen <- sub("^(.*?)_.*", "\\1", tip_gen_sp)
    # set tip labels to gen_sp
    tree$tip.label <- sub("^[^_]*_[^_]*_(.*)", "\\1", tree$tip.label)
    print(paste("original tree length =", length(tree$tip.label)))

    # return the naturalis data for the current tree
    nat_samp <- nat_samp_data[[grep(name, names(nat_samp_data))]]
    # subset nat data to species present in the tree
    nat_tree_intersect_species <-
      nat_samp[nat_samp$species %in% intersect(nat_samp$species, tip_gen_sp), ]
    print(paste(
      "nat phylo species intersect =",
      nrow(nat_tree_intersect_species)
    ))
    # subset nat data to genera present in the tree
    nat_tree_intersect_genus <-
      nat_samp[nat_samp$genus %in% intersect(nat_samp$genus, tip_gen), ]
    # keep only the first occurring species in each genera
    nat_tree_intersect_genus <-
      nat_tree_intersect_genus[!duplicated(nat_tree_intersect_genus$genus), ]
    print(paste(
      "nat phylo genus intersect =", nrow(nat_tree_intersect_genus)
    ))

    if (sub_sample) {
      # if ntaxa > sample_size, randomly sample sample_size taxa
      nat_tree_intersect_species <-
        sub_sample_nat_tree_intersect(nat_tree_intersect_species, "species")
      # if ntaxa > sample_size, randomly sample sample_size taxa
      nat_tree_intersect_genus <-
        sub_sample_nat_tree_intersect(nat_tree_intersect_genus, "genus")
    }

    if (export) {
      write.csv(
        nat_tree_intersect_species,
        paste(substr(name, 1, 3), "nat_species.csv", sep = "_"),
        row.names = FALSE
      )
      print(paste("exported labels: ", substr(name, 1, 3), "_nat_species.csv"))
      write.csv(
        nat_tree_intersect_genus,
        paste(substr(name, 1, 3), "nat_genus.csv", sep = "_"),
        row.names = FALSE
      )
      print(paste("exported labels: ", substr(name, 1, 3), "_nat_genus.csv"))
    }

    # subset the trees
    # drop tree tips not present in the nat species subset
    tree_species_sub <-
      drop.tip(tree, setdiff(
        tree$tip.label,
        nat_tree_intersect_species$species
      ))
    # if some tips are duplicated, remove all copies but one
    tree_species_sub <- unique_tips(tree_species_sub)
    print(paste(
      "tree_pruned_species length =",
      length(tree_species_sub$tip.label)
    ))

    # keep one tip per genus
    tree_tip_info <- # construct df with tip label, genus and species
      data.frame(
        tip = tree$tip.label,
        tip_gen_sp = tip_gen_sp, tip_genus = tip_gen
      )
    # Randomly select one species from each genus
    unique_genus <- unique(tree_tip_info$tip_genus)
    selected_tips <- sapply(unique_genus, function(genus) {
      species_options <- tree_tip_info$tip[tree_tip_info$tip_genus == genus]
      sample(species_options, 1) # Randomly select one species
    })
    # drop all but one tip from each genera
    tree_pruned <- drop.tip(tree, setdiff(tree$tip.label, selected_tips))
    # set tip labels to genus
    tree_pruned$tip.label <- sub("^(.*?)_.*", "\\1", tree_pruned$tip.label)
    print(paste(
      "tree_pruned_1_tip_per_genus length =",
      length(tree_pruned$tip.label)
    ))
    # drop tips not present in the nat genus subset
    tree_genus_sub <-
      drop.tip(tree_pruned, setdiff(
        tree_pruned$tip.label,
        nat_tree_intersect_genus$genus
      ))
    # if some tips are duplicated, remove all copies but one
    tree_genus_sub <- unique_tips(tree_genus_sub)
    print("Tree length")
    print(paste(
      "pruned_tree intersect with naturalis tree intersect length =",
      length(tree_genus_sub$tip.label)
    ))
    if (export) {
      write.nexus(
        tree_genus_sub,
        file = paste(substr(name, 1, 3), "nat_genus.tre", sep = "_")
      )
      print(paste0("exported tree: ", substr(name, 1, 3), "_nat_genus.tre"))
      write.nexus(
        tree_species_sub,
        file = paste(substr(name, 1, 3), "nat_species.tre", sep = "_")
      )
      print(paste0("exported tree: ", substr(name, 1, 3), "_nat_species.tre"))
    }
    genus_trees[[name]] <- tree_genus_sub
    species_trees[[name]] <- tree_species_sub
  }
  return(c(genus_trees, species_trees))
}

# nat_tree_intersect(export = FALSE)

nat_full_tree_intersect <- function(export) {
  trees <- load_trees()
  nat_full <- load_shape_data_full()
  print(nat_full)
}

# nat_full_tree_intersect(export = FALSE)

generate_tree_labs <- function(
    trees, label_data, export_labs, level = "genus") {
  tree_names <- c("janssens", "zuntini")
  tree_labs <- list()
  for (name in tree_names) {
    print(name)
    tree <- trees[[grep(name, names(trees))]]

    tree_label_intersect <- label_data[
      label_data[[level]] %in% tree$tip.label,
    ]
    print(paste("nrow filtered label data", nrow(tree_label_intersect)))
    print(paste("ntips tree", length(tree$tip.label)))
    tree_label_intersect <- tree_label_intersect[, c(level, "shape")]
    tree_labs[[name]] <- tree_label_intersect
    if (export_labs) {
      write.table(
        tree_label_intersect,
        file = paste(substr(name, 1, 3), "nat", level, ".txt", sep = "_"),
        sep = "\t",
        row.names = FALSE,
        col.names = FALSE,
        quote = FALSE
      )
    }
  }
  return(tree_labs)
}

drop_ambiguous_tips <- function(
    trees, tree_labs, export_trees, level = "genus") {
  trees_unambig <- list()
  print(trees)
  for (i in seq_along(trees)) {
    tree <- trees[[i]]
    print(tree)
    name <- names(trees[i])
    print(name)
    tree_lab <- tree_labs[[i]]
    tree_unambig <- drop.tip(tree, setdiff(tree$tip.label, tree_lab[[level]]))
    print(paste("ntips before dropping ambiguous", length(tree$tip.label)))
    print(paste("no. tips in unambiguous label data", nrow(tree_lab)))
    print(paste(
      "ntips after dropping ambiguous",
      length(tree_unambig$tip.label)
    ))
    trees_unambig[[i]] <- tree_unambig

    if (export_trees) {
      write.nexus(tree_unambig, file = paste(substr(name, 1, 3),
        "nat_genus.tre",
        sep = "_"
      ))
    }
  }
  return(trees_unambig)
}


label_phylogenies <- function() {
  # Generate labels for trees from union label data
  label_data <- read.csv(paste0(
    "shape_data/Naturalis/",
    "jan_zun_nat_ang_11-07-25/",
    "jan_zun_union_nat_species_11-07-25_labelled.csv"
  ))
  trees_list <- nat_tree_intersect(export = FALSE, sub_sample = FALSE)
  g_trees <- trees_list[1:2]
  sp_trees <- trees_list[3:4]

  tree_labs <- generate_tree_labs(
    sp_trees, label_data,
    export_labs = TRUE, level = "species"
  )
  trees_unambig <- drop_ambiguous_tips(
    sp_trees, tree_labs,
    export_trees = TRUE, level = "species"
  )
}

get_nat_tree_intersect <- function() {
  nat_tree_intersect(tree_path = "phylo_data/raw_trees")
}

load_apg <- function() {
  apg <- read.csv("leaf_data/APG_IV/APG_IV_ang_fams.csv")
}


tree_shape_intersect <- function(tree_path, tree, shape_data) {
  if (grepl("Zuntini", tree_path)) {
    # get the genera present in the phylogenetic tree
    tree_genera <- c()
    for (i in seq_along(tree$tip.label)) {
      tree_genera[i] <-
        paste(strsplit(tree$tip.label[i], "_")[[1]][3], collapse = "_")
    }
  }
  # get genera from shape dataset
  shape_data_genera <- unique(shape_data$genus)
  genera_intersect <- intersect(shape_data_genera, tree_genera)

  return(genera_intersect)
}

get_random_rows <- function(df_full, value_list, column_name) {
  # subset the data to just the ID and taxon columns to save time
  df <- df_full[, c("CoreId", column_name)]
  result_list <- lapply(value_list, function(value) {
    # Filter the data frame based on the current value
    filtered_df <- df[df[[column_name]] == value, ]

    # Check if there are matching rows
    if (nrow(filtered_df) > 0) {
      # Sample a random row
      random_row <- filtered_df[sample(nrow(filtered_df), 1), ]
      return(random_row)
    } else {
      # Return NA if no matching rows
      return(NA)
    }
  })

  # Combine results into a data frame
  result_df <- do.call(rbind, result_list)
  return(result_df)
}

get_sample_from_shape_data <- function(sample_ids, shape_data) {
  sample_full <- shape_data[shape_data$CoreId %in% sample_ids$CoreId, ]
  sample_full$gen_sp <- paste0(sample_full$genus, sample_full$specificEpithet)
  sample_full_unique <- sample_full[!duplicated(sample_full$gen_sp), ]
  print(nrow(sample_full))
  print(nrow(sample_full_unique))
  return(sample_full_unique)
}

generate_phylo_nat_intersect <- function() {
  tree_path <- paste0(
    "Zuntini_2024/trees/",
    "4_young_tree_smoothing_10_",
    "pruned_for_diversification_analyses.tre"
  )
  shape_dataset <- "Naturalis"
  tree <- load_tree(tree_path)
  shape_data <- load_shape_data(shape_dataset)
  apg <- load_apg()
  genera_intersect <- tree_shape_intersect(tree_path, tree, shape_data)
  img_sample <- get_random_rows(shape_data, genera_intersect, "genus")
  sample_full_unique <- get_sample_from_shape_data(img_sample, shape_data)
  write.csv(sample_full_unique, "zuntini_naturalis_sample_16-09-24.csv")
}

compare_phylo_taxa <- function() {
  jan_labels <- read.csv(
    paste0(
      "phylogenies/final_data/labels/",
      "jan_phylo_nat_class_21-1-24.txt"
    ),
    sep = "\t",
    header = FALSE
  )
  names(jan_labels) <- c("gen_sp", "shape")
  zun_labels <- read.csv(
    paste0(
      "phylogenies/final_data/labels/",
      "zuntini_genera_phylo_nat_class_10-09-24.txt"
    ),
    sep = "\t",
    header = FALSE
  )
  names(zun_labels) <- c("genus", "shape")
  print(jan_labels)
  print(zun_labels)
  jan_labels$genus <- sub("_.*", "", jan_labels$gen_sp)
  genus_intersect <- intersect(jan_labels$genus, zun_labels$genus)
}

get_label_union <- function(x, y) {
  ## Get the union of labels from two datasets
  x <- read.csv("zun_nat_species_11-07-25.csv")
  y <- read.csv("jan_nat_species_11-07-25.csv")
  union <- rbind(x, y)
  union <- unique(union) # remove duplicate rows n.b. some duplicate species
  print(nrow(union)) # may remain
  write.csv(union, "jan_zun_union_nat_genus.csv", row.names = FALSE)
}

label_phylogenies_alt <- function(export = FALSE) {
  ## Label  subsampled trees with the union of labels from two datasets
  label_data <- read.csv(paste0(
    "shape_data/Naturalis/jan_zun_nat_ang_11-07-25/",
    "jan_zun_union_nat_species_11-07-25_labelled.csv"
  )) # union

  jan_sp <- read.nexus(paste0(
    "shape_data/Naturalis/jan_zun_nat_ang_11-07-25/",
    "unlabelled_tree_data/jan_nat_species_11-07-25.tre"
  ))
  zun_sp <- read.nexus(paste0(
    "shape_data/Naturalis/jan_zun_nat_ang_11-07-25/",
    "unlabelled_tree_data/zun_nat_species_11-07-25.tre"
  ))

  trees <- list(jan = jan_sp, zun = zun_sp)
  for (tree_name in names(trees)) {
    # export labels for each
    tree <- trees[[tree_name]]
    tree_label_intersect <- label_data[label_data$species %in% tree$tip.label, ]
    print(paste("nrow filtered label data", nrow(tree_label_intersect)))
    tree_label_intersect <- tree_label_intersect[, c("species", "shape")]
    if (export) {
      write.table(
        tree_label_intersect,
        file = paste(tree_name, "nat_species.txt", sep = "_"),
        sep = "\t",
        row.names = FALSE,
        col.names = FALSE,
        quote = FALSE
      )
      print(paste("exported labels: ", tree_name, "_nat_species.txt"))
    }

    # prune ambiguous tips and export trees
    print(paste("ntips tree", length(tree$tip.label)))
    tree_unambig <- drop.tip(
      tree,
      setdiff(tree$tip.label, tree_label_intersect$species)
    )
    print(paste(
      "ntips after dropping ambiguous",
      length(tree_unambig$tip.label)
    ))
    if (export) {
      write.nexus(
        tree_unambig,
        file = paste(tree_name, "nat_species.tre", sep = "_")
      )
      print(paste0("exported tree: ", tree_name, "_nat_species.tre"))
    }
  }
}

#### Main #####
trees <- nat_tree_intersect(export = FALSE)
genus_trees <- trees[1:2]
print(genus_trees)
jan_tips <- genus_trees$janssens$tip.label
zun_tips <- genus_trees$zuntini$tip.label
print(paste("jan tips:", length(jan_tips)))
print(paste("zun tips:", length(zun_tips)))
union <- union(jan_tips, zun_tips)
print(paste("union tips:", length(union)))
# label_trees()
# label_phylogenies()
# label_phylogenies_alt(export = TRUE)
# get_label_union()
