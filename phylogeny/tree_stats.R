# This script calculates tree statistics and generates a plot of the trees
library(ape)
library(treestats)
library(ggplot2)
library(ggtree)
library(dplyr)
library(svglite)
library(stringr)
library(RColorBrewer)

show_tip_labs <- TRUE # set to True to show tip labs on tree plots
show_order_heatmap <- FALSE # set to True to show order heatmap on tree plots
scale_bar <- TRUE # show scale bar on tree plots
legend <- FALSE # show legend on tree plots
tree_plot <- 1 # choose from tree_files which tree to plot
branch_width <- 0.1 # set the branch width for the tree plot 0.1 default
tree_files <- c(
  "zun_nat_species_11-07-25.tre",
  "jan_nat_species_11-07-25.tre",
  "zun_genus_phylo_nat_class_26-09-24.tre",
  "jan_genus_phylo_nat_class_26-09-24.tre",
  "geeta_phylo_geeta_class_23-04-24.tre"
  # "jan_phylo_nat_class_21-01-24.tre"
)

import_trees <- function() {
  # import trees
  tre_names <- sort(list.files("phylo_data/trees_final"))
  tre_names_filt <- tre_names[!grepl("shuff|each|cenrich", tre_names)] #
  summary <- data.frame(dataset = tre_names_filt)
  return(summary)
}

import_labels <- function() {
  labels <- sort(list.files("shape_data/labels_final"))
  labels_filt <- labels[!grepl("shuff|each|cenrich", labels)]
  label_data <- data.frame()
  for (label in labels_filt) {
    data <- read.delim(paste0(
      "shape_data/labels_final/",
      label
    ), header = FALSE, sep = "\t")
    data$dataset <- sub("^(.*class).*", "\\1", label) # format dataset name
    data$dataset <- sub("\\.txt$", "", data$dataset)
    label_data <- rbind(label_data, data)
  }
  names(label_data) <- c("label", "shape_num", "dataset")
  label_data <- label_data[, c("dataset", "label", "shape_num")]
  # split string to return second to last part separated by _ or return whole
  # string if no _ the genus is always in the tip label, so we can make a
  # separate column for it
  label_data$genus <- sapply(
    strsplit(label_data$label, "_"),
    function(x) if (length(x) >= 2) x[length(x) - 1] else x
  )
  label_data <- label_data %>%
    mutate(shape = case_when(
      shape_num == 0 ~ "unlobed",
      shape_num == 1 ~ "lobed",
      shape_num == 2 ~ "dissected",
      shape_num == 3 ~ "compound",
    ))
  return(label_data)
}

map_higher_order_labels <- function(labels) {
  # add higher order taxon classification to the tree labels using data from
  # Naturalis
  tax_map <- read.csv( # get taxon data for all unique genera in Naturalis
    "shape_data/Naturalis/Naturalis_unique_genera_11-02-25.csv"
  )
  tax_map <- tax_map[!duplicated(tax_map$genus), ] # remove duplicate genera
  labels <- left_join(labels, tax_map, by = "genus")
  return(labels)
}

get_treeness <- function(s) {
  # calculate treeness - sum of all internal branch lengths (e.g. branches
  # not leading to a tip) divided by the sum over all branch lengths
  for (d in s$dataset) {
    tree <- read.nexus(paste0("phylo_data/trees_final/", d))
    if (class(tree) == "phylo") {
      s[s$dataset == d, "treeness"] <- treeness(tree)
      s[s$dataset == d, "mean_branch_length"] <- mean_branch_length(tree)
      s[s$dataset == d, "var_branch_length"] <- var_branch_length(tree)
      s[s$dataset == d, "avg_leaf_depth"] <- average_leaf_depth(tree)
      s[s$dataset == d, "var_leaf_depth"] <- var_leaf_depth(tree)
      s[s$dataset == d, "pigot_rho"] <- pigot_rho(tree)
      s[s$dataset == d, "max_depth"] <- max(node.depth.edgelength(tree))
    } else if (class(tree) == "multiPhylo") {
      tness_subtree_list <- c()
      mbl_list <- c()
      vbl_list <- c()
      ald_list <- c()
      vld_list <- c()
      pr_list <- c()
      md_list <- c()
      for (i in seq_along(tree)) { # Append subtree treeness
        tness_subtree_list <- c(tness_subtree_list, treeness(tree[[i]]))
        mbl_list <- c(mbl_list, mean_branch_length(tree[[i]]))
        vbl_list <- c(vbl_list, var_branch_length(tree[[i]]))
        ald_list <- c(ald_list, average_leaf_depth(tree[[i]]))
        vld_list <- c(vld_list, var_leaf_depth(tree[[i]]))
        pr_list <- c(pr_list, pigot_rho(tree[[i]]))
        md_list <- c(md_list, max(node.depth.edgelength(tree[[i]])))
      } # Calculate the mean treeness, ignoring NAs
      s[s$dataset == d, "treeness"] <- mean(tness_subtree_list)
      s[s$dataset == d, "mean_branch_length"] <- mean(mbl_list)
      s[s$dataset == d, "var_branch_length"] <- mean(vbl_list)
      s[s$dataset == d, "avg_leaf_depth"] <- mean(ald_list)
      s[s$dataset == d, "var_leaf_depth"] <- mean(vld_list)
      s[s$dataset == d, "pigot_rho"] <- mean(pr_list)
      s[s$dataset == d, "max_depth"] <- mean(md_list)
    }
  }
  return(s)
}

plot_trees <- function(summary) {
  layout(matrix(1:length(summary$dataset), ncol = 2, byrow = TRUE))
  par()
  for (dataset in summary$dataset) {
    tree <- read.nexus(paste0("phylo_data/trees_final/", dataset))
    if (class(tree) == "phylo") {
      plot(tree, type = "fan", main = dataset)
    } else if (class(tree) == "multiPhylo") {
      plot(tree[[1]], type = "fan", main = dataset)
    }
  }
}

plot_ggtrees <- function(summary) {
  tiplab_text_size <- 0.2 # set the size for tip labels and heatmap labels
  # set the vertical justification for tip labels and heatmap labels to
  # ensure alignmnet with tree tips
  tiplab_vjust <- 0.25
  dataset <- tree_files[tree_plot]
  label_data <- import_labels()
  label_data <- map_higher_order_labels(label_data)
  trees <- list() # Initialize an empty list to store trees
  heatmap_data <- c()

  data_name <- sub("^(.*class).*", "\\1", dataset)
  data_name <- sub("\\.tre$", "", data_name)
  tree_path <- paste0("phylo_data/trees_final/", dataset)
  tree <- read.nexus(tree_path)
  labels <- label_data[label_data$dataset == data_name, ]
  # Create a data frame with the tree tip labels in the tree order
  # Handle both phylo and multiPhylo objects for tip labels
  tip_labels <- if (inherits(tree, "multiPhylo")) {
    tree[[1]]$tip.label
  } else {
    tree$tip.label
  }
  label_data_ordered <- data.frame("label" = tip_labels)
  # Join with the labels data frame to get the order
  label_data_ordered <- left_join(label_data_ordered, labels, by = "label")
  # here we specify the taxonomic level to be used for the heatmap
  heatmap_data <- data.frame("order" = label_data_ordered$order)
  rownames(heatmap_data) <- label_data_ordered$label

  if (inherits(tree, "phylo")) {
    tree <- left_join(tree, labels, by = "label")
    trees[[dataset]] <- tree
  } else if (inherits(tree, "multiPhylo")) {
    tree <- left_join(tree[[1]], labels, by = "label")
    trees[[dataset]] <- tree # Select first tree if multiPhylo
  }

  class(trees) <- "multiPhylo"
  p <- ggtree(trees,
    layout = "circular",
    size = branch_width
  ) +
    aes(colour = shape) +
    facet_wrap(~.id, scale = "free", ncol = 4) +
    # theme_tree2() +
    # geom_tippoint(aes(colour=factor(order))) +
    scale_color_manual(values = c(
      "unlobed" = "#0173B2",
      "lobed" = "#DE8F05", "dissected" = "#029E73",
      "compound" = "#D55E00"
    ))

  if (!legend) {
    p <- p + theme(legend.position = "none")
  }
  if (scale_bar) { # width=0.1 for geeta, 100 for jan, zun
    if (grepl("geeta", data_name)) {
      p <- p + geom_treescale(x = 0, y = 0, width = 0.1, offset = 5)
    } else {
      p <- p + geom_treescale(x = 0, y = 0, width = 100, offset = 5)
    }
  }
  if (show_tip_labs) {
    p <- p + geom_tiplab(size = tiplab_text_size, vjust = tiplab_vjust)
  }
  # gives the tip labels in the order they appear in ggplot
  tips_plot_order <- rev(get_taxa_name(p))
  tips_ape_order <- rownames(heatmap_data)
  # find the index of each ape tip label in the plot
  idx_ape_in_plot <- match(tips_ape_order, tips_plot_order)
  # calculate the angle for each tip label based on the plot order, not
  # order in the ape phylo object
  ang <- ((360 * idx_ape_in_plot) / length(idx_ape_in_plot))
  ang[ang > 90 & ang < 270] <- ang[ang > 90 & ang < 270] - 180


  # extend the angle list to include NULL values for internal nodes
  total_nodes <- Ntip(trees[[1]]) + Nnode(trees[[1]])
  add_nodes <- total_nodes - length(ang)
  ang_ext <- c(ang, rep(list(NULL), add_nodes))

  if (show_order_heatmap) {
    p <- gheatmap(p, heatmap_data,
      offset = 0.05, width = 0.1, color = NULL,
      colnames = TRUE, hjust = 0.5, colnames_offset_x = 5
    ) +
      scale_fill_viridis_d(option = "D") +
      geom_text(aes(label = order, angle = ang_ext),
        color = "white", size = tiplab_text_size, nudge_x = 20,
        vjust = tiplab_vjust, hjust = 0.5
      ) +
      guides(fill = "none") # remove heatmap legend
  }
  ggsave(
    file = paste0(dataset, ".svg"), plot = p, width = 10, height = 10,
    dpi = 10000
  ) # text below a certain size will not be rendered with .pdf
  print(paste("Tree plot exported as", dataset, ".svg"))
  ggsave(
    file = paste0(dataset, ".pdf"), plot = p, width = 10, height = 10,
    dpi = 600
  )
  print(p)
}

get_shape_counts <- function(summary) {
  label_data <- import_labels()
  tree_names <- summary$dataset
  shape_freq_df <- data.frame()
  for (dataset in tree_names) {
    data_name <- sub("^(.*class).*", "\\1", dataset) # format data_name
    data_name <- sub("\\.tre$", "", data_name)
    labels <- label_data[label_data$dataset == data_name, ]
    shape_freq <- as.data.frame(table(labels$shape_num))
    shape_freq$dataset <- dataset
    shape_freq <- reshape(shape_freq,
      idvar = "dataset", timevar = "Var1",
      direction = "wide"
    )
    shape_freq_df <- bind_rows(shape_freq_df, shape_freq)
  }
  shape_freq_df[is.na(shape_freq_df)] <- 0
  shape_freq_df$n_tips <- rowSums(shape_freq_df[
    grep("^Freq", names(shape_freq_df), value = TRUE)
  ])
  colnames(shape_freq_df) <- c(
    "dataset", "u", "l", "d", "c", "ld", "lc",
    "ldc", "n_tips"
  )
  # Calculate proportions for each shape column (except dataset and n_tips)
  # shape_cols <- setdiff(colnames(shape_freq_df), c("dataset", "n_tips"))
  # for (col in shape_cols) {
  #   prop <- shape_freq_df[[col]] / shape_freq_df$n_tips
  #   # Create a new column with "count (xx%)" format
  #   shape_freq_df[[col]] <- sprintf(
  #     "%d (%.1f%%)", shape_freq_df[[col]], 100 * prop
  #   )
  # }
  summary <- merge(summary, shape_freq_df, by = "dataset")
}
summary <- import_trees()
summary <- get_treeness(summary)
summary <- get_shape_counts(summary)

write.csv(summary, "tree_statistics.csv", row.names = FALSE)
plot_ggtrees(summary)
