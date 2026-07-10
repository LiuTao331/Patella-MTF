#!/usr/bin/env Rscript
# =============================================================================
# Figure 4c-d – CNN Feature Importance (Top 10 Principal Components)
# Generates two bar plots showing the Gini importance of the top 10 principal
# components for the 2D CNN and 3D CNN features, as described in the paper.
#
# Input files:
#   X_2d_for_r.csv : 2D CNN PCA features (numeric matrix, rows = samples)
#   X_3d_for_r.csv : 3D CNN PCA features (numeric matrix)
#   y_for_r.csv    : labels, must contain a column named "Label" (0=Female,1=Male)
#   The feature matrices should have columns ordered as PC1, PC2, ...
#   (column names, if present, will be used as labels; otherwise "PC1", "PC2" ...)
#
# Usage:
#   Rscript fig4_cnn_importance.R \
#       --x2d_csv <X_2d_for_r.csv> \
#       --x3d_csv <X_3d_for_r.csv> \
#       --y_csv   <y_for_r.csv> \
#       --output_dir <output_directory> \
#       [--ntree 500]
#
# Dependencies: ggplot2, dplyr, randomForest, readr, optparse
# Note: Default font is system sans‑serif. Change base_family in theme_nc()
#       to "Arial" if available and preferred.
# =============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(randomForest)
  library(readr)
  library(optparse)
})

# 1. Command-line arguments ---------------------------------------------------
option_list <- list(
  make_option(c("--x2d_csv"), type = "character", default = NULL,
              help = "Path to 2D CNN PCA features CSV", metavar = "FILE"),
  make_option(c("--x3d_csv"), type = "character", default = NULL,
              help = "Path to 3D CNN PCA features CSV", metavar = "FILE"),
  make_option(c("--y_csv"), type = "character", default = NULL,
              help = "Path to labels CSV (must contain a 'Label' column)", metavar = "FILE"),
  make_option(c("--output_dir"), type = "character", default = "./fig4_output",
              help = "Directory to save figures [default %default]", metavar = "DIR"),
  make_option(c("--ntree"), type = "integer", default = 500,
              help = "Number of trees for random forest [default %default]", metavar = "N")
)

opt <- parse_args(OptionParser(option_list = option_list), positional_arguments = FALSE)

# Validate required arguments
required_args <- c("x2d_csv", "x3d_csv", "y_csv")
missing_args <- required_args[!required_args %in% names(opt) | sapply(opt[required_args], is.null)]
if (length(missing_args) > 0) {
  stop("Missing required arguments: ", paste(missing_args, collapse = ", "))
}

if (!dir.exists(opt$output_dir)) dir.create(opt$output_dir, recursive = TRUE)

# 2. Helper function: auto‑detect CSV encoding --------------------------------
read_csv_auto <- function(file_path) {
  encodings <- c("UTF-8", "GBK", "GB2312", "latin1")
  for (enc in encodings) {
    df <- tryCatch(
      readr::read_csv(file_path, locale = readr::locale(encoding = enc), show_col_types = FALSE),
      error = function(e) NULL,
      warning = function(w) NULL
    )
    if (!is.null(df) && nrow(df) > 0) {
      message("Successfully read ", file_path, " with encoding ", enc)
      return(as.data.frame(df))
    }
  }
  # Fallback to default UTF-8
  df <- tryCatch(
    readr::read_csv(file_path, show_col_types = FALSE),
    error = function(e) stop("Unable to read file: ", file_path,
                             "\nAttempted encodings: ", paste(encodings, collapse = ", "),
                             "\nOriginal error: ", e$message)
  )
  message("Read ", file_path, " with default encoding")
  return(as.data.frame(df))
}

# 3. Global style ------------------------------------------------------------
nc_red   <- "#E64B35"
nc_blue  <- "#4DBBD5"
nc_green <- "#00A087"
nc_dark  <- "#3C5488"

theme_nc <- function(base_size = 14, base_family = "") {
  theme_classic(base_size = base_size, base_family = base_family) +
    theme(
      axis.line.x.bottom = element_line(linewidth = 1.0, color = "black"),
      axis.line.y.left   = element_line(linewidth = 1.0, color = "black"),
      axis.line.x.top    = element_blank(),
      axis.line.y.right  = element_blank(),
      axis.ticks.length  = unit(0.1, "cm"),
      axis.ticks         = element_line(linewidth = 1.0, color = "black"),
      axis.text          = element_text(size = base_size, color = "black"),
      axis.title         = element_text(size = base_size, face = "bold"),
      plot.title         = element_text(size = base_size + 2, face = "bold", hjust = 0.5),
      legend.position    = "none",
      panel.background   = element_rect(fill = "white"),
      plot.background    = element_rect(fill = "white"),
      panel.grid         = element_blank()
    )
}

save_plot <- function(plot, filename, width = 8, height = 6, dpi = 300) {
  ggsave(file.path(opt$output_dir, paste0(filename, ".png")),
         plot = plot, width = width, height = height, dpi = dpi, bg = "white")
  ggsave(file.path(opt$output_dir, paste0(filename, ".pdf")),
         plot = plot, width = width, height = height, device = "pdf", bg = "white")
}

# 4. Load data and validate ---------------------------------------------------
X_2d <- as.matrix(read_csv_auto(opt$x2d_csv))
X_3d <- as.matrix(read_csv_auto(opt$x3d_csv))
y_df <- read_csv_auto(opt$y_csv)

# Check for Label column
if (!"Label" %in% names(y_df)) {
  stop("'Label' column not found in ", opt$y_csv)
}

y <- y_df$Label
if (!all(unique(y) %in% c(0, 1))) {
  stop("'Label' column must contain only 0 and 1. Found: ", paste(unique(y), collapse = ", "))
}

# Row count alignment – critical for valid correspondence
stopifnot("Number of rows in X_2d does not match labels" = nrow(X_2d) == nrow(y_df))
stopifnot("Number of rows in X_3d does not match labels" = nrow(X_3d) == nrow(y_df))

# Ensure numeric matrices
if (!is.numeric(X_2d) || !is.numeric(X_3d)) {
  stop("Feature matrices must be numeric. Please check the input files.")
}

y_factor <- factor(y, levels = c(0, 1), labels = c("Female", "Male"))

message("Data dimensions:")
message("  2D CNN: ", nrow(X_2d), " x ", ncol(X_2d))
message("  3D CNN: ", nrow(X_3d), " x ", ncol(X_3d))
message("  Labels: ", length(y_factor))

# 5. Random forest importance (with improved column naming) -------------------
get_top_importance <- function(X, y_factor, ntree = 500, top_n = 10) {
  set.seed(42)
  rf_model <- randomForest(x = X, y = y_factor, ntree = ntree, importance = TRUE)
  imp <- importance(rf_model, type = 2)  # Mean Decrease Gini
  imp_vec <- imp[, 1]
  
  idx_sorted <- order(imp_vec, decreasing = TRUE)[1:top_n]
  top_imps <- imp_vec[idx_sorted]
  
  # Use column names if available, otherwise create "PC1","PC2",...
  if (!is.null(colnames(X))) {
    pc_names <- colnames(X)[idx_sorted]
  } else {
    pc_names <- paste0("PC", idx_sorted)
  }
  
  # Ensure unique factor levels (in case of duplicate names)
  data.frame(Feature = factor(pc_names, levels = rev(pc_names)), 
             Importance = top_imps)
}

df_2d <- get_top_importance(X_2d, y_factor, ntree = opt$ntree, top_n = 10)
df_3d <- get_top_importance(X_3d, y_factor, ntree = opt$ntree, top_n = 10)

# 6. Plotting functions ------------------------------------------------------
plot_importance <- function(df, color, title_text, x_max) {
  ggplot(df, aes(x = Importance, y = Feature)) +
    geom_bar(stat = "identity", fill = color, width = 0.7, alpha = 0.85) +
    geom_text(aes(label = sprintf("%.3f", Importance)), 
              hjust = -0.1, size = 4, color = "gray20") +
    labs(title = title_text, x = "Gini Importance", y = NULL) +
    scale_x_continuous(limits = c(0, x_max * 1.12), expand = c(0, 0)) +
    theme_nc(base_size = 14)
}

max_imp <- max(c(df_2d$Importance, df_3d$Importance))

p_2d <- plot_importance(df_2d, nc_blue, "2D CNN - Top 10 Important PCs", max_imp)
p_3d <- plot_importance(df_3d, nc_green, "3D CNN - Top 10 Important PCs", max_imp)

# 7. Save figures ------------------------------------------------------------
save_plot(p_2d, "figure_2dcnn_importance_top10", width = 8, height = 6)
save_plot(p_3d, "figure_3dcnn_importance_top10", width = 8, height = 6)

message("CNN importance figures saved to: ", opt$output_dir)