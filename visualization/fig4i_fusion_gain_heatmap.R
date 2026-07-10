#!/usr/bin/env Rscript
# =============================================================================
# Figure 4i – Prediction Probability Heatmap (Single‑Modality vs Fusion)
# Orders samples by true sex then fusion probability, showing sharper
# discrimination in the Transformer Fusion model compared to single modalities.
#
# Input files (expected in --data_dir):
#   X_cli_for_r.csv   – conventional morphometric features
#   X_rad_for_r.csv   – radiomic features
#   X_2d_for_r.csv    – 2D CNN features
#   X_3d_for_r.csv    – 3D CNN features
#   X_fusion_for_r.csv – fusion model features
#   y_for_r.csv       – labels (0 = Female, 1 = Male), column named "Label"
#
# Usage:
#   Rscript fig4i_fusion_gain_heatmap.R \
#       --data_dir /path/to/data \
#       --output_dir /path/to/output
#
# Dependencies: glmnet, pheatmap, readr, optparse
# Note: Font defaults to system sans‑serif.  If Arial is available,
#       it will be used for PDF output.
# =============================================================================

suppressPackageStartupMessages({
  library(glmnet)
  library(pheatmap)
  library(readr)
  library(optparse)
})

# 1. Command-line arguments ---------------------------------------------------
option_list <- list(
  make_option(c("--data_dir"), type = "character", default = NULL,
              help = "Directory containing input CSV files", metavar = "DIR"),
  make_option(c("--output_dir"), type = "character", default = "./fig4i_output",
              help = "Directory to save figures [default %default]", metavar = "DIR")
)

opt <- parse_args(OptionParser(option_list = option_list), positional_arguments = FALSE)

if (is.null(opt$data_dir)) {
  stop("--data_dir must be provided.")
}

if (!dir.exists(opt$output_dir)) dir.create(opt$output_dir, recursive = TRUE)

# 2. Helper: auto‑detect CSV encoding -----------------------------------------
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
  df <- tryCatch(
    readr::read_csv(file_path, show_col_types = FALSE),
    error = function(e) stop("Unable to read file: ", file_path)
  )
  message("Read ", file_path, " with default encoding")
  return(as.data.frame(df))
}

# 3. NC colour palette --------------------------------------------------------
nc_red  <- "#E64B35"
nc_blue <- "#4DBBD5"

# 4. Load data ----------------------------------------------------------------
data_dir <- opt$data_dir
required_files <- c("X_cli_for_r.csv", "X_rad_for_r.csv", "X_2d_for_r.csv",
                    "X_3d_for_r.csv", "X_fusion_for_r.csv", "y_for_r.csv")

for (fname in required_files) {
  if (!file.exists(file.path(data_dir, fname))) {
    stop("Missing file: ", file.path(data_dir, fname))
  }
}

X_cli <- as.matrix(read_csv_auto(file.path(data_dir, "X_cli_for_r.csv")))
X_rad <- as.matrix(read_csv_auto(file.path(data_dir, "X_rad_for_r.csv")))
X_2d  <- as.matrix(read_csv_auto(file.path(data_dir, "X_2d_for_r.csv")))
X_3d  <- as.matrix(read_csv_auto(file.path(data_dir, "X_3d_for_r.csv")))
X_fus <- as.matrix(read_csv_auto(file.path(data_dir, "X_fusion_for_r.csv")))
y_df  <- read_csv_auto(file.path(data_dir, "y_for_r.csv"))

if (!"Label" %in% names(y_df)) stop("'Label' column not found in y_for_r.csv")
y <- y_df$Label
if (!all(unique(y) %in% c(0, 1))) stop("Label must contain only 0 and 1")

# Validate sample sizes
n_samples <- length(y)
if (nrow(X_cli) != n_samples || nrow(X_rad) != n_samples || nrow(X_2d) != n_samples ||
    nrow(X_3d) != n_samples || nrow(X_fus) != n_samples) {
  stop("Mismatch in number of samples across feature files and labels.")
}

message("Data loaded: ", n_samples, " samples")
message("Conventional: ", ncol(X_cli), " features")
message("Radiomics:    ", ncol(X_rad), " features")
message("2D CNN:       ", ncol(X_2d), " features")
message("3D CNN:       ", ncol(X_3d), " features")
message("Fusion:       ", ncol(X_fus), " features")

# 5. Standardise features -----------------------------------------------------
standardize <- function(X) {
  means <- colMeans(X, na.rm = TRUE)
  sds   <- apply(X, 2, sd, na.rm = TRUE)
  sds[sds == 0] <- 1
  scale(X, center = means, scale = sds)
}

X_cli_std <- standardize(X_cli)
X_rad_std <- standardize(X_rad)
X_2d_std  <- standardize(X_2d)
X_3d_std  <- standardize(X_3d)
X_fus_std <- standardize(X_fus)

# 6. L1‑regularised logistic regression (glmnet) ------------------------------
set.seed(42)
cv_cli <- cv.glmnet(X_cli_std, y, family = "binomial", alpha = 1)
cv_rad <- cv.glmnet(X_rad_std, y, family = "binomial", alpha = 1)
cv_2d  <- cv.glmnet(X_2d_std,  y, family = "binomial", alpha = 1)
cv_3d  <- cv.glmnet(X_3d_std,  y, family = "binomial", alpha = 1)
cv_fus <- cv.glmnet(X_fus_std, y, family = "binomial", alpha = 1)

prob_cli <- predict(cv_cli, newx = X_cli_std, s = "lambda.min", type = "response")[, 1]
prob_rad <- predict(cv_rad, newx = X_rad_std, s = "lambda.min", type = "response")[, 1]
prob_2d  <- predict(cv_2d,  newx = X_2d_std,  s = "lambda.min", type = "response")[, 1]
prob_3d  <- predict(cv_3d,  newx = X_3d_std,  s = "lambda.min", type = "response")[, 1]
prob_fus <- predict(cv_fus, newx = X_fus_std, s = "lambda.min", type = "response")[, 1]

# 7. Build probability matrix and order rows ----------------------------------
df_prob <- data.frame(
  Sample       = paste0("S", seq_len(n_samples)),
  True_Label   = y,
  Conventional = prob_cli,
  Radiomics    = prob_rad,
  `2D CNN`     = prob_2d,
  `3D CNN`     = prob_3d,
  Fusion       = prob_fus,
  check.names  = FALSE
)

# Order: first by true sex (female -> male), then by fusion probability (ascending)
df_prob <- df_prob[order(df_prob$True_Label, df_prob$Fusion), ]
heat_matrix <- as.matrix(df_prob[, c("Conventional", "Radiomics", "2D CNN", "3D CNN", "Fusion")])
rownames(heat_matrix) <- df_prob$Sample

# 8. Row annotation -----------------------------------------------------------
annotation_row <- data.frame(
  True_Sex = factor(df_prob$True_Label, levels = c(0, 1), labels = c("Female", "Male"))
)
rownames(annotation_row) <- df_prob$Sample

ann_colors <- list(True_Sex = c(Female = nc_red, Male = nc_blue))

# 9. Generate heatmap (without Cairo dependency) ------------------------------
# Use system default font; Arial will be used if available
p <- pheatmap(heat_matrix,
              cluster_rows = FALSE,
              cluster_cols = FALSE,
              scale = "none",
              color = colorRampPalette(c(nc_red, "white", nc_blue))(100),
              border_color = NA,
              annotation_row = annotation_row,
              annotation_colors = ann_colors,
              show_rownames = FALSE,
              fontsize = 12,
              fontsize_col = 14,
              angle_col = 0,
              main = "Prediction Probability: Single-Modality vs Fusion",
              legend = TRUE,
              annotation_legend = TRUE,
              annotation_names_row = FALSE)

# 10. Save as PDF and PNG -----------------------------------------------------
pdf_path <- file.path(opt$output_dir, "figure_fusion_gain_heatmap.pdf")
png_path <- file.path(opt$output_dir, "figure_fusion_gain_heatmap.png")

# Try to use Arial if available, else default sans
family_pdf <- if ("Arial" %in% postscriptFonts()) "Arial" else "sans"

pdf(pdf_path, width = 12, height = 8, family = family_pdf)
print(p)
dev.off()

png(png_path, width = 12, height = 8, units = "in", res = 300)
print(p)
dev.off()

message("Heatmap saved to: ", opt$output_dir)