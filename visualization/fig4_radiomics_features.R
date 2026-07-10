#!/usr/bin/env Rscript
# =============================================================================
# Figure 4 (panels a and b) – Radiomic Feature Visualisation
# Generates:
#   a) Spearman correlation heatmap of selected radiomic features
#   b) LASSO coefficient bar plot (positive → blue, negative → red)
#
# Input requirements:
#   --feature_csv: CSV with a 'Label' column (0 = Female, 1 = Male) and
#                  feature columns (numeric).
#   --coef_csv:    CSV with columns 'Feature' and 'Coefficient'.
#                  'Feature' names must exactly match the feature columns
#                  in the feature CSV.
#
# Usage:
#   Rscript fig4_radiomics_features.R \
#       --feature_csv <features.csv> \
#       --coef_csv <coefficients.csv> \
#       --output_dir <output_directory>
#
# Dependencies: ggplot2, dplyr, tidyr, readr, optparse
# Note: Default font is system sans‑serif. Change base_family in theme_nc()
#       to "Arial" if available and preferred.
# =============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(optparse)
})

# 1. Command-line arguments ---------------------------------------------------
option_list <- list(
  make_option(c("--feature_csv"), type = "character", default = NULL,
              help = "Path to radiomic features CSV (must contain 'Label' column)", metavar = "FILE"),
  make_option(c("--coef_csv"), type = "character", default = NULL,
              help = "Path to LASSO coefficients CSV (columns: Feature, Coefficient)", metavar = "FILE"),
  make_option(c("--output_dir"), type = "character", default = "./fig4_output",
              help = "Directory to save figures [default %default]", metavar = "DIR")
)

opt <- parse_args(OptionParser(option_list = option_list), positional_arguments = FALSE)

if (is.null(opt$feature_csv) || is.null(opt$coef_csv)) {
  stop("Both --feature_csv and --coef_csv must be provided.")
}

if (!dir.exists(opt$output_dir)) dir.create(opt$output_dir, recursive = TRUE)

# 2. Helper functions ---------------------------------------------------------
# Auto‑detect CSV encoding (robust fallback)
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
      return(df)
    }
  }
  # Final fallback: default UTF-8 with explicit error capture
  tryCatch({
    df <- readr::read_csv(file_path, show_col_types = FALSE)
    message("Read ", file_path, " with default encoding")
    return(df)
  }, error = function(e) {
    stop("Unable to read file: ", file_path, 
         "\nAttempted encodings: ", paste(encodings, collapse = ", "),
         "\nOriginal error: ", e$message)
  })
}

# 3. Global style ------------------------------------------------------------
nc_red   <- "#E64B35"
nc_blue  <- "#4DBBD5"

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
      legend.text        = element_text(size = base_size),
      legend.title       = element_text(size = base_size, face = "bold"),
      panel.background   = element_rect(fill = "white"),
      plot.background    = element_rect(fill = "white"),
      panel.grid         = element_blank()
    )
}

save_plot <- function(plot, filename, width, height, dpi = 300) {
  ggsave(file.path(opt$output_dir, paste0(filename, ".png")),
         plot = plot, width = width, height = height, dpi = dpi, bg = "white")
  ggsave(file.path(opt$output_dir, paste0(filename, ".pdf")),
         plot = plot, width = width, height = height, device = "pdf", bg = "white")
}

# 4. Data loading and validation ----------------------------------------------
df_feat <- read_csv_auto(opt$feature_csv)
df_coef <- read_csv_auto(opt$coef_csv)

# Check required columns
stopifnot(
  "'Label' column missing in feature CSV" = "Label" %in% names(df_feat),
  "Feature column missing in coefficient CSV" = "Feature" %in% names(df_coef),
  "Coefficient column missing in coefficient CSV" = "Coefficient" %in% names(df_coef)
)

# Validate Label values
if (!all(unique(df_feat$Label) %in% c(0, 1))) {
  stop("'Label' column must contain only 0 and 1. Found: ", paste(unique(df_feat$Label), collapse = ", "))
}

# Identify feature columns and convert to numeric
feature_cols <- setdiff(names(df_feat), "Label")
df_feat <- df_feat %>%
  mutate(across(all_of(feature_cols), as.numeric))

# Check for non‑numeric columns that became entirely NA
na_cols <- feature_cols[sapply(df_feat[feature_cols], function(x) all(is.na(x)))]
if (length(na_cols) > 0) {
  warning("The following feature columns could not be converted to numeric and contain only NAs: ",
          paste(na_cols, collapse = ", "))
}

n_features <- length(feature_cols)
message("Number of feature columns: ", n_features)

# Verify that all LASSO‑selected features exist in the feature table
missing_features <- setdiff(df_coef$Feature, feature_cols)
if (length(missing_features) > 0) {
  stop("The following features in the coefficient table are missing from the feature CSV:\n",
       paste(missing_features, collapse = ", "))
}

# Create sex factor
df_feat$Sex <- factor(df_feat$Label, levels = c(0, 1), labels = c("Female", "Male"))

# Order coefficients by absolute value (descending)
df_coef <- df_coef %>%
  arrange(desc(abs(Coefficient)))
features_ordered <- df_coef$Feature

# 5. Figure 4a: Spearman correlation heatmap ---------------------------------
X <- df_feat[, feature_cols, drop = FALSE]

# Convert to matrix (numeric guaranteed by earlier conversion)
cor_matrix <- cor(as.matrix(X), method = "spearman")

# Convert to long format without additional packages
cor_melt <- as.data.frame(as.table(cor_matrix))
names(cor_melt) <- c("Feature1", "Feature2", "Correlation")

p1 <- ggplot(cor_melt, aes(x = Feature1, y = Feature2, fill = Correlation)) +
  geom_tile(color = "white", linewidth = 0.3) +
  scale_fill_gradient2(low = nc_blue, mid = "white", high = nc_red,
                       midpoint = 0, limit = c(-1, 1), name = "Spearman r") +
  labs(title = paste0("Spearman Correlation of ", n_features, " Selected Radiomics Features")) +
  theme_nc(base_size = 12) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 9),
        axis.text.y = element_text(size = 9),
        legend.title = element_text(face = "bold"),
        plot.title   = element_text(size = 14))

heatmap_width  <- max(8, n_features * 0.5)
heatmap_height <- max(6, n_features * 0.5)
save_plot(p1, "figure_radiomics_correlation", width = heatmap_width, height = heatmap_height)

# 6. Figure 4b: LASSO coefficient bar plot -----------------------------------
df_coef_plot <- df_coef
df_coef_plot$Feature <- factor(df_coef_plot$Feature, levels = rev(df_coef_plot$Feature))
# Create a sign factor for automatic legend (more robust than manual coordinates)
df_coef_plot$Sign <- ifelse(df_coef_plot$Coefficient > 0,
                            "Positive (Male)",
                            "Negative (Female)")

x_range <- diff(range(df_coef_plot$Coefficient))
label_offset <- x_range * 0.03

p2 <- ggplot(df_coef_plot, aes(x = Coefficient, y = Feature, fill = Sign)) +
  geom_col(width = 0.7, alpha = 0.9) +
  geom_vline(xintercept = 0, linewidth = 0.8, color = "black") +
  geom_text(aes(label = sprintf("%.3f", Coefficient),
                x = Coefficient + ifelse(Coefficient > 0, label_offset, -label_offset)),
            size = 3.5, color = "black") +
  scale_fill_manual(values = c("Positive (Male)" = nc_blue,
                               "Negative (Female)" = nc_red),
                    name = NULL) +
  labs(title = "LASSO Coefficients of Selected Radiomics Features",
       x = "Coefficient", y = NULL) +
  theme_nc(base_size = 14) +
  theme(axis.text.y = element_text(size = 11),
        legend.position = "right")    # automatic, avoids overlap

bar_height <- max(6, n_features * 0.4)
save_plot(p2, "figure_radiomics_lasso_coef", width = 10, height = bar_height)

message("All radiomics figures saved to: ", opt$output_dir)