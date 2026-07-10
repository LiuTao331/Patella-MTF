#!/usr/bin/env Rscript
# =============================================================================
# Fig3 – Patellar Morphometric Parameter Comparison by Sex
# This script generates boxplots, violin plots, density curves, and summary
# statistics comparing six automated patellar measurements between females
# and males, as described in the paper.
#
# Usage:
#   Rscript fig3_patellar_morphology.R \
#       --train_file <train.csv> \
#       --test_file <test.csv> \
#       --output_dir <output_directory> \
#       --sex_col <sex_column_name> \
#       --param_cols <col1> <col2> ... <col6>
#
# Dependencies: tidyverse, ggplot2, ggpubr, patchwork, optparse
# Note: The default font is the system sans‑serif font.  If you need Arial,
#       install the required font package and change base_family to "Arial".
# =============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(ggplot2)
  library(ggpubr)
  library(patchwork)
  library(optparse)
})

# 1. Command-line arguments ---------------------------------------------------
option_list <- list(
  make_option(c("--train_file"), type = "character", default = NULL,
              help = "Path to training set measurements CSV", metavar = "FILE"),
  make_option(c("--test_file"), type = "character", default = NULL,
              help = "Path to internal test set measurements CSV", metavar = "FILE"),
  make_option(c("--output_dir"), type = "character", default = "./fig3_output",
              help = "Directory to save figures and statistics [default %default]", metavar = "DIR"),
  make_option(c("--sex_col"), type = "character", default = "sex",
              help = "Name of the sex column [default %default]", metavar = "COL"),
  make_option(c("--param_cols"), type = "character", nargs = 6,
              default = c("Patellar length (mm)", "Patellar width (mm)", 
                          "Patellar thickness (mm)", "Patellar volume (mm³)",
                          "Patellar surface area (mm²)", "Patellar coronal perimeter (mm)"),
              help = "Six measurement column names (space‑separated)", metavar = "COL")
)

opt <- parse_args(OptionParser(option_list = option_list), positional_arguments = FALSE)

if (is.null(opt$train_file) || is.null(opt$test_file)) {
  stop("Both --train_file and --test_file must be provided.")
}

if (!dir.exists(opt$output_dir)) dir.create(opt$output_dir, recursive = TRUE)

# 2. Helper function: auto‑detect CSV encoding ---------------------------------
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
  # Fallback to default encoding
  df <- tryCatch(
    readr::read_csv(file_path, show_col_types = FALSE),
    error = function(e) stop("Unable to read file: ", file_path, " with any encoding.")
  )
  if (!is.null(df) && nrow(df) > 0) {
    message("Successfully read ", file_path, " with default encoding")
    return(df)
  } else {
    stop("Unable to read file: ", file_path)
  }
}

# 3. Load and prepare data ----------------------------------------------------
train_data <- read_csv_auto(opt$train_file) %>% mutate(dataset = "Train")
test_data  <- read_csv_auto(opt$test_file)  %>% mutate(dataset = "Test")
all_data   <- bind_rows(train_data, test_data)

# Check that required columns exist
required_cols <- c(opt$sex_col, opt$param_cols)
missing_cols <- required_cols[!required_cols %in% names(all_data)]
if (length(missing_cols) > 0) {
  stop("The following required columns are missing: ", paste(missing_cols, collapse = ", "),
       ". Available columns: ", paste(names(all_data), collapse = ", "))
}

# Convert sex to factor with standard labels
all_data <- all_data %>%
  mutate(sex_label = factor(.data[[opt$sex_col]], levels = c(0, 1), labels = c("Female", "Male")))

# Select only the needed columns and drop rows with missing measurements
analysis_data <- all_data %>%
  select(sex_label, dataset, all_of(opt$param_cols)) %>%
  drop_na()

# 4. Theme and colours --------------------------------------------------------
theme_nc <- function(base_size = 14, base_family = "") {
  theme_classic(base_size = base_size, base_family = base_family) %+replace%
    theme(
      axis.line         = element_line(colour = "black", linewidth = 1.0),
      axis.ticks        = element_line(colour = "black", linewidth = 1.0),
      axis.ticks.length = unit(0.1, "cm"),
      legend.position   = "right",
      legend.title      = element_text(size = base_size, face = "bold"),
      legend.text       = element_text(size = base_size),
      plot.title        = element_text(size = base_size + 1, face = "bold", hjust = 0.5),
      plot.subtitle     = element_text(size = base_size, hjust = 0.5),
      axis.title        = element_text(size = base_size, face = "bold"),
      axis.text         = element_text(size = base_size, colour = "black"),
      strip.text        = element_text(size = base_size, face = "bold"),
      strip.background  = element_blank()
    )
}

sex_colors <- c("Female" = "#E64B35FF", "Male" = "#4DBBD5FF")

# 5. Long‑format data ---------------------------------------------------------
long_data <- analysis_data %>%
  pivot_longer(cols = all_of(opt$param_cols), names_to = "parameter", values_to = "value")

# 6. Generate and save figures ------------------------------------------------
save_plot <- function(plot, filename, width, height, dpi = 300) {
  ggsave(file.path(opt$output_dir, paste0(filename, ".png")),
         plot = plot, width = width, height = height, dpi = dpi, bg = "white")
  ggsave(file.path(opt$output_dir, paste0(filename, ".pdf")),
         plot = plot, width = width, height = height, device = "pdf", bg = "white")
}

# 6.1 Faceted boxplot
p_box <- ggplot(long_data, aes(x = sex_label, y = value, fill = sex_label)) +
  geom_boxplot(alpha = 0.7, outlier.shape = NA) +
  geom_jitter(width = 0.2, alpha = 0.3, size = 0.8) +
  scale_fill_manual(values = sex_colors) +
  facet_wrap(~ parameter, scales = "free_y", ncol = 2) +
  labs(title = "Patellar Morphology by Sex", x = "Sex", y = "Value") +
  theme_nc() +
  theme(legend.position = "none")

save_plot(p_box, "FigS1_Boxplot_Facet", width = 14, height = 12)

# 6.2 Faceted violin + boxplot
p_violin <- ggplot(long_data, aes(x = sex_label, y = value, fill = sex_label)) +
  geom_violin(trim = FALSE, alpha = 0.6) +
  geom_boxplot(width = 0.15, fill = "white", outlier.size = 0.5) +
  scale_fill_manual(values = sex_colors) +
  facet_wrap(~ parameter, scales = "free_y", ncol = 2) +
  labs(title = "Patellar Morphology by Sex", x = "Sex", y = "Value") +
  theme_nc() +
  theme(legend.position = "none")

save_plot(p_violin, "FigS2_Violin_Facet", width = 14, height = 12)

# 6.3 Single‑parameter detailed comparison (boxplot + mean annotation + significance)
for (param in opt$param_cols) {
  df_sub <- analysis_data %>% select(sex_label, value = all_of(param))
  
  stats <- df_sub %>%
    group_by(sex_label) %>%
    summarise(
      mean_val = mean(value, na.rm = TRUE),
      sd_val   = sd(value, na.rm = TRUE),
      .groups  = "drop"
    )
  
  caption_text <- sprintf(
    "Female: %.2f ± %.2f\nMale: %.2f ± %.2f",
    stats$mean_val[stats$sex_label == "Female"],
    stats$sd_val[stats$sex_label == "Female"],
    stats$mean_val[stats$sex_label == "Male"],
    stats$sd_val[stats$sex_label == "Male"]
  )
  
  p <- ggplot(df_sub, aes(x = sex_label, y = value, fill = sex_label)) +
    geom_boxplot(alpha = 0.7, outlier.shape = NA) +
    geom_jitter(width = 0.2, alpha = 0.4, size = 1.5, color = "gray30") +
    scale_fill_manual(values = sex_colors) +
    labs(title = param, x = "Sex", y = param) +
    theme_nc() +
    theme(legend.position = "none") +
    stat_compare_means(
      aes(group = sex_label),
      comparisons = list(c("Female", "Male")),
      method = "wilcox.test",
      method.args = list(exact = FALSE),
      label = "p.signif",
      tip.length = 0.01,
      size = 5.5,
      family = ""
    ) +
    annotate("text", x = Inf, y = -Inf, label = caption_text,
             hjust = 1.1, vjust = -0.5, size = 4.5, family = "")
  
  safe_name <- gsub("[^A-Za-z]", "", param)
  save_plot(p, paste0("FigS3_", safe_name, "_Sex"), width = 7, height = 6)
}

# 6.4 Density curves
for (param in opt$param_cols) {
  p_dens <- ggplot(analysis_data, aes(x = .data[[param]], fill = sex_label)) +
    geom_density(alpha = 0.5) +
    scale_fill_manual(values = sex_colors) +
    labs(title = param, x = param, y = "Density") +
    theme_nc() +
    theme(legend.position = c(0.85, 0.85))
  
  safe_name <- gsub("[^A-Za-z]", "", param)
  save_plot(p_dens, paste0("FigS4_", safe_name, "_Density"), width = 7, height = 6)
}

# 6.5 Training vs test set comparison
long_data_split <- analysis_data %>%
  pivot_longer(cols = all_of(opt$param_cols), names_to = "parameter", values_to = "value")

p_split <- ggplot(long_data_split, aes(x = sex_label, y = value, fill = sex_label)) +
  geom_boxplot(alpha = 0.6) +
  scale_fill_manual(values = sex_colors) +
  facet_grid(dataset ~ parameter, scales = "free_y") +
  labs(title = "Training vs Test Sets: Sex Comparison", x = "Sex", y = "Value") +
  theme_nc(base_size = 12) +
  theme(legend.position = "none")

save_plot(p_split, "FigS5_TrainTest_Comparison", width = 16, height = 9)

# 7. Summary statistics table -------------------------------------------------
summary_table <- analysis_data %>%
  group_by(sex_label) %>%
  summarise(across(all_of(opt$param_cols), 
                   list(mean = ~ mean(.x, na.rm = TRUE),
                        sd   = ~ sd(.x, na.rm = TRUE)),
                   .names = "{.col}_{.fn}")) %>%
  pivot_longer(cols = -sex_label, 
               names_to = c("parameter", ".value"),
               names_pattern = "(.*)_(.*)") %>%
  mutate(mean_sd = sprintf("%.2f ± %.2f", mean, sd)) %>%
  select(sex_label, parameter, mean_sd) %>%
  pivot_wider(names_from = sex_label, values_from = mean_sd)

write.csv(summary_table, file = file.path(opt$output_dir, "TableS1_Summary_by_Sex.csv"), 
          row.names = FALSE, fileEncoding = "UTF-8")

message("All figures and statistics saved to: ", opt$output_dir)