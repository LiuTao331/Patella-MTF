#!/usr/bin/env Rscript
# =====================================================================
# Fig2 – Patella CT Cohort Demographics and Statistics
# Implements the data distribution figures (age histograms, violin plots,
# region bar charts) and descriptive statistics reported in the paper.
#
# Usage:
#   Rscript fig2_data_distribution.R \
#       --train_path <train.csv> \
#       --test_path <test.csv> \
#       --integrated_path <integrated.xlsx> \
#       --external_path <external.csv> \
#       --output_dir <output_directory>
#
# Dependencies: ggplot2, dplyr, readr, readxl, patchwork, forcats, optparse
# Note: The default font is set to the system sans‑serif font. If you need
# Arial, install the ttf-mscorefonts-installer package (Linux) or equivalent,
# and change base_family = "Arial" in theme_nc().
# =====================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(readr)
  library(readxl)
  library(patchwork)
  library(forcats)
  library(optparse)
})

# 1. Command-line options ----------------------------------------------------
option_list <- list(
  make_option(c("--train_path"), type = "character", default = NULL,
              help = "Path to training set CSV (columns: number, sex, age, zone)", metavar = "FILE"),
  make_option(c("--test_path"), type = "character", default = NULL,
              help = "Path to internal test set CSV (same columns)", metavar = "FILE"),
  make_option(c("--integrated_path"), type = "character", default = NULL,
              help = "Path to integrated region Excel file (with zone column)", metavar = "FILE"),
  make_option(c("--external_path"), type = "character", default = NULL,
              help = "Path to external cohort CSV (columns: number, sex, age, zone)", metavar = "FILE"),
  make_option(c("--output_dir"), type = "character", default = "./fig2_output",
              help = "Directory to save figures and statistics [default %default]", metavar = "DIR")
)

opt <- parse_args(OptionParser(option_list = option_list), positional_arguments = FALSE)

# Validate required inputs
required_args <- c("train_path", "test_path", "integrated_path", "external_path")
missing_args <- required_args[!required_args %in% names(opt) | sapply(opt[required_args], is.null)]
if (length(missing_args) > 0) {
  stop("Missing required arguments: ", paste(missing_args, collapse = ", "),
       call. = FALSE)
}

# Create output directory
if (!dir.exists(opt$output_dir)) dir.create(opt$output_dir, recursive = TRUE)

# 2. Theme and colour palettes -----------------------------------------------
# Use the system default sans font for portability (change to "Arial" if available)
theme_nc <- function(base_size = 14, base_family = "") {
  theme_classic(base_size = base_size, base_family = base_family) %+replace%
    theme(
      axis.line         = element_line(colour = "black", linewidth = 1.0),
      axis.ticks        = element_line(colour = "black", linewidth = 1.0),
      axis.ticks.length = unit(0.1, "cm"),
      legend.position   = "right",
      legend.title      = element_text(size = base_size, face = "bold"),
      legend.text       = element_text(size = base_size),
      plot.title        = element_text(size = base_size + 1, face = "bold", hjust = 0.5,
                                       margin = margin(b = 8)),
      plot.subtitle     = element_text(size = base_size, hjust = 0.5),
      axis.title        = element_text(size = base_size, face = "bold"),
      axis.text         = element_text(size = base_size, colour = "black"),
      strip.text        = element_text(size = base_size, face = "bold",
                                       margin = margin(t = 4, b = 4)),
      strip.background  = element_blank()
    )
}

sex_colors <- c("Female" = "#E64B35FF", "Male" = "#4DBBD5FF")

npg_palette <- c(
  "#E64B35FF", "#4DBBD5FF", "#00A087FF", "#3C5488FF",
  "#F39B7FFF", "#8491B4FF", "#91D1C2FF", "#DC0000FF",
  "#7E6148FF", "#B09C85FF"
)

# 3. Data loading and column checks ------------------------------------------
train_df <- read_csv(opt$train_path, show_col_types = FALSE) %>%
  mutate(dataset = "Train")
stopifnot(all(c("sex", "age") %in% colnames(train_df)))

test_df <- read_csv(opt$test_path, show_col_types = FALSE) %>%
  mutate(dataset = "Test")
stopifnot(all(c("sex", "age") %in% colnames(test_df)))

all_samples <- bind_rows(train_df, test_df) %>%
  mutate(
    sex = factor(sex, levels = c(0, 1), labels = c("Female", "Male")),
    age = as.numeric(age)
  )

train_only <- all_samples %>% filter(dataset == "Train")
test_only  <- all_samples %>% filter(dataset == "Test")

integrated_df <- read_excel(opt$integrated_path) %>%
  mutate(
    sex  = factor(sex, levels = c(0, 1), labels = c("Female", "Male")),
    age  = as.numeric(age),
    zone = as.character(zone),
    # Data correction: zone 632133 is a typo; the correct code is 632123 (Haidong)
    zone = if_else(zone == "632133", "632123", zone)
  )
stopifnot("zone" %in% colnames(integrated_df))

external_df <- read_csv(opt$external_path, show_col_types = FALSE) %>%
  mutate(
    sex     = factor(sex, levels = c(0, 1), labels = c("Female", "Male")),
    age     = as.numeric(age),
    zone    = as.character(zone),
    dataset = "External"
  )
stopifnot(all(c("zone") %in% colnames(external_df)))

# 4. Region mapping (Qinghai Province) --------------------------------------
qh_region_map <- c(
  "630101" = "Xining", "630102" = "Xining", "630103" = "Xining",
  "630104" = "Xining", "630105" = "Xining", "630120" = "Xining",
  "630121" = "Xining", "630122" = "Xining", "630123" = "Xining",
  "630100" = "Xining",
  "632121" = "Haidong", "632122" = "Haidong", "632123" = "Haidong",
  "632124" = "Haidong", "632125" = "Haidong", "632126" = "Haidong",
  "632127" = "Haidong", "632128" = "Haidong",
  "620422" = "Haidong",
  "632221" = "Haibei", "632222" = "Haibei", "632223" = "Haibei",
  "632224" = "Haibei", "632225" = "Haibei",
  "632321" = "Huangnan", "632322" = "Huangnan", "632323" = "Huangnan",
  "632324" = "Huangnan", "632325" = "Huangnan", "632421" = "Huangnan",
  "632521" = "Hainan", "632522" = "Hainan", "632523" = "Hainan",
  "632524" = "Hainan", "632525" = "Hainan",
  "632621" = "Golog", "632622" = "Golog", "632623" = "Golog",
  "632624" = "Golog", "632625" = "Golog", "632626" = "Golog",
  "632721" = "Yushu", "632722" = "Yushu", "632723" = "Yushu",
  "632724" = "Yushu", "632725" = "Yushu", "632726" = "Yushu",
  "632801" = "Haixi", "632802" = "Haixi", "632821" = "Haixi",
  "632822" = "Haixi", "632823" = "Haixi", "632824" = "Haixi",
  "632825" = "Haixi", "632826" = "Haixi"
)

integrated_df <- integrated_df %>%
  mutate(region = recode(zone, !!!qh_region_map, .default = "Other"))

external_df <- external_df %>%
  mutate(region = recode(zone, !!!qh_region_map, .default = "Other"))

# Check for unmapped zones
other_internal <- integrated_df %>% filter(region == "Other") %>% distinct(zone) %>% pull(zone)
if (length(other_internal) > 0) {
  message("Internal dataset - zones classified as 'Other': ", paste(other_internal, collapse = ", "))
} else {
  message("Internal dataset: all zones mapped successfully.")
}

other_external <- external_df %>% filter(region == "Other") %>% distinct(zone) %>% pull(zone)
if (length(other_external) > 0) {
  message("External dataset - zones classified as 'Other': ", paste(other_external, collapse = ", "))
} else {
  message("External dataset: all zones mapped successfully.")
}

# Unified region colour map
all_regions <- union(unique(integrated_df$region), unique(external_df$region))
region_color_map <- setNames(
  rep(npg_palette, length.out = length(all_regions)),
  all_regions
)

# 5. Plotting functions ------------------------------------------------------
plot_age_hist <- function(data, title = "Age Distribution") {
  data_with_range <- data %>%
    group_by(sex) %>%
    mutate(sex_label = paste0(sex, " (", min(age, na.rm = TRUE), "-",
                              max(age, na.rm = TRUE), " yrs)")) %>%
    ungroup()
  
  p <- ggplot(data_with_range, aes(x = age, fill = sex)) +
    geom_histogram(binwidth = 5, color = "black", linewidth = 0.3, alpha = 0.8) +
    geom_text(stat = "bin", binwidth = 5, aes(label = after_stat(count)),
              vjust = -0.5, size = 4, family = "") +
    scale_fill_manual(values = sex_colors) +
    facet_wrap(~ sex_label, ncol = 2, scales = "free_y") +
    labs(x = "Age (years)", y = "Count", title = title) +
    theme_nc() +
    theme(legend.position = "none")
  
  y_max <- max(ggplot_build(p)$data[[1]]$count, na.rm = TRUE) * 1.15
  p + scale_y_continuous(expand = expansion(mult = c(0, 0.05)), limits = c(0, y_max))
}

plot_violin <- function(data, title = "Training set") {
  ggplot(data, aes(x = sex, y = age, fill = sex)) +
    geom_violin(trim = FALSE, alpha = 0.7, color = "black", linewidth = 0.3) +
    geom_boxplot(width = 0.12, fill = "white", outlier.shape = NA,
                 color = "black", linewidth = 0.3) +
    scale_fill_manual(values = sex_colors) +
    labs(x = "Sex", y = "Age (years)", title = title) +
    theme_nc() +
    theme(legend.position = "none")
}

plot_region_bar <- function(data, title = "Region Distribution", color_map = region_color_map) {
  region_counts <- data %>%
    count(region, sort = TRUE) %>%
    mutate(region = fct_reorder(region, n))
  
  ggplot(region_counts, aes(x = n, y = region, fill = region)) +
    geom_col(color = "black", linewidth = 0.3, width = 0.7) +
    geom_text(aes(label = n), hjust = -0.2, size = 4.5, family = "") +
    scale_fill_manual(values = color_map) +
    labs(x = "Count", y = NULL, title = title) +
    theme_nc() +
    theme(legend.position = "none", axis.text.y = element_text(size = 14)) +
    scale_x_continuous(expand = expansion(mult = c(0, 0.15)))
}

# 6. Generate and save figures -----------------------------------------------
message("Generating figures...")

p1 <- plot_age_hist(all_samples, "Age Distribution-Internal")
p2 <- plot_violin(train_only, "Training set")
p3 <- plot_violin(test_only, "Internal test set")
p4 <- plot_region_bar(integrated_df, "Region Distribution-Internal")

p5 <- plot_age_hist(external_df, "Age Distribution-External")
p6 <- plot_violin(external_df, "External test set")
p7 <- plot_region_bar(external_df, "Region Distribution-External")

# Use the standard pdf device for maximum portability
save_plot <- function(plot, filename, width = 8, height = 6, dpi = 300) {
  ggsave(file.path(opt$output_dir, paste0(filename, ".png")),
         plot = plot, width = width, height = height, dpi = dpi, bg = "white")
  ggsave(file.path(opt$output_dir, paste0(filename, ".pdf")),
         plot = plot, width = width, height = height, device = "pdf", bg = "white")
}

save_plot(p1, "Fig1_Age_Histogram_Internal", width = 10, height = 5)
save_plot(p2, "Fig2_Train_Violin", width = 6, height = 6)
save_plot(p3, "Fig3_Internal_Test_Violin", width = 6, height = 6)
save_plot(p4, "Fig4_Region_Distribution_Internal", width = 8, height = 6)

save_plot(p5, "Fig5_Age_Histogram_External", width = 10, height = 5)
save_plot(p6, "Fig6_External_Test_Violin", width = 6, height = 6)
save_plot(p7, "Fig7_Region_Distribution_External", width = 8, height = 6)

# Combined internal figure
combined <- (p1 | p2) / (p3 | p4) +
  plot_annotation(tag_levels = 'A') &
  theme(plot.tag = element_text(face = "bold", size = 16, family = ""))

ggsave(file.path(opt$output_dir, "Combined_Figure.png"),
       plot = combined, width = 14, height = 12, dpi = 300, bg = "white")
ggsave(file.path(opt$output_dir, "Combined_Figure.pdf"),
       plot = combined, width = 14, height = 12, device = "pdf", bg = "white")

message("All figures saved to: ", opt$output_dir)

# 7. Descriptive statistics and Wilcoxon tests -------------------------------
output_file <- file.path(opt$output_dir, "summary_statistics.txt")
sink(output_file, split = TRUE)

cat("=================================================================\n")
cat("Descriptive Statistics for Patella CT Dataset Figures\n")
cat("Generated on:", as.character(Sys.time()), "\n")
cat("=================================================================\n")

age_summary <- function(data, dataset_label) {
  ages <- data$age[!is.na(data$age)]
  cat("\n========================================\n")
  cat("Age Summary -", dataset_label, "\n")
  cat("========================================\n")
  cat(sprintf("Overall (n = %d): %.1f ± %.1f years, median = %.1f (IQR = %.1f), range = %.0f–%.0f\n",
              length(ages), mean(ages), sd(ages), median(ages), IQR(ages), min(ages), max(ages)))
  
  for (s in c("Female", "Male")) {
    sub <- data %>% filter(sex == s) %>% pull(age)
    sub <- sub[!is.na(sub)]
    cat(sprintf("%s (n = %d): %.1f ± %.1f years, median = %.1f (IQR = %.1f), range = %.0f–%.0f\n",
                s, length(sub), mean(sub), sd(sub), median(sub), IQR(sub), min(sub), max(sub)))
  }
}

violin_summary <- function(data, dataset_label) {
  cat("\n----------------------------------------\n")
  cat("Violin/Box equivalent -", dataset_label, "\n")
  cat("----------------------------------------\n")
  for (s in c("Female", "Male")) {
    sub <- data %>% filter(sex == s) %>% pull(age)
    sub <- sub[!is.na(sub)]
    if (length(sub) == 0) next
    cat(sprintf("%s (n = %d): median = %.1f, Q1 = %.1f, Q3 = %.1f, range = %.0f–%.0f\n",
                s, length(sub), median(sub), quantile(sub, 0.25), quantile(sub, 0.75),
                min(sub), max(sub)))
  }
}

region_summary <- function(data, dataset_label) {
  cat("\n========================================\n")
  cat("Region Distribution -", dataset_label, "\n")
  cat("========================================\n")
  total <- nrow(data)
  region_tab <- data %>%
    count(region, sort = TRUE) %>%
    mutate(
      pct = n / total * 100,
      label = sprintf("%s: %d (%.1f%%)", region, n, pct)
    )
  for (i in 1:nrow(region_tab)) {
    cat(region_tab$label[i], "\n")
  }
  cat(sprintf("Total regions = %d\n", nrow(region_tab)))
  other_zones <- data %>% filter(region == "Other") %>% distinct(zone) %>% pull(zone)
  if (length(other_zones) > 0) {
    cat("'Other' includes zones: ", paste(other_zones, collapse = ", "), "\n")
  } else {
    cat("All zones successfully mapped.\n")
  }
}

sex_distribution <- function(data, dataset_label) {
  cat("\n----------------------------------------\n")
  cat("Sex Distribution -", dataset_label, "\n")
  cat("----------------------------------------\n")
  sex_count <- data %>%
    count(sex) %>%
    mutate(pct = n / sum(n) * 100)
  for (i in 1:nrow(sex_count)) {
    cat(sprintf("%s: %d (%.1f%%)\n", sex_count$sex[i], sex_count$n[i], sex_count$pct[i]))
  }
}

wilcox_sex_age <- function(data, dataset_label) {
  data_clean <- data %>% filter(!is.na(age))
  if (nrow(data_clean) < 3) {
    cat("Not enough data for Wilcoxon test.\n")
    return()
  }
  test <- wilcox.test(age ~ sex, data = data_clean, exact = FALSE)
  cat(sprintf("Wilcoxon rank sum test (two-sided): W = %.1f, p = %.4f\n",
              test$statistic, test$p.value))
  if (test$p.value > 0.05) {
    cat("No significant age difference between sexes in the", dataset_label, ".\n")
  } else {
    cat("Significant age difference detected between sexes in the", dataset_label, ".\n")
  }
}

cat("\n>>> Fig1_Age_Histogram_Internal (all internal samples)\n")
age_summary(all_samples, "Internal Overall")
sex_distribution(all_samples, "Internal Overall")

cat("\n>>> Fig2_Train_Violin\n")
violin_summary(train_only, "Internal Training Set")
sex_distribution(train_only, "Internal Training Set")
wilcox_sex_age(train_only, "Internal Training Set")

cat("\n>>> Fig3_Internal_Test_Violin\n")
violin_summary(test_only, "Internal Test Set")
sex_distribution(test_only, "Internal Test Set")
wilcox_sex_age(test_only, "Internal Test Set")

cat("\n>>> Fig4_Region_Distribution_Internal\n")
region_summary(integrated_df, "Internal (Integrated)")

cat("\n>>> Fig5_Age_Histogram_External\n")
age_summary(external_df, "External Dataset")
sex_distribution(external_df, "External Dataset")

cat("\n>>> Fig6_External_Test_Violin\n")
violin_summary(external_df, "External Dataset")
wilcox_sex_age(external_df, "External Dataset")

cat("\n>>> Fig7_Region_Distribution_External\n")
region_summary(external_df, "External Dataset")

sink()

message("Statistical summary saved to: ", output_file)
message("All tasks completed successfully.")