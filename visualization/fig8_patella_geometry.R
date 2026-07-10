#!/usr/bin/env Rscript
# =============================================================================
# Fig8 – Patella Geometric Parameter Measurement and Visualisation
# This script reads a patellar STL mesh, computes six morphometric parameters
# (volume obtained from the closed mesh via vcgVolume), and generates
# orthogonal projection views, a coronal perimeter plot, a parameter table,
# and an optional 3D volume rendering.
#
# Note:
#   - Projection outlines are convex hulls for simplicity; the actual paper
#     uses CT reconstruction projections preserving anatomical details.
#   - The 3D volume rendering requires an interactive graphics device or
#     virtual framebuffer (e.g., Xvfb). Use --skip_3d to omit.
#   - The STL mesh should be a watertight, manifold surface. A basic check
#     is performed; if holes exist, results may be inaccurate.
#
# Usage:
#   Rscript fig8_patella_geometry.R \
#       --stl_file <path/to/mesh.stl> \
#       --output_dir <output_directory> \
#       [--skip_3d]
#
# Dependencies:
#   Rvcg (>= 0.20), rgl, ggplot2, dplyr, tidyr, gridExtra, grid, geometry,
#   optparse
# =============================================================================

suppressPackageStartupMessages({
  library(optparse)
  library(rgl)
  library(Rvcg)
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(gridExtra)
  library(grid)
  library(geometry)
})

# 1. Command-line arguments ---------------------------------------------------
option_list <- list(
  make_option(c("--stl_file"), type = "character", default = NULL,
              help = "Path to the patella STL mesh", metavar = "FILE"),
  make_option(c("--output_dir"), type = "character", default = "./fig8_output",
              help = "Directory to save figures and data [default %default]", metavar = "DIR"),
  make_option(c("--skip_3d"), action = "store_true", default = FALSE,
              help = "Skip the 3D volume rendering step")
)

opt <- parse_args(OptionParser(option_list = option_list), positional_arguments = FALSE)

if (is.null(opt$stl_file)) {
  stop("--stl_file must be provided.")
}

if (!dir.exists(opt$output_dir)) dir.create(opt$output_dir, recursive = TRUE)

# 2. Global style and font ----------------------------------------------------
# Use system sans-serif for portability; change to "Arial" if available
base_family <- "sans"

theme_nc <- function(base_size = 14, base_family = base_family) {
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
      strip.text        = element_text(size = base_size, face = "bold"),
      strip.background  = element_blank()
    )
}

# 3. Read and check STL mesh --------------------------------------------------
cat("Reading STL file:", opt$stl_file, "\n")
patella_mesh <- vcgImport(opt$stl_file)
cat(sprintf("Vertices: %d\n", ncol(patella_mesh$vb)))
cat(sprintf("Faces: %d\n", ncol(patella_mesh$it)))

# Basic watertight check
if (vcgIsWatertight(patella_mesh)) {
  cat("Mesh is watertight.\n")
} else {
  cat("WARNING: Mesh is not watertight – volume and surface area may be inaccurate.\n")
}

# 4. Compute morphometric parameters ------------------------------------------
calculate_patella_metrics <- function(mesh) {
  vertices <- t(mesh$vb[1:3, ])
  if (nrow(vertices) == 0) stop("Unable to extract vertices from mesh.")

  bbox_min <- apply(vertices, 2, min)
  bbox_max <- apply(vertices, 2, max)
  bbox <- rbind(bbox_min, bbox_max)
  colnames(bbox) <- c("x", "y", "z")

  length <- abs(bbox_max[3] - bbox_min[3])
  width  <- abs(bbox_max[1] - bbox_min[1])
  thickness <- abs(bbox_max[2] - bbox_min[2])
  center <- colMeans(vertices)

  # Surface area via triangle summation (fast, exact for manifold mesh)
  if (!is.null(mesh$it)) {
    triangles <- t(mesh$it)
    triangle_areas <- numeric(nrow(triangles))
    for (i in 1:nrow(triangles)) {
      v1 <- vertices[triangles[i, 1], ]
      v2 <- vertices[triangles[i, 2], ]
      v3 <- vertices[triangles[i, 3], ]
      edge1 <- v2 - v1
      edge2 <- v3 - v1
      cross_prod <- c(
        edge1[2] * edge2[3] - edge1[3] * edge2[2],
        edge1[3] * edge2[1] - edge1[1] * edge2[3],
        edge1[1] * edge2[2] - edge1[2] * edge2[1]
      )
      triangle_areas[i] <- 0.5 * sqrt(sum(cross_prod^2))
    }
    total_surface_area <- sum(triangle_areas)

    # Coronal perimeter via convex hull of XY projection
    # (Note: convex hull simplifies the actual outline)
    front_proj <- vertices[, c(1, 2)]
    hull_idx <- chull(front_proj)
    hull_pts <- front_proj[hull_idx, ]
    anterior_area <- abs(polyarea(hull_pts[, 1], hull_pts[, 2]))

    perimeter <- 0
    n_hull <- length(hull_idx)
    for (i in 1:n_hull) {
      j <- ifelse(i == n_hull, 1, i + 1)
      perimeter <- perimeter + sqrt(sum((hull_pts[i, ] - hull_pts[j, ])^2))
    }
  } else {
    stop("Mesh has no faces – cannot compute surface area or perimeter.")
  }

  # Exact volume using vcgVolume (requires watertight mesh)
  volume_exact <- vcgVolume(mesh)
  if (is.na(volume_exact)) {
    cat("WARNING: vcgVolume returned NA; falling back to ellipsoid approximation.\n")
    volume_exact <- (4/3) * pi * (width/2) * (thickness/2) * (length/2)
  }

  return(list(
    length = length, width = width, thickness = thickness,
    volume = volume_exact,
    anterior_area = anterior_area,
    total_surface_area = total_surface_area,
    perimeter = perimeter,
    bbox = bbox, center = center, vertices = vertices,
    hull_pts = hull_pts
  ))
}

metrics <- calculate_patella_metrics(patella_mesh)

cat("\n=== Patellar Geometric Parameters ===\n")
cat(sprintf("Patellar length: %.2f mm\n", metrics$length))
cat(sprintf("Patellar width: %.2f mm\n", metrics$width))
cat(sprintf("Patellar thickness: %.2f mm\n", metrics$thickness))
cat(sprintf("Patellar volume: %.2f mm3\n", metrics$volume))
cat(sprintf("Patellar surface area: %.2f mm2\n", metrics$total_surface_area))
cat(sprintf("Patellar coronal perimeter: %.2f mm\n", metrics$perimeter))

# 5. Generate 2D projection plots ---------------------------------------------
create_single_projection <- function(vertices, metrics, view = c("sagittal", "coronal", "axial")) {
  view <- match.arg(view)
  df <- as.data.frame(vertices)
  colnames(df) <- c("x", "y", "z")
  bbox <- metrics$bbox

  # Helper to add dimension arrows (keeps code DRY)
  add_dim_arrow <- function(p, x0, x1, y0, y1, label, color, ...) {
    p + annotate("segment", x = x0, xend = x1, y = y0, yend = y1,
                 arrow = arrow(ends = "both", length = unit(0.15, "cm")),
                 color = color, linewidth = 1) +
      annotate("text", x = (x0 + x1)/2, y = y0 + (y1 - y0) * 0.05,
               label = label, color = color, size = 5, fontface = "bold", family = base_family)
  }

  if (view == "sagittal") {
    hull_idx <- chull(df$x, df$z)
    hull_data <- df[hull_idx, c("x", "z")]
    p <- ggplot(df, aes(x = x, y = z)) +
      geom_polygon(data = hull_data, aes(x = x, y = z),
                   fill = "lightblue", alpha = 0.3, color = "blue", linewidth = 0.5) +
      geom_point(alpha = 0.1, size = 0.5, color = "blue")
    # Width arrow (X axis)
    p <- add_dim_arrow(p, bbox[1,1], bbox[2,1],
                       bbox[2,3] + 0.15 * diff(range(df$z)),
                       bbox[2,3] + 0.15 * diff(range(df$z)),
                       sprintf("Width: %.1f mm", metrics$width), "#4DBBD5")
    # Length arrow (Z axis)
    p <- add_dim_arrow(p, bbox[1,1] - 0.1 * diff(range(df$x)),
                       bbox[1,1] - 0.1 * diff(range(df$x)),
                       bbox[1,3], bbox[2,3],
                       sprintf("Length: %.1f mm", metrics$length), "#E64B35")
    p <- p + labs(title = "Sagittal View", x = "Medial-Lateral (mm)", y = "Superior-Inferior (mm)")

  } else if (view == "coronal") {
    hull_idx <- chull(df$y, df$z)
    hull_data <- df[hull_idx, c("y", "z")]
    p <- ggplot(df, aes(x = y, y = z)) +
      geom_polygon(data = hull_data, aes(x = y, y = z),
                   fill = "lightgreen", alpha = 0.3, color = "darkgreen", linewidth = 0.5) +
      geom_point(alpha = 0.1, size = 0.5, color = "darkgreen")
    p <- add_dim_arrow(p, bbox[1,2], bbox[2,2],
                       bbox[2,3] + 0.15 * diff(range(df$z)),
                       bbox[2,3] + 0.15 * diff(range(df$z)),
                       sprintf("Thickness: %.1f mm", metrics$thickness), "#F39B7F")
    p <- add_dim_arrow(p, bbox[1,2] - 0.1 * diff(range(df$y)),
                       bbox[1,2] - 0.1 * diff(range(df$y)),
                       bbox[1,3], bbox[2,3],
                       sprintf("Length: %.1f mm", metrics$length), "#E64B35")
    p <- p + labs(title = "Coronal View", x = "Anterior-Posterior (mm)", y = "Superior-Inferior (mm)")

  } else {  # axial
    hull_idx <- chull(df$x, df$y)
    hull_data <- df[hull_idx, c("x", "y")]
    p <- ggplot(df, aes(x = x, y = y)) +
      geom_polygon(data = hull_data, aes(x = x, y = y),
                   fill = "lavender", alpha = 0.3, color = "purple", linewidth = 0.5) +
      geom_point(alpha = 0.1, size = 0.5, color = "purple")
    p <- add_dim_arrow(p, bbox[1,1], bbox[2,1],
                       bbox[2,2] + 0.15 * diff(range(df$y)),
                       bbox[2,2] + 0.15 * diff(range(df$y)),
                       sprintf("Width: %.1f mm", metrics$width), "#4DBBD5")
    p <- add_dim_arrow(p, bbox[1,1] - 0.1 * diff(range(df$x)),
                       bbox[1,1] - 0.1 * diff(range(df$x)),
                       bbox[1,2], bbox[2,2],
                       sprintf("Thickness: %.1f mm", metrics$thickness), "#F39B7F")
    p <- p + labs(title = "Axial View", x = "Medial-Lateral (mm)", y = "Anterior-Posterior (mm)")
  }

  p + theme_nc() + coord_fixed(ratio = 1)
}

# Save each projection
cat("Generating projection plots...\n")
views <- c("sagittal", "coronal", "axial")
for (v in views) {
  p <- create_single_projection(metrics$vertices, metrics, v)
  ggsave(file.path(opt$output_dir, paste0("Patella_", v, "_View.png")),
         p, width = 6, height = 6, dpi = 300, bg = "white")
  ggsave(file.path(opt$output_dir, paste0("Patella_", v, "_View.pdf")),
         p, width = 6, height = 6, device = "pdf", bg = "white")
}

# 6. Coronal perimeter plot ---------------------------------------------------
create_perimeter_plot <- function(metrics) {
  df <- as.data.frame(metrics$vertices)
  colnames(df) <- c("x", "y", "z")
  hull_idx <- chull(df$x, df$y)
  hull_data <- df[hull_idx, c("x", "y")]

  ggplot(df, aes(x = x, y = y)) +
    geom_polygon(data = hull_data, aes(x = x, y = y),
                 fill = NA, color = "#3C5488", linewidth = 1.2) +
    geom_path(data = hull_data, aes(x = x, y = y),
              color = "#3C5488", linewidth = 1.2) +
    geom_point(data = hull_data, aes(x = x, y = y), 
               color = "#DC0000", size = 2) +
    annotate("text", x = mean(hull_data$x), y = mean(hull_data$y),
             label = sprintf("Coronal Perimeter = %.2f mm", metrics$perimeter),
             color = "#3C5488", size = 6, fontface = "bold", family = base_family) +
    labs(title = "Patellar Coronal Perimeter",
         x = "Medial-Lateral (mm)", y = "Anterior-Posterior (mm)") +
    theme_nc() + coord_fixed(ratio = 1)
}

p_peri <- create_perimeter_plot(metrics)
ggsave(file.path(opt$output_dir, "Patella_Coronal_Perimeter.png"), p_peri, width = 6, height = 6, dpi = 300, bg = "white")
ggsave(file.path(opt$output_dir, "Patella_Coronal_Perimeter.pdf"), p_peri, width = 6, height = 6, device = "pdf", bg = "white")

# 7. Parameter table (CSV + image) --------------------------------------------
param_df <- data.frame(
  Parameter = c("Patellar length (mm)", "Patellar width (mm)",
                "Patellar thickness (mm)", "Patellar volume (mm3)",
                "Patellar surface area (mm2)", "Patellar coronal perimeter (mm)"),
  Value = c(sprintf("%.2f", metrics$length),
            sprintf("%.2f", metrics$width),
            sprintf("%.2f", metrics$thickness),
            sprintf("%.2f", metrics$volume),
            sprintf("%.2f", metrics$total_surface_area),
            sprintf("%.2f", metrics$perimeter))
)

write.csv(param_df, file.path(opt$output_dir, "Patella_Measurements.csv"), row.names = FALSE, fileEncoding = "UTF-8")

# Create table image
tbl <- tableGrob(param_df, rows = NULL,
                 theme = ttheme_minimal(
                   core = list(fg_params = list(fontsize = 14, fontfamily = base_family),
                               bg_params = list(fill = c("white", "gray95"))),
                   colhead = list(fg_params = list(fontsize = 15, fontface = "bold", fontfamily = base_family))
                 ))
title <- textGrob("Patella Geometric Parameters", 
                  gp = gpar(fontsize = 18, fontface = "bold", fontfamily = base_family))
final_tbl <- grid.arrange(title, tbl, ncol = 1, heights = c(0.15, 0.85))
ggsave(file.path(opt$output_dir, "Patella_Parameters_Table.png"), final_tbl, width = 8, height = 4, dpi = 300, bg = "white")
ggsave(file.path(opt$output_dir, "Patella_Parameters_Table.pdf"), final_tbl, width = 8, height = 4, device = "pdf", bg = "white")

# 8. 3D volume rendering (optional) -------------------------------------------
if (!opt$skip_3d) {
  cat("Attempting 3D volume rendering...\n")
  create_3d_view_enhanced <- function(mesh, metrics, output_dir) {
    span <- max(metrics$bbox[2,] - metrics$bbox[1,])
    vox_size <- span / 40
    cat(sprintf("Voxel size: %.2f mm\n", vox_size))

    voxel_success <- FALSE
    voxel_mesh <- NULL

    # Use vcgVoxel (voxelSize argument; requires Rvcg >= 0.20)
    tryCatch({
      voxel_mesh <- vcgVoxel(mesh, voxelSize = vox_size)
      voxel_success <- TRUE
    }, error = function(e) {
      cat("Voxelisation with default size failed, trying larger...\n")
      tryCatch({
        voxel_mesh <<- vcgVoxel(mesh, voxelSize = vox_size * 1.5)
        voxel_success <<- TRUE
      }, error = function(e) {
        cat("Voxelisation still failed. Falling back to standard mesh rendering.\n")
      })
    })

    open3d()
    par3d(windowRect = c(100, 100, 1000, 800))
    bg3d("white")

    if (voxel_success && !is.null(voxel_mesh)) {
      shade3d(voxel_mesh, color = "#4DBBD5", alpha = 0.35, specular = "#555555", shininess = 40)
      wire3d(mesh, color = "#222222", lit = FALSE, linewidth = 1.2)
    } else {
      shade3d(mesh, color = "#4DBBD5", alpha = 0.6, specular = "#333333", shininess = 50)
      wire3d(mesh, color = "#222222", lit = FALSE, linewidth = 1.2)
    }

    axes3d(edges = "bbox", labels = TRUE, tick = TRUE, nticks = 5,
           xlab = "Medial-Lateral (mm)", ylab = "Anterior-Posterior (mm)", 
           zlab = "Superior-Inferior (mm)",
           col = "black", lwd = 2, cex = 1.2)
    title3d(main = paste0("Patellar volume ≈ ", round(metrics$volume), " mm³"),
            col = "black", cex = 1.8, font = 2)

    view3d(theta = 35, phi = 25, zoom = 0.8)
    rgl.snapshot(file.path(output_dir, "Patella_3D_Volume.png"))
    view3d(theta = 0, phi = 0, zoom = 0.8)
    rgl.snapshot(file.path(output_dir, "Patella_3D_Volume_Top.png"))
    close3d()
  }

  create_3d_view_enhanced(patella_mesh, metrics, opt$output_dir)
} else {
  cat("Skipping 3D volume rendering.\n")
}

cat("\nAll outputs saved to:", opt$output_dir, "\n")