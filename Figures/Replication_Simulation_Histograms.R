# Load packages
  library(tidyverse)
  library(scales)
  library(lemon)

# Set data path here
  data_path <- ""

# Load data
  load(paste0(data_path,"Sim_Histograms/Results_Figure5_Simulation_Macro.RData"))

  results_all <- results_all |>
                   mutate(Method = str_replace(Method, "Rel._lasso", "Relaxed~Lasso"))
  
    
# Colors for plot
  my_color <- c("#000000", "#E69F00", "#3a5795", "#56B4E9", "#009E73", "#0072B2",
                "#D55E00", "#CC79A7")

# Tidy data set
  results_all_tidy <- pivot_longer(results_all, !Method) |>
                        mutate(name   = fct_relevel(name,"nb_5", "nb_50", "nb_100")) |>
                        mutate(name   = fct_recode(name, "n[beta]~'='~5" = "nb_5",
                                                         "n[beta]~'='~50"  = "nb_50",
                                                         "n[beta]~'='~100" = "nb_100")) |>
                        mutate(Method = fct_relevel(Method, "FC-Flex", "BSS", 
                                                            "Relaxed~Lasso", "Lasso")) |>
                        mutate(Method = fct_recode(Method, "Elastic~Net" = "E-Net",
                                                           "FC^{Flex}"   = "FC-Flex",
                                                           "Best~Subset" = "BSS"))
 # Make summary for values
   summary_vals <- results_all_tidy |>
                    mutate(true_val = as.numeric(str_extract(name, "\\d+"))) |>
                    group_by(Method, name) |>
                    summarise_at(vars(value, true_val), mean)
  
  
# Make ggplot
  p_sim <- ggplot(results_all_tidy) +
              geom_histogram(aes(x = value, y = ..density..), 
                             col = alpha("grey", 0.8), fill = "grey", bins = 20, alpha = .7) +
              facet_rep_grid(Method ~ name, labeller = label_parsed, scales = "free", repeat.tick.labels = T) +
              geom_vline(data = summary_vals, aes(xintercept = value, col = "Estimated number of predictors"),
                         alpha = 1, linetype = 2, size = 0.7)                                   +
              geom_vline(data = summary_vals, aes(xintercept = true_val, col = "True number of predictors"),   
                         alpha = 1, linetype = 2, size = 0.7)                                                +
              theme_bw() +
              theme(text             = element_text(size = 10),
                    axis.text        = element_text(size = 8),
                    strip.text.x     = element_text(size = 8),
                    strip.text.y     = element_text(size = 8),
                    axis.title.y     = element_text(margin  = margin(t = 0, r = 5, b = 0, l = 0)),
                    axis.title.x     = element_text(margin  = margin(t = 5, r = 0, b = 0, l = 0)),
                    strip.background = element_rect(fill    = alpha("gray", 0.2)),
                    axis.text.y      = element_blank(),
                    axis.ticks.y     = element_blank(),
                    panel.grid.major = element_blank(),
                    panel.grid.minor = element_blank(),
                    panel.spacing = unit(0.75, "lines"),
                    legend.position = "") +
              #scale_x_continuous(limits = c(0, 101), breaks = c(0, 25, 50, 75, 100)) +
              scale_y_continuous(expand = c(0, 0)) +
              scale_color_manual(values = c("Estimated number of predictors" = my_color[7], 
                                            "True number of predictors"      = my_color[3])) +
              labs(x = "Number of predictors",
                   y = "Density")
  p_sim

  
  # Save plot
  # setwd("")
  # pdf(file = "p_sim.pdf", width = 8, height = 10, pointsize = 8)
  # p_sim
  # dev.off()
