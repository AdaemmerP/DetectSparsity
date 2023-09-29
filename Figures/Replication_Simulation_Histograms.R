# Load packages
  library(tidyverse)
  library(scales)
  library(lemon)

# Source file with data paths
  source("data_path.R")

# Load data
  load(paste0(data_path,"Sim_Histograms/Histogram_Macro_n400.RData"))

  results_all <- results_all |>
                   mutate(Method = str_replace(Method, "Rel._lasso", "Relaxed~Lasso"))


# Colors for plot
  my_color <- c("#000000", "#E69F00", "#3a5795", "#56B4E9", "#009E73", "#0072B2",
                "#D55E00", "#CC79A7")

 # Tidy data set Macro
  results_all_tidy <- pivot_longer(results_all, !Method) |>
                        mutate(name   = fct_relevel(name,"nb_5", "nb_50", "nb_100")) |>
                        mutate(name   = fct_recode(name, "n[beta]~'='~5" = "nb_5",
                                                         "n[beta]~'='~50"  = "nb_50",
                                                         "n[beta]~'='~100" = "nb_100")) |>
                        mutate(Method = fct_relevel(Method, "FC-Flex", "BSS",
                                                           "Relaxed Lasso", "Lasso", "Elastic Net")) |>
                        mutate(Method = fct_recode(Method, "Elastic~Net" = "Elastic Net", # "Elastic Net", # "E-Net",
                                                           "Relaxed~Lasso" = "Relaxed Lasso",
                                                           "FC^{Flex}"   = "FC-Flex",
                                                           "Best~Subset" = "BSS",
                                                            "VFSS" = "BFSS" ))

# Remove FC^FLEX
  results_all_tidy <- results_all_tidy |> 
    filter(!(Method %in% "FC^{Flex}"))
  

# Tidy data set Financial
  results_all_tidy <- pivot_longer(results_all, !Method) |>
                        mutate(name   = fct_relevel(name,"nb_3", "nb_6", "nb_13")) |>
                        mutate(name   = fct_recode(name, "n[beta]~'='~3" = "nb_3",
                                                   "n[beta]~'='~6"  = "nb_6",
                                                   "n[beta]~'='~13" = "nb_13")) |>
                        mutate(Method = fct_relevel(Method, "FC-Flex", "BSS",
                                                    "Relaxed Lasso", "Lasso", "Elastic Net")) |>
                        mutate(Method = fct_recode(Method,
                                                   "Relaxed~Lasso" = "Relaxed Lasso",
                                                   "Elastic~Net" = "Elastic Net",
                                                   "FC^{Flex}"   = "FC-Flex",
                                                   "Best~Subset" = "BSS"))
  
  # Remove FC^FLEX
  results_all_tidy <- results_all_tidy |> 
    filter(!(Method %in% "FC^{Flex}"))

  
 # Make summary for values
   summary_vals <- results_all_tidy |>
                    mutate(true_val = as.numeric(str_extract(name, "\\d+"))) |>
                    group_by(Method, name) |>
                    summarise_at(vars(value, true_val), mean)


# Make ggplot
  p_sim <- ggplot(results_all_tidy) +
              geom_histogram(aes(x = value, y = ..density..),
                             col = alpha("grey", 0.8), fill = "grey", alpha = .7, bins = 20) +#) + # ,  binwidth = 1 (finance)
              #geom_density(aes(x = value, y = ..density..)) +
              facet_rep_grid(Method ~ name, labeller = label_parsed, scales = "free_y", repeat.tick.labels = T) +
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
             # scale_x_continuous(limits = c(0, 13), breaks = c(0, 3, 6, 9, 13)) + # For financial data
              scale_y_continuous(expand = c(0, 0)) +
              scale_color_manual(values = c("Estimated number of predictors" = my_color[7],
                                            "True number of predictors"      = my_color[3])) +
              labs(x = "Number of predictors",
                   y = "Density")
  p_sim


  # Save plot
   setwd(save_path)
   pdf(file = "p_sim_macro_n400.pdf", width = 8, height = 9, pointsize = 8)
   p_sim
   dev.off()
