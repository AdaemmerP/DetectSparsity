# Load libraries
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(scales)
  library(stringr)

#--------------------- Plots for weak predictors (nrs) ------------------------#
# Set data path here to "Results"
  data_path <- ""
  
# Make plots
  setwd(paste(data_path, "GW_Data/1960/", sep = ""))

# File names
  file_names <- list.files()

# Load first results
  df_all <- read.csv(file_names[[1]])[, 1:5]

# Loop to add results
  for(jj in 2:length(file_names)){

    df_all <- left_join(df_all, read.csv(file_names[[jj]])[, 1:5], by = c("date", "y_true"))

  }

  df_all <- df_all %>%
            mutate(date = as.Date(date))

  df_nr <- df_all                                  %>%
             dplyr::select(date, nr_comby, nr_bss_ebic,
                                 nr_lasso_rel, nr_lasso,
                                 nr_glp, nr_enet,
                                 nr_ridge, nr_ew) %>%
             pivot_longer(!date, names_to = "method", values_to = "value")

# Make factors to order facets
  flevels <- c("nr_comby", "nr_bss_ebic",
              "nr_lasso_rel", "nr_lasso",
              "nr_glp", "nr_enet",
              "nr_ridge", "nr_ew")

  flabels <- c(expression("FC"^{Flex}), "Best~Subset",
                        "Relaxed~Lasso",  "Lasso",
                        "Bayesian~FSS",  "Elastic~Net",
                        "Ridge", expression("FC"^{EW}))

  df_nr$method_new <- factor(df_nr$method, levels = flevels, labels = flabels)


 # Colors for plots
  my_color <- c("#000000", "#E69F00", "#3a5795", "#56B4E9", "#009E73", "#0072B2",
               "#D55E00", "#CC79A7")

# Make plot with number of predictors
  nr_weakpr <- ggplot(data = df_nr) +
                 geom_point(aes(x = date, y = value), color = my_color[6], size = .05) +
                 theme_bw() +
                 theme(text             = element_text(size = 10),
                       axis.text        = element_text(size = 8),
                       strip.text.x     = element_text(size = 10),
                       axis.title.y     = element_text(margin  = margin(t = 0, r = 5, b = 0, l = 0)),
                       axis.title.x     = element_text(margin  = margin(t = 5, r = 0, b = 0, l = 0)),
                       axis.text.x      = element_text(hjust = 0.6),
                       strip.background = element_blank(),
                       panel.grid.minor = element_blank(),
                       legend.position = "none",
                       panel.spacing.x  = unit(2, "lines")) +
                 ylab("# of nonzero coefficients") +
                 xlab("Year") +
                 facet_wrap(~method_new, labeller = label_parsed, ncol = 2, scales = "free") +
                 coord_cartesian(ylim = c(0, 12)) +
                 scale_y_continuous(breaks = c(0, 2, 4, 6, 8, 10, 12)) +
                 scale_x_date(expand        = c(0,0),
                              date_breaks   = "10 year",
                              labels        = date_format("%Y"))
  nr_weakpr

 #---------------- Plots for relative inclusions

 # File names
  file_names <- list.files()

 # Load first results
  df_all <- tibble() 

 # Loop to add results
   for(jj in 1:length(file_names)){

     dftmp   <- read.csv(file_names[[jj]])
     newname <- word(colnames(dftmp)[str_detect(colnames(dftmp), "fcast_")], 2, sep ="_")
     dftmp   <- dftmp %>%
               mutate(method = newname)
     dftmp   <- dftmp %>%
                dplyr::select(-date, -y_true, -starts_with("fcast_"), -hmean, -starts_with("nr_")) %>%
                      pivot_longer(-method)
     df_all <- rbind(df_all, dftmp)


   }

  df_gg <- df_all %>%
            group_by(method, name) %>%
            summarize(mean_val = mean(value))

# Make factors to order facets
  flevels <- c("comby", "bss",
              "lassorel", "lasso",
              "glp", "enet",
              "ridge", "ew")

  flabels <- c(expression("FC"^{Flex}), "Best~Subset",
              "Relaxed~Lasso",  "Lasso",
              "Bayesian~FSS",  "Elastic~Net",
              "Ridge", expression("FC"^{EW}))


  df_gg$method_new <- factor(df_gg$method, levels = flevels, labels = flabels)


  inkl_weakpr <- ggplot(data = df_gg, aes(x = name, y = mean_val)) +
                  geom_bar(stat = "identity", width = 0.6, fill = my_color[6], alpha = 0.5)+
                  theme_bw() +
                  theme(text             = element_text(size = 10),
                        axis.text        = element_text(size = 8),
                        strip.text.x     = element_text(size = 10),
                        axis.title.y     = element_text(margin  = margin(t = 0,   r = 7.5, b = 0, l = 0)),
                        axis.title.x     = element_text(margin  = margin(t = 7.5, r = 0, b = 0, l = 0)),
                        strip.background = element_blank(),
                        panel.grid.minor = element_blank(),
                        panel.grid.major = element_blank(),
                        legend.position = "none",
                        panel.spacing.x  = unit(2, "lines")) +
                  ylab("Relative inclusion") +
                  xlab("Predictor") +
                  scale_x_discrete(guide = guide_axis(n.dodge=2)) +
                  facet_wrap(~method_new, labeller = label_parsed, ncol = 2, scales = "free") +
                  coord_cartesian(ylim = c(0, 1))
  inkl_weakpr

#------------------ Plots for included predictor by Pyun (2019) ---------------#

 # Set data path
  setwd(paste(data_path, "GWP_Data/1999/", sep = ""))
  
 # File names
  file_names <- list.files()

 # Load first results
  df_all <- read.csv(file_names[[1]])[, 1:5]

 # Loop to add results
   for(jj in 2:length(file_names)){

     df_all <- left_join(df_all, read.csv(file_names[[jj]])[, 1:5], by = c("date"))

   }

   df_all <- df_all %>%
               mutate(date = as.Date(date))

   df_nr <- df_all  %>%
               dplyr::select(date, nr_comby, nr_bss_ebic,
                             nr_lasso_rel, nr_lasso,
                             nr_glp, nr_enet,
                             nr_ridge, nr_ew) %>%
               pivot_longer(!date, names_to = "method", values_to = "value")

# Make factors to order facets
   flevels = c("nr_comby", "nr_bss_ebic",
               "nr_lasso_rel", "nr_lasso",
               "nr_glp", "nr_enet",
               "nr_ridge", "nr_ew")

   flabels = c(expression("FC"^{Flex}), "Best~Subset",
               "Relaxed~Lasso",  "Lasso",
               "Bayesian~FSS",  "Elastic~Net",
               "Ridge", expression("FC"^{EW}))



   df_nr$method_new <- factor(df_nr$method, levels = flevels, labels = flabels)


 # Colors for plots
   my_color <- c("#000000", "#E69F00", "#3a5795", "#56B4E9", "#009E73", "#0072B2",
                 "#D55E00", "#CC79A7")

 # Make plot with number of predictors
   nr_strongp <- ggplot(data = df_nr) +
                 geom_point(aes(x = date, y = value), color = my_color[6], size = .1) +
                 theme_bw() +
                 theme(text             = element_text(size = 10),
                       axis.text        = element_text(size = 8),
                       strip.text.x     = element_text(size = 10),
                       axis.title.y     = element_text(margin  = margin(t = 0, r = 5, b = 0, l = 0)),
                       axis.title.x     = element_text(margin  = margin(t = 5, r = 0, b = 0, l = 0)),
                       strip.background = element_blank(),
                       panel.grid.minor = element_blank(),
                       legend.position = "none",
                       panel.spacing.x  = unit(1.25, "lines")) +
                 ylab("# of nonzero coefficients") +
                 xlab("Year")  +
                 facet_wrap(~method_new, labeller = label_parsed, ncol = 2, scales = "free") +
                 coord_cartesian(ylim = c(0, 13)) +
                 scale_y_continuous(breaks = c(0, 2, 4, 6, 8, 10, 12)) +
                 scale_x_date(expand        = c(0,0),
                              date_breaks   = "5 year",
                              labels        = date_format("%Y"))

   nr_strongp

#---------------- Plots for relative inclusions

   # File names
   file_names <- list.files()

   # Load first results
   df_all <- tibble() 

   # Loop to add results
   for(jj in 1:length(file_names)){

     dftmp   <- read.csv(file_names[[jj]]) %>%
                  rename(vrp = "Pyun_FC")

     newname <- word(colnames(dftmp)[str_detect(colnames(dftmp), "fcast_")], 2, sep ="_")
     dftmp   <- dftmp %>%
                  mutate(method = newname)
     dftmp   <- dftmp %>%
                 dplyr::select(-date, -y_true, -starts_with("fcast_"), -hmean, -starts_with("nr_")) %>%
                 pivot_longer(-method)
     df_all <- rbind(df_all, dftmp)


   }

   df_gg <- df_all %>%
             group_by(method, name) %>%
             summarize(mean_val = mean(value))

   # Make factors to order facets
   flevels <- c("comby", "bss",
               "lassorel", "lasso",
               "glp", "enet",
               "ridge", "ew")


   flabels <- c(expression("FC"^{Flex}), "Best~Subset",
               "Relaxed~Lasso",  "Lasso",
               "Bayesian~FSS",  "Elastic~Net",
               "Ridge", expression("FC"^{EW}))


   df_gg$method_new <- factor(df_gg$method, levels = flevels, labels = flabels)

   df_gg$name = factor(df_gg$name, levels = unique(df_gg$name)[c(1:10, 12, 13, 11)], labels = unique(df_gg$name)[c(1:10, 12, 13, 11)])


   inkl_strongpr <- ggplot(data = df_gg, aes(x = name, y = mean_val)) +
                       geom_bar(stat = "identity", width = 0.6, fill = my_color[6], alpha = 0.5)+
                       theme_bw() +
                       theme(text             = element_text(size = 10),
                             axis.text        = element_text(size = 8),
                             strip.text.x     = element_text(size = 10),
                             axis.title.y     = element_text(margin  = margin(t = 0,   r = 7.5, b = 0, l = 0)),
                             axis.title.x     = element_text(margin  = margin(t = 7.5, r = 0, b = 0, l = 0)),
                             strip.background = element_blank(),
                             panel.grid.minor = element_blank(),
                             panel.grid.major = element_blank(),
                             legend.position = "none",
                             panel.spacing.x  = unit(2, "lines")) +
                       ylab("Relative inclusion") +
                       xlab("Predictor") +
                       scale_x_discrete(guide = guide_axis(n.dodge=2)) +
                       facet_wrap(~method_new, labeller = label_parsed, ncol = 2, scales = "free") +
                       coord_cartesian(ylim = c(0, 1))
   inkl_strongpr


    