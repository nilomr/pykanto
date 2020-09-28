
# fieldwork-resources.R
#
# This script provides the source code necessary to 
# check wether recorders are affecting egg laying 
# and make plots to this effect
# The functions below are very specific to this task and
# are very 'dirty'.
# 
# Copyright (c) Nilo Merino recalde, 2020, except where indicated
# Date Created: 2020-04-30


# --------------------------------------------------------------------------
# REQUIRES
# --------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(tidyverse)
  library(googlesheets4)
  library(devtools)
  library(jcolors)
  library(readxl)
  library(gridExtra)
  library(lubridate)
  library(janitor)
  library(lme4)
  library(ggrepel)
  library(RColorBrewer)
  library(patchwork)
  library(sjPlot)
  library(ochRe)
  library(ghibli)
  library(wesanderson)
  library(kableExtra)
  library(knitr)
})


# --------------------------------------------------------------------------
# PATHS
# --------------------------------------------------------------------------

data_path <- file.path(getwd(), "resources", "brood_data")

sheet_path <- file.path(getwd(), "resources", "fieldwork", year(today()))

figures_path <- file.path(getwd(), "resources", "fieldwork", year(today()))


if (!dir.exists(data_path)) {
  dir.create(data_path, recursive = TRUE)
}

if (!dir.exists(sheet_path)) {
  dir.create(sheet_path, recursive = TRUE)
}

if (!dir.exists(figures_path)) {
  dir.create(figures_path, recursive = TRUE)
}

# --------------------------------------------------------------------------
# FUNCTIONS
# --------------------------------------------------------------------------

#' Violin plot comparing the effect of the presence of recorders in a given
#' variable. Colours datapoints by date.
#'
#' @param data a tibble with the data
#' @param species_code one of c("b", "g")
#' @param variable column name of the dependent variable
#' @return a plot
plot_nest_data <-
  function(data,
           species_code,
           variable,
           colour = "#d6ad7a",
           palette = "OrRd",
           breaks = 5,
           text_size = 12) {
    
    getPalette = colorRampPalette(brewer.pal(9, "OrRd"))
    
    varname <- gsub("_", " ", variable)
    
    if (species_code == "g") {
      species <- "great tits"
    } else if (species_code == "b") {
      species <- "blue tits"
    } else
      print("Species code not valid")
    
    plot <-
      data %>%
      filter(species == species_code) %>%
      filter(lay_date < move_by | is.na(move_by)) %>%
      ggplot(aes(x = recorded, y = eval(parse(text = variable)))) +
      geom_violin(fill = "#d6ad7a",
                  colour = NA,
                  alpha = 0.2,
                  adjust = 2) +
      geom_jitter(
        alpha = 1,
        aes(colour = factor(lay_date)),
        position = position_jitter(
          width = 0.2,
          height = 0.05,
          seed = 2
        )
      ) +
      theme_minimal(base_size = text_size) +
      theme(legend.position = "none") +
      scale_colour_manual(values = rev(getPalette(length(
        unique(data$lay_date)
      )))) +
      theme(
        panel.grid.minor = element_blank(),
        panel.grid.major.x = element_blank(),
        axis.text.x = element_text(face = "bold", size = text_size - (text_size / 100 * 20)),
        axis.title = element_text(size = text_size - (text_size / 100 * 15)),
        plot.title = element_text(size = text_size + (text_size / 100 * 10),
                                  face = "bold"),
        plot.subtitle = element_text(
          color = "black",
          size = text_size - (text_size / 100 * 20),
          face = "italic"
        )
      ) +
      scale_y_continuous(breaks = scales::pretty_breaks(n = breaks)) +
      labs(
        y = paste0(stringr::str_to_sentence(varname), "\n"),
        x = "\nRecorder in nestbox?",
        title = paste0("Does the presence of recorders", "\n", "affect ", varname, "s?"),
        subtitle = paste0("Data for ", species, ". Darker = earlier lay date")
      ) 
    
    return(plot)
  }



#' Plot effect size estimates from linear regression model
#'
#' @param model model data frame
#' @param plot_title a string with the title
#' @return a plot
plot_recorder_model <- function(model, plot_title){
  plot_model(model) +
    labs(title = plot_title)  +
    geom_hline(
      yintercept = 0,
      colour = "black",
      size = 2,
      alpha = 0.1
    ) +
    theme_minimal() +
    theme(
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.background = element_blank(),
      plot.title.position = "plot"
    )
}

#' Get total birds identified each year in a range
#'
#' @param data cleaned ebmp csv for the desired range of years
#' @param years int, range of years
#' @return a data frame with total and identified males
get_id_number <- function(data, years) {
  noid_greti_m <- vector(mode = "numeric", length = length(years))
  noid_greti_f <- vector(mode = "numeric", length = length(years))
  total_gretis <- vector(mode = "numeric", length = length(years))
  parent = c('father', 'mother')
  for (i in seq_along(years)) {
    total <-
      (
        data %>%
          filter(year == years[i]) %>%
          select(all_of(parent)) %>%
          count() %>%
          sum()
      )
    no_id_m <-
      (
        data %>%
          filter(year == years[i]) %>%
          select('father') %>%
          is.na() %>%
          sum()
      )
    no_id_f <-
      (
        data %>%
          filter(year == years[i]) %>%
          select('mother') %>%
          is.na() %>%
          sum()
      )
    total_gretis[i] <- total
    noid_greti_m[i] <- no_id_m
    noid_greti_f[i] <- no_id_f
  }
  
  id_gretis_m <- total_gretis - noid_greti_m
  id_gretis_f <- total_gretis - noid_greti_f
  
  id_data <-
    do.call(
      rbind,
      Map(
        data.frame,
        year = years,
        total = total_gretis,
        identified_male = id_gretis_m,
        identified_female = id_gretis_f
      )
    )
  
  return(id_data)
}

