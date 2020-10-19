




library(here)
library(readxl)
library(ggplot2)
library(lubridate)

devtools::install('/home/nilomr/projects/ganttrify/')
library(ganttrify)




date_breaks_y <- seq.Date(from = ymd('2020-01-01'),
                          to = ymd('2023-01-01'),
                          by = "1 year")


chart <- (ganttrify(
  project = read_excel(here(
    'reports', 'text', 'dphil_transfer', 'thesis_plan_gantt.xlsx'
  )),
  spots = read_excel(here(
    'reports', 'text', 'dphil_transfer', 'thesis_plan_gantt_spots.xlsx'
  )),
  by_date = TRUE,
  project_start_date = "2020-01",
  mark_years = T,
  font_family = "Roboto Condensed",
  size_wp = 5,
  size_activity = 4.2,
  size_text_relative = 2.2,
  month_number = FALSE,
  colour_stripe = "#e8e8e8",
  colour_palette = c(
    '#00A08A',
    '#F2AD00',
    '#5BBCD6',
    '#F98400',
    '#AB608A',
    '#c93030'
  )
) + theme(axis.title.x = element_blank(), 
        panel.grid.minor.x = element_blank(),
        panel.grid.major.x = element_blank())
    )

chart

out_path <- here('reports', 'figures', "transfer_gantt")

ggsave(paste0(out_path,'.pdf'), plot = chart, device = cairo_pdf, units = "cm", width = 35 , height = 26)
ggsave(paste0(out_path,'.svg'), plot = chart, device = svg, units = "cm", width = 35 , height = 26)
