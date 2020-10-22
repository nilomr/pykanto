




library(here)
library(readxl)
library(ggplot2)
library(lubridate)

devtools::install('/home/nilomr/projects/ganttrify/')
library(ganttrify)
library(dplyr)




date_breaks_y <- seq.Date(from = ymd('2020-01-01'),
                          to = ymd('2023-01-01'),
                          by = "1 year")

extra_segments <- read_excel(here(
  'reports', 'text', 'dphil_transfer', 'thesis_plan_extra.xlsx'
)) %>% mutate(start_date = as.Date(start_date),
              end_date = as.Date(end_date))

size_activity = 4.2


chart <- (ganttrify(
  project = read_excel(here(
    'reports', 'text', 'dphil_transfer', 'thesis_plan_gantt.xlsx'
  )),
  # spots = read_excel(here(
  #   'reports', 'text', 'dphil_transfer', 'thesis_plan_gantt_spots.xlsx'
  # )),
  by_date = TRUE,
  project_start_date = "2020-01",
  mark_years = T,
  font_family = "Roboto Condensed",
  size_wp = 5,
  size_activity = size_activity,
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
        panel.grid.major.x = element_blank()) +
  ggplot2::geom_segment(data = extra_segments ,
                        lineend = "round",
                        size = size_activity,
                        alpha = 0.3) +
  geom_vline(xintercept = as.numeric(as.Date("2020-10-22")), size = 1, colour = 'grey', linetype= 'dashed', alpha = 0.5) +
  annotate("text", x=as.Date("2020-09-08"), label="Today", y=0.8, colour="grey", alpha = 0.8, angle=0, size=6) 
)

chart

out_path <- here('reports', 'figures', "transfer_report", "transfer_gantt")

ggsave(paste0(out_path,'.pdf'), plot = chart, device = cairo_pdf, units = "cm", width = 35 , height = 26)
ggsave(paste0(out_path,'.svg'), plot = chart, device = svg, units = "cm", width = 35 , height = 26)
