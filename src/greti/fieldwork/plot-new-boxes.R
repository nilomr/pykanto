#!/usr/bin/env Rscript

# 1. Settings ====

renv::load(project = "/home/nilomr/projects/0.0_great-tit-song")
setwd("/home/nilomr/projects/0.0_great-tit-song")

# Source file
source("./src/viz/plot-new-gretis.R")


# 2. Nestboxes to be recorded ====

# Prepare data for plotting
newones <-
  read_csv(file.path(sheet_path, "toberecorded.csv"), 
           col_types = cols()) %>% trans_coords(., 27700, 4326)

## This is my google key, remove before sharing
# ggmap::register_google(key = "AIzaSyDfAbhsGQ32byJTmQnpOFjQ_5yr3ViVYeI", write = TRUE)

# Centre the map here
wytham <- c(lon = -1.321898, lat = 51.771250)

# Make subtitle for plot
now <- gsub('.{3}$', '', now(tzone = "UTC"))

subtitle <- paste(
  nrow(newones),
  "new nestboxes",
  "as of",
  now,
  sep = " "
)

# Make plot
new_boxes_map <-
  get_googlemap(
    center = wytham,
    zoom = 14,
    scale = 4,
    maptype = "satellite",
    color = "color"
  ) %>%
  ggmap(base_layer = ggplot(
    aes(x = lon, y = lat), data = newones)) %>%
  +geom_point(colour = "white", size = 2) %>%
  +geom_text_repel(
    aes(label = Nestbox),
    direction = 'both',
    box.padding = 0.40,
    colour = "white",
    segment.alpha = 0.5,
    seed = 30
  ) %>%
  +theme_nothing() %>%
  +ggtitle("New great tit nestboxes", subtitle = subtitle) %>%
  +theme(plot.title = element_text(hjust = 0, size = 30, face = "bold")) %>%
  +theme(plot.subtitle = element_text(hjust = 0, size = 15)) %>%
  +theme(plot.margin = unit(c(1, 1, 1, 1), "cm"))

# Save plot
now <- gsub('.{3}$', '', now(tzone = "UTC"))  %>% sub(" ", "_", .)
filename <- paste0("newboxes_map_", now, ".png")

ggsave(
  new_boxes_map,
  path = figures_path,
  filename = filename,
  width = 30,
  height = 30,
  units = "cm",
  dpi = 350
)

# Make and save list

list_path <-
  file.path("resources",
            "fieldwork",
            year(today()),
            paste0("new_", now, ".pdf"))

newones %>% select(-x, -y) %>% 
  knitr::kable() %>%  
  kable_styling() %>%
  save_kable(file = list_path)



# 3. Plot recorded vs remaining nest boxes ====


# Prepare data for plotting
recorded <-
  read_excel(file.path(sheet_path, "already-recorded.xlsx")) %>%
  mutate_at(c('x', 'y'), as.numeric) %>%
  filter(str_detect(Nestbox, "Nestbox", negate = TRUE)) %>%
  trans_coords(., 27700, 4326)

## This is my google key, remove before sharing
# ggmap::register_google(key = "AIzaSyDfAbhsGQ32byJTmQnpOFjQ_5yr3ViVYeI", write = TRUE)

# Centre the map here
wytham <- c(lon = -1.321898, lat = 51.771250)

# Make subtitle for plot
now <- gsub('.{3}$', '', now(tzone = "UTC"))

subtitle <- paste(
  nrow(recorded),
  "nestboxes <b style='color:#e09200'>recorded</b> and",
  nrow(newones),
  "<b style='color:#4184b0'>to be recorded</b>",
  "as of",
  now,
  sep = " "
)

# Make plot
recorded_map <-
  get_googlemap(
    center = wytham,
    zoom = 14,
    scale = 4,
    maptype = "satellite",
    color = "color"
  ) %>%
  ggmap(base_layer = ggplot(aes(x = lon, y = lat), data = recorded)) %>%
  +geom_point(colour = "#e09200",
              size = 2,
              alpha = 0.8) %>%
  +geom_point(
    data = newones,
    colour = "#4184b0",
    size = 2,
    alpha = 0.65
  ) %>%
  +theme_void() %>%
  +theme(plot.margin = unit(c(1, 1, 1, 1), "cm")) %>%
  +labs(title = "Great Tit Song Recording Season",
        subtitle = subtitle) %>%
  +theme(
    plot.title = element_markdown(
      lineheight = 1.1,
      hjust = 0,
      size = 25,
      face = "bold"
    ),
    plot.subtitle = element_markdown(
      lineheight = 2,
      hjust = 0,
      padding = unit(c(0, 0, 8, 0), "pt"),
      size = 16
    )
  )

# Save plot
nowplot <- now %>% sub(" ", "_", .)
filename <- paste0("recorded_newboxes_map_", nowplot, ".png")

ggsave(
  recorded_map,
  path = figures_path,
  filename = filename,
  width = 30,
  height = 30,
  units = "cm",
  dpi = 350
)

print("Done!")

rm(list = ls())