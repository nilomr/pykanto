
# new-nestboxes.R
#
# This script provides part of the source code necessary to 
# make lists and plot new nestboxes as reported by field workers
# Note: The functions below are very specific to this task and
# are very dirty.
# 
# Copyright (c) Nilo Merino recalde, 2020, except where indicated
# Date Created: 2020-04-02


# --------------------------------------------------------------------------
# REQUIRES
# --------------------------------------------------------------------------

library(rprojroot)
library(tidyverse)
library(lubridate)
library(ggmap)
library(rgdal)
library(ggrepel)
library(plotKML)
library(pgirmess)
library(kableExtra)

# Python setup
library(reticulate)
use_condaenv("0.0_great-tit-song-segment")

# --------------------------------------------------------------------------
# PATHS
# --------------------------------------------------------------------------

data_path <- file.path(getwd(), "data")
sheet_path <- file.path(data_path, "resources", "fieldwork", year(today()))
figures_path <- file.path(getwd(), "resources", "fieldwork", year(today()))
gpx_path <- file.path(figures_path, "gpx-files")


if (!dir.exists(sheet_path)) {
  dir.create(sheet_path, recursive = TRUE)
}

if (!dir.exists(figures_path)) {
  dir.create(figures_path, recursive = TRUE)
}

if (!dir.exists(gpx_path)) {
  dir.create(gpx_path, recursive = TRUE)
}

# --------------------------------------------------------------------------
# FUNCTIONS
# --------------------------------------------------------------------------

#' Runs and checks if the python script downloaded any new data from the summary spreadsheet
#' in the last 50 seconds -- this is ugly as hell but seems to be robust (as long as 
#' you don't check it constantly!)
#'
#' @param new_boxes_list a list of .csv files
#' @return Whether there are at least 10 new nestboxes 
check_new_nestboxes <- function(new_boxes_list) {
  Sys.sleep(2)
  if (length(new_boxes_list) == 0) {
    stop("The list is empty")
    }
  last <- read_csv(last(new_boxes_list)) %>%
    mutate(Added = ymd_hm(Added)) %>% select(Added)
  last_date <- (last$Added[1])
  now <- now(tzone = "UTC")
if (as.double(difftime(now, last_date, units = "s")) < 60) {
    print("TRUE: There are at least 10 new nestboxes")
  } else {
    stop("FALSE: There are fewer than 10 new nestboxes")
  }
}


#' Makes tibble with new nestboxes and their coordinates
#' @param new_boxes_list a list of .csv files
#' @return a tibble
get_new_nestboxes <- function(new_boxes_list) {
  if (length(new_boxes_list) == 0) {
    stop("There are no files here")
  } else if (length(new_boxes_list) == 1) {
    new_boxes <- read_csv(new_boxes_list) %>%
      mutate(Added = ymd_hm(Added)) %>%
      trans_coords(., 27700, 4326)
    return(new_boxes)
  } else if (length(new_boxes_list) > 1) {
    new_boxes <- read_csv(last(new_boxes_list)) %>%
      mutate(Added = ymd_hm(Added)) %>%
      trans_coords(., 27700, 4326)
    return(new_boxes)
  }
}


#' Automatically creates subtitles for plots depending on whether 
#' there were one or more .csv files with new nestboxes in the directory.
#' #' @param new_boxes_list a list of .csv files
#' #' @param new_boxes a tibble created by get_new_nestboxes()
#' #' @return a string containing a subtitle for a plot
make_subtitle <- function(new_boxes_list, new_boxes) {
  now <- gsub('.{3}$', '', now(tzone = "UTC"))
  times <- tibble(now = now)
  if (length(new_boxes_list) == 1) {
    subtitle <- paste(nrow(new_boxes),
                      "new nestboxes",
                      times$now,
                      sep = " ")
    return(subtitle)
  } else if (length(new_boxes_list) > 1) {
    previous <- read_csv(nth(new_boxes_list,-2L)) %>%
      mutate(Added = ymd_hm(Added)) %>% select(Added)
    previous_date <- gsub('.{3}$', '', previous$Added[1])
    
    times <- times %>% mutate(last = previous_date)
    subtitle <- paste(
      nrow(new_boxes),
      "new nestboxes",
      "in the period between",
      times$last,
      "and",
      times$now,
      sep = " "
    )
    return(subtitle)
  }
}

#' Transforms coordinates 
#'
#' @param tibble a tibble with x, y columns
#' @param origin epsg code of original coordinates
#' @param new epsg code to be transformed to 
#' @return a tibble with transformed lon lat columns 
#' @examples
#' new_boxes_lonlat <- trans_coords(new_boxes, 27700, 4326)
trans_coords <- function(tibble, origin, new) {
  coords_new_boxes <-
    tibble %>%
    select(x, y) %>%
    rename(lon = x, lat = y)
  
  coordinates(coords_new_boxes) <- c("lon", "lat")
  
  proj4string(coords_new_boxes) <-
    CRS(paste0("+init=epsg:", as.character(origin)))
  
  CRS.new <- CRS(paste0("+init=epsg:", as.character(new)))
  
  coords_new_boxes_latlong <-
    spTransform(coords_new_boxes, CRS.new)
  
  new_coords <- as_tibble(coords_new_boxes_latlong@coords)
  
  tibble_new <-
    tibble %>% mutate(lon = new_coords$lon, lat = new_coords$lat)
  
  return(tibble_new)
}

