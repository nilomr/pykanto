# Read .wav metadata ------------------------------------------------------

#  The code below is by Stuart K. Grange and 'uses Phil Harvey's Perl \code{exiftool} to extract a 
#  file's metadata and then formats the output into a table which can be 
#  conveniently used within R'.


file_metadata <- function(file) {
  
  # Check for programme
  detect_exiftool()
  
  # Initialize progress bar
  # if (.progress) progress_bar <- dplyr::progress_estimated(length(file))
  
  # Ensure path is expanded, sometimes in necessary and then do
  df <- fs::path_expand(file) %>% 
    purrr::map_dfr(file_metadata_worker) %>% 
    as_tibble()
  
  return(df)
  
}


file_metadata_worker <- function(file, .progress) {
  
  # Get file basename
  file_basename <- basename(file)
  
  # Escape characters for bash
  file <- gsub("$", "\\$", file, fixed = TRUE)
  file <- gsub("`", "\\`", file, fixed = TRUE)
  
  # Build system command
  # Watch the different types of quotes here
  command <- stringr::str_c("exiftool -json ", '"', file, '"')
  
  # Use system command
  string <- system(command, intern = TRUE)
  
  # Split string into variable and value
  df <- jsonlite::fromJSON(string)
  
  # If there are duplicated variables, append a suffix
  names(df) <- str_to_underscore(names(df))
  names(df) <- make.names(names(df), unique = TRUE)
  
  return(df)
  
}


detect_exiftool <- function() {
  
  # Test
  text <- suppressWarnings(system("which exiftool", intern = TRUE, ignore.stderr = TRUE))
  
  # Raise error if not installed
  if (length(text) == 0 || !grepl("exiftool", text)) {
    stop("'exiftool' system programme not detected...", call. = FALSE)
  }
  
  # No return
  
}

#  This also from Stuart K. Grange - 
#  see https://rdrr.io/github/skgrange/threadr/src/R/str_functions.R

str_to_underscore <- function(x) {
  
  x <- gsub("([A-Za-z])([A-Z])([a-z])", "\\1_\\2\\3", x)
  x <- gsub(".", "_", x, fixed = TRUE)
  x <- gsub(":", "_", x, fixed = TRUE)
  x <- gsub("\\$", "_", x)
  x <- gsub(" ", "_", x)
  x <- gsub("__", "_", x)
  x <- gsub("([a-z])([A-Z])", "\\1_\\2", x)
  x <- stringr::str_to_lower(x)
  x <- stringr::str_trim(x)
  return(x)
  
}

# ================================================


#' @name   set_project_dir
#' @title  Attempt to set project directory within Rmd files
#' @author Gene Leynes and Scott Rodgers
#'
#' @param project_name   Directory name of the current project
#'
#' @description
#' 		Used within Knitr (Rmd) files when knitting
#' 		
#' @details
#' 		Used within Knitr (Rmd) files when the report file is not at the top
#' 		level of the project.  It changes the directory to be up one level
#' 		while knitting the report so that it can find directories like
#' 		./data
#' 
#' ## 2014-03-31   SJR   Extracted this function from 00_Initialize.R
#' ## 2014-05-05   GWL   Adding function to geneorama package
#' ##
#' 
#' ##
#' ## Usage: Call this function within Rmd files if they are located higher than 
#' ##        the project root directory. This will navigate up one directory until
#' ##        it reaches either the root directory or "project_name"
#' ##
#' 

set_project_dir <- function(project_name){
  while (basename(getwd()) != project_name &&
         basename(getwd()) != basename(normalizePath(".."))){
    setwd("..")
  }
}

