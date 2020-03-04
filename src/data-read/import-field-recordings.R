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

# ----------------------------------------------------
