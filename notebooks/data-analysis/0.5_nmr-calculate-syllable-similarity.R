
# Code to calculate acoustic distances, after Keen et al 2021

# Settings and paths --------------------------------------
# # clear the R workspace
rm(list = ls())
x <- gc()
x <- c(
    "parallel",
    "vegan",
    "bioacoustics",
    "warbleR" ,
    "ggplot2",
    "Rraven",
    "cluster",
    "randomForest",
    "MASS",
    "fossil",
    "pbapply",
    "adehabitatHR",
    "Sim.DiffProc",
    "caret",
    "e1071",
    "tidyverse"
  )

# load packages 
aa <- lapply(x, function(y) {
  if (!y %in% installed.packages()[, "Package"])  {
    install.packages(y)
  }
  try(require(y, character.only = T), silent = T)
})

# Paths
root.dir = rprojroot::find_rstudio_root_file()
dataset_name <-  "GRETI_HQ_2020_notes"
data_dir <-  file.path(root.dir, "data", "processed", dataset_name, 'WAV')
out_dir = file.path(root.dir, "data", 'note_dfs', dataset_name)

est.file.name <- file.path(out_dir, 'selection_table.RDS')
specs_dir <-  file.path(out_dir, 'spectrograms')

if (!dir.exists(specs_dir)) {
  dir.create(specs_dir, recursive = TRUE)
}

ncores = parallel::detectCores() - 1

# WarbleR options
warbleR_options(
  wav.path = data_dir,
  f = 32000,
  wl = 512,
  flim = c(1.2, 8),
  ovlp = 95,
  bp = c(1.2, 8),
  parallel = ncores
)

# Make selection table
wi <- wav_info()
summary(wi)
est <- selection_table(whole.recs = T, extended = T, confirm.extended = F)
save(est,file=est.file.name)

# Test subset ====================
# est = est %>% slice(1:20)
# Test subset 


# Prepare acoustic measures  ------------------------

# Measure spectral parameters
sp <- specan(est, parallel=ncores, threshold = 15)
# Spectrogram cross-correlation
xc <- xcorr(est, bp=c(1,5), parallel = ncores)
# MDS for cross-correlation 
xc.mds <- cmdscale(1 - xc, k = 5)
# Translate MDS output into 5-D coordinates that we'll use as features
colnames(xc.mds) <- paste0("xc.dim.", 1:5)
# Dynamic time warping of frequency contours
dtw.dist <- dfDTW(est, pb=TRUE, parallel = ncores, img=FALSE, threshold = 15)
# MDS on DTW distance 
dtw.mds <- cmdscale(dtw.dist, k = 5)
# Translate DTW MDS into 5-D coordinates we'll use as features
colnames(dtw.mds) <- paste0("dtw.dim.", 1:5)
# Get cepstral coefficients and descriptive statistics
cps.cf <- mfcc_stats(est, parallel = ncores)
# put data and features together 
prms <- data.frame(est[, c("sound.files", "Call.Type")], sp[, -c(1:4)], xc.mds, dtw.mds, cps.cf[, -c(1:2)])

## save acoustic parms so we can just load these later
saveRDS(prms, file.path(out_dir,"acoustic_parameters.RDS"))
write.csv(prms,file.path(out_dir,"acoustic_parameters.csv"), row.names = FALSE)


# Scale, centre, remove corr, merge and save ------------------------

parameters <-
  read.csv(file.path(out_dir, "acoustic_parameters.csv"),
           stringsAsFactors = FALSE)

# Create dataframe that has all feature measurements
acous.meas <- parameters

# Remove colinear, boxcox transform and scale/center
# Remove any columns that have filenames, etc when doing this (here this is just column 1)
preparameters <-
  preProcess(acous.meas[,-c(1)], method = c("center", "scale", "BoxCox", "corr"))
parameters <- predict(preparameters, acous.meas)
nums <- unlist(lapply(parameters, is.numeric))
parameters <- parameters[, nums]

# Remove highly correlated vars
cm <- cor(parameters, use = "pairwise.complete.obs")
high.corr <- findCorrelation(cm, cutoff = .95)
print("Removed colinear parameters (r > 0.95)")
names(parameters)[high.corr]
parameters <-
  parameters[,!names(parameters) %in% names(parameters)[high.corr]]

# Prepare syllable df ------------------------------------------------
# (join individual note information)
parameters = parameters %>% add_column(note = acous.meas[, 1], .before = 1)
parameters$note <-
  lapply(parameters$note, function(x)
    gsub("\\..*", "", x))

df_list = list()
for (i in 0:2) {
  df_list[[as.character(i)]] = parameters %>% filter(str_detect(note, paste0(i, "$"))) %>% 
    rename_at(vars(-note), ~ paste(.x, as.character(i), sep = "_")) %>% 
    mutate(note = gsub('-[^-]*$', '', note))
}

parameters =  df_list %>% reduce(left_join, by = "note") %>% drop_na() # There shouldn't be any missing values, but will check later just in case.

# Read csv containing syllable metadata
metadata = read.csv(file.path(out_dir, paste0(dataset_name, ".csv")), stringsAsFactors = FALSE) %>% 
  select(key, silence_1, silence_2) %>% mutate(note = gsub('-[^-]*$', '', key)) %>% 
  distinct(note, .keep_all = TRUE) %>% select(-key)
# Join both dfs
parameters = left_join(parameters, metadata)

# Save this file so we can skip these steps later
out_name <- file.path(out_dir,"transformed_noncolinear_param.csv")
write.csv(parameters, out_name, row.names = FALSE)


# Load measurements -----------------------------------------------
# Load data if previous step already done
out_name <- file.path(out_dir, "transformed_noncolinear_param.csv")
parameters <- read.csv(out_name, stringsAsFactors = FALSE)


# Random Forest ---------------------------------------------------
URF_output = randomForest(parameters[,-c(1)], proximity=T, ntree = 10000)
# Save random forest so we can easily load it later
rf_out <- file.path(out_dir, "syllables_URF.rda")
save(URF_output, file = rf_out)

