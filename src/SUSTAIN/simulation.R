## load packages
rm(list = ls())

library(dplyr)
library(catlearn)


source("slpSUSTAIN_trace.R") 


sub_ids <- c(1,4,7,10,13,16,19,22)

for (sub in sub_ids) {
  tr_sustain <- read.csv(paste0("data/tr_sub", sub, ".csv"))
  
  ## import best fitting parameters  
  load(sprintf("results/parameters_con%d_sub%d.RData", 1, sub))
  
  par <- paramEvolve$optim$bestmem
  
  st <- list(lambda = c(1, 1, 1, 1),           # tuning of receptive fields
             tau = 0.0,                  # unsupervised threshold
             dims = c(1, 1, 1, 1),
             w = NA,
             cluster = NA,
             colskip = 1,
             r = par[1],                 # attentional focus
             beta = par[2],              # lateral inhibition
             d = par[3],                 # decisiton consistency
             eta = par[4])               # learning rate  
  
  
  simData <- slpSUSTAIN_trace(st, tr_sustain, xtdo = TRUE) 
  
  save(simData, file = sprintf("results/simulations_con%d_sub%d.RData", 1, sub))  
  
}



######################
# slp_src <- getFromNamespace("slpSUSTAIN", "catlearn")
# slp_src




