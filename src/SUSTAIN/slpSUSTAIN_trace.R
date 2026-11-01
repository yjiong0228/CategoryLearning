.prob.response <- function(C.out, decision.consistency) {
  prob.denom <- sum(exp(decision.consistency * C.out))  # denominator
  prob.nom <- exp(decision.consistency * C.out)  # nominator
  prob <-  prob.nom / prob.denom
  return(prob)
}

## Probability of endorsement (Eq. A11)

.prob.endorse <- function(R, threshold) {
  out <- R / (R + threshold)
  return(out)
}

## Calculating stimulus distance from a cluster (Eq. 4)
## AW: OK, 2018-03-21
.calc.distances <- function(input, cluster, fac.dims, fac.na) {
  mu <- matrix(0, nrow = nrow(cluster),
               ncol = length(unique(fac.dims)))
  for (k in 1:nrow(cluster)) {
    mu[k, ] <- as.vector(tapply(abs(input - cluster[k, fac.na]), fac.dims,
                                sum)) * 0.5 ## Equation 4
  }
  return(mu)
}


## Utility function for selecting a random cluster between those
## who have tied activations

.random <- function(act) {
  tmp <- which(act == max(act))
  ret <- tmp[sample(length(tmp), 1)]
  return(ret)
}

## Calculating cluster activation and related values (Eq.5, 6, A6)
## act - Activation of each cluster (Eq. 5)
## out - Activations after cluster competition (Eq. 6)
## rec - Recognition score from A6 equation in Appendix in Love and Gureckis (2007)

## mu.lambda - Product of mu and lambda, calculated within function
## but also needed in later calculation, so returned.

## AW: OK, 2018-03-21
.cluster.activation <- function(lambda, r, beta, mu, ties) {
  mu.lambda <- sweep(mu, MARGIN = 2, -lambda, `*`)
  nom <- sweep(exp(mu.lambda), MARGIN = 2, lambda ^ r, `*`)
  act <- apply(nom, MARGIN = 1, sum) / sum(lambda ^ r) # Equation 5
  out <- (act ^ beta / sum(act^beta)) * act # Equation 6
  rec <- sum(out) # Equation A6
  switch(
    ties,
    "random" = winner <- .random(act),
    "first" = winner <- which.max(act)
  )
  out[-winner] <- 0 # For all other non-winning clusters = 0
  clus <- list("act" = act,
               "out" = out,
               "rec" = rec,
               "mu.lambda" = mu.lambda,
               "winner" = winner)
  return(clus)
}



slpSUSTAIN_trace <- function (st, tr, xtdo = FALSE, ties = "random") 
{
  tr <- data.matrix(tr)
  
  colskip <- st$colskip + 1
  lambda <- st$lambda
  w <- st$w
  cluster <- st$cluster
  maxcat <- st$maxcat
  threshold <- st$k
  if (is.null(maxcat)) 
    maxcat <- 0
  if (is.null(threshold)) 
    threshold <- 0.19
  fac.dims <- rep(seq_along(st$dims), st$dims)
  fac.na <- seq(sum(st$dims))
  sim <- NULL
  xout <- rep(0, nrow(tr))
  activations <- rep(0, nrow(tr))
  prob.o <- NULL
  rec <- rep(0, nrow(tr))
  endorse <- rep(0, nrow(tr))
  if (length(st$dims) != length(lambda)) {
    stop(paste("length of dimensions mismatch. \n", "dim, lambda, w, ", 
               "and cluster are not of equal lengths.\n"), call. = TRUE)
  }
  
  
  cluster_hist <- vector("list", nrow(tr))
  lambda_hist <- matrix(NA_real_, nrow = nrow(tr), ncol = length(lambda))
  colnames(lambda_hist) <- paste0("lambda_dim", seq_along(lambda))
  w_hist <- vector("list", nrow(tr))
  
  for (i in 1:nrow(tr)) {
    trial <- tr[i, ]
    if (trial["ctrl"] %in% c(0, 1, 2)) {
      fac.queried <- seq(sum(st$dims) + 1, ncol(tr) - 
                           colskip + 1)
    }
    else {
      fac.queried <- fac.na
    }
    if (i == 1 && trial["ctrl"] %in% 0:2 && max(fac.queried) + 
        colskip - 1 != ncol(tr)) {
      stop("colskip is not correct.\nCheck number of optional columns!", 
           call. = TRUE)
    }
    input <- as.vector(trial[colskip:(colskip + sum(st$dims) - 
                                        1)])
    if (trial["ctrl"] %in% c(1, 4)) {
      w <- st$w
      lambda <- st$lambda
      cluster <- st$cluster
      if (length(cluster) == 1) {
        if (is.na(cluster)) {
          cluster <- matrix(as.vector(trial[colskip:(ncol(tr))]), 
                            nrow = 1)
        }
      }
      if (length(w) == 1) {
        if (is.na(w)) {
          w <- matrix(rep(0, ncol(cluster)), nrow = 1)
        }
      }
    }
    mu <- .calc.distances(input, cluster, fac.dims, fac.na)
    c.act <- .cluster.activation(lambda, st$r, st$beta, 
                                 mu, ties)
    C.out <- w[which.max(c.act$act), ] * c.act$out[which.max(c.act$act)]
    
    if (trial["ctrl"] %in% c(0, 1, 2)) {
      prob.r <- .prob.response(C.out[fac.queried], st$d)
    }
    else {
      prob.r <- .prob.response(C.out, st$d)
    }
    target <- as.vector(trial[colskip:(length(trial))])
    target[target == 1] <- pmax(C.out[which(target == 1)], 
                                1)
    target[target == 0] <- pmin(C.out[which(target == 0)], 
                                0)
    new.cluster <- FALSE
    exist.cluster <- FALSE
    
    if (trial["ctrl"] == 0) {
      t.queried <- which.max(C.out[fac.queried])
      if (target[fac.queried][t.queried] < 1) 
        new.cluster <- TRUE
      in.order <- C.out[fac.queried][order(C.out[fac.queried])]
      if (in.order[1] == in.order[2]) 
        new.cluster <- TRUE
    }
    if (trial["ctrl"] == 3 & maxcat > 0) {
      if (max(c.act$act) < st$tau & maxcat > nrow(cluster)) 
        new.cluster <- TRUE
    }
    else if (trial["ctrl"] == 3 & maxcat == 0) {
      if (max(c.act$act) < st$tau) 
        new.cluster <- TRUE
    }
    if (new.cluster) {
      sim <- rowSums(abs(sweep(cluster, 2, as.vector(trial[colskip:(length(trial))]), 
                               FUN = "-")))
      if (any(sim == 0)) {
        new.cluster <- FALSE
        exist.cluster <- TRUE
      }
    }
    if (new.cluster == TRUE) {
      cluster <- rbind(cluster, as.vector(trial[colskip:(length(trial))]))
      w <- rbind(w, rep(0, length(st$w)))
      mu <- rbind(mu, vector(mode = "numeric", length = length(st$dims)))
      c.act <- .cluster.activation(lambda, st$r, st$beta, 
                                   mu, ties)
    }
    ifelse(exist.cluster == TRUE, win <- which(sim == 0)[1], 
           win <- c.act$winner)
    
    if (trial["ctrl"] %in% c(0, 1, 3, 4)) {
      cluster[win, fac.na] <- cluster[win, fac.na] + (st$eta * 
                                                        (input - cluster[win, fac.na]))
      lambda <- lambda + (st$eta * exp(c.act$mu.lambda[win, 
      ]) * (1 + c.act$mu.lambda[win, ]))
      w[win, fac.queried] <- w[win, fac.queried] + (st$eta * 
                                                      (target[fac.queried] - C.out[fac.queried]) * 
                                                      c.act$out[win])
    }
    xout[i] <- win
    activations[i] <- c.act$out[win]
    prob.o <- rbind(prob.o, prob.r)
    rec[i] <- c.act$rec
    endorse[i] <- .prob.response(c.act$rec, threshold)
    
    cluster_hist[[i]] <- cluster
    lambda_hist[i, ] <- lambda
    w_hist[[i]] <- w
  }
  
  rownames(prob.o) <- NULL
  if (xtdo) {
    extdo <- cbind(probabilities = prob.o, winning = xout, 
                   activation = activations, recognition_score = rec, 
                   endorsement = endorse)
  }
  if (xtdo) {
    ret <- list(xtdo = extdo, lambda = lambda, lambda_hist  = lambda_hist,
                w = w, w_hist = w_hist,
                cluster = cluster, cluster_hist = cluster_hist)
  }
  else {
    ret <- list(probs = prob.o, lambda = lambda, w = w, 
                cluster = cluster)
  }
  return(ret)
}
