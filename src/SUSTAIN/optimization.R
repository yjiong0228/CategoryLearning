## Set-up ---------------------------------------------------------------------
rm(list = ls()) ## Clear all variables in the workspace


library(dplyr)
library(catlearn)
library(DEoptim)
library(parallel)




# data ------------------------------------------------------------
data <- read.csv("Task2_processed.csv")

df <- data %>% 
  select(condition, iSub, feature1, feature2, feature3, feature4, category, choice, feedback)

for (iCon in 1:3) {
  df_con <- df %>% 
    filter(condition == iCon)
  
  if (iCon == 1) {
    # iCon=1：category 1/2→1，3/4→2 → 生成c1/c2
    df_con <- df_con %>% 
      mutate(category = case_when(
        category %in% c(1, 2) ~ 1,
        category %in% c(3, 4) ~ 2
      ),
      c1 = as.integer(category == 1),
      c2 = as.integer(category == 2)
      )
  }
  else {
    # iCon=2/3：category 1/2/3/4 → 生成c1/c2/c3/c4
    df_con <- df_con %>% 
      mutate(
        c1 = as.integer(category == 1),
        c2 = as.integer(category == 2),
        c3 = as.integer(category == 3),
        c4 = as.integer(category == 4)
      )
  }
  
  write.csv(df_con, file = paste0("df_con", iCon, ".csv"), row.names = FALSE)
}


## Fitting SUSTAIN function ------------------------------------------------------------
fitSUSTAIN <- function(par, tr, choice_vec) {
  library(catlearn)
  
  st.all <- list(
    # 4 个连续维度，每个维度一个“槽”
    dims   = c(1, 1, 1, 1),
    lambda = c(1, 1, 1, 1),
    
    # 这些都跟 trial 类型有关
    tau    = 0,      # 只有无监督学习才用，这里全是监督就设 0
    maxcat = 0,      # 不限制 cluster 数
    
    # 初始 cluster 和权重，按文档设为 NA，让函数自己初始化
    w       = NA,
    cluster = NA,
    
    # 没有 optional 列 → colskip = 1
    # 注意：这是文档要求的值，内部会再 +1
    colskip = 1,
    
    # 4 个你要拟合的自由参数
    r    = par[1],
    beta = par[2],
    d    = par[3],
    eta  = par[4]
  )
  
  tr_mat <- data.matrix(tr)

  out.list <- slpSUSTAIN(st = st.all, tr = tr_mat, xtdo = FALSE)
  
  probs <- as.matrix(out.list$probs)  # nTrial × K
  
  # 安全检查：行数要和 choice_vec 长度一致
  if (nrow(probs) != length(choice_vec)) {
    stop("长度不一致：nrow(probs) != length(choice_vec)")
  }
  
  # choice_vec 取值应该是 1..K，对应 probs 的列索引
  # 构造 (行, 列) 的索引矩阵
  idx <- cbind(seq_along(choice_vec), choice_vec)
  
  # 模型对被试真实选择的概率
  prob_chosen <- probs[idx]
  
  # 数值稳定性
  prob_chosen[prob_chosen < 1e-10] <- 1e-10
  
  # 返回负 log-likelihood
  -sum(log(prob_chosen))
}



## optimization------------------------------------------------------------

for (iCon in 1:3) {
  df_con <- read.csv(paste0("df_con", iCon, ".csv"))
  
  sub_ids <- sort(unique(df_con$iSub))
  
  for (sub in sub_ids) {
    df_sub <- df_con %>%
      filter(iSub == sub)
    
    # 构造 ctrl
    tr_tmp <- df_sub %>%
      mutate(ctrl = 0L) %>%
      mutate(ctrl = replace(ctrl, 1, 1L))  # 第一 trial ctrl = 1
    
    # 构造 SUSTAIN 训练矩阵
    if (iCon == 1) {
      tr_sustain <- tr_tmp %>%
        select(ctrl, feature1, feature2, feature3, feature4, c1, c2, choice, feedback) %>%
        data.matrix()
    } else {
      tr_sustain <- tr_tmp %>%
        select(ctrl, feature1, feature2, feature3, feature4, c1, c2, c3, c4, choice, feedback) %>%
        data.matrix()
    }
    
    dir.create("data", showWarnings = FALSE)
    write.csv(tr_sustain, file = paste0("data/tr_sub", sub, ".csv"), row.names = FALSE)
      
    # choice_vec <- df_sub$choice
    # 
    # cl <- makeCluster(detectCores() - 1)  # 留1个核心给系统，避免卡死
    # clusterExport(cl, c("fitSUSTAIN", "tr_sustain", "choice_vec"))
    # 
    # paramEvolve <- DEoptim(fitSUSTAIN,
    #                        tr = tr_sustain[, 1:(ncol(tr_sustain)-2)],
    #                        choice_vec = tr_sustain[, ncol(tr_sustain)-1],
    #                        lower = c(0, 0, 0, 0),
    #                        upper = c(10, 10, 30, 1),
    #                        control = list(parallelType = 1, itermax = 1000,
    #                                       c = 0.5, strategy = 6, p = 0.30))
    # 
    # dir.create("results", showWarnings = FALSE)
    # save(paramEvolve, file = sprintf("results/parameters_con%d_sub%d.RData", iCon, sub))
  }
}

stopCluster(cl)
