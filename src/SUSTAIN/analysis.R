
library(dplyr)
library(ggplot2)
library(purrr)


sub_ids <- c(1,4,7,10,13,16,19,22)

# for (sub in sub_ids) {
  
  dir.create(paste0("plots/sub", sub), showWarnings = FALSE)
  
  load(sprintf("results/simulations_con%d_sub%d.RData", 1, sub))
  
  
  cluster_hist <- simData$cluster_hist
  nTrials <- length(cluster_hist)
  
  cluster_count_df <- data.frame(
    trial    = 1:nTrials,
    nCluster = sapply(cluster_hist, nrow)
  )
  
  p_ncluster <- ggplot(cluster_count_df, aes(x = trial, y = nCluster)) +
    geom_line() +
    geom_point(size = 0.8) +
    theme_minimal() +
    labs(
      x = "Trial",
      y = "Number of clusters",
      title = "SUSTAIN: number of clusters over time"
    )
  ggsave(paste0("plots/sub", sub, "/nclusters.png"), plot = p_ncluster, width = 6, height = 4, dpi = 300)
  
  
  ###############################

  cluster_hist <- simData$cluster_hist
  nTrials <- length(cluster_hist)
  
  traj_list <- map2(
    .x = cluster_hist,
    .y = 1:nTrials,
    .f = function(cl_mat, t) {
      cl_mat <- as.matrix(cl_mat)
      if (is.null(dim(cl_mat))) return(NULL)   # 没有 cluster 的情况
      
      nC <- nrow(cl_mat)
      data.frame(
        trial   = t,
        cluster = 1:nC,
        dim1    = cl_mat[, 1],
        dim2    = cl_mat[, 2],
        dim3    = cl_mat[, 3],
        dim4    = cl_mat[, 4]
      )
    }
  )
  
  cluster_traj <- bind_rows(traj_list)
  
  # 看一下结构
  # head(cluster_traj)
  
  
  ############################################################################################################
  # library(plotly)
  # 
  # # 一个画单个 cluster 的 3D 轨迹函数
  # plot_cluster_3d <- function(cluster_id) {
  #   df <- cluster_traj %>% 
  #     filter(cluster == cluster_id) %>% 
  #     arrange(trial)
  #   
  #   plot_ly(
  #     df,
  #     x = ~dim1,
  #     y = ~dim2,
  #     z = ~dim3,
  #     color = ~trial,              # 颜色按时间渐变
  #     colors = colorRamp(c("blue", "red")),  # 蓝→红，你可以换别的
  #     type = "scatter3d",
  #     mode = "lines+markers"
  #   ) %>%
  #     layout(
  #       title = paste("Cluster", cluster_id, "trajectory (dims 1–3)"),
  #       scene = list(
  #         xaxis = list(title = "Dim 1"),
  #         yaxis = list(title = "Dim 2"),
  #         zaxis = list(title = "Dim 3")
  #       )
  #     )
  # }
  # 
  # # 举例：画 cluster 1 的 3D 轨迹
  # plot_cluster_3d(1)
  
  
  
  ############################################################################################
  
  p1 <- ggplot(cluster_traj, aes(x = trial, y = dim1, color = trial, group = cluster)) +
    geom_line() +
    facet_wrap(~ cluster, scales = "free_y") +
    theme_minimal() +
    labs(
      x = "Trial",
      y = "Dim 1 value",
      color = "Trial",
      title = "Cluster dim1 trajectories over time (faceted by cluster)"
    ) +
    ylim(0, 1)
  ggsave(paste0("plots/sub", sub, "/clusters_dim1.png"), plot = p1, width = 6, height = 4, dpi = 300)
  
  
  p2 <- ggplot(cluster_traj, aes(x = trial, y = dim2, color = trial, group = cluster)) +
    geom_line() +
    facet_wrap(~ cluster, scales = "free_y") +
    theme_minimal() +
    labs(
      x = "Trial",
      y = "Dim 2 value",
      color = "Trial",
      title = "Cluster dim2 trajectories over time (faceted by cluster)"
    ) +
    ylim(0, 1)
  ggsave(paste0("plots/sub", sub, "/clusters_dim2.png"), plot = p2, width = 6, height = 4, dpi = 300)
  
  
  p3 <- ggplot(cluster_traj, aes(x = trial, y = dim3, color = trial, group = cluster)) +
    geom_line() +
    facet_wrap(~ cluster, scales = "free_y") +
    theme_minimal() +
    labs(
      x = "Trial",
      y = "Dim 3 value",
      color = "Trial",
      title = "Cluster dim3 trajectories over time (faceted by cluster)"
    ) +
    ylim(0, 1)
  ggsave(paste0("plots/sub", sub, "/clusters_dim3.png"), plot = p3, width = 6, height = 4, dpi = 300)
  
  p4 <- ggplot(cluster_traj, aes(x = trial, y = dim4, color = trial, group = cluster)) +
    geom_line() +
    facet_wrap(~ cluster, scales = "free_y") +
    theme_minimal() +
    labs(
      x = "Trial",
      y = "Dim 4 value",
      color = "Trial",
      title = "Cluster dim4 trajectories over time (faceted by cluster)"
    ) +
    ylim(0, 1)
  ggsave(paste0("plots/sub", sub, "/clusters_dim4.png"), plot = p4, width = 6, height = 4, dpi = 300)
  
  
  
  ######################################################################################################
  
  library(patchwork)
  
  final_cluster <- as.matrix(simData$cluster)
  
  cluster_df <- data.frame(
    cluster = factor(1:nrow(final_cluster)),  # cluster ID，方便标颜色或标注
    dim1 = final_cluster[, 1],
    dim2 = final_cluster[, 2],
    dim3 = final_cluster[, 3],
    dim4 = final_cluster[, 4]
  )
  
  # head(cluster_df)
  
  
  # 图 1：dim1 vs dim2
  p12 <- ggplot(cluster_df, aes(x = dim1, y = dim2, label = cluster)) +
    geom_point(size = 3) +
    geom_text(vjust = -0.8, size = 3) +  # 在点旁边标 cluster 编号
    coord_fixed(xlim = c(0, 1), ylim = c(0, 1)) +  # 坐标范围固定到 [0,1]
    theme_minimal() +
    labs(
      x = "Dim 1",
      y = "Dim 2"
    )
  
  # 图 2：dim1 vs dim3
  p13 <- ggplot(cluster_df, aes(x = dim1, y = dim3, label = cluster)) +
    geom_point(size = 3) +
    geom_text(vjust = -0.8, size = 3) +
    coord_fixed(xlim = c(0, 1), ylim = c(0, 1)) +
    theme_minimal() +
    labs(
      x = "Dim 1",
      y = "Dim 3"
    )
  
  # 图 3：dim1 vs dim4
  p14 <- ggplot(cluster_df, aes(x = dim1, y = dim4, label = cluster)) +
    geom_point(size = 3) +
    geom_text(vjust = -0.8, size = 3) +
    coord_fixed(xlim = c(0, 1), ylim = c(0, 1)) +
    theme_minimal() +
    labs(
      x = "Dim 1",
      y = "Dim 4"
    )
  
  # 3. 拼成一行显示 ---------------------------
                                                                                                                                                     
  p_finalpos <- p12 + p13 + p14 +
    plot_annotation(
      title = "Final cluster positions",
      theme = theme(plot.title = element_text(hjust = 0.5))  # hjust=0.5 居中
    )
  
  ggsave(paste0("plots/sub", sub, "/clusters_final.png"), plot = p_finalpos, width = 6, height = 4, dpi = 300)
  
  
  
  ####################################################################################
  
  sub_ids <- c(1,4,7,10,13,16,19,22)
  plot_list <- list()
  
  for (sub in sub_ids) {
  
    load(sprintf("results/simulations_con%d_sub%d.RData", 1, sub))
    
    lambda_raw <- simData$lambda_hist 
    rs <- rowSums(lambda_raw)
    
    # 避免出现除以 0（理论上不会，但保险一下）
    rs[rs == 0] <- NA  
    
    lambda_norm <- lambda_raw / rs                 # 每行除以行和 → 相加=1
    colnames(lambda_norm) <- paste0("lambda_dim", 1:ncol(lambda_norm))
    
    lambda_df <- as.data.frame(lambda_norm)
    lambda_df$trial <- seq_len(nrow(lambda_df))
    
    library(dplyr)
    library(tidyr)
    library(ggplot2)
    
    lambda_long <- lambda_df %>%
      tidyr::pivot_longer(
        cols = starts_with("lambda_dim"),
        names_to = "dimension",
        values_to = "lambda_norm"
      )
    
    p_att <- ggplot(lambda_long, aes(x = trial, y = lambda_norm, color = dimension)) +
      geom_line(linewidth = 1) +
      labs(
        title = paste0("S", sub),
        x = "Trial",
        y = "Attention weight"
      ) +
      theme_classic() +
      theme(
            legend.position = "none",  # 子图隐藏图例
            plot.title = element_text(size = 10, hjust = 0.5),  # 子图标题居中
            axis.title = element_text(size = 9),  # 轴标题缩小
            axis.text = element_text(size = 8)    # 轴刻度缩小
          )
    plot_list[[as.character(sub)]] <- p_att
  }
  
  # 所有被试拼图
  combined_plot <- wrap_plots(plot_list, nrow = 2, ncol = 4) +
    plot_annotation(
      title = "SUSTAIN normalized attention weight ",  # 总标题
      theme = theme(plot.title = element_text(size = 14, hjust = 0.5))
    ) +
    plot_layout(guides = "collect") &  # 收集所有图例并统一展示
    theme(
      legend.position = "right",  # 图例位置：right（右侧）/bottom（底部）
      legend.text = element_text(size = 9),
      legend.key.size = unit(0.8, "cm")
    )
  ggsave("plots/attention.png", plot = combined_plot, width = 12, height = 6, dpi = 300)
  
    
    
  ####################################################################################
  library(dplyr)
  library(purrr)
  library(tidyr)
  library(ggplot2)
  
  w_hist <- simData$w_hist
  nTrials <- length(w_hist)
  
  # 把每个 trial 的 w 矩阵展开
  w_list <- map2(
    .x = w_hist,
    .y = 1:nTrials,
    .f = function(w_mat, t) {
      if (is.null(w_mat)) return(NULL)
      w_mat <- as.matrix(w_mat)
      nC <- nrow(w_mat)
      # 这里只取第 5,6 列：对 c1, c2 的权重
      data.frame(
        trial   = t,
        cluster = 1:nC,
        w_c1    = w_mat[, 5],
        w_c2    = w_mat[, 6]
      )
    }
  )
  
  w_traj <- bind_rows(w_list)
  
  # head(w_traj)
  
  
  w_long <- w_traj %>%
    pivot_longer(
      cols      = c(w_c1, w_c2),
      names_to  = "category",
      values_to = "w"
    ) %>%
    mutate(category = ifelse(category == "w_c1", "Category 1", "Category 2"))
  
  p_w <- ggplot(w_long, aes(x = trial, y = w, color = category)) +
    geom_line() +
    facet_wrap(~ cluster, scales = "free_y") +
    theme_minimal() +
    labs(
      x = "Trial",
      y = "Association weight w",
      color = "Category",
      title = "Cluster–category association weights over time"
    )
  ggsave(paste0("plots/sub", sub, "/asso_weight.png"), plot = p_w, width = 6, height = 4, dpi = 300)

  
# }
  
  
  ####################################################################################
  library(zoo) 
  
  sub_ids <- c(1,4,7,10,13,16,19,22)
  plot_list <- list()
  
  for (sub in sub_ids) {

    tr_sustain <- read.csv(paste0("data/tr_sub", sub, ".csv"))
    
    true_cat <- ifelse(tr_sustain$c1 == 1, 1, 2)
    human_choice <- tr_sustain$choice
    human_correct <- as.integer(human_choice == true_cat)
    
    load(sprintf("results/simulations_con%d_sub%d.RData", 1, sub))
    
    xtdo <- simData$xtdo    # xtdo 是一个矩阵 / data.frame
    
    prob_mat  <- as.matrix(xtdo[, 1:2])
    model_p_correct <- prob_mat[cbind(seq_along(true_cat), true_cat)]
    
    ## 滑动平均
    k <- 16
    
    human_ma <- zoo::rollmean(human_correct, k = k, align = "right", fill = NA)
    model_ma <- zoo::rollmean(model_p_correct, k = k, align = "right", fill = NA)
    

    df_plot <- data.frame(
      trial         = seq_along(human_correct),
      human_correct = human_ma,
      model_correct = model_ma
    )
    
    df_long <- df_plot %>%
      pivot_longer(
        cols = c(human_correct, model_correct),
        names_to = "series",
        values_to = "accuracy"
      )
    
    ## 画图
    p_acc <- ggplot(df_long, aes(x = trial, y = accuracy, color = series)) +
      geom_line(linewidth = 1) +
      scale_color_manual(
        values = c(human_correct = "black", model_correct = "red"),
        labels = c("Human (16-trial MA)", "Model (16-trial MA)"),
        name   = NULL
      ) +
      coord_cartesian(ylim = c(0, 1)) +
      labs(
        title = paste0("S", sub),
        x = "Trial",
        y = "Accuracy"
      ) +
      theme_classic() +
      theme(
        legend.position = "none",  # 子图隐藏图例
        plot.title = element_text(size = 10, hjust = 0.5),  # 子图标题居中
        axis.title = element_text(size = 9),  # 轴标题缩小
        axis.text = element_text(size = 8)    # 轴刻度缩小
      )
    plot_list[[as.character(sub)]] <- p_acc
  }
  
  # 所有被试拼图
  combined_plot <- wrap_plots(plot_list, nrow = 2, ncol = 4) +
    plot_annotation(
      title = "Model vs Human Accuracy (16-trial Moving Average)",  # 总标题
      theme = theme(plot.title = element_text(size = 14, hjust = 0.5))
    ) +
    plot_layout(guides = "collect") &  # 收集所有图例并统一展示
    theme(
      legend.position = "right",  # 图例位置：right（右侧）/bottom（底部）
      legend.text = element_text(size = 9),
      legend.key.size = unit(0.8, "cm")
    )
  ggsave("plots/accuracy.png", plot = combined_plot, width = 12, height = 6, dpi = 300)
  
  
  ####################################################################################
  data <- read.csv("Task2_processed.csv")
  
  sub_ids <- c(1,4,7,10,13,16,19,22)
  plot_list <- list()
  
  for (sub in sub_ids) {
    df_sub <- data %>% 
      filter(iSub == sub)
    
    use_mat <- as.matrix(df_sub[, c("feature1_use", "feature2_use",
                                      "feature3_use", "feature4_use")])
    
    k <- 16  # 窗长
    
    use_ma <- apply(use_mat, 2, function(x) rollmean(x, k = k, align = "right", fill = NA))
    
    df_use <- data.frame(
      trial = seq_len(nrow(df_sub)),
      feature1_use_ma = use_ma[, 1],
      feature2_use_ma = use_ma[, 2],
      feature3_use_ma = use_ma[, 3],
      feature4_use_ma = use_ma[, 4]
    )
    
    df_use_long <- df_use %>%
      pivot_longer(
        cols = starts_with("feature"),
        names_to = "feature",
        values_to = "use_rate"
      )
    
    p_use <- ggplot(df_use_long, aes(x = trial, y = use_rate, color = feature)) +
      geom_line(linewidth = 1) +
      coord_cartesian(ylim = c(0, 1)) +
      labs(
        title = paste0("S", sub),
        x = "Trial",
        y = "Use rate"
      ) +
      theme_classic() +
      theme(
        legend.position = "none",  # 子图隐藏图例
        plot.title = element_text(size = 10, hjust = 0.5),  # 子图标题居中
        axis.title = element_text(size = 9),  # 轴标题缩小
        axis.text = element_text(size = 8)    # 轴刻度缩小
      )
    plot_list[[as.character(sub)]] <- p_use
  }
  
  # 所有被试拼图
  combined_plot <- wrap_plots(plot_list, nrow = 2, ncol = 4) +
    plot_annotation(
      title = "Oral feature use (16-trial Moving Average)",  # 总标题
      theme = theme(plot.title = element_text(size = 14, hjust = 0.5))
    ) +
    plot_layout(guides = "collect") &  # 收集所有图例并统一展示
    theme(
      legend.position = "right",  # 图例位置：right（右侧）/bottom（底部）
      legend.text = element_text(size = 9),
      legend.key.size = unit(0.8, "cm")
    )
  ggsave("plots/feature_use.png", plot = combined_plot, width = 12, height = 6, dpi = 300)
  
  
  ####################################################################################
  
  sub_ids <- c(1,4,7,10,13,16,19,22)
  plot_list <- list()
  
  for (sub in sub_ids) {
  
    oral_ent <- rep(NA, nrow(use_ma))
    for (t in seq_len(nrow(use_ma))) {
      v <- use_ma[t, ]
      if (all(is.na(v))) next
      
      if (sum(v, na.rm = TRUE) == 0) next  # 该窗口完全没汇报任何特征 → 跳过
      
      p <- v / sum(v, na.rm = TRUE)
      p <- p[p > 0]      # 避免 log(0)
      oral_ent[t] <- -sum(p * log(p))
    }
    
    model_ent <- rep(NA, nrow(lambda_df))
    for (t in seq_len(nrow(lambda_df))) {
      v <- lambda_df[t, ]
      if (all(is.na(v))) next
      
      if (sum(v, na.rm = TRUE) == 0) next
      
      p <- v / sum(v, na.rm = TRUE)
      p <- p[p > 0]
      model_ent[t] <- -sum(p * log(p))
    }
    
    ## 整合成一个 dataframe
    df_ent <- data.frame(
      trial     = seq_len(nrow(use_ma)),
      H_oral    = oral_ent,
      H_model   = model_ent
    )
    df_ent_cc <- df_ent %>% filter(!is.na(H_oral), !is.na(H_model))
    
    ## 相关系数
    cor_val <- cor(df_ent_cc$H_oral, df_ent_cc$H_model, method = "pearson")
    
    ## 画图
    p_scatter <- ggplot(df_ent_cc, aes(x = H_oral, y = H_model)) +
      geom_point(alpha = 0.6) +
      geom_smooth(method = "lm", se = TRUE, linewidth = 1) +
      labs(
        title = sprintf("Sub %d: , r = %.3f", sub, cor_val),
        x = "H(oral feature use, 16-trial MA)",
        y = "H(SUSTAIN attention, 16-trial MA)"
      ) +
      theme_classic() +
      theme(
        legend.position = "none",  # 子图隐藏图例
        plot.title = element_text(size = 10, hjust = 0.5),  # 子图标题居中
        axis.title = element_text(size = 9),  # 轴标题缩小
        axis.text = element_text(size = 8)    # 轴刻度缩小
      )
    plot_list[[as.character(sub)]] <- p_scatter
  }
  
  # 所有被试拼图
  combined_plot <- wrap_plots(plot_list, nrow = 2, ncol = 4) +
    plot_annotation(
      title = "Entropy(oral) vs Entropy(model)",  # 总标题
      theme = theme(plot.title = element_text(size = 14, hjust = 0.5))
    ) +
    plot_layout(guides = "collect") &  # 收集所有图例并统一展示
    theme(
      legend.position = "right",  # 图例位置：right（右侧）/bottom（底部）
      legend.text = element_text(size = 9),
      legend.key.size = unit(0.8, "cm")
    )
  ggsave("plots/correlation.png", plot = combined_plot, width = 12, height = 6, dpi = 300)
  
  
  
  ####################################################################################
  
  data <- read.csv("Task2_processed.csv")
  
  sub_ids <- c(1,4,7,10,13,16,19,22)
  cat1_plot_list <- list()
  cat2_plot_list <- list()
  
  for (sub in sub_ids) {

    load(sprintf("results/simulations_con%d_sub%d.RData", 1, sub))
    
    xtdo         <- simData$xtdo
    cluster_hist <- simData$cluster_hist
    winning      <- xtdo[, "winning"]
    nTrials      <- length(winning)
    
    # 取 probabilities.* 列，算模型在该 trial 选的是 1 还是 2
    prob_mat     <- as.matrix(xtdo[, 1:2])
    model_choice <- max.col(prob_mat)             # 1 或 2
    
    # cluster 中前 4 列是四个 feature 维度坐标
    fac_na <- 1:4
    
    # 2) 每个 trial 的 winning cluster 在4个维度上的坐标
    win_pos <- matrix(NA, nTrials, 4)
    for (t in 1:nTrials) {
      clus_mat <- cluster_hist[[t]]
      win_idx  <- winning[t]
      win_pos[t, ] <- clus_mat[win_idx, fac_na]
    }
    colnames(win_pos) <- paste0("cluster_dim", 1:4)
    
    # 3) 组织成长格式，并给每个 trial 标记它属于哪个 category（模型选择）
    df_model <- data.frame(
      trial        = 1:nTrials,
      model_choice = model_choice,
      cluster_dim1 = win_pos[, 1],
      cluster_dim2 = win_pos[, 2],
      cluster_dim3 = win_pos[, 3],
      cluster_dim4 = win_pos[, 4]
    )
    
    df_model_long <- df_model %>%
      pivot_longer(
        cols = starts_with("cluster_dim"),
        names_to = "dim",
        values_to = "coord"
      ) %>%
      mutate(
        dim = factor(dim,
                     levels = c("cluster_dim1","cluster_dim2","cluster_dim3","cluster_dim4"),
                     labels = c("Feature 1","Feature 2","Feature 3","Feature 4")),
        category = factor(model_choice,
                          levels = c(1, 2),
                          labels = c("Category 1", "Category 2"))
      )
    
    
    df_sub <- data %>% 
      filter(iSub == sub)
    
    oral_df <- data.frame(
      trial        = 1:nrow(df_sub),
      choice       = df_sub$choice,
      oral_dim1    = df_sub$feature1_oral,
      oral_dim2    = df_sub$feature2_oral,
      oral_dim3    = df_sub$feature3_oral,
      oral_dim4    = df_sub$feature4_oral
    )
    
    df_oral_long <- oral_df %>%
      pivot_longer(
        cols = starts_with("oral_dim"),
        names_to = "dim",
        values_to = "coord"
      ) %>%
      mutate(
        dim = factor(dim,
                     levels = c("oral_dim1","oral_dim2","oral_dim3","oral_dim4"),
                     labels = c("Feature 1","Feature 2","Feature 3","Feature 4")),
        category = factor(choice,
                          levels = c(1, 2),
                          labels = c("Category 1", "Category 2"))
      )
    
    df_model_long2 <- df_model_long %>%
      mutate(source = "Model")
    
    df_oral_long2 <- df_oral_long %>%
      mutate(source = "Oral")
    
    df_cat1 <- bind_rows(
      df_model_long2 %>% filter(category == "Category 1"),
      df_oral_long2  %>% filter(category == "Category 1")
    )
    
    df_cat2 <- bind_rows(
      df_model_long2 %>% filter(category == "Category 2"),
      df_oral_long2  %>% filter(category == "Category 2")
    )
    
    p_cat1 <- ggplot(df_cat1, aes(x = trial, y = coord, color = source)) +
      geom_line(linewidth = 0.9, na.rm = TRUE) +
      facet_wrap(~ dim, ncol = 2, scales = "free_y") +
      scale_color_manual(
        values = c("Model" = "darkgreen", "Oral" = "black"),
        name   = NULL
      ) +
      labs(
        title = sprintf("S%d", sub),
        x = "Trial",
        y = "Winning cluster coordinate"
      ) +
      theme_classic() +
      theme(
        legend.position = "none",  # 子图隐藏图例
        plot.title = element_text(size = 10, hjust = 0.5),  # 子图标题居中
        axis.title = element_text(size = 9),  # 轴标题缩小
        axis.text = element_text(size = 8)    # 轴刻度缩小
      )
    
    p_cat2 <- ggplot(df_cat2, aes(x = trial, y = coord, color = source)) +
      geom_line(linewidth = 0.9, na.rm = TRUE) +
      facet_wrap(~ dim, ncol = 2, scales = "free_y") +
      scale_color_manual(
        values = c("Model" = "darkgreen", "Oral" = "black"),
        name   = NULL
      ) +
      labs(
        title = sprintf("S%d", sub),
        x = "Trial",
        y = "Oral coordinate"
      ) +
      theme_classic() +
      theme(
        legend.position = "none",  # 子图隐藏图例
        plot.title = element_text(size = 10, hjust = 0.5),  # 子图标题居中
        axis.title = element_text(size = 9),  # 轴标题缩小
        axis.text = element_text(size = 8)    # 轴刻度缩小
      )
    
    cat1_plot_list[[as.character(sub)]] <- p_cat1
    cat2_plot_list[[as.character(sub)]] <- p_cat2
  }
  
  # 所有被试拼图
  cat1_combined_plot <- wrap_plots(cat1_plot_list, nrow = 2, ncol = 4) +
    plot_annotation(
      title = "Category 1\nModel winning cluster vs Oral coordinates",  # 总标题
      theme = theme(plot.title = element_text(size = 14, hjust = 0.5))
    ) +
    plot_layout(guides = "collect") &  # 收集所有图例并统一展示
    theme(
      legend.position = "right",  # 图例位置：right（右侧）/bottom（底部）
      legend.text = element_text(size = 9),
      legend.key.size = unit(0.8, "cm")
    )
  ggsave("plots/cat1_aling.png", plot = cat1_combined_plot, width = 12, height = 6, dpi = 300)
  
  cat2_combined_plot <- wrap_plots(cat2_plot_list, nrow = 2, ncol = 4) +
    plot_annotation(
      title = "Category 2\nModel winning cluster vs Oral coordinates",  # 总标题
      theme = theme(plot.title = element_text(size = 14, hjust = 0.5))
    ) +
    plot_layout(guides = "collect") &  # 收集所有图例并统一展示
    theme(
      legend.position = "right",  # 图例位置：right（右侧）/bottom（底部）
      legend.text = element_text(size = 9),
      legend.key.size = unit(0.8, "cm")
    )
  ggsave("plots/cat2_aling.png", plot = cat2_combined_plot, width = 12, height = 6, dpi = 300)
  