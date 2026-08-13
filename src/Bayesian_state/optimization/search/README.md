# 参数搜索

本包包含共享的被试级搜索执行框架，以及 coordinate descent 和 exhaustive grid search
两种搜索算法。Objective 定义和候选构造保留在上一层，因为它们是多种搜索算法共享的输入，
本身并不是搜索算法。
