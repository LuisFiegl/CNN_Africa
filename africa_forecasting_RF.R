library(tidyverse)
library(mlr3verse)
library(ranger)
setwd("/dss/dssfs04/lwp-dss-0002/pn98na/pn98na-dss-0000/racek/CNNs_for_conflict_pred/Syria")
est_df = readRDS(paste0("data/est_df_", cell_size ,".rds"))


est_df = est_df %>% 
  #mutate(log_pop = log(pop)) %>% 
  mutate(ucdp1_log_lag_1 = log(ucdp1_lag_1 + 1), ucdp2_log_lag_1 = log(ucdp2_lag_1 + 1), 
         ucdp1_log_12lags = log(ucdp1_12lags + 1), ucdp2_log_12lags = log(ucdp2_12lags + 1)) %>% 
  select(id, time_point, intersect_area, long, lat, pop,
         landcover_grass_shrub, landcover_bare, landcover_built, landcover_crop, landcover_tree, landcover_water,
         ucdp1_bin,
         ucdp1_log_lag_1, ucdp2_log_lag_1,
         ucdp1_log_12lags, ucdp2_log_12lags,
         #ucdp1_last_conflict, ucdp2_last_conflict,
         month, year
  ) %>% 
  mutate(obs_id = row_number()) %>% 
  relocate(obs_id)

write.csv(est_df %>% select(-pop, -starts_with(("landcover_"))), "data/cnn_est_df.csv")






task = as_task_classif(est_df, target = "ucdp1_bin", positive = "1")  
task$select(setdiff(task$feature_names, c("id", "time_point", "month", "year")))

rf_model = lrn("classif.ranger",predict_type = "prob")
rf_model$param_set$values = list(num.trees = 500, num.threads = 16)


train_ids = est_df %>% filter(year < 2018) %>% pull(obs_id)
test_ids = est_df %>% filter(year >= 2018) %>% pull(obs_id)

rf_model$train(task, row_ids = train_ids)

prediction_rf = rf_model$predict(task, row_ids = test_ids)

prediction_df = as.data.table(prediction_rf) %>% 
  rename(pred_rf = prob.1)


saveRDS(prediction_df, "results/prediction_df.rds")


prediction_rf$score(list(msr("classif.acc"),
                         msr("classif.auc"),
                         msr("classif.prauc")
))