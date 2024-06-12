library(tidyverse)
library(mlr3verse)
library(ranger)
est_df = readRDS(paste0("./est_df.rds"))
est_df = est_df %>%
  filter(year <=2020) %>%
  filter(year >2015) %>%
  mutate(obs_id = row_number())

task = as_task_classif(est_df, target = "ucdp_12_bin", positive = "1")  
task$select(setdiff(task$feature_names, c("id", "month", "year")))

rf_model = lrn("classif.ranger",predict_type = "prob")
rf_model$param_set$values = list(num.trees = 500, num.threads = 16)


train_ids = est_df %>% filter(year <= 2019) %>% pull(obs_id)
test_ids = est_df %>% filter(year > 2019) %>% pull(obs_id)

rf_model$train(task, row_ids = train_ids)

prediction_rf = rf_model$predict(task, row_ids = test_ids)

prediction_df = as.data.table(prediction_rf) %>% 
  rename(pred_rf = prob.1)


#saveRDS(prediction_df, "results/prediction_df.rds")


prediction_rf$score(list(msr("classif.acc"),
                         msr("classif.auc"),
                         msr("classif.prauc")
))