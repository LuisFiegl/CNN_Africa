# Random Forest model

library(tidyverse)
library(mlr3verse)
library(ranger)
library(caret)
est_df = readRDS(paste0("C:/Users/Luis/Downloads/est_df.rds"))
est_df = est_df %>%
  filter(year <=2020) %>%
  filter(year >2015) %>%
  select(!c(ucdp_deaths_12, col, row)) %>%
  mutate(obs_id = row_number())


# For re-scaling remote sensing feature:
# List of 11 remote sensing features to scale
remote_sensing_features <- c(
  'lag_pop', 'lag_rainfall', 'lag_nighttimes', 'lag_landcover_missing', 'lag_landcover_crop', 'lag_landcover_built', 
  'lag_landcover_water','lag_landcover_grass_shrub', 'lag_landcover_tree', 'lag_landcover_sea', 
  'lag_landcover_bare'
)

# Create a preprocessing object with the range method (min-max scaling)
pre_process <- preProcess(est_df[, remote_sensing_features], method = c("range"))

# Apply the scaling to the dataset
est_df <- predict(pre_process, est_df)

remote_sensing_data <- est_df %>%
  select(all_of(remote_sensing_features))

# Calculate the correlation matrix
correlation_matrix <- cor(remote_sensing_data, use = "pairwise.complete.obs")
print(correlation_matrix)


# random forest model
task = as_task_classif(est_df, target = "ucdp_12_bin", positive = "1")  
task$select(setdiff(task$feature_names, c("id", "month", "year", "obs_id")))
#'lag_pop', 'lag_rainfall', 'lag_nighttimes','lag_landcover_missing', 'lag_landcover_grass_shrub','lag_landcover_crop', 'lag_landcover_built', 'lag_landcover_water','lag_landcover_tree', 'lag_landcover_sea', 'lag_landcover_bare'

rf_model = lrn("classif.ranger",predict_type = "prob")
rf_model$param_set$values = list(num.trees = 500, num.threads = 16, mtry = 8, importance = "permutation")


train_ids = est_df %>% filter(year <= 2019) %>% pull(obs_id)
test_ids = est_df %>% filter(year > 2019) %>% pull(obs_id)

rf_model$train(task, row_ids = train_ids)

prediction_rf = rf_model$predict(task, row_ids = test_ids)

prediction_df = as.data.table(prediction_rf) %>% 
  rename(pred_rf = prob.1)


#saveRDS(prediction_df, "results/prediction_df.rds")

# results
prediction_rf$score(list(msr("classif.acc"),
                         msr("classif.auc"),
                         msr("classif.prauc"),
                         msr("classif.fbeta", beta = 1)
))

importance_scores <- rf_model$importance()
print(importance_scores)

# You can also visualize the feature importance using a bar plot
importance_df = as.data.frame(importance_scores)
importance_df$feature = rownames(importance_df)

top_features = importance_df %>%
  top_n(15, wt = importance_scores) %>%  # Select top 15 features
  arrange(desc(importance_scores))        # Sort in descending order

# Plot the top 15 features
ggplot(top_features, aes(x = reorder(feature, importance_scores), y = importance_scores)) +
  geom_bar(stat = "identity", fill = "darkblue") +
  coord_flip() +
  labs(title = "Top 15 Features Based on \nPermutation Importance Scores",
       x = "Feature", 
       y = "Importance Score") +
  theme_minimal()




# Find optimal threshold
train_data <- est_df %>% filter(year <= 2019)
train_labels <- train_data$ucdp_12_bin
train_probs <- rf_model$predict(task, row_ids = train_ids)$prob[, "1"]

library(PRROC)
pr_curve <- pr.curve(scores.class0 = train_probs, weights.class0 = train_labels, curve = TRUE)
plot(pr_curve)

# Extract the precision, recall, and thresholds
precision <- pr_curve$curve[, 1]
recall <- pr_curve$curve[, 2]
thresholds <- pr_curve$curve[, 3]  # This is available only if curve = TRUE was used

# Compute F1 score for each threshold
f1_scores <- 2 * (precision * recall) / (precision + recall)

# Find the threshold that maximizes the F1 score
best_index <- which.max(f1_scores)
best_threshold <- thresholds[best_index]
best_f1_score <- f1_scores[best_index]

# Create a data frame for plotting
f1_data <- data.frame(
  Threshold = thresholds,
  F1_Score = f1_scores
)

# Plot F1 Scores vs Thresholds
ggplot(f1_data, aes(x = Threshold, y = F1_Score)) +
  geom_line(color = "darkblue", linewidth = 1.5) +
  labs(
    x = "Threshold",
    y = "F1 Score") +
  theme_minimal()

# Print the best threshold and corresponding F1 score
print(paste("Best threshold:", best_threshold))
print(paste("Best F1 score:", best_f1_score))



# Test set for prediction
#threshold = 0.5
threshold = best_threshold
prediction_df = prediction_df[, .(row_ids, pred_rf, pred_class = as.integer(pred_rf > threshold))]

test_df = est_df %>% 
  filter(obs_id %in% test_ids) %>%
  mutate(row_ids = obs_id)

test_with_predictions = test_df %>%
  left_join(prediction_df, by = "row_ids")

#saveRDS(test_with_predictions, paste0("D:/consulting/code_consulting/Africa/rf_pred_results/rf_predictions_new2.rds"))





conf_matrix = confusionMatrix(factor(prediction_df$pred_class), factor(task$truth(test_ids)))
print(conf_matrix)
f1_score_test <- conf_matrix$byClass["F1"]
print(paste("F1 Score on test data with optimal threshold:", f1_score_test))



