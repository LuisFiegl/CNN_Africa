# File for the confusion matrix map plot

library(dplyr)
library(ggplot2)
library(lubridate)
full_df = readRDS("C:/Users/Luis/Downloads/est_df.rds")
full_df = full_df %>%
  filter(year >=2015)

library(terra)
library(sf)
polygons = readRDS("C:/Users/Luis/OneDrive/Desktop/CNN_Africa/africa_cell_polygons.rds")
polygons <- polygons %>%
  mutate(id = row_number())

polygons = polygons %>% select(geom_intersect, id)
polygons <- st_as_sf(polygons, sf_column_name = "geom_intersect")


# Map with binary if at least one attack per year
binary_2015 <- full_df %>%
  filter(year == 2020) %>%
  filter(month == 10) %>%
  group_by(id) %>%
  summarize(at_least_one = ifelse(sum(ucdp_12_bin) > 0, 1, 0))

polygons1 <- left_join(polygons, binary_2015, mapping, by = "id")

ggplot() + geom_sf(data = polygons1, aes(fill = at_least_one))


# Map for binary predictions
preds = readRDS("C:/Users/Luis/Downloads/19val_predictions.rds")

preds = readRDS("C:/Users/Luis/Downloads/rf_predictions_new2.rds")
preds = preds %>%
  rename(predictions = pred_class)

pred_df <- preds %>%
  filter(month == 8) %>%
  group_by(id) %>%
  summarize(correctly_predicted = ifelse(sum(predictions) > 0, 
                                         ifelse(sum(ucdp_12_bin) > 0, "Correctly Predicted Fatality", "Incorrectly Predicted as Fatality"), 
                                         ifelse(sum(ucdp_12_bin) > 0, "Missed Fatality", "Correctly Predicted as 0")))


polygons2 <- left_join(polygons, pred_df, mapping, by = "id")

ggplot() + geom_sf(data = polygons2, aes(fill = correctly_predicted), color = "azure3", size = 0.001)+
  scale_fill_manual(values = c("Correctly Predicted Fatality" = "springgreen2", "Incorrectly Predicted as Fatality" = "dodgerblue3", "Missed Fatality" = "firebrick2", "Correctly Predicted as 0" = "white")) +
  labs(fill = "August 2020 predictions \nof our binary target")+
  theme_minimal()+
  theme(
    axis.text = element_blank(),      # Remove axis text (coordinates)
    axis.ticks = element_blank(),     # Remove axis ticks
    panel.grid.major = element_blank(),     # Optional: Remove grid lines
    panel.grid.minor = element_blank(),
    legend.position = "right",         # Ensure legend is positioned (optional)
    legend.text = element_text(size = 8),        # Smaller legend text
    legend.title = element_text(size = 10),      # Smaller legend title
    legend.key.size = unit(0.5, "lines")
  )
