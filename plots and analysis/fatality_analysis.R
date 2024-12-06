# File for analysis of our fatalities-data and remote sensing features

library(dplyr)
library(ggplot2)
library(lubridate)
full_df = readRDS("C:/Users/Luis/OneDrive/Desktop/CNN_Africa/est_df.rds")
full_df = full_df %>%
  filter(year >=2015)

library(terra)
library(sf)
polygons = readRDS("C:/Users/Luis/OneDrive/Desktop/CNN_Africa/africa_cell_polygons.rds")
polygons <- polygons %>%
  mutate(id = row_number())
polygons = polygons %>% select(geom_intersect, id)
polygons <- st_as_sf(polygons, sf_column_name = "geom_intersect")


#How many killings happened for each year?
killings_per_year = full_df %>%
  group_by(year) %>%
  summarize(number_of_killings = sum(ucdp_deaths_12), total_obs = n())

print(killings_per_year)


#How many cells had killings per year?
number_cells_with_killings_per_year = full_df %>%
  filter(ucdp_12_bin==1) %>%
  group_by(year) %>%
  summarize(number_of_cells_with_killings = n_distinct(id), cells_with_deaths_monthly_basis = n())

print(number_cells_with_killings_per_year) # about 1 percent of cells had killings per year, tendency increasing
# number of killings increased by 7%, while the number of affected cells increased by 40% -> killings are more spreaded among the map


#How many unique cells of all our cells had killings from 2015 to 2020?
unique_cells = full_df %>%
  filter(ucdp_12_bin==1) %>%
  summarize(num_unique_ids = n_distinct(id))

print(unique_cells) # only 1102 cells (of 10639) had at least one death in the 6 years, meaning they were affected by some sort of terror


# Plot killings per month
plot1 = full_df %>%
  group_by(year, month) %>%
  summarize(number_of_killings = sum(ucdp_deaths_12))

plot1 <- plot1 %>%
  mutate(year_month = as.Date(ymd(paste(year, month, "01", sep = "-"))))

ggplot(plot1, aes(x = year_month, y = number_of_killings)) +
  geom_line(color = "darkblue")+
  #geom_smooth(method = "lm", color = "red", se = FALSE) +
  labs(title = "Development of Total Fatality [sum over all cells]",
       x = "Date",
       y = "Fatality") +
  scale_x_date(date_labels = "%Y-%m", date_breaks = "8 months") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  theme_minimal()


# Plot cells with killings per month
plot2 = full_df %>%
  group_by(year, month) %>%
  summarize(number_of_cells = sum(ucdp_12_bin))

plot2 <- plot2 %>%
  mutate(year_month = as.Date(ymd(paste(year, month, "01", sep = "-"))))

ggplot(plot2, aes(x = year_month, y = number_of_cells)) +
  geom_line(color = "darkblue") +
  labs(
       x = "Date",
       y = "Number of Cells") +
  scale_x_date(date_labels = "%Y-%m", date_breaks = "8 months") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  theme_minimal()


# Map with binary if at least one attack per year
# choose a year
binary_2015 <- full_df %>%
  #filter(year == 2020) %>%
  #filter(month == 12) %>%
  group_by(id) %>%
  summarize(at_least_one = factor(ifelse(sum(ucdp_12_bin) > 0, 1, 0)))

polygons1 <- left_join(polygons, binary_2015, mapping, by = "id")

ggplot() + geom_sf(data = polygons1, aes(fill = at_least_one))+
  scale_fill_manual(values = c("0" = "darkblue", "1" = "yellow")) +
  labs(fill = "At Least One Fatality")+
  theme_minimal()+
  theme(
    axis.text = element_blank(),      # Remove axis text (coordinates)
    axis.ticks = element_blank(),     # Remove axis ticks
    panel.grid = element_blank(),     # Optional: Remove grid lines
    legend.position = "none"         # Ensure legend is positioned (optional)
  )


# Map with number of attacks per year
sum_2015 <- full_df %>%
  filter(year == 2015) %>%
  filter(month == 2) %>%
  group_by(id) %>%
  summarize(number_deaths = sum(ucdp_deaths_12))

polygons2 <- left_join(polygons, sum_2015, mapping, by = "id")

ggplot() + geom_sf(data = polygons2, aes(fill = number_deaths))+
  scale_fill_gradient(low = "darkblue", high = "yellow", na.value = "white") +
  labs(fill = "Number of Deaths")+
  theme_minimal()+
  theme(
    axis.text = element_blank(),      # Remove axis text (coordinates)
    axis.ticks = element_blank(),     # Remove axis ticks
    panel.grid = element_blank(),     # Optional: Remove grid lines
    legend.position = "right"         # Ensure legend is positioned (optional)
  )


# Map for most prevalent cells
# Map with number of attacks per year
sum_2015 <- full_df %>%
  #filter(year == 2015) %>%
  #filter(month == 2) %>%
  group_by(id) %>%
  summarize(number_deaths = sum(ucdp_12_bin))

polygons2 <- left_join(polygons, sum_2015, mapping, by = "id")

ggplot() + geom_sf(data = polygons2, aes(fill = number_deaths))+
  scale_fill_gradient(low = "white", high = "black", na.value = "white") +
  labs(fill = "Number of Deaths")+
  theme_minimal()+
  theme(
    axis.text = element_blank(),      # Remove axis text (coordinates)
    axis.ticks = element_blank(),     # Remove axis ticks
    panel.grid = element_blank(),     # Optional: Remove grid lines
    legend.position = "right"         # Ensure legend is positioned (optional)
  )


# Map for population
library(scales)  # Make sure to load the scales package
pop_map <- full_df %>%
  filter(year == 2020) %>%
  #filter(month == 8) %>%
  group_by(id) %>%
  summarize(avg_pop_this_year = mean(lag_pop))%>%
  mutate(avg_pop_this_year_winsorized = ifelse(avg_pop_this_year > quantile(avg_pop_this_year, 0.95), 
                                               quantile(avg_pop_this_year, 0.95), 
                                               avg_pop_this_year))

polygons3 <- left_join(polygons, pop_map, mapping, by = "id")

ggplot() + 
  geom_sf(data = polygons3, aes(fill = avg_pop_this_year_winsorized))+
  scale_fill_gradient(low = "darkblue", high = "yellow", na.value = "white", 
                      labels = label_number(scale = 1, accuracy = 1))+
  labs(fill = "Population Count")+
  theme_minimal()+
  theme(
    axis.text = element_blank(),      # Remove axis text (coordinates)
    axis.ticks = element_blank(),     # Remove axis ticks
    panel.grid = element_blank(),     # Optional: Remove grid lines
    legend.position = "right"         # Ensure legend is positioned (optional)
  )


# Map for rainfall/etc
rf_map <- full_df %>%
  filter(year == 2020) %>%
  group_by(id) %>%
  summarize(avg_value = mean(lag_landcover_water))

polygons4 <- left_join(polygons, rf_map, mapping, by = "id")

ggplot() + geom_sf(data = polygons4, aes(fill = avg_value))+
  scale_fill_gradient(low = "darkblue", high = "yellow", na.value = "white", limits = c(0, 1))+
  labs(fill = "Weight (between 0 and 1)")+
  theme_minimal()+
  theme(
    axis.text = element_blank(),      # Remove axis text (coordinates)
    axis.ticks = element_blank(),     # Remove axis ticks
    panel.grid = element_blank(),     # Optional: Remove grid lines
    legend.position = "left"         # Ensure legend is positioned (optional)
  )


# 2015 landcover map
lc_df = readRDS("C:/Users/Luis/Downloads/df_landcover_info_2018_newest_vertical.rds")
rf_map <- lc_df %>%
  group_by(id) %>%
  summarize(avg_value = mean(landcover_crop))

polygons5 <- left_join(polygons, rf_map, mapping, by = "id")

ggplot() + geom_sf(data = polygons5, aes(fill = avg_value))

lc_matrix = readRDS("C:/Users/Luis/Downloads/landcover_matrix_2018_newest_vertical.rds")



# further analysis

na_count <- sapply(landcover_complete, function(x) sum(is.na(x)))
print(na_count)

sum_columns_by_row_id <- function(df, id_value, columns) {
  row_values <- df[df$id == id_value, columns, drop = FALSE]
  sum_values <- sum(row_values, na.rm = TRUE)
  return(sum_values)
}

na_missings = landcover_complete[is.na(landcover_complete$landcover_missing), ]
ids_miss = na_missings$id

for(idd in ids_miss){
  sum = sum_columns_by_row_id(landcover_complete, id_value = idd, columns = c("landcover_grass_shrub", "landcover_crop", "landcover_built", "landcover_bare", "landcover_water", "landcover_tree", "landcover_sea"))
  if(round(sum, 10) != 1){
    print(sum)
    print(idd)
    }
}

landcover_complete <- landcover_complete %>%
  mutate(landcover_missing = if_else(is.na(landcover_missing), 0, landcover_missing))

na_count <- sapply(landcover_complete, function(x) sum(is.na(x)))
print(na_count)


# population data analysis
tabpop = readRDS("C:/Users/Luis/OneDrive/Desktop/CNN_Africa/data/df_pop_info.rds")
matrix_pop = readRDS("C:/Users/Luis/OneDrive/Desktop/CNN_Africa/data/population_matrix.rds")

tabpop = tabpop %>%
  filter(year == 2020)

tabpop[10491,]
matrix_pop[6, 10491, , ]


# Summary columns
summary_stats <- full_df %>%
  summarize(
    max_value = max(lag_pop, na.rm = TRUE),  # Get the maximum value
    min_value = min(lag_pop, na.rm = TRUE)   # Get the minimum value
  )

# Print the results
print(summary_stats)

mapping = readRDS("C:/Users/Luis/Downloads/df_nighttime_info.rds")
mega_city = full_df %>%
  filter(lag_pop >=18465290)
View(mega_city)

# check nas
sapply(full_df, function(x) sum(is.na(x)))



####### landcover summarized map
library(tidyr)
rf_map <- full_df %>%
  filter(year == 2020) %>%
  group_by(id) %>%
  summarize(across(c(lag_landcover_missing, lag_landcover_grass_shrub, lag_landcover_crop, lag_landcover_built, lag_landcover_bare, lag_landcover_water, lag_landcover_tree, lag_landcover_sea), mean, na.rm = TRUE), .groups = 'drop') %>%
  pivot_longer(cols = c(lag_landcover_missing, lag_landcover_grass_shrub, lag_landcover_crop, lag_landcover_built, lag_landcover_bare, lag_landcover_water, lag_landcover_tree, lag_landcover_sea),
               names_to = "category", values_to = "value") %>%
  group_by(id) %>%
  slice_max(order_by = value, n = 1) %>%
  ungroup()

# Custom colors for each category
custom_colors <- c(
  "lag_landcover_missing" = "gray",
  "lag_landcover_grass_shrub" = "green",
  "lag_landcover_crop" = "yellowgreen",
  "lag_landcover_built" = "red",
  "lag_landcover_bare" = "tan",
  "lag_landcover_water" = "blue",
  "lag_landcover_tree" = "darkgreen",
  "lag_landcover_sea" = "cyan"
)

custom_labels <- c(
  "lag_landcover_missing" = "missing",
  "lag_landcover_grass_shrub" = "grass_shrub",
  "lag_landcover_crop" = "crop",
  "lag_landcover_built" = "built",
  "lag_landcover_bare" = "bare",
  "lag_landcover_water" = "water",
  "lag_landcover_tree" = "tree",
  "lag_landcover_sea" = "sea"
)

polygons4 <- left_join(polygons, rf_map, by = "id")

# Plotting the map
ggplot() +
  geom_sf(data = polygons4, aes(fill = category)) +
  scale_fill_manual(values = custom_colors, labels = custom_labels, na.value = "white") +
  labs(fill = "Dominant Landcover Class") +
  theme_minimal() +
  theme(
    axis.text = element_blank(),      # Remove axis text (coordinates)
    axis.ticks = element_blank(),     # Remove axis ticks
    panel.grid = element_blank(),     # Optional: Remove grid lines
    legend.position = "none"         # Ensure legend is positioned (optional)
  )