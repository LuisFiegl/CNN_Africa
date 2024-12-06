# This file was used to match all the remote sensing features into one Tabular
# dataset and one Remote Sensing Feature Matrix

library(tidyverse)

full_df = readRDS("D://consulting//code_consulting//Africa//data//tabular_df.rds")
mapping = readRDS("D:/consulting/code_consulting/Africa/data/mapping_ids.rds")
#syr_df = readRDS("D:/consulting/code_consulting/Africa/data/est_df_250.rds")
df_polygons =  readRDS("D:/consulting/code_consulting/Africa/data/africa_cell_polygons.rds")
df_polygons <- df_polygons %>%
  mutate(id = row_number()) %>%
  select(id, intersect_area, xcoord, ycoord)

full_df <- left_join(full_df, mapping, by = "gid") # adding proper id column
full_df = full_df %>% select(!c(gid, month_cons, region))

full_df = left_join(full_df, df_polygons, by = "id")%>%
  select(!c(geom_intersect))

full_df = full_df %>%
  arrange(desc(col), desc(row))

##################
#match population#
##################
pop_df = readRDS(paste0("D:/consulting/code_consulting/Africa/data/df_pop_info.rds"))

full_df = full_df %>% 
  left_join(pop_df, by = c("id", "year"))

##################
#match rainfall#
##################
rainfall_df = readRDS(paste0("D:/consulting/code_consulting/Africa/data/df_rainfall_info.rds"))

full_df = full_df %>% 
  left_join(rainfall_df, by = c("id", "year"))

##################
#match nighttime#
##################
nighttime_df = readRDS(paste0("D:/consulting/code_consulting/Africa/data/df_nighttime_info.rds"))
nighttime_df$year[42557:nrow(nighttime_df)] <- 2019 # adjust for missing data in 2019

full_df = full_df %>% 
  left_join(nighttime_df, by = c("id", "year"))

##################
#match landcover#
##################
landcover_df = readRDS(paste0("D:/consulting/code_consulting/Africa/data/df_landcover_complete.rds"))

full_df = full_df %>% 
  left_join(landcover_df, by = c("id", "year"))

lag_vars = c("pop", "rainfall", "nighttimes", "landcover_missing", "landcover_grass_shrub", "landcover_crop", "landcover_built", "landcover_water", "landcover_tree", "landcover_sea", "landcover_bare")

df_lagged <- full_df %>%
  mutate(year = year + 1) %>%
  select(id, year, month, all_of(lag_vars)) %>%
  rename_with(~paste0("lag_", .), all_of(lag_vars))

df_combined <- full_df %>%
  left_join(df_lagged, by = c("id", "year", "month"))%>%
  select(-one_of(lag_vars))

saveRDS(df_combined, paste0("D:/consulting/code_consulting/Africa/data/est_df.rds"))


# MATRIX COMBINATION

pop_matrix = readRDS(paste0("D:/consulting/code_consulting/Africa/data/population_matrix.rds"))
rainfall_matrix = readRDS(paste0("D:/consulting/code_consulting/Africa/data/rainfall_matrix.rds"))
nighttime_matrix = readRDS(paste0("D:/consulting/code_consulting/Africa/data/nighttime_matrix.rds"))
landcover_matrix = readRDS(paste0("D:/consulting/code_consulting/Africa/data/landcover_matrix_complete.rds"))

full_matrix = array(NA, dim=c(5, 10639, 25, 25, 11))


# 5 10639    25    25     8

for (year in seq(1, 5)){
  print(year)
  full_matrix[year, , , , 1] = landcover_matrix[year, , , , 1]
  full_matrix[year, , , , 2] = landcover_matrix[year, , , , 2]
  full_matrix[year, , , , 3] = landcover_matrix[year, , , , 3]
  full_matrix[year, , , , 4] = landcover_matrix[year, , , , 4]
  full_matrix[year, , , , 5] = landcover_matrix[year, , , , 5]
  full_matrix[year, , , , 6] = landcover_matrix[year, , , , 6]
  full_matrix[year, , , , 7] = landcover_matrix[year, , , , 7]
  full_matrix[year, , , , 8] = landcover_matrix[year, , , , 8]
  full_matrix[year, , , , 9] = rainfall_matrix[year, , , ]
  full_matrix[year, , , , 10] = nighttime_matrix[year, , , ]
  full_matrix[year, , , , 11] = pop_matrix[year, , , ]
}

saveRDS(full_matrix, paste0("D:/consulting/code_consulting/Africa/data/full_matrix.rds"))

library(rhdf5)
h5_file <- 'D:/consulting/code_consulting/Africa/data/full_matrix.h5'
h5createFile(h5_file)
h5write(full_matrix, h5_file, "full_matrix")

h5ls(h5_file)
