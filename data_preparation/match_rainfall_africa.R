# This file is used to match the rainfall data to our cell polygons

library(tidyverse)
library(sf)
library(terra)

setwd("C:/Users/Luis/OneDrive/Desktop/CNN_Africa")

df_polygons =  readRDS("data/data/africa_cell_polygons.rds")
df_polygons <- df_polygons %>%
  mutate(id = row_number()) %>%
  arrange(desc(col))

gids_col_order = df_polygons$gid

### to create mapping for match_Africa_data.R later
mapping_ids = df_polygons %>% dplyr::select(id, gid) %>% st_set_geometry(NULL)
saveRDS(mapping_ids, paste0("data/mapping_ids.rds"))
###

df_polygons = df_polygons %>%
  select(!gid)

df_ids = df_polygons %>% dplyr::select(id) %>% st_set_geometry(NULL)

years = seq(2015, 2019)
rainfall_complete = tibble()

rainfall_matrix = array(NA, dim=c(length(years), length(df_ids$id), 25, 25))
matrix_counter = 1

for(year_loop in years){
  print(year_loop)
  path = paste0("data/rainfall/chirps-v2.0.", year_loop, ".tif")
  rainfall = rast(path)
  rainfall = project(rainfall, "EPSG:4326", method = "bilinear")
  rainfall = app(rainfall, fun=function(x){ x[is.na(x)] = 0; return(x)} )
  
  rainfall_info = extract(rainfall, vect(st_geometry(df_polygons)), bind = TRUE, weights = TRUE) %>%
    rename(rainfall = paste0("lyr.1")) %>%
    filter(!is.nan(rainfall)) %>%
    mutate(rainfall = rainfall*weight) %>%
    group_by(ID) %>%
    summarize(rainfall = sum(rainfall)) %>%
    ungroup() %>%
    mutate(year = year_loop) %>%
    rename(id = ID) %>%
    relocate(year, .after = id)
  
  rainfall_info$id = df_ids$id
  
  rainfall_info = rainfall_info %>%
    mutate(rainfall = replace_na(rainfall,0), year = replace_na(year, year_loop))
  
  rainfall_complete = bind_rows(rainfall_complete, rainfall_info)
  
  
  #create image data for each cell:
  i=1
  for(gidd in gids_col_order){
    file = paste0("cell_", gidd, ".tif")
    cell_path = paste0("data/cell_rasters/", file)
    cell = rast(cell_path)
    
    cell_rainfall = terra::resample(rainfall, cell, method="sum")
    
    d <- dim(cell_rainfall)
    rainfall_values = matrix(values(cell_rainfall), d[1], d[2], byrow=TRUE)
    rainfall_matrix[matrix_counter, i, ,] = rainfall_values
    i = i+1
  }
  matrix_counter = matrix_counter + 1
}
saveRDS(rainfall_complete, paste0("data/df_rainfall_info.rds"))
saveRDS(rainfall_matrix, paste0("data/rainfall_matrix.rds"))