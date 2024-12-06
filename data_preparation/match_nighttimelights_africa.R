# This file is used to match the nighttime lights data to our cell polygons

library(tidyverse)
library(sf)
library(terra)
#library(stars)

df_polygons =  readRDS("data/data/africa_cell_polygons.rds")
df_polygons <- df_polygons %>%
  mutate(id = row_number()) %>%
  arrange(desc(col))

gids_col_order = df_polygons$gid

df_polygons = df_polygons %>%
  select(!gid)

df_ids = df_polygons %>% dplyr::select(id) %>% st_set_geometry(NULL)

years = seq(2015, 2019)
nighttime_complete = tibble()

nighttime_matrix = array(NA, dim=c(length(years), length(df_ids$id), 25, 25))
matrix_counter = 1

for(year_loop in years){
  print(year_loop)
  if (year_loop == 2019){
    year_loop = 2018}
  path = paste0("data/night-time-lights/Harmonized_DN_NTL_", year_loop, "_simVIIRS.tif")
  nighttimes = rast(path)
  nighttimes = project(nighttimes, "EPSG:4326", method = "bilinear")
  nighttimes = app(nighttimes, fun=function(x){ x[is.na(x)] = 0; return(x)} ) #we manually checked for NAs and didnt find any in all of the years
  
  nighttimes_info = extract(nighttimes, vect(st_geometry(df_polygons)), bind = TRUE, weights = TRUE) %>%
    rename(nighttimes = paste0("lyr.1")) %>%
    filter(!is.nan(nighttimes)) %>%
    mutate(nighttimes = nighttimes*weight) %>%
    group_by(ID) %>%
    summarize(nighttimes = sum(nighttimes)) %>%
    ungroup() %>%
    mutate(year = year_loop) %>%
    rename(id = ID) %>%
    relocate(year, .after = id)
  
  nighttimes_info$id = df_ids$id
  
  nighttimes_info = nighttimes_info %>%
    mutate(nighttimes = replace_na(nighttimes,0), year = replace_na(year, year_loop))
  
  # the tabular dataset
  nighttime_complete = bind_rows(nighttime_complete, nighttimes_info)
  
  
  #create image data for each cell (matrix):
  i=1
  for(gidd in gids_col_order){
    file = paste0("cell_", gidd, ".tif")
    cell_path = paste0("data/cell_rasters/", file)
    cell = rast(cell_path)
    
    cell_nighttimes = terra::resample(nighttimes, cell, method="sum")
    
    d <- dim(cell_nighttimes)
    nighttimes_values = matrix(values(cell_nighttimes), d[1], d[2], byrow=TRUE)
    nighttime_matrix[matrix_counter, i, ,] = nighttimes_values
    i = i+1
  }
  matrix_counter = matrix_counter + 1
}
saveRDS(nighttime_complete, paste0("data/df_nighttime_info.rds"))
saveRDS(nighttime_matrix, paste0("data/nighttime_matrix.rds"))