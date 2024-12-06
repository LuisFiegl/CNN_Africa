# This file is used to match the population data to our cell polygons

library(tidyverse)
library(sf)
library(terra)
#library(stars)

setwd("C:/Users/12lui/Documents/consulting")
df_polygons =  readRDS("data/africa_cell_polygons.rds")
df_polygons <- df_polygons %>%
  mutate(id = row_number()) %>%
  arrange(desc(col))

gids_col_order = df_polygons$gid

df_polygons <- df_polygons %>%
  select(!gid)

df_ids = df_polygons %>% dplyr::select(id) %>% st_set_geometry(NULL)

years = seq(2015, 2019)
population_complete = tibble()

population_matrix = array(NA, dim=c(length(years), length(df_ids$id), 25, 25))
matrix_counter = 1

for(year_loop in years){
  print(year_loop)
  path = paste0("data/population/ppp_", year_loop, "_1km_Aggregated.tif")
  pop = rast(path)
  pop = project(pop, "EPSG:4326", method = "bilinear")
  pop = app(pop, fun=function(x){ x[is.na(x)] = 0; return(x)} )
  
  pop_info = extract(pop, vect(st_geometry(df_polygons)), bind = TRUE, weights = TRUE) %>%
    #rename(pop = paste0("syr_ppp_", year_loop, "_UNadj")) %>%
    rename(pop = paste0("lyr.1")) %>%
    filter(!is.nan(pop)) %>%
    mutate(pop = pop*weight) %>%
    group_by(ID) %>%
    summarize(pop = sum(pop)) %>%
    ungroup() %>%
    mutate(year = year_loop) %>%
    rename(id = ID) %>%
    relocate(year, .after = id)
  
  pop_info$id = df_ids$id
    
  pop_info = pop_info %>%
    mutate(pop = replace_na(pop,0), year = replace_na(year, year_loop))
  
  population_complete = bind_rows(population_complete, pop_info)
  
  
  #create image data for each cell:
  i=1
  for(gidd in gids_col_order){
    file = paste0("cell_", gidd, ".tif")
    cell_path = paste0("data/cell_rasters/", file)
    cell = rast(cell_path)
    
    cell_pop = terra::resample(pop, cell, method="sum")
    
    d <- dim(cell_pop)
    pop_values = matrix(values(cell_pop), d[1], d[2], byrow=TRUE)
    population_matrix[matrix_counter, i, ,] = pop_values
    i = i+1
  }
  matrix_counter = matrix_counter + 1
}

saveRDS(population_complete, paste0("data/df_pop_info.rds"))
saveRDS(population_matrix, paste0("data/population_matrix.rds"))
