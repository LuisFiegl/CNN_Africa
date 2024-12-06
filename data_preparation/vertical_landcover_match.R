# This file is used to fit the landcover data to our cell polygons (vertical version)

library(tidyverse)
library(sf)
library(terra)
#library(stars)

#setwd("C:/Users/Luis/OneDrive/Desktop/CNN_Africa")
cell_size = 250
df_polygons =  readRDS("D:/consulting/code_consulting/Africa/data_new/data/africa_cell_polygons.rds")

# Creating dataframes for the bounding boxes

n = 30
unique_col_numbers <- unique(df_polygons$col)
num_unique <- length(unique_col_numbers)
cols_per_region <- ceiling(num_unique / n)

df_polygons_test <- df_polygons %>% 
  mutate(region = cut(col, breaks = seq(324, 463+cols_per_region, by = cols_per_region), labels = FALSE, include.lowest = TRUE))

number_of_regions = length(unique(df_polygons_test$region))
xmin = c()
ymin = c()
xmax = c()
ymax = c()
for (i in seq(1, number_of_regions)){
  df_region_i = df_polygons_test %>% filter(region == i)
  region_box <- st_bbox(df_region_i$geom_intersect)
  xmin <- append(xmin, as.numeric(region_box[1]))
  ymin <- append(ymin, as.numeric(region_box[2]))
  xmax <- append(xmax, as.numeric(region_box[3]))
  ymax <- append(ymax, as.numeric(region_box[4]))
}
bounding_boxes = data.frame(xmin = xmin-1,
                            ymin = ymin-1,
                            xmax = xmax+1,
                            ymax = ymax+1)

write.csv(bounding_boxes,"D:\\consulting\\code_consulting\\Africa\\data\\bounding_boxes_vertical.csv", row.names = FALSE)


# Creating landcover_complete and landcover_matrix

landcover_complete = tibble()

years = seq(2015, 2019)
landcover_matrix = array(NA, dim=c(length(years), length(df_polygons$gid), 25, 25, 8))
matrix_counter = 1
df_polygons_subparts <- df_polygons %>% 
  mutate(region = cut(col, breaks = seq(324, 463+cols_per_region, by = cols_per_region), labels = FALSE, include.lowest = TRUE))

df_polygons_subparts = df_polygons_subparts %>%
  arrange(desc(col))

file_name_list = unique(df_polygons_subparts$region)
#file_name_list =c("9","8","7","6","5","4","3","2","1")

#### 

# Read the GeoTIFF files into separate SpatRaster objects
year = 2015 # note that this file one processes one year at a time. For the years 2015 to 2019, it must be run 5 times
i = 0
for (sub_part in file_name_list){
  print("Sub-part")
  print(sub_part)
  if (sub_part %in% seq(12, 22)){
    file_name1 = paste0("D:\\consulting\\code_consulting\\Africa\\landcover_vertical\\landcover_vertical_",year,"/",year, "africa_landcover", sub_part, "-0000000000-0000000000.tif")
    file_name2 = paste0("D:\\consulting\\code_consulting\\Africa\\landcover_vertical\\landcover_vertical_",year,"/",year, "africa_landcover", sub_part, "-0000065536-0000000000.tif")
    raster1 = rast(file_name1)
    raster2 = rast(file_name2)
    merged_raster <- merge(raster1, raster2)
    landcover = project(merged_raster, "EPSG:4326", method = "near")
    
  }
  else {
    file_name = paste0("D:\\consulting\\code_consulting\\Africa\\landcover_vertical\\landcover_vertical_",year,"/",year, "africa_landcover", sub_part, ".tif")
    raster = rast(file_name)
    landcover = project(raster, "EPSG:4326", method = "near")
  }
  
  # Our 8 landcover categories
  landcover2 <- app(landcover, fun=function(x){ 
    x[is.na(x)] = 0
    x[x>=111 & x<=126] = 111
    x[x==20 | x==30 | x==90] = 20
    x[x==70| x==100]=0
    return(x)} 
  )
  
  categories = c(0, 60, 20, 111, 40, 50, 80, 200)
  category_df <- data.frame(value = sort(categories), 
                            cover=c("missing", "grass_shrub", "crop", "built", #0  20  40  50
                                    "bare", "water", "tree", "sea")) #60  80 111 200
  
  levels(landcover2) = category_df
  
  df_polygons_sub = df_polygons_subparts %>%
    filter(region == sub_part)
  
  landcover_info = terra::extract(landcover2, vect(st_geometry(df_polygons_sub)), weights = TRUE) %>% 
    rename(landcover_class = cover) %>% 
    group_by(ID, landcover_class) %>% 
    summarize(n = sum(weight)) %>% 
    ungroup()
  
  landcover_info = landcover_info %>% 
    pivot_wider(id_cols = ID, names_from = landcover_class, names_prefix = "landcover_", values_from = n) %>% 
    mutate(across(starts_with("landcover"), ~ replace_na(.x,0))) %>% 
    mutate(sum_landcover = rowSums(dplyr::pick(starts_with("landcover")) )) %>% 
    mutate(across(starts_with("landcover"), ~ .x/sum_landcover)) %>% 
    select(-sum_landcover) %>% 
    rename(id = ID) %>% 
    mutate(year = year) %>% 
    relocate(year, .after = id)
  
  gids_col_order = df_polygons_sub$gid
  landcover_complete = bind_rows(landcover_complete, landcover_info)
  landcover_seq = segregate(landcover2)
  
  number_cells = length(list.files("D:\\consulting\\code_consulting\\Africa\\data\\cell_rasters"))
  
  print("continue with matrix")
  for(gidd in gids_col_order){
    file = paste0("cell_", gidd, ".tif")
    cell_path = paste0("D:\\consulting\\code_consulting\\Africa\\data\\cell_rasters/", file)
    
    cell = rast(cell_path)
    cell_landcover = terra::resample(landcover_seq, cell, method="sum")
    
    d <- dim(cell_landcover)
    
    landcover_values_layers = values(cell_landcover)
    landcover_shares_layers = t(apply(landcover_values_layers, 1, function(i) i/sum(i)))  #for some reason the output needs to be transposed
    
    cell_landcover_matrix = array(NA, dim=d)
    for(j in seq(1, dim(landcover_shares_layers)[2], by = 1)){
      #print(j)
      cell_landcover_matrix[,,j] = matrix(landcover_shares_layers[,j], d[1], d[2], byrow=TRUE)
    }
    
    landcover_matrix[matrix_counter, i, , ,] = cell_landcover_matrix
    i =i+1
    print(i)
  }
}
matrix_counter = matrix_counter + 1

mapping = readRDS("D:/consulting/code_consulting/Africa/data/mapping_ids.rds")
landcover_complete = landcover_complete %>% select(!c(id))

landcover_complete$gid = df_polygons_subparts$gid

landcover_complete <- left_join(landcover_complete, mapping, by = "gid")
landcover_complete = landcover_complete %>% select(!c(gid))

# Data Quality Checks
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

saveRDS(landcover_complete, paste0("D:\\consulting\\code_consulting\\Africa\\data\\df_landcover_info_2015_vertical.rds"))
saveRDS(landcover_matrix, paste0("D:\\consulting\\code_consulting\\Africa\\data\\landcover_matrix_2015_vertical.rds"))