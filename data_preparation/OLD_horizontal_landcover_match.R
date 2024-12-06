# This is the old file with which we processed the landcover data horizontally

library(tidyverse)
library(sf)
library(terra)
#library(stars)

#setwd("C:/Users/Luis/OneDrive/Desktop/CNN_Africa")
cell_size = 250
df_polygons =  readRDS("D:/consulting/code_consulting/Africa/data_new/data/africa_cell_polygons.rds")

# Creating dataframes for the bounding boxes

n = 20
unique_row_numbers <- unique(df_polygons$row)
num_unique <- length(unique_row_numbers)
rows_per_region <- ceiling(num_unique / n)

df_polygons_test <- df_polygons %>% 
  mutate(region = cut(row, breaks = seq(110, 255+rows_per_region, by = rows_per_region), labels = FALSE, include.lowest = TRUE))

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
bounding_boxes = data.frame(xmin = xmin,
                            ymin = ymin,
                            xmax = xmax,
                            ymax = ymax)

write.csv(bounding_boxes,"D:\\consulting\\code_consulting\\Africa\\data\bounding_boxes.csv", row.names = FALSE)

# For regions that were too large:

df_polygons_test_row11 = df_polygons_test %>%
  filter(region == 11)

df_polygons_test_row11_1 = df_polygons_test_row11 %>%
  filter(row <= 193)

df_polygons_test_row11_2 = df_polygons_test_row11 %>%
  filter((row > 193) & (row < 196))

df_polygons_test_row11_3 = df_polygons_test_row11 %>%
  filter(row >= 196)

df_polygons_test_row12 = df_polygons_test %>%
  filter(region == 12)

df_polygons_test_row12_1 = df_polygons_test_row12 %>%
  filter(row <= 202)

df_polygons_test_row12_2 = df_polygons_test_row12 %>%
  filter(row > 202)

dataframes_splitted_further = list(df_polygons_test_row11_1, df_polygons_test_row11_2, df_polygons_test_row11_3, df_polygons_test_row12_1, df_polygons_test_row12_2)

xmin = c()
ymin = c()
xmax = c()
ymax = c()
for(i in 1:length(dataframes_splitted_further))
{
  region_box <- st_bbox(dataframes_splitted_further[[i]]$geom_intersect)
  xmin <- append(xmin, as.numeric(region_box[1]))
  ymin <- append(ymin, as.numeric(region_box[2]))
  xmax <- append(xmax, as.numeric(region_box[3]))
  ymax <- append(ymax, as.numeric(region_box[4]))
}

names_alt = c("11_1", "11_2", "11_3", "12_1", "12_2")
bounding_boxes_alt = data.frame(xmin = xmin,
                                ymin = ymin,
                                xmax = xmax,
                                ymax = ymax,
                                names = names_alt)

write.csv(bounding_boxes_alt,"data/data/bounding_boxes_alt.csv", row.names = FALSE)

# Creating landcover_complete and landcover_matrix

landcover_complete = tibble()

years = seq(2015, 2019)
landcover_matrix = array(NA, dim=c(length(years), length(df_polygons$gid), 25, 25, 8))
matrix_counter = 1
df_polygons_subparts <- df_polygons %>% 
  mutate(region = cut(row, breaks = seq(110, 255+rows_per_region, by = rows_per_region), labels = FALSE, include.lowest = TRUE),
         region = case_when(
           between(row, 191, 193) ~ "11_1",
           between(row, 194, 195) ~ "11_2",
           between(row, 196, 198) ~ "11_3",
           between(row, 199, 202) ~ "12_1",
           between(row, 203, 206) ~ "12_2",
           TRUE ~ as.character(region)
         ))

file_name_list = unique(df_polygons_subparts$region)
#file_name_list =c("9","8","7","6","5","4","3","2","1")

#### 

# Read the GeoTIFF files into separate SpatRaster objects
year = 2019
i = 0
for (sub_part in file_name_list){
  print("Sub-part")
  print(sub_part)
  if (sub_part %in% c("11_1", "11_2", "11_3", "12_1", "12_2", "13")){
    file_name1 = paste0("D:\\consulting\\code_consulting\\Africa\\data\\Landcover_",year,"/",year, "_africa_landcover_", sub_part, "-0000000000-0000000000.tif")
    file_name2 = paste0("D:\\consulting\\code_consulting\\Africa\\data\\Landcover_",year,"/",year, "_africa_landcover_", sub_part, "-0000000000-0000065536.tif")
    raster1 = rast(file_name1)
    raster2 = rast(file_name2)
    merged_raster <- merge(raster1, raster2)
    landcover = project(merged_raster, "EPSG:4326", method = "near")
    
  }
  else {
    file_name = paste0("D:\\consulting\\code_consulting\\Africa\\data\\Landcover_",year,"/",year, "_africa_landcover_", sub_part, ".tif")
    raster = rast(file_name)
    landcover = project(raster, "EPSG:4326", method = "near")
  }
  
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
  
  landcover_complete = bind_rows(landcover_complete, landcover_info)
  landcover_seq = segregate(landcover2)
  
  #existing_categories <- names(landcover_seq)
  #missing_categories <- setdiff(categories, existing_categories)
  #for (cat in missing_categories) {
    #empty_layer <- landcover2[[1]]
   # values(empty_layer) <- 0
   # names(empty_layer) <- cat
    #landcover_seq <- c(landcover_seq, empty_layer)
#  }
  #landcover_seq <- landcover_seq[[order(as.numeric(names(landcover_seq)))]]
  
  number_cells = length(list.files("D:\\consulting\\code_consulting\\Africa\\data\\cell_rasters"))
  
  
  print("continue with matrix")
  cell_raster_files = list.files("D:\\consulting\\code_consulting\\Africa\\data\\cell_rasters")
  for(file in cell_raster_files){
    gid_number <- as.integer(gsub("[^0-9]", "", file))
    
    if (!(gid_number %in% df_polygons_sub$gid)) {
      next
    }
    
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

landcover_complete = landcover_complete %>%
  select(!id) %>%
  mutate(id= row_number())

saveRDS(landcover_complete, paste0("D:\\consulting\\code_consulting\\Africa\\data\\df_landcover_info_2019_final.rds"))
saveRDS(landcover_matrix, paste0("D:\\consulting\\code_consulting\\Africa\\data\\landcover_matrix_2019_final.rds"))