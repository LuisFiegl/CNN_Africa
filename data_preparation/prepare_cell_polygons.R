library(terra)
library(tidyverse)
library(sf)

setwd("C:/R - Workspace/New_CNN_for_Conflict")


prio_polygons = st_read("data/priogrid_cellshp/priogrid_cell.shp") 

#filter out countries not in study
c_shapes_df = read_csv("data/CShapes-2.0.csv") 
prio_df = read_csv("data/PRIO-GRID Yearly Variables for 2014-2014 - 2022-07-28.csv")
countries_c_shapes = c_shapes_df %>% filter(gweyear >= 1997) %>% distinct(cntry_name, gwcode, .keep_all = TRUE) %>% select(cntry_name, gwcode, the_geom)
countries_c_shapes = st_as_sf(countries_c_shapes, wkt = "the_geom")
african_countries_with_data = c("Algeria", "Angola", "Benin", "Botswana", "Burkina Faso (Upper Volta)",
                                "Burundi", "Cameroon", "Central African Republic", "Chad", "Congo",
                                "Congo, Democratic Republic of (Zaire)", "Cote D'Ivoire", "Djibouti", "Egypt", "Equatorial Guinea",
                                "Eritrea", "Swaziland (Eswatini)", "Ethiopia", "Gabon", "Gambia",
                                "Ghana", "Guinea", "Guinea-Bissau", "Kenya", "Lesotho",
                                "Liberia", "Libya", "Madagascar (Malagasy)", "Malawi", "Mali",
                                "Mauritania", "Morocco", "Mozambique", "Namibia", "Niger",
                                "Nigeria", "Rwanda", "Senegal", "Sierra Leone", "Somalia",
                                "South Africa", "Sudan", "Tanzania (Tanganyika)", "Togo", "Tunisia",
                                "Uganda", "Zambia", "Zimbabwe (Rhodesia)", "South Sudan")  #South Sudan starting in 2011 in ACLED!!!
additional_african_countries = c("Cape Verde","Comoros", "Mauritius", "Reunion") 
#according to UN definition:
northern_africa = c("Algeria", "Egypt", "Libya", "Morocco", "Sudan", "Tunisia")
eastern_africa = c("Burundi", "Comoros", "Djibouti", "Eritrea", "Ethiopia", "Kenya", "Madagascar (Malagasy)", "Malawi",
                   "Mauritius", "Mozambique", "Reunion", "Rwanda", "Somalia", "South Sudan", "Tanzania (Tanganyika)", "Uganda",
                   "Zambia", "Zimbabwe (Rhodesia)")
middle_africa = c("Angola", "Cameroon", "Central African Republic", "Chad", "Congo, Democratic Republic of (Zaire)", "Congo",
                  "Equatorial Guinea", "Gabon")
southern_africa = c("Botswana", "Swaziland (Eswatini)", "Lesotho", "Namibia", "South Africa")
western_africa = c("Benin", "Burkina Faso (Upper Volta)", "Cape Verde", "Cote D'Ivoire", "Gambia", "Ghana", "Guinea", "Guinea-Bissau",
                   "Liberia", "Mali",  "Mauritania", "Niger", "Nigeria",  "Senegal", "Sierra Leone", "Togo")
sum(length(northern_africa), length(eastern_africa), length(middle_africa), length(southern_africa), length(western_africa))
sum(length(african_countries_with_data), length(additional_african_countries))

african_countries_gwno = countries_c_shapes %>% st_set_geometry(NULL) %>% 
  filter(cntry_name %in% african_countries_with_data | cntry_name %in% additional_african_countries) 
african_countries_with_data_gwno = countries_c_shapes  %>% st_set_geometry(NULL) %>% 
  filter(cntry_name %in% african_countries_with_data) %>% 
  mutate(region = ifelse(cntry_name %in% northern_africa, 1,
                         ifelse(cntry_name %in% eastern_africa, 2,
                                ifelse(cntry_name %in% middle_africa, 3,
                                       ifelse(cntry_name %in% southern_africa, 4, 5))))
         
  )
# south_sudan_gwo = countries_c_shapes %>% 
#   filter(cntry_name == "South Sudan")

african_prio_grid_cells_with_data = prio_df %>% 
  distinct(gid, gwno) %>% 
  filter(gwno %in% african_countries_with_data_gwno$gwcode) %>% 
  left_join(
    african_countries_with_data_gwno %>% select(gwcode, region) %>% rename(gwno = gwcode),
    by = "gwno"
  )

africa_prio_grid_polygons = prio_polygons %>% 
  filter(gid %in% african_prio_grid_cells_with_data$gid) %>% 
  left_join(
    african_prio_grid_cells_with_data %>% select(gid, region),
    by = "gid"
  ) %>% 
  filter(gid != 62356) #filter out the small island

g = ggplot() + geom_sf(data = africa_prio_grid_polygons)
ggsave("results/full_africa_prio_grid_data.png", g, width = 8, height = 8)


sf_use_s2(FALSE)
african_borders = st_union(countries_c_shapes %>% filter(cntry_name %in% african_countries_with_data))

intersecting_polygons = st_intersection(africa_prio_grid_polygons, african_borders)
intersecting_polygons$intersect_area <- as.numeric(units::set_units(st_area(intersecting_polygons), "km^2"))


intersecting_polygons$geom_intersect = intersecting_polygons$geometry
intersecting_polygons = intersecting_polygons %>% st_set_geometry("geom_intersect")
intersecting_polygons$geometry = NULL


g = ggplot() + 
  geom_sf(data = intersecting_polygons,  aes(fill = intersect_area))
ggsave("results/intersected_cells.png", g, width = 8, height = 8)

sf_use_s2(TRUE)

intersecting_polygons$geom_non_intersect = africa_prio_grid_polygons$geometry

###---> this seems good

saveRDS(intersecting_polygons, paste0("data/africa_cell_polygons.rds"))


#######################
#create individual .tif files for each cell
create_and_write_raster = function(input_vector){
  print(input_vector$gid)
  intersect_polygon = vect(input_vector$geom_intersect)
  #input_vector = input_vector %>% st_set_geometry(NULL)
  
  raster_object = rast(vect(input_vector$geom_non_intersect), nrows = 25, ncols = 25, vals = 0)
  names(raster_object)[1] = "empty"
  
  cells_covered = terra::extract(raster_object, intersect_polygon, weights = TRUE, cells = TRUE)
  list_ids = cells_covered$cell
  
  raster_object[list_ids] = 1
  crs(raster_object) = "epsg:4326"
  
  writeRaster(raster_object, paste0("data/cell_rasters/cell_", input_vector$gid, ".tif"), overwrite=TRUE)
}
#input_vector = intersecting_polygons[1,]
apply(intersecting_polygons , 1, create_and_write_raster)



