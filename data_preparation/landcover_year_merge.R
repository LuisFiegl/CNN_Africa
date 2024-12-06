# Since we process the landcover data one year at a time, we used this file to merge all the years together again

# Tabular part
df_15=readRDS("D:\\consulting\\code_consulting\\Africa\\landcover_vertical\\df_landcover_info_2015_newest_vertical.rds")
View(df_15)

df_16=readRDS("D:\\consulting\\code_consulting\\Africa\\landcover_vertical\\df_landcover_info_2016_newest_vertical.rds")
View(df_16)
df_17=readRDS("D:\\consulting\\code_consulting\\Africa\\landcover_vertical\\df_landcover_info_2017_newest_vertical.rds")
df_18=readRDS("D:\\consulting\\code_consulting\\Africa\\landcover_vertical\\df_landcover_info_2018_newest_vertical.rds")
df_19=readRDS("D:\\consulting\\code_consulting\\Africa\\landcover_vertical\\df_landcover_info_2019_newest_vertical.rds")

library(tidyverse)
landcover_complete = bind_rows(df_15, df_16, df_17, df_18, df_19)


# Matrix part
matrix=readRDS("D:\\consulting\\code_consulting\\Africa\\landcover_vertical\\landcover_matrix_2015_newest_vertical.rds")
dim(matrix)

matrix16=readRDS("D:\\consulting\\code_consulting\\Africa\\landcover_vertical\\landcover_matrix_2016_newest_vertical.rds")
matrix17=readRDS("D:\\consulting\\code_consulting\\Africa\\landcover_vertical\\landcover_matrix_2017_newest_vertical.rds")
matrix18=readRDS("D:\\consulting\\code_consulting\\Africa\\landcover_vertical\\landcover_matrix_2018_newest_vertical.rds")
matrix19=readRDS("D:\\consulting\\code_consulting\\Africa\\landcover_vertical\\landcover_matrix_2019_newest_vertical.rds")

dim(matrix19)

matrix[2,,,,] = matrix16[1,,,,]
matrix[3,,,,] = matrix17[1,,,,]
matrix[4,,,,] = matrix18[1,,,,]
matrix[5,,,,] = matrix19[1,,,,]

saveRDS(landcover_complete, paste0("D:\\consulting\\code_consulting\\Africa\\data\\df_landcover_complete.rds"))
saveRDS(matrix, paste0("D:\\consulting\\code_consulting\\Africa\\data\\landcover_matrix_complete.rds"))
