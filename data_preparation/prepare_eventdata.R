library(tidyverse)
library(sf)


full_df = readRDS("data/full_df_deaths.rds")
neighbour_df = readRDS("data/neighbour_deaths_df.rds")


tabular_df = full_df %>% 
  left_join(neighbour_df, by = c("gid", "month_cons")) %>% 
  filter(year >= 2000 & year <= 2020) %>% 
  filter(gid != 62356) %>% 
  select(gid, col, row, year, month, month_cons, region, ucdp_12_bin, ucdp_deaths_12, 
         starts_with("ucdp_deaths_12_lag_"), starts_with("ucdp_deaths_12_neighbour_lag_"))

saveRDS(tabular_df, "data/tabular_df.rds")