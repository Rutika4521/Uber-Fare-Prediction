library(arrow)
library(dplyr)

file_paths <- list.files(path = "D:/DS CP/Data", pattern = "*.parquet", full.names = TRUE)
sample_data <- function(file, sample_size = 1000000) {
  data <- read_parquet(file)  # Load dataset
  # Take a random sample (or full data if the file has fewer rows)
  sampled_data <- data %>% sample_n(min(nrow(data), sample_size))
  return(sampled_data)
}

sampled_data_list<-lapply(file_paths, sample_data)
merged_data <- bind_rows(sampled_data_list)

write_parquet(merged_data, "merged_fare_data.parquet")

#EDA
install.packages("DataExplorer")
install.packages("corrplot")

library(ggplot2)
library(DataExplorer) # For quick EDA
library(corrplot)

str(merged_data)
summary(merged_data)
glimpse(merged_data)

colSums(is.na(merged_data))
plot_missing(merged_data)

nrow(merged_data) - nrow(distinct(merged_data))

boxplot(merged_data$fare_amount, main = "Fare Amount Boxplot before cleaning", col = "skyblue", breaks=50)
boxplot(merged_data$trip_distance, main = "Trip Distance Distribution before cleaning", col = "green", breaks = 50)
hist(merged_data$trip_distance, main = "Trip Distance Distribution before Cleaning", col = "green", breaks = 50)

numeric_df <- select(merged_data, where(is.numeric))
corr_matrix <- cor(na.omit(numeric_df))
corrplot(corr_matrix, method = "color", tl.cex = 0.8)

