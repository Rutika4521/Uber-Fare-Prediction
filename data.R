library(dplyr)
library(lubridate)
library(ggplot2)
library(DataExplorer)
library(corrplot)

uber<-read_parquet("C:/Users/HP/Desktop/SY II/Data Science/R language/cleaned_uber_data.parquet")
uber <-uber %>%
  mutate(
    pickup_datetime = as.POSIXct(tpep_pickup_datetime, format="%Y-%m-%d %H:%M:%S"),
    dropoff_datetime = as.POSIXct(tpep_dropoff_datetime, format="%Y-%m-%d %H:%M:%S"),
    day_of_week = weekdays(pickup_datetime),  # Extract Day Name
    hour_of_day = hour(pickup_datetime),  # Extract Hour
    is_weekend = ifelse(day_of_week %in% c("Saturday", "Sunday"), 1, 0),  # Weekend Indicator
    is_peak_hour = ifelse(hour_of_day %in% c(7,8,9,17,18,19), 1, 0),  # Peak Rush Hour Indicator
    trip_duration = as.numeric(difftime(dropoff_datetime, pickup_datetime, units="mins")) # Trip duration in minutes
  )

uber <- uber %>% filter(trip_duration > 0)

str(uber)
# Categorize trip distance
uber <-uber %>%
  mutate(
    trip_distance_category = case_when(
      trip_distance <= 2 ~ "Short",
      trip_distance > 2 & trip_distance <= 5 ~ "Medium",
      trip_distance > 5 ~ "Long"
    )
  )

# Remove unnecessary columns
uber <-uber %>%
  select(-c(tpep_pickup_datetime, tpep_dropoff_datetime, store_and_fwd_flag))

duration_cutoff <- quantile(uber$trip_duration, 0.99, na.rm = TRUE)
uber <- uber %>% filter(trip_duration <= duration_cutoff)

dist_cutoff <- quantile(uber$trip_distance, 0.99, na.rm = TRUE)
uber <- uber %>% filter(trip_distance <= dist_cutoff)

t_cutoff <- quantile(uber$total_amount, 0.99, na.rm = TRUE)
uber <- uber %>% filter(total_amount <= t_cutoff)

summary(uber)


# Missing values check
plot_missing(uber)

# Distribution of fare_amount
ggplot(uber, aes(x = fare_amount)) +
  geom_histogram(bins = 50, fill = "blue", alpha = 0.7) +
  labs(title = "Fare Amount Distribution", x = "Fare ($)", y = "Count")

# Boxplot for trip duration
ggplot(uber, aes(x = trip_duration)) +
  geom_histogram(fill = "red",bins=30, alpha = 0.6) +
  labs(title = "Trip Duration Boxplot", y = "Minutes")

# Correlation heatmap
numeric_cols <-uber %>% select_if(is.numeric)
corrplot(cor(numeric_cols, use = "complete.obs"), method = "color",tl.cex = 0.8)

# Step 4: Data Preprocessing for Modeling

# Handle missing values (remove or impute)
uber <-uber %>% drop_na()

# Remove outliers using IQR method
Q1 <- quantile(uber$fare_amount, 0.25)
Q3 <- quantile(uber$fare_amount, 0.75)
IQR_value <- Q3 - Q1
uber <-uber %>% filter(fare_amount >= (Q1 - 1.5 * IQR_value) & fare_amount <= (Q3 + 1.5 * IQR_value))

# Normalize trip duration (Optional for models)
uber <-uber %>%
  mutate(trip_duration_scaled = scale(trip_duration))

# Save cleaned dataset for model training
write_parquet(uber, "cleaned_uber_data1.parquet")

