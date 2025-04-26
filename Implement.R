# Required libraries
library(dplyr)
library(Metrics)
library(caret)
library(arrow)

data <- read_parquet("C:/Users/HP/Desktop/SY II/Data Science/R language/cleaned_uber_data1.parquet")
summary(data)
# ========================
# INNOVATIVE RIDGE REGRESSION
# ========================

# Normalize numeric features
normalize <- function(x) {
  return((x - mean(x)) / sd(x))
}

# Huber loss derivative
huber_gradient <- function(r, delta=1.0) {
  return(ifelse(abs(r) <= delta, r, delta * sign(r)))
}

# Custom Ridge Regression with feature-weighted penalty and Huber loss
custom_ridge_regression <- function(X, y, lambda=1.0, learning_rate=0.05, epochs=300, delta=1.0) {
  n <- nrow(X)
  p <- ncol(X)
  
  # Add intercept
  X <- cbind(1, X)
  p <- p + 1
  
  # Feature importance weights (based on inverse variance)
  feature_var <- apply(X[, -1], 2, var)
  penalty_weights <- 1 / (feature_var + 1e-5)  # Avoid division by zero
  penalty_weights <- c(0, penalty_weights)     # No penalty for intercept
  
  # Initialize coefficients
  beta <- rep(0, p)
  
  for (epoch in 1:epochs) {
    y_pred <- X %*% beta
    residuals <- as.vector(y_pred - y)
    
    # Apply Huber loss derivative
    grad_r <- huber_gradient(residuals, delta)
    
    # Multiply each residual gradient to its corresponding feature row
    grad_loss <- colMeans(sweep(X, 1, grad_r, FUN = "*"))
    
    # Ridge penalty gradient (skip intercept)
    grad_penalty <- 2 * lambda * penalty_weights * beta
    
    # Total gradient
    grad_total <- grad_loss + grad_penalty
    
    # Adaptive learning rate
    lr <- learning_rate / sqrt(epoch)
    
    # Update coefficients
    beta <- beta - lr * grad_total
  }
  
  return(beta)
}

# Prediction function
predict_custom <- function(X, beta) {
  X <- cbind(1, X)
  return(X %*% beta)
}

# ===============
# DATA PREP
# ===============

# Filter and prepare data
df <- data %>%
  select(fare_amount, trip_distance, trip_duration, day_of_week,passenger_count, hour_of_day, is_weekend, is_peak_hour) %>%
  filter(fare_amount > 0)

# Encode day_of_week as dummy vars
df <- df %>%
  mutate(day_of_week = factor(day_of_week)) %>%
  mutate(across(c(trip_distance, passenger_count, trip_duration, hour_of_day), normalize)) %>%
  cbind(model.matrix(~ day_of_week - 1, data = df)) %>%
  select(-day_of_week)

# Split into training/testing
set.seed(123)
indexes <- sample(1:nrow(df), 0.8 * nrow(df))
train_df <- df[indexes, ]
test_df <- df[-indexes, ]

# Split X and y
X_train <- as.matrix(train_df[, -1])
y_train <- train_df$fare_amount

X_test <- as.matrix(test_df[, -1])
y_test <- test_df$fare_amount

feature_means <- sapply(train_df[, c("trip_distance", "trip_duration", "passenger_count", "hour_of_day")], mean)
feature_sds   <- sapply(train_df[, c("trip_distance", "trip_duration", "passenger_count", "hour_of_day")], sd)
saveRDS(feature_means, "feature_means.rds")
saveRDS(feature_sds, "feature_sds.rds")

# ===============
# TRAIN & PREDICT
# ===============
coeffs <- custom_ridge_regression(X_train, y_train, lambda=2, learning_rate=0.1, epochs=200)
preds <- predict_custom(X_test, coeffs)

saveRDS(as.numeric(coeffs), "custom_ridge_huber_weights.rds")
length(coeffs)
# ===============
# EVALUATION
# ===============
rmse_val <- sqrt(mean((preds - y_test)^2))
mae_val <- mean(abs(preds - y_test))

cat("🔍 Custom Ridge Regression Results:\n")
cat("RMSE:", round(rmse_val, 3), "\n")
cat("MAE :", round(mae_val, 3), "\n")

# Predicted vs Actual Plot
ggplot(data = data.frame(Predicted = preds, Actual = y_test), aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.4, color = "#2C3E50") +
  geom_abline(intercept = 0, slope = 1, color = "#E74C3C", linetype = "dashed", size = 1) +
  labs(
    title = "Predicted vs Actual Fare",
    x = "Actual Fare",
    y = "Predicted Fare"
  ) +
  theme_minimal()

# Residuals = Actual - Predicted
residuals <- y_test - preds

# Residual Distribution
ggplot(data = data.frame(Residuals = residuals), aes(x = Residuals)) +
  geom_histogram(binwidth = 1, fill = "#3498DB", color = "black", alpha = 0.7) +
  labs(
    title = "Residuals Distribution",
    x = "Residual (Actual - Predicted)",
    y = "Frequency"
  ) +
  theme_minimal()

library(reshape2)
# Load coefficients excluding intercept
feature_names <- colnames(X_train)
coeff_values <- coeffs[-1]  # Remove intercept

importance_df <- data.frame(
  Feature = feature_names,
  Coefficient = coeff_values,
  Importance = abs(coeff_values)
)

# Heatmap
ggplot(importance_df, aes(x = reorder(Feature, Importance), y = Importance, fill = Importance)) +
  geom_col(show.legend = FALSE) +
  coord_flip() +
  scale_fill_gradient(low = "#AED6F1", high = "#1F618D") +
  labs(
    title = "Feature Importance Heatmap",
    x = "Feature",
    y = "Absolute Coefficient (Importance)"
  ) +
  theme_minimal()


library(shiny)

# Load model weights
model_weights <- tryCatch({
  readRDS("C:/Users/HP/Desktop/SY II/Data Science/R language/custom_ridge_huber_weights.rds")
}, error = function(e) {
  NULL
})

# Custom prediction function
predict_custom_model <- function(input_values, weights) {
  if (is.null(weights) || !is.numeric(weights)) {
    stop("Model weights are missing or invalid.")
  }
  
  
  # Dummy variables for day_of_week
  day_vector <- c("Friday", "Monday", "Saturday", "Sunday", "Thursday", "Tuesday", "Wednesday")
  day_dummy <- as.numeric(day_vector == input_values$day_of_week)
  
  # Normalize numeric inputs with hardcoded training means/sd
  normalize_val <- function(x, mean, sd) {
    return((x - mean) / sd)
  }
  
  feature_means <- readRDS("feature_means.rds")
  feature_sds <- readRDS("feature_sds.rds")
  
  # Normalize numeric features
  trip_distance <- normalize_val(input_values$trip_distance, feature_means["trip_distance"], feature_sds["trip_distance"])
  trip_duration <- normalize_val(input_values$trip_duration, feature_means["trip_duration"], feature_sds["trip_duration"])
  passenger_count <- normalize_val(input_values$passenger_count, feature_means["passenger_count"], feature_sds["passenger_count"])
  hour_of_day <- normalize_val(input_values$hour_of_day, feature_means["hour_of_day"], feature_sds["hour_of_day"])

  # Combine all features
  features <- c(1,  # intercept
                trip_distance,
                trip_duration,
                passenger_count,
                hour_of_day,
                0,  # is_weekend (optional — set to 0 if unused)
                0,  # is_peak_hour (optional — set to 0 if unused)
                day_dummy)

  features <- matrix(as.numeric(features), nrow = 1)
  prediction <- as.numeric(features %*% weights)
  return(prediction)
}

# Define UI
ui <- fluidPage(
  titlePanel("Uber Fare Prediction"),
  sidebarLayout(
    sidebarPanel(
      numericInput("trip_distance", "Trip Distance:", 2.0),
      selectInput("distance_unit", "Distance Unit:", choices = c("Miles", "Kilometers")),
      numericInput("trip_duration", "Trip Duration (minutes):", 15),
      numericInput("passenger_count", "Passenger Count:", 1, min = 1, max = 6),
      selectInput("day_of_week", "Day of Week:", choices = c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")),
      sliderInput("hour_of_day", "Hour of Day:", min = 0, max = 23, value = 14),
      actionButton("predict_btn", "Predict Fare")
    ),
    mainPanel(
      h3("Predicted Fare:"),
      verbatimTextOutput("fare_output")
    )
  )
)

# Define Server
server <- function(input, output) {
  observeEvent(input$predict_btn, {
    req(model_weights)
    
    distance_miles <- if (input$distance_unit == "Kilometers") {
      input$trip_distance * 0.621371
    } else {
      input$trip_distance
    }

    user_input <- list(
      trip_distance = distance_miles,
      trip_duration = input$trip_duration,
      passenger_count = input$passenger_count,
      day_of_week = input$day_of_week,
      hour_of_day = input$hour_of_day
    )

    prediction <- tryCatch({
      predict_custom_model(user_input, model_weights)
    }, error = function(e) {
      print(e$message)
      NA
    })
    
    fare_inr <- round(prediction * 85.38, 2)

    output$fare_output <- renderText({
      if (is.na(fare_inr)) {
        "⚠️ Error predicting fare. Please check model weights or input data."
      } else {
        paste0("Predicted Fare: ₹", round(fare_inr, 2))
      }
    })
  })
}

# Launch App
shinyApp(ui = ui, server = server)
