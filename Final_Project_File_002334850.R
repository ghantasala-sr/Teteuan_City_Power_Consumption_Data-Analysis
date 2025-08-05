# Power Consumption of Tetouan City - Data Science Engineering Methods and Tools

# Install required packages
#install.packages(c("tidyverse", "lubridate", "scales", "gridExtra", 
#                   "car", "broom", "splines", "caret", "corrplot", "grid"))
# After installing the required packages , comment the above line.

# Load required libraries
library(tidyverse)
library(lubridate)
library(scales)
library(gridExtra)
library(car)
library(broom)
library(splines)
library(caret)
library(corrplot)
library(grid)

# Read the CSV file
# Make sure to set the working directory which contains the .csv file.
power_data <- read.csv("Tetuan City power consumption.csv", stringsAsFactors = FALSE)

#Feature Engineering & Data variables extraction

# Convert DateTime to proper datetime format
power_data$DateTime <- mdy_hm(power_data$DateTime)

# Extract date and time components - Feature Extraction
power_data$Date <- as.Date(power_data$DateTime)
power_data$Hour <- hour(power_data$DateTime)
power_data$Month <- month(power_data$DateTime)
power_data$DayOfWeek <- wday(power_data$DateTime, label = TRUE)
power_data$WeekdayType <- ifelse(power_data$DayOfWeek %in% c("Sat", "Sun"), "Weekend", "Weekday")
power_data$IsWeekend <- ifelse(power_data$DayOfWeek %in% c(1, 7), 1, 0)
power_data$IsPeakHour <- ifelse(power_data$Hour %in% c(18, 19, 20, 21), 1, 0)
# Create time of day categories
power_data$TimeOfDay <- case_when(
  power_data$Hour >= 0 & power_data$Hour < 6 ~ "Night",
  power_data$Hour >= 6 & power_data$Hour < 12 ~ "Morning",
  power_data$Hour >= 12 & power_data$Hour < 18 ~ "Afternoon",
  power_data$Hour >= 18 & power_data$Hour < 24 ~ "Evening"
)

# Create season categories
power_data$Season <- case_when(
  power_data$Month %in% c(12, 1, 2) ~ "Winter",
  power_data$Month %in% c(3, 4, 5) ~ "Spring",
  power_data$Month %in% c(6, 7, 8) ~ "Summer",
  power_data$Month %in% c(9, 10, 11) ~ "Fall"
)

# Create temperature categories for chi-square test
power_data$TempCategory <- cut(power_data$Temperature, 
                               breaks = c(0, 10, 20, 30, 40, Inf),
                               labels = c("Very Cold", "Cold", "Moderate", "Warm", "Hot"),
                               include.lowest = TRUE)

# Create consumption categories for chi-square test
power_data$Zone1Category <- cut(power_data$`Zone.1.Power.Consumption`, 
                                breaks = quantile(power_data$`Zone.1.Power.Consumption`, probs = seq(0, 1, 0.25)),
                                labels = c("Low", "Medium-Low", "Medium-High", "High"),
                                include.lowest = TRUE)

###############################################################################

###############################################################################

#Exploratory Data Analysis 
#includes summary statistics, boxplots, histograms and power consumption of a
#particular day

numerical_vars <- c("Temperature", "Humidity", "Wind.Speed", 
                    "general.diffuse.flows", "diffuse.flows", 
                    "Zone.1.Power.Consumption", "Zone.2..Power.Consumption", 
                    "Zone.3..Power.Consumption")

summary_stats <- power_data %>%
  summarize(across(all_of(numerical_vars), 
                   list(min = min, 
                        q1 = ~quantile(., 0.25), 
                        median = median, 
                        mean = mean, 
                        q3 = ~quantile(., 0.75), 
                        max = max, 
                        sd = sd)))

summary_long <- summary_stats %>%
  pivot_longer(everything(), 
               names_to = c("variable", "stat"), 
               names_pattern = "(.*)_(.*)",
               values_to = "value")

# Print nicely formatted summary
summary_table <- summary_long %>%
  pivot_wider(names_from = stat, values_from = value) %>%
  arrange(variable)

print(summary_table)

hist_plots <- list()
for (var in numerical_vars) {
  p <- ggplot(power_data, aes_string(x = var)) +
    geom_histogram(aes(y = after_stat(density)), bins = 30, fill = "steelblue", alpha = 0.7) +
    geom_density(color = "red", size = 1) +
    geom_vline(aes(xintercept = mean(get(var))), color = "black", linetype = "dashed", size = 1) +
    geom_vline(aes(xintercept = median(get(var))), color = "green", linetype = "dashed", size = 1) +
    theme_minimal() +
    theme(plot.title = element_text(size = 10)) +  # Smaller title for subplots
    labs(
      title = paste("Distribution of", var),
      x = var,
      y = "Density"
    )
  hist_plots[[var]] <- p
}

# Display histograms as a grid
grid.arrange(grobs = hist_plots, ncol = 2, 
             top = textGrob("Distributions of Numerical Variables", gp = gpar(fontsize = 14, fontface = "bold")))

# Create boxplot plots for zone variables by season
boxplot_plots <- list()
for (zone_var in c("Zone.1.Power.Consumption", "Zone.2..Power.Consumption", "Zone.3..Power.Consumption")) {
  p <- ggplot(power_data, aes_string(x = "Season", y = zone_var, fill = "Season")) +
    geom_boxplot() +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      plot.title = element_text(size = 10),  # Smaller title
      legend.position = "none"  # Remove redundant legends
    ) +
    labs(
      title = paste("Distribution of", zone_var, "by Season"),
      x = "Season",
      y = zone_var
    )
  boxplot_plots[[zone_var]] <- p
}

# Display boxplots as a grid
grid.arrange(grobs = boxplot_plots, ncol = 2,
             top = textGrob("Power Consumption by Season", gp = gpar(fontsize = 14, fontface = "bold")))

correlation_matrix <- cor(power_data[, numerical_vars])
corrplot(correlation_matrix, method = "color", type = "upper", 
         tl.col = "black", tl.srt = 45, 
         title = "Correlation Matrix of Numerical Variables")

# Calculate overall zone statistics
zone_stats <- power_data %>%
  summarize(
    Zone1_Mean = mean(Zone.1.Power.Consumption),
    Zone2_Mean = mean(Zone.2..Power.Consumption),
    Zone3_Mean = mean(Zone.3..Power.Consumption),
    Zone1_to_Zone2_Ratio = Zone1_Mean / Zone2_Mean,
    Zone1_to_Zone3_Ratio = Zone1_Mean / Zone3_Mean,
    Zone2_to_Zone3_Ratio = Zone2_Mean / Zone3_Mean
  )

cat("1. Power consumption ratios between zones:\n")
cat("   - Zone 1 to Zone 2 ratio:", round(zone_stats$Zone1_to_Zone2_Ratio, 2), "\n")
cat("   - Zone 1 to Zone 3 ratio:", round(zone_stats$Zone1_to_Zone3_Ratio, 2), "\n")
cat("   - Zone 2 to Zone 3 ratio:", round(zone_stats$Zone2_to_Zone3_Ratio, 2), "\n\n")

hourly_avg <- power_data %>%
  group_by(Hour) %>%
  summarize(
    Zone1_Avg = mean(Zone.1.Power.Consumption),
    Zone2_Avg = mean(Zone.2..Power.Consumption),
    Zone3_Avg = mean(Zone.3..Power.Consumption)
  ) %>%
  pivot_longer(
    cols = starts_with("Zone"),
    names_to = "Zone",
    values_to = "Average_Consumption"
  )
# Find peak hours
peak_hour <- hourly_avg %>%
  group_by(Zone) %>%
  filter(Average_Consumption == max(Average_Consumption)) %>%
  select(Zone, Hour, Average_Consumption)

cat("2. Peak consumption hours:\n")
print(peak_hour)

hourly_consumption <- power_data %>%
  group_by(Hour) %>%
  summarize(
    Zone1 = mean(`Zone.1.Power.Consumption`),
    Zone2 = mean(`Zone.2..Power.Consumption`),
    Zone3 = mean(`Zone.3..Power.Consumption`)
  )

p1 <- hourly_consumption %>%
  pivot_longer(cols = c(Zone1, Zone2, Zone3), names_to = "Zone", values_to = "Consumption") %>%
  ggplot(aes(x = Hour, y = Consumption, color = Zone, group = Zone)) +
  geom_line(linewidth = 1) +
  geom_point() +
  scale_y_continuous(labels = comma) +
  theme_minimal() +
  labs(
    title = "Hourly Power Consumption by Zone",
    x = "Hour of Day",
    y = "Average Power Consumption",
    color = "Zone"
  ) +
  theme(legend.position = "bottom")

print(p1)

monthly_consumption <- power_data %>%
  group_by(Month) %>%
  summarize(
    Zone1 = mean(`Zone.1.Power.Consumption`),
    Zone2 = mean(`Zone.2..Power.Consumption`),
    Zone3 = mean(`Zone.3..Power.Consumption`),
    Temperature = mean(Temperature)
  )
p3 <- monthly_consumption %>%
  pivot_longer(cols = c(Zone1, Zone2, Zone3), names_to = "Zone", values_to = "Consumption") %>%
  ggplot(aes(x = factor(Month), y = Consumption, fill = Zone)) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_y_continuous(labels = comma) +
  scale_x_discrete(labels = month.abb) +
  theme_minimal() +
  labs(
    title = "Monthly Average Power Consumption by Zone",
    x = "Month",
    y = "Average Power Consumption",
    fill = "Zone"
  ) +
  theme(legend.position = "bottom")
print(p3)

# Create a function to generate time-based visualizations for a specific day
analyze_specific_day <- function(date_to_analyze = "2017-06-01") {
  day_data <- power_data %>%
    filter(Date == as.Date(date_to_analyze))
  
  if(nrow(day_data) == 0) {
    return(paste("No data available for", date_to_analyze))
  }
  
  p_day <- ggplot(day_data) +
    geom_line(aes(x = DateTime, y = `Zone.1.Power.Consumption`, color = "Zone 1")) +
    geom_line(aes(x = DateTime, y = `Zone.2..Power.Consumption`, color = "Zone 2")) +
    geom_line(aes(x = DateTime, y = `Zone.3..Power.Consumption`, color = "Zone 3")) +
    scale_y_continuous(labels = comma) +
    theme_minimal() +
    labs(
      title = paste("Power Consumption on", date_to_analyze),
      x = "Time",
      y = "Power Consumption",
      color = "Zone"
    ) +
    theme(legend.position = "bottom")
  
  ggsave(paste0("daily_pattern_", date_to_analyze, ".png"), p_day, width = 10, height = 6)
  
  return(p_day)
}

# Generate visualization for a specific day
specific_day_plot <- analyze_specific_day("2017-06-01")
print(specific_day_plot)

###############################################################################
# Statistical Analysis
###############################################################################

cat("ANOVA TEST\n")
cat("Hypothesis : Mean power consumption is the same across all seasons\n")
cat("H0: μWinter = μSpring = μSummer = μFall\n")
cat("H1: At least one season has a different mean power consumption\n\n")

# Perform ANOVA by season
anova_season <- aov(`Zone.1.Power.Consumption` ~ Season, data = power_data)
print(summary(anova_season))

# Tukey HSD post-hoc test if ANOVA is significant
tukey_season <- TukeyHSD(anova_season)
print(tukey_season)

# Create visualization for ANOVA by season
season_summary <- power_data %>%
  group_by(Season) %>%
  summarize(
    Mean = mean(`Zone.1.Power.Consumption`),
    SE = sd(`Zone.1.Power.Consumption`) / sqrt(n())
  )

anova_season_plot <- ggplot(season_summary, aes(x = Season, y = Mean, fill = Season)) +
  geom_bar(stat = "identity") +
  geom_errorbar(aes(ymin = Mean - SE, ymax = Mean + SE), width = 0.2) +
  theme_minimal() +
  labs(
    title = "ANOVA: Mean Zone 1 Power Consumption by Season",
    subtitle = paste("p-value =", format.pval(summary(anova_season)[[1]]$"Pr(>F)"[1], digits = 3)),
    x = "Season",
    y = "Mean Power Consumption"
  )

print(anova_season_plot)

###############################################################################


cat("CHI-SQUARE TEST OF INDEPENDENCE\n")
cat("Hypothesis 1: Temperature category and Zone 1 power consumption category are independent\n")
cat("H0: Temperature category and Zone 1 consumption category are independent\n")
cat("H1: Temperature category and Zone 1 consumption category are not independent\n\n")

# Create contingency table
contingency_table <- table(power_data$TempCategory, power_data$Zone1Category)
print(contingency_table)

# Perform Chi-square test
chi_square_result <- chisq.test(contingency_table)
print(chi_square_result)

# Create a visualization for the Chi-square test
chi_square_data <- as.data.frame(contingency_table)
names(chi_square_data) <- c("Temperature", "Consumption", "Frequency")

chi_square_plot <- ggplot(chi_square_data, aes(x = Temperature, y = Consumption)) +
  geom_tile(aes(fill = Frequency), color = "white") +
  geom_text(aes(label = Frequency), vjust = 1) +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  theme_minimal() +
  labs(
    title = "Chi-square Test: Temperature vs. Zone 1 Consumption",
    subtitle = paste("p-value =", format.pval(chi_square_result$p.value, digits = 3)),
    x = "Temperature Category",
    y = "Zone 1 Consumption Category"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

print(chi_square_plot)

###############################################################################

cat("PAIRED T-TEST: HighTemp & LowTemp \n")

cat("\nHypothesis 7: There is no difference in power consumption between high and low temperature days\n")
cat("H0: μHighTemp - μLowTemp = 0\n")
cat("H1: μHighTemp - μLowTemp ≠ 0\n\n")

# Create high and low temperature groups
temp_median <- median(power_data$Temperature)
high_temp_data <- power_data %>% filter(Temperature > temp_median) %>% pull(`Zone.1.Power.Consumption`)
low_temp_data <- power_data %>% filter(Temperature <= temp_median) %>% pull(`Zone.1.Power.Consumption`)

# Perform independent t-test between high and low temperature days
ttest_temp <- t.test(high_temp_data, low_temp_data)
print(ttest_temp)

# Create visualization for t-test between high and low temperature days
temp_group_summary <- power_data %>%
  mutate(TempGroup = ifelse(Temperature > temp_median, "High Temperature", "Low Temperature")) %>%
  group_by(TempGroup) %>%
  summarize(
    Mean = mean(`Zone.1.Power.Consumption`),
    SE = sd(`Zone.1.Power.Consumption`) / sqrt(n())
  )

ttest_temp_plot <- ggplot(temp_group_summary, aes(x = TempGroup, y = Mean, fill = TempGroup)) +
  geom_bar(stat = "identity") +
  geom_errorbar(aes(ymin = Mean - SE, ymax = Mean + SE), width = 0.2) +
  theme_minimal() +
  labs(
    title = "t-test: Zone 1 Power Consumption by Temperature Group",
    subtitle = paste("p-value =", format.pval(ttest_temp$p.value, digits = 3)),
    x = "Temperature Group",
    y = "Mean Power Consumption"
  )

print(ttest_temp_plot)

###############################################################################

cat("PAIRED T-TEST: ZONE 2 vs ZONE 3\n")

cat("Hypothesis: There is no difference in power consumption between Zone 2 and Zone 3\n")
cat("H0: μZone2 - μZone3 = 0\n")
cat("H1: μZone2 - μZone3 ≠ 0\n\n")

# Perform paired t-test between Zone 2 and Zone 3
ttest_zones_2_3 <- t.test(power_data$`Zone.2..Power.Consumption`, 
                          power_data$`Zone.3..Power.Consumption`, 
                          paired = TRUE)
print(ttest_zones_2_3)

# Calculate difference statistics for visualization
zone_diff <- power_data$`Zone.2..Power.Consumption` - power_data$`Zone.3..Power.Consumption`
zone_diff_mean <- mean(zone_diff)
zone_diff_sd <- sd(zone_diff)
zone_diff_se <- zone_diff_sd / sqrt(length(zone_diff))
conf_interval <- ttest_zones_2_3$conf.int

# Summary statistics
cat("\nSummary statistics:\n")
cat("Mean difference (Zone 2 - Zone 3):", zone_diff_mean, "\n")
cat("Standard deviation of difference:", zone_diff_sd, "\n")
cat("95% Confidence interval:", conf_interval[1], "to", conf_interval[2], "\n")

# Create visualization for paired t-test between Zone 2 and Zone 3
ttest_zones_2_3_plot <- ggplot(data.frame(diff = zone_diff), aes(x = diff)) +
  geom_histogram(aes(y = ..density..), bins = 50, fill = "lightblue", color = "black") +
  geom_density(color = "red", size = 1) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "blue", size = 1) +
  geom_vline(xintercept = zone_diff_mean, linetype = "solid", color = "green", size = 1) +
  geom_vline(xintercept = conf_interval[1], linetype = "dotted", color = "darkgreen", size = 0.8) +
  geom_vline(xintercept = conf_interval[2], linetype = "dotted", color = "darkgreen", size = 0.8) +
  theme_minimal() +
  labs(
    title = "Paired t-test: Zone 2 - Zone 3 Power Consumption",
    subtitle = paste("p-value =", format.pval(ttest_zones_2_3$p.value, digits = 3),
                     "| Mean Difference =", round(zone_diff_mean, 2)),
    x = "Difference (Zone 2 - Zone 3)",
    y = "Density"
  )

print(ttest_zones_2_3_plot)

# Create a boxplot comparing the two zones
zone_comparison <- power_data %>%
  select(`Zone.2..Power.Consumption`, `Zone.3..Power.Consumption`) %>%
  rename(Zone2 = `Zone.2..Power.Consumption`, Zone3 = `Zone.3..Power.Consumption`) %>%
  pivot_longer(cols = c(Zone2, Zone3), names_to = "Zone", values_to = "Power")

boxplot_zones_2_3 <- ggplot(zone_comparison, aes(x = Zone, y = Power, fill = Zone)) +
  geom_boxplot() +
  theme_minimal() +
  scale_fill_manual(values = c("Zone2" = "skyblue", "Zone3" = "lightgreen")) +
  labs(
    title = "Power Consumption: Zone 2 vs. Zone 3",
    subtitle = paste("Paired t-test p-value =", format.pval(ttest_zones_2_3$p.value, digits = 3)),
    y = "Power Consumption",
    x = "Zone"
  )

print(boxplot_zones_2_3)

# Time series view of both zones to visualize patterns
# Sample a week of data for better visualization
sample_data <- power_data %>%
  filter(Date >= min(Date) & Date < min(Date) + 7) %>%
  select(DateTime, `Zone.2..Power.Consumption`, `Zone.3..Power.Consumption`) %>%
  rename(Zone2 = `Zone.2..Power.Consumption`, Zone3 = `Zone.3..Power.Consumption`) %>%
  pivot_longer(cols = c(Zone2, Zone3), names_to = "Zone", values_to = "Power")

time_series_plot <- ggplot(sample_data, aes(x = DateTime, y = Power, color = Zone)) +
  geom_line() +
  scale_color_manual(values = c("Zone2" = "blue", "Zone3" = "green")) +
  theme_minimal() +
  labs(
    title = "Time Series: Zone 2 vs. Zone 3 Power Consumption",
    subtitle = "One week sample",
    x = "Date Time",
    y = "Power Consumption",
    color = "Zone"
  )

print(time_series_plot)

# Create a combined visualization
combined_plots <- grid.arrange(
  ttest_zones_2_3_plot, 
  boxplot_zones_2_3,
  time_series_plot,
  ncol = 2
)

print(combined_plots)


cat("ANOVA: COMPARING MULTIPLE GROUPS\n")

# Question : Is there a significant difference in power consumption across seasons?
a1 <- aov(Zone.1.Power.Consumption ~ Season, data = power_data)
summary_a1 <- summary(a1)
print(summary_a1)

# Post-hoc Tukey test
tukey_a1 <- TukeyHSD(a1)
print(tukey_a1)

# Prepare data for visualization
season_summary <- power_data %>%
  group_by(Season) %>%
  summarize(
    Mean = mean(Zone.1.Power.Consumption),
    SD = sd(Zone.1.Power.Consumption),
    SE = SD / sqrt(n()),
    Count = n()
  )

# Reorder seasons chronologically
season_summary$Season <- factor(season_summary$Season, levels = c("Winter", "Spring", "Summer", "Fall"))

# Create a bar plot with error bars
a1_plot <- ggplot(season_summary, aes(x = Season, y = Mean, fill = Season)) +
  geom_bar(stat = "identity") +
  geom_errorbar(aes(ymin = Mean - SE, ymax = Mean + SE), width = 0.2) +
  scale_fill_brewer(palette = "Set3") +
  scale_y_continuous(labels = comma) +
  theme_minimal() +
  labs(
    title = "Power Consumption by Season",
    subtitle = paste("ANOVA p-value:", format.pval(summary_a1[[1]]$"Pr(>F)"[1], digits = 3)),
    x = "Season",
    y = "Mean Zone 1 Power Consumption",
    fill = "Season"
  ) +
  theme(
    plot.title = element_text(face = "bold"),
    legend.position = "none"
  )

print(a1_plot)


###############################################################################
# Multiple Linear Regression - Prediction Modelling
###############################################################################

# Define zone columns and numerical predictors
zone_cols <- c("Zone.1.Power.Consumption", "Zone.2..Power.Consumption", "Zone.3..Power.Consumption")
num_predictors <- c("Temperature", "Humidity", "Wind.Speed", "general.diffuse.flows", "diffuse.flows")

# Additional features we'll use
cat_predictors <- c("Hour", "Month", "DayOfWeek", "IsWeekend", "TimeOfDay", "Season")

# Create dummy variables for categorical predictors with multiple levels
# First, ensure factors
power_data$TimeOfDay <- as.factor(power_data$TimeOfDay)
power_data$Season <- as.factor(power_data$Season)
power_data$DayOfWeek <- as.factor(power_data$DayOfWeek)

# Split data into training and test sets (70/30)
set.seed(123)
train_indices <- createDataPartition(power_data$Zone.1.Power.Consumption, p = 0.7, list = FALSE)
train_data <- power_data[train_indices, ]
test_data <- power_data[-train_indices, ]

# Function to scale numerical predictors
scale_data <- function(train, test, predictors) {
  means <- sapply(train[, predictors], mean)
  sds <- sapply(train[, predictors], sd)
  
  # Scale training data
  train_scaled <- train
  for (col in predictors) {
    train_scaled[[col]] <- (train[[col]] - means[[col]]) / sds[[col]]
  }
  
  # Scale test data using training parameters
  test_scaled <- test
  for (col in predictors) {
    test_scaled[[col]] <- (test[[col]] - means[[col]]) / sds[[col]]
  }
  
  return(list(train = train_scaled, test = test_scaled))
}

# Scale the data
scaled_data <- scale_data(train_data, test_data, num_predictors)
train_scaled <- scaled_data$train
test_scaled <- scaled_data$test

# Function to build and evaluate MLR model
build_evaluate_mlr <- function(zone_col) {
  # Create formula - include numerical predictors and categorical variables
  formula <- as.formula(paste(
    zone_col, "~", 
    paste(c(num_predictors, "Hour", "Month", "IsWeekend", "TimeOfDay", "Season"), collapse = " + ")
  ))
  
  # Train the model
  model <- lm(formula, data = train_scaled)
  
  # Make predictions on test data
  predictions <- predict(model, newdata = test_scaled)
  
  # Calculate metrics
  rmse <- sqrt(mean((predictions - test_scaled[[zone_col]])^2))
  mae <- mean(abs(predictions - test_scaled[[zone_col]]))
  r2 <- cor(predictions, test_scaled[[zone_col]])^2
  mape <- mean(abs((test_scaled[[zone_col]] - predictions) / test_scaled[[zone_col]])) * 100
  accuracy <- 100 - mape
  
  # Return results
  return(list(
    model = model,
    predictions = predictions,
    rmse = rmse,
    mae = mae,
    r2 = r2,
    mape = mape,
    accuracy = accuracy
  ))
}

# Build and evaluate models for all zones
zone1_model <- build_evaluate_mlr(zone_cols[1])
zone2_model <- build_evaluate_mlr(zone_cols[2])
zone3_model <- build_evaluate_mlr(zone_cols[3])

cat("MULTIPLE LINEAR REGRESSION PERFORMANCE METRICS\n")

cat("Zone 1 MLR Performance:\n")
cat("  RMSE:", round(zone1_model$rmse, 2), "\n")
cat("  MAE:", round(zone1_model$mae, 2), "\n")
cat("  R²:", round(zone1_model$r2, 4), "\n")
cat("  MAPE:", round(zone1_model$mape, 2), "%\n")
cat("  Accuracy:", round(zone1_model$accuracy, 2), "%\n\n")

cat("Zone 2 MLR Performance:\n")
cat("  RMSE:", round(zone2_model$rmse, 2), "\n")
cat("  MAE:", round(zone2_model$mae, 2), "\n")
cat("  R²:", round(zone2_model$r2, 4), "\n")
cat("  MAPE:", round(zone2_model$mape, 2), "%\n")
cat("  Accuracy:", round(zone2_model$accuracy, 2), "%\n\n")

cat("Zone 3 MLR Performance:\n")
cat("  RMSE:", round(zone3_model$rmse, 2), "\n")
cat("  MAE:", round(zone3_model$mae, 2), "\n")
cat("  R²:", round(zone3_model$r2, 4), "\n")
cat("  MAPE:", round(zone3_model$mape, 2), "%\n")
cat("  Accuracy:", round(zone3_model$accuracy, 2), "%\n\n")

# Create a comparison table
performance_table <- data.frame(
  Zone = c("Zone 1", "Zone 2", "Zone 3"),
  RMSE = c(zone1_model$rmse, zone2_model$rmse, zone3_model$rmse),
  MAE = c(zone1_model$mae, zone2_model$mae, zone3_model$mae),
  R_Squared = c(zone1_model$r2, zone2_model$r2, zone3_model$r2),
  MAPE = c(zone1_model$mape, zone2_model$mape, zone3_model$mape),
  Accuracy = c(zone1_model$accuracy, zone2_model$accuracy, zone3_model$accuracy)
)

# Print the table
print(performance_table)

# Visualize actual vs predicted values
# Create a list to hold all three prediction plots
prediction_plots <- list()

for (i in 1:3) {
  zone_col <- zone_cols[i]
  model <- get(paste0("zone", i, "_model"))
  
  plot_data <- data.frame(
    Actual = test_scaled[[zone_col]],
    Predicted = model$predictions
  )
  
  p <- ggplot(plot_data, aes(x = Actual, y = Predicted)) +
    geom_point(alpha = 0.5, color = "steelblue") +
    geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
    theme_minimal() +
    labs(
      title = paste("Zone", i, "MLR: Actual vs. Predicted"),
      subtitle = paste("R² =", round(model$r2, 4), "Accuracy =", round(model$accuracy, 2), "%"),
      x = "Actual Power Consumption",
      y = "Predicted Power Consumption"
    )
  
  prediction_plots[[i]] <- p
}

# Display all three plots in a single figure
grid.arrange(grobs = prediction_plots, ncol = 2, 
             top = "Multiple Linear Regression Performance Across Zones")

# Function to predict power consumption for all three zones
predict_power_consumption <- function(new_data, zone1_model, zone2_model, zone3_model, scaling_params) {
  # Check if required columns exist
  required_cols <- c("Temperature", "Humidity", "Wind.Speed", 
                     "general.diffuse.flows", "diffuse.flows", 
                     "Hour", "Month", "IsWeekend", "TimeOfDay", "Season")
  
  missing_cols <- setdiff(required_cols, names(new_data))
  if (length(missing_cols) > 0) {
    stop("Missing required columns: ", paste(missing_cols, collapse = ", "))
  }
  
  # Scale numerical predictors using the same parameters as training
  num_predictors <- c("Temperature", "Humidity", "Wind.Speed", 
                      "general.diffuse.flows", "diffuse.flows")
  
  # Create a copy of the new data
  scaled_data <- new_data
  
  # Scale each numerical predictor
  for (col in num_predictors) {
    if (col %in% names(new_data)) {
      scaled_data[[col]] <- (new_data[[col]] - scaling_params$means[[col]]) / 
        scaling_params$sds[[col]]
    }
  }
  
  # Make predictions for each zone
  predictions <- data.frame(
    Zone1_Prediction = predict(zone1_model$model, newdata = scaled_data),
    Zone2_Prediction = predict(zone2_model$model, newdata = scaled_data),
    Zone3_Prediction = predict(zone3_model$model, newdata = scaled_data)
  )
  
  
  # Return predictions
  return(predictions)
}

# Usage example:

# First, we need to get the scaling parameters
scaling_params <- list(
  means = sapply(train_data[, num_predictors], mean),
  sds = sapply(train_data[, num_predictors], sd)
)

# Create sample new data
new_data <- data.frame(
  Temperature = c(15, 25, 35),
  Humidity = c(50, 60, 70),
  Wind.Speed = c(1, 2, 3),
  general.diffuse.flows = c(100, 200, 300),
  diffuse.flows = c(50, 75, 100),
  Hour = c(9, 15, 20),
  Month = c(1, 6, 12),
  IsWeekend = c(0, 0, 1),
  TimeOfDay = factor(c("Morning", "Afternoon", "Evening"), 
                     levels = levels(power_data$TimeOfDay)),
  Season = factor(c("Winter", "Summer", "Winter"), 
                  levels = levels(power_data$Season))
)

# Make predictions
predicted_consumption <- predict_power_consumption(
  new_data = new_data,
  zone1_model = zone1_model,
  zone2_model = zone2_model,
  zone3_model = zone3_model,
  scaling_params = scaling_params
)

# Print predictions
print("Predicted Power Consumption:")
print(predicted_consumption)


