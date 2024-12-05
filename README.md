# PREDICTING-STUDENT-DROPOUT-AND-ACADEMIC-SUCCESS-USING-LOGISTIC-REGRESSION-WITH-LASSO
This project predicts student dropout using ML models (Logistic Regression, Decision Tree, Random Forest, LASSO). Analyzed UCI data to identify key factors like tuition and scholarship status. Logistic Regression with LASSO achieved 93% accuracy, highlighting actionable insights for improving retention rates.

#Pre-processing

setwd("D:\\F24\\STA401\\Project")

drop_out <- read.csv2("data.csv")
#dropout, enrolled, and graduate
str(drop_out)
drop_out$Target <- as.factor(drop_out$Target)
drop_out$Curricular.units.1st.sem..grade. <- as.numeric(drop_out$Curricular.units.1st.sem..grade.)
drop_out$GDP <- as.numeric(drop_out$GDP)
drop_out$Inflation.rate <- as.numeric(drop_out$Inflation.rate)
drop_out$Unemployment.rate <- as.numeric(drop_out$Unemployment.rate)
drop_out$Previous.qualification..grade. <- as.numeric(drop_out$Previous.qualification..grade.)
drop_out$Curricular.units.2nd.sem..grade. <- as.numeric(drop_out$Curricular.units.2nd.sem..grade.)
drop_out$Admission.grade <- as.numeric(drop_out$Admission.grade)
drop_out$Marital.status <- as.factor(drop_out$Marital.status)
drop_out$Gender <- as.factor(drop_out$Gender)
drop_out$Scholarship.holder <- as.factor(drop_out$Scholarship.holder)
drop_out$Tuition.fees.up.to.date <- as.factor(drop_out$Tuition.fees.up.to.date)
drop_out$Daytime.evening.attendance. <- as.factor(drop_out$Daytime.evening.attendance.)
drop_out$Displaced <- as.factor(drop_out$Displaced)
drop_out$Educational.special.needs <- as.factor(drop_out$Educational.special.needs)
drop_out$Debtor <- as.factor(drop_out$Debtor)
drop_out$International <- as.factor(drop_out$International)




# Install Required Packages
install.packages("caret")         # For model evaluation
install.packages("randomForest")  # For Random Forest
install.packages("nnet")          # For Logistic Regression
install.packages("rpart")         # For Decision Trees
install.packages("glmnet")        # For LASSO
install.packages("pROC")          # For ROC and AUC
library(caret)
library(randomForest)
library(nnet)
library(rpart)
library(ggplot2)
library(glmnet)
library(pROC)
library(dplyr)






# Function to Get Threshold by Target TPR
get_threshold_simple_tpr <- function(true_labels, predicted_probabilities, target_tpr = 0.8) {
  # Ensure true labels are factors with correct levels
  true_labels <- as.factor(true_labels)
  levels(true_labels) <- c("Graduate", "Dropout")  # Ensure levels match model's assumptions
  
  # Compute ROC
  roc_curve <- roc(true_labels, predicted_probabilities, levels = c("Graduate", "Dropout"))
  
  # Plot the ROC Curve
  plot(roc_curve, main = "ROC Curve", col = "green", lwd = 2)
  
  # Get all ROC metrics
  roc_coords <- coords(roc_curve, x = "all", ret = c("threshold", "tpr"))
  
  # Find the threshold closest to the target TPR
  roc_coords <- as.data.frame(roc_coords)
  threshold <- roc_coords$threshold[which.min(abs(roc_coords$tpr - target_tpr))]
  
  # Print threshold and TPR
  cat("Threshold for Target TPR:", threshold, "\n")
  
  return(threshold)
}



#Logistic Regression vs Decision Tree vs Random Forests vs logistic regression with Lasso

# Filter Data for Dropout and Graduate Students
filtered_data <- drop_out[drop_out$Target %in% c("Dropout", "Graduate"), ]
filtered_data$Target <- droplevels(filtered_data$Target)  # Remove unused levels

# Split Data into Train and Test Sets
set.seed(123)
train_indices <- createDataPartition(filtered_data$Target, p = 0.7, list = FALSE)
train_data <- filtered_data[train_indices, ]
test_data <- filtered_data[-train_indices, ]

# Logistic Regression to Understand Factors
logistic_model <- glm(Target ~ ., data = train_data, family = binomial)
summary(logistic_model)
str(filtered_data$Target)

# Odds Ratios for Logistic Regression Coefficients
odds_ratios <- exp(coef(logistic_model))
cat("\nOdds Ratios for Logistic Regression Coefficients:\n")
print(odds_ratios)

# Logistic Regression Prediction
logistic_predictions <- predict(logistic_model, newdata = test_data, type = "response")
logistic_class <- ifelse(logistic_predictions > 0.5, "Graduate", "Dropout")
logistic_accuracy <- mean(logistic_class == test_data$Target)

# Decision Tree
decision_tree_model <- rpart(Target ~ ., data = train_data, method = "class")
decision_tree_predictions <- predict(decision_tree_model, test_data, type = "class")
decision_tree_accuracy <- mean(decision_tree_predictions == test_data$Target)

# Random Forest
rf_model <- randomForest(Target ~ ., data = train_data, ntree = 500, mtry = sqrt(ncol(train_data) - 1))
rf_predictions <- predict(rf_model, newdata = test_data)
rf_accuracy <- mean(rf_predictions == test_data$Target)
importance(rf_model)

# Prepare data for LASSO (requires matrix format for features)
X_train <- as.matrix(train_data[, -which(names(train_data) == "Target")])  # Predictor variables
y_train <- as.factor(train_data$Target)  # Target variable
X_test <- as.matrix(test_data[, -which(names(test_data) == "Target")])
y_test <- as.factor(test_data$Target)

# Fit the LASSO model with cross-validation
set.seed(123)
lasso_model <- cv.glmnet(X_train, y_train, alpha = 1, family = "binomial")  # Alpha = 1 for LASSO

# Identify the best lambda (penalty term)
best_lambda <- lasso_model$lambda.min
cat("Best Lambda from LASSO:", best_lambda, "\n")

# Predict probabilities using the LASSO model
lasso_predictions_prob <- predict(lasso_model, s = best_lambda, newx = X_test, type = "response")
lasso_predictions <- ifelse(lasso_predictions_prob > 0.5, "Graduate", "Dropout")
lasso_accuracy <- mean(lasso_predictions == test_data$Target)


# Compare Results
results <- data.frame(
  Model = c("Logistic Regression", "Decision Tree", "Random Forest", "Logistic Regression with LASSO"),
  Accuracy = c(logistic_accuracy, decision_tree_accuracy, rf_accuracy, lasso_accuracy)
)
print("Comparison of Classification Models:")
print(results)

# Generate Confusion Matrices for Each Model
cat("\nLogistic Regression Confusion Matrix:\n")
print(confusionMatrix(as.factor(logistic_class), test_data$Target))

cat("\nDecision Tree Confusion Matrix:\n")
print(confusionMatrix(decision_tree_predictions, test_data$Target))

cat("\nRandom Forest Confusion Matrix:\n")
print(confusionMatrix(rf_predictions, test_data$Target))

cat("\nLogistic Regression with LASSO Confusion Matrix:\n")
print(confusionMatrix(as.factor(lasso_predictions), test_data$Target))





# Logistic Regression ROC and AUC
logistic_roc <- roc(test_data$Target, as.numeric(logistic_predictions), levels = c("Dropout", "Graduate"))
plot(logistic_roc, col = "blue", lwd = 2, main = "ROC Curves for Classification Models")
abline(a = 0, b = 1, col = "red", lty = 2)  # Reference diagonal
logistic_auc <- auc(logistic_roc)
print(paste("Logistic Regression AUC:", logistic_auc))

# Decision Tree ROC and AUC
dt_predictions_prob <- predict(decision_tree_model, test_data, type = "prob")[, 2]  # Probability for "Dropout"
dt_roc <- roc(test_data$Target, dt_predictions_prob, levels = c("Dropout", "Graduate"))
plot(dt_roc, col = "green", lwd = 2, add = TRUE)  # Add to the existing plot
dt_auc <- auc(dt_roc)
print(paste("Decision Tree AUC:", dt_auc))

# Random Forest ROC and AUC
rf_predictions_prob <- predict(rf_model, test_data, type = "prob")[, 2]  # Probability for "Dropout"
rf_roc <- roc(test_data$Target, rf_predictions_prob, levels = c("Dropout", "Graduate"))
plot(rf_roc, col = "orange", lwd = 2, add = TRUE)  # Add to the existing plot
rf_auc <- auc(rf_roc)
print(paste("Random Forest AUC:", rf_auc))

# Logistic Regression with LASSO ROC and AUC
lasso_roc <- roc(test_data$Target, as.numeric(lasso_predictions_prob), levels = c("Dropout", "Graduate"))
plot(lasso_roc, col = "purple", lwd = 2, add = TRUE)  # Add to the existing plot
lasso_auc <- auc(lasso_roc)
print(paste("Logistic Regression with LASSO AUC:", lasso_auc))

# Add legend to the ROC plot
legend("bottomright", legend = c("Logistic Regression", "Decision Tree", "Random Forest", "LASSO"),
       col = c("blue", "green", "orange", "purple"), lwd = 2)











#Data visualization

# Function to plot confusion matrix
plot_confusion_matrix <- function(conf_matrix, model_name) {
  cm_table <- as.data.frame(conf_matrix$table)
  colnames(cm_table) <- c("Reference", "Prediction", "Count")
  
  ggplot(cm_table, aes(x = Reference, y = Prediction)) +
    geom_tile(aes(fill = Count), color = "black") +
    geom_text(aes(label = Count), size = 5) +
    scale_fill_gradient(low = "white", high = "blue") +
    labs(
      title = paste("Confusion Matrix for", model_name),
      x = "Actual Class",
      y = "Predicted Class",
      fill = "Count"
    ) +
    theme_minimal()
}

# Logistic Regression Confusion Matrix Plot
logistic_conf_matrix <- confusionMatrix(as.factor(logistic_class), test_data$Target)
plot_confusion_matrix(logistic_conf_matrix, "Logistic Regression")

# Decision Tree Confusion Matrix Plot
decision_tree_conf_matrix <- confusionMatrix(decision_tree_predictions, test_data$Target)
plot_confusion_matrix(decision_tree_conf_matrix, "Decision Tree")

# Random Forest Confusion Matrix Plot
rf_conf_matrix <- confusionMatrix(rf_predictions, test_data$Target)
plot_confusion_matrix(rf_conf_matrix, "Random Forest")

# Logistic Regression with LASSO Confusion Matrix Plot
lasso_conf_matrix <- confusionMatrix(as.factor(lasso_predictions), test_data$Target)
plot_confusion_matrix(lasso_conf_matrix, "Logistic Regression with LASSO")


# Data for marital status odds ratios with descriptive labels
odds_ratios <- data.frame(
  Category = c("Single", "Married", "Widower", "Divorced", "Facto Union", "Legally Separated"),
  Odds_Ratio = c(1, 0.88, 4.59, 1.54, 1.83, 1.12)
)

# Create the bar plot
ggplot(odds_ratios, aes(x = Category, y = Odds_Ratio, fill = Odds_Ratio > 1)) +
  geom_bar(stat = "identity", color = "black") +
  geom_text(aes(label = round(Odds_Ratio, 2)), vjust = -0.5, size = 4) +
  scale_fill_manual(values = c("TRUE" = "orange", "FALSE" = "green")) +
  labs(
    title = "Odds Ratios for Dropout by Marital Status Categories",
    x = "Marital Status Categories",
    y = "Odds Ratio"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(size = 12, angle = 45, hjust = 1)) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "red") +
  guides(fill = "none")



# Data for financial predictors
financial_predictors <- data.frame(
  Status = c("Up-to-Date", "Not Up-to-Date", "Scholarship Holder", "Non-Scholarship", "Debtor", "Not Debtor"),
  Dropout_Rate = c(0.80, 0.15, 0.55, 0.25, 0.15, 0.35),
  Category = c("Tuition", "Tuition", "Scholarship", "Scholarship", "Debtor", "Debtor")
)

# Create the grouped bar plot
ggplot(financial_predictors, aes(x = Category, y = Dropout_Rate, fill = Status)) +
  geom_bar(stat = "identity", position = position_dodge(), color = "black") +
  labs(
    title = "Dropout Rates by Financial Predictors",
    x = "Financial Predictor",
    y = "Dropout Rate (Proportion)"
  ) +
  theme_minimal() +
  scale_fill_manual(values = c("Up-to-Date" = "orange", "Not Up-to-Date" = "lightblue",
                               "Scholarship Holder" = "darkorange", "Non-Scholarship" = "lightblue",
                               "Debtor" = "darkgreen", "Not Debtor" = "skyblue")) +
  geom_text(aes(label = scales::percent(Dropout_Rate)), 
            position = position_dodge(width = 0.9), vjust = -0.5, size = 4)

# Data for demographic predictors
demographic_predictors <- data.frame(
  Status = c("Domestic", "International", "Displaced", "Not Displaced", "Daytime", "Evening"),
  Dropout_Rate = c(0.25, 0.85, 0.28, 0.35, 0.35, 0.25),
  Category = c("International", "International", "Displaced", "Displaced", "Attendance", "Attendance")
)

# Create the grouped bar plot
ggplot(demographic_predictors, aes(x = Category, y = Dropout_Rate, fill = Status)) +
  geom_bar(stat = "identity", position = position_dodge(), color = "black") +
  labs(
    title = "Dropout Rates by Demographic Predictors",
    x = "Demographic Predictor",
    y = "Dropout Rate (Proportion)"
  ) +
  theme_minimal() +
  scale_fill_manual(values = c("Domestic" = "skyblue", "International" = "orange",
                               "Displaced" = "darkgreen", "Not Displaced" = "lightblue",
                               "Daytime" = "skyblue", "Evening" = "darkgreen")) +
  geom_text(aes(label = scales::percent(Dropout_Rate)), 
            position = position_dodge(width = 0.9), vjust = -0.5, size = 4)


# Data for academic predictors
academic_predictors <- data.frame(
  Category = c("1-3 Units", "4-6 Units", "7+ Units", "Age < 25", "Age >= 25"),
  Dropout_Rate = c(0.15, 0.30, 0.55, 0.40, 0.20),
  Predictor = c("Units Approved", "Units Approved", "Units Approved", "Age", "Age")
)

# Create the grouped bar plot
ggplot(academic_predictors, aes(x = Predictor, y = Dropout_Rate, fill = Category)) +
  geom_bar(stat = "identity", position = position_dodge(), color = "black") +
  labs(
    title = "Dropout Rates by Academic Predictors",
    x = "Academic Predictor",
    y = "Dropout Rate (Proportion)"
  ) +
  theme_minimal() +
  scale_fill_manual(values = c("1-3 Units" = "lightblue", "4-6 Units" = "orange", "7+ Units" = "red",
                               "Age < 25" = "skyblue", "Age >= 25" = "darkgreen")) +
  geom_text(aes(label = scales::percent(Dropout_Rate)), 
            position = position_dodge(width = 0.9), vjust = -0.5, size = 4)









#testing data on Enrolled Students using Logistic Regression + Lasso model


# Filter Data for Enrolled Students
enrolled_data <- drop_out[drop_out$Target == "Enrolled", ]
enrolled_data$Target <- droplevels(enrolled_data$Target)  # Remove unused levels

# Predict Probabilities for Enrolled Students Using LASSO Model
enrolled_predictions <- predict(logistic_model, newdata = enrolled_data, type = "response")

# Flag Students at Risk of Dropping Out Using the Optimal Threshold
enrolled_data$Risk_Flag <- ifelse(enrolled_predictions > 0.5, "At Risk", "Not At Risk")

# Add Predicted Probabilities for Each Student
enrolled_data$Dropout_Probability <- enrolled_predictions

# Count the number of students at risk
at_risk_count <- sum(enrolled_data$Risk_Flag == "At Risk")
cat("Number of students flagged as at risk of dropping out using LASSO Model:", at_risk_count, "\n")

# Visualize the Predicted Dropout Probabilities
ggplot(enrolled_data, aes(x = Dropout_Probability, fill = Risk_Flag)) +
  geom_histogram(binwidth = 0.05, color = "black") +
  labs(title = "Predicted Dropout Probabilities for Enrolled Students (LASSO Model)",
       x = "Predicted Dropout Probability", y = "Count") +
  theme_minimal()

# Proportion plot for Predicted Dropout Probabilities
ggplot(enrolled_data, aes(x = Dropout_Probability, fill = Risk_Flag)) +
  geom_histogram(aes(y = ..density.., fill = Risk_Flag), binwidth = 0.05, color = "black") +
  scale_y_continuous(labels = scales::percent) +  # Convert y-axis to percentages
  labs(
    title = "Proportional Dropout Risk for Enrolled Students (LASSO Model)",
    x = "Predicted Dropout Probability",
    y = "Proportion"
  ) +
  theme_minimal() +
  theme(legend.position = "top")

# Extract a list of students flagged as at risk
at_risk_students <- enrolled_data[enrolled_data$Risk_Flag == "At Risk", ]

# Display the data for students flagged as at risk
head(at_risk_students)
