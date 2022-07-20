#Loan Default Predict Group Project
rm(list=ls())
cat("\014")
 
# Read the data frame
train <- read.csv(file='train_v2.csv', stringsAsFactors = FALSE)
test <- read.csv(file='test_v2.csv', stringsAsFactors = FALSE)

### Data Preprocessing:
## 1. Add a binary default status column
## 2. Check if there are duplicate
## 3. Identify missing data
## 4. delete rows where there are more than 10% of missing data
## 5. replaced NA with mean of each column
## 6. Normalize/Scaling data frame

# Train and test are the original dataframe
# Train.new and test.new are the dataframe after preprocessing, use them for analysis
# Train.nona and test.nona deleted all rows with any missing value, not ideal.
# Train.norm and test.norm are normalized dataframe

## 1. 
# Add default status: 0 represent no default and 1 represent default occuring.
train$default <- train$loss
train$default[train$default != 0] <- 1

## 2. 
# Check duplicate
train[duplicated(train$id),] #This will give you duplicate rows
dim(train[duplicated(train$id),])[1] #This will give you the number of duplicates

test[duplicated(test$id),]
dim(test[duplicated(test$id),])[1]

# After checking, there are no duplicate values in datasets
test = test[,!duplicated(names(test))]#remove duplicated values
train = train[,!duplicated(names(train))]

## 3
# Identify number of NA in data frame
sum(is.na(train))
sum(is.na(test))

# See if there are missing id
sum(is.na(train$id))
sum(is.na(test$id))

# See if missing loss/default data
sum(is.na(train$loss))
sum(is.na(test$loss))

## 4.
# Delete ID 
# Generate % of NA in each row
# Delete rows with more than 10% of NA and make a new dataframe
row.names(train) <- train[,1]
train <- train[,-1]
row.names(test) <- test[,1]
test <- test[,-1]

# % of missing value = (na in the same row) / (# of columns)
train$missing <- rowSums(is.na(train))/ncol(train)
train$missing
train.new <- train[train$missing < 0.1, ]

test$missing <- rowSums(is.na(test))/ncol(test)
test$missing
test.new <- test[test$missing < 0.1, ]

## 4.5
# If we remove all rows with any NA value, only ~50% of data remains, so this might not be ideal
train.nona <- train[train$missing == 0, ]
test.nona <- test[test$missing == 0, ]

## 5.
# Replacing rest of na with mean of the same column 
for(i in 1:ncol(train.new)){
  train.new[is.na(train.new[,i]), i] <- mean(train[,i], na.rm = TRUE)
}

for(i in 1:ncol(test.new)){
  test.new[is.na(test.new[,i]), i] <- mean(test[,i], na.rm = TRUE)
}

## 6.
# Normalization and scaling
train.norm <- data.frame(sapply(train.new, scale))
row.names(train.norm) <- row.names(train.new)

test.norm <- data.frame(sapply(test.new, scale))
row.names(test.norm) <- row.names(test.new)

### Regression 
## Perform pca before regression since there are over 700 predictors
## Create data for pca by removing predictors that have 0 variance

# Remove columns that have categorical variables from scaled training set
pca.data <- data.frame(t(na.omit(t(train.norm[ , -c(770,771,772)]))))

# Perform pca
pca <- prcomp(pca.data)
summary(pca)

# Visualize the results of pca by creating scree plot
library(ggplot2)

# First, create vector of variances for each generated principal component
pca.var <- pca$sdev^2

# Proportion of variance explained by each principal component
pve <- pca.var/sum(pca.var)
pca.pve <- data.frame(pve, component = c(1:758)) 

# Make a scree plot
g <- ggplot(pca.pve, aes(component, pve)) 
g + geom_point() + labs(title="Scree Plot", x="Component Number", y="PVE")

# Generate cumulative point plot
plot(cumsum(pve), xlab = "Principal Component",
     ylab = "Cumulative Proportion of Variance Explained",
     type = "b")

sum(pve[1:150])
# 96% of variance is explained by 150 principal components.

# Fit a logistic regression model with PC1, PC2, ... , PC150 
# Make new data frame consisted of PC1, PC2, ... , PC150
pca.data <- cbind(pca.data, pca$x)
pca.train <- as.data.frame(pca.data[, 759:908])
data <- data.frame(default = train.new$default, pca.train)

# Fit a logistic regression model
pca.log <- glm(default ~ ., data, family = binomial)
summary(pca.log)

# Predict default variable with the model
pred <- predict(pca.log, type = "response")
default.pred <- factor(ifelse(pred >= 0.5, "1", "0"))
summary(default.pred)
table(train.new$default, default.pred)
# Accuracy -> 94742/104445 = 0.9070994
# This logistic regression model using pca works well.

# Prediction on test data using our regression model
pca.test <- data.frame(t(na.omit(t(test.norm[ , -770]))))
test.p <- predict(pca, newdata = pca.test)
pred.test <- predict(pca.log, newdata = data.frame(test.p) , type = "response")
default.pred.test <- factor(ifelse(pred.test >= 0.5, "1", "0"))
summary(default.pred.test)
# Result ->  0: 209797, 1: 141