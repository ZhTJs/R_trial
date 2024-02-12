install.packages("ggplot2", "dplyr", "glmnet", "glm", "data.table")

library(data.table) # used for reading and manipulation of data
library(e1071)
library(dplyr)      # used for data manipulation and joining
library(ggplot2)    # used for ploting 
library(caret)      # used for modeling
library(corrplot)   # used for making correlation plot
library(cowplot)    # used for combining multiple plots
library(glmnet)     # used for LASSO regression model

rm(list = ls())

traindata <- read.csv("train.csv")
testdata <- read.csv("test.csv")

# linear model

linear_reg <- lm(traindata$UNITS~ traindata$VISITS+)


#dataset size and variables

dim(traindata)
dim(testdata)

#'data.frame':	520800 obs. of  12 variables:
#$ WEEK_END_DATE: chr  "01-Apr-09" "01-Dec-10" "01-Jul-09" "01-Jun-11" ...
#$ STORE_ID     : int  10019 10019 10019 10019 10019 10019 10019 10019 10019 10019 ...
#$ UPC          : num  1.11e+09 1.11e+09 1.11e+09 1.11e+09 1.11e+09 ...
#$ UNITS        : int  11 8 11 9 16 19 8 4 7 1 ...
#$ VISITS       : int  10 8 9 9 14 17 8 4 6 1 ...
#$ HHS          : int  10 8 9 9 14 17 8 4 6 1 ...
#$ SPEND        : num  13.1 9.6 10.8 10.9 15.7 ...
#$ PRICE        : num  1.19 1.2 0.98 1.21 0.98 0.97 1.2 0.99 1.21 0.89 ...
#$ BASE_PRICE   : num  1.19 1.2 0.98 1.21 1.19 1.22 1.2 1.22 1.21 0.99 ...
#$ FEATURE      : int  0 0 0 0 1 0 0 0 0 0 ...
#$ DISPLAY      : int  0 0 0 0 0 1 0 0 0 0 ...
#$ TPR_ONLY     : int  0 0 0 0 0 0 0 1 0 1 ...

#head and variable types

str(traindata)
str(testdata)

#check if there are missing valeus and find the location

combi <- rbind(traindata, testdata)

colSums(is.na(combi))

#Descriptive analysis for dependent variable 'UNITS'

summary(traindata$UNITS)
summary(testdata$UNITS)

summary(combi$UNITS)

ggplot(combi) +
  geom_histogram(aes(UNITS), binwidth = 20, fill = "black") +
  xlab("Unit sales")

#Independent Variables(numeric variables)

p1 = ggplot(combi) + geom_histogram(aes(VISITS), binwidth = 50, fill = "blue")
p2 = ggplot(combi) + geom_histogram(aes(HHS), binwidth = 40, fill = "blue")
p3 = ggplot(combi) + geom_histogram(aes(SPEND), binwidth = 20, fill = "blue")
p4 = ggplot(combi) + geom_histogram(aes(PRICE), binwidth = 0.5, fill = "blue")
p5 = ggplot(combi) + geom_histogram(aes(BASE_PRICE), binwidth = 0.1, fill = "blue")

#Transform numeric variables

skewness(traindata$VISITS)
skewness(traindata$PRICE)

traindata$VISITS = log(traindata$VISITS + 1)

#Independent Variables(categorical variables)

combi$DISPLAY[combi$DISPLAY == "0"] = "No Display"
combi$DISPLAY[combi$DISPLAY == "1"] = "Promotional Display"
p6 = ggplot(combi %>% group_by(DISPLAY) %>% summarise(Count = n())) + 
  geom_bar(aes(DISPLAY, Count), stat = "identity", fill = "coral1")

combi$FEATURE[combi$FEATURE == "0"] = "No Circular"
combi$FEATURE[combi$FEATURE == "1"] = "In-store Circular"
p7 = ggplot(combi %>% group_by(FEATURE) %>% summarise(Count = n())) + 
  geom_bar(aes(FEATURE, Count), stat = "identity", fill = "coral1")

plot_grid(p6, p7, ncol = 2)

#And other independent variables...

#Correlated Variables

cor_train = cor(traindata[, !names(traindata) %in% c("UPC", "WEEK_END_DATE")])
corrplot(cor_train, method = "pie", type = "lower", tl.cex = 0.9)

#LASSO 1 test

#set.seed(1235)
#my_control = trainControl(method="cv", number=5)
#Grid = expand.grid(alpha = 1, lambda = seq(0.001,0.1,by = 0.0002))

#lasso_mod = train(x = traindata[, c('STORE_ID', 'UPC', 'VISITS', 'SPEND', 'PRICE', 
#                                    'BASE_PRICE', 'FEATURE', 'DISPLAY')], y = traindata$UNITS,
#                  method='glmnet', trControl= my_control, tuneGrid = Grid)
#plot(lasso_mod)
#summary(lasso_mod)


#LASSO 2 test

# Define the response variable and the matrix of predictor variables

lasso_y_train <- traindata$UNITS
lasso_x_train <- data.matrix(traindata[, c('STORE_ID', 'VISITS', 'SPEND', 
                                           'PRICE',  'BASE_PRICE', 'FEATURE', 'DISPLAY')])
lasso_y_test <- testdata$UNITS
lasso_x_test <- data.matrix(testdata[, c('STORE_ID', 'VISITS', 'SPEND', 
                                         'PRICE',  'BASE_PRICE', 'FEATURE', 'DISPLAY')])




lambda.array <- 10^seq(10, -2, length = 1000)
lasso_reg <- glmnet(lasso_x_train, lasso_y_train, alpha = 1, family = "poisson", lambda = lambda.array)
summary(lasso_reg)


#Lambdas in relation to the coefficients

plot(lasso_reg, xvar = 'lambda', lable = T)

#Goodness of fit 

plot(lasso_reg, xvar = 'dev', lable = T)


# Perform k-fold cross-validation to find the optimal lambda value

cv_train <- cv.glmnet(lasso_x_train, lasso_y_train, alpha = 1)
plot(cv_train)
bestlam <- cv_train$lambda.min
lasso_bestlam <- glmnet(lasso_x_train, lasso_y_train, alpha = 1, lambda = bestlam)
coef(lasso_bestlam)

#Predicted values

lasso_y_predicted <- predict(cv_train, s = bestlam, newx = lasso_x_test)

testx <- matrix(c(24, 2.5, 3.5, 18.5), nrow=1, ncol=7) 

#SSE SST

sst <- sum((lasso_y_test - mean(lasso_y_test))^2)
sse <- sum((lasso_y_predicted - lasso_y_test)^2)
1 - sse/sst

#MSE

MSE = (sum(lasso_y_predicted - lasso_y_test)^2)/length(lasso_y_predicted)



