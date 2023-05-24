# Script Name:        Dimensionality_Reduction.R
# Created on:         May_20_2023
# Author:             Dr. Martin Calvino
# Purpose:            Implement PCA & UMAP to explore inherent structure (group of applicants based on race) 
#                     of home loan data from Bank of America during 2018-2019-2020
#                     to infer the race (Asian vs. African American) of loan applicants during 2021
# Version:            v1.05.20.2023
# Dataset:            Dataset was downloded from: https://ffiec.cfpb.gov/data-browser/data/2021?category=states
# Documentation:      Dataset field's description can be found at https://ffiec.cfpb.gov/documentation/2018/lar-data-fields/


# load libraries
library(tidyverse)
library(psych)
library(GGally)
library(yardstick)

# load datasets 2018-2019-2020-2021
# home loan data is nationwide
# Bank of America 2018 as boa18
boa18 <- read.csv(file.choose()) # 462,401 observations x 99 variables
boa19 <- read.csv(file.choose()) # 466,552 observations x 99 variables
boa20 <- read.csv(file.choose()) # 373,621 observations x 99 variables
boa21 <- read.csv(file.choose()) # 368,728 observations x 99 variables

# combine datasets from 2018-2019-2020 into a single data frame named boa
boa <- rbind(boa18, boa19, boa20) # 1,302,574 observations x 99 variables

# inspect the names of variables
colnames(boa)

# select variables of interest (feature selection)
median_family_income <- boa$ffiec_msa_md_median_family_income
income <- boa$income*1000
loanAmount <- boa$loan_amount
property_value <- boa$property_value
ltv <- boa$loan_to_value_ratio
house.age <- boa$tract_median_age_of_housing_units
interest_rate <- boa$interest_rate
loan_coast <- boa$total_loan_costs
origination_charges <- boa$origination_charges
loan_term <- boa$loan_term
race <- boa$derived_race

# because PCA & UMAP only works with numeric variables (continuous data),
# create a new data frame with numeric variables and leaving behind all categorical variables
# in the original dataset except race
boa.numeric <- data.frame(income,
                          median_family_income,
                          loanAmount,
                          property_value,
                          ltv,
                          house.age,
                          loan_coast,
                          origination_charges,
                          loan_term,
                          interest_rate,
                          race)

# for our imaginary example, let's focus on Asian & African American loan applicants only
boa.numeric <- filter(boa.numeric, race == "Asian" | race == "Black or African American")

# inspect the first 100 rows of boa.numeric
View(boa.numeric[1:100, ])

# inspect summary statistics
summary(boa.numeric[, c(1:10)])


# identify & remove outliers for income, loan amount and property value variables
boxplot(boa.numeric[, c(1:10)])

# remove outliers for the income variable
outliers.inc <- boxplot(boa.numeric[, 1])$out
boa.numeric <- boa.numeric[-which(boa.numeric[, 1] %in% outliers.inc), ]
boxplot(boa.numeric[, c(1:10)])

# remove outliers for the loan amount variable
outliers.la <- boxplot(boa.numeric[, 3])$out
boa.numeric <- boa.numeric[-which(boa.numeric[, 3] %in% outliers.la), ]
boxplot(boa.numeric[, c(1:10)])

# remove outliers for the property value variable
outliers.pv <- boxplot(boa.numeric[, 4])$out
boa.numeric <- boa.numeric[-which(boa.numeric[, 4] %in% outliers.pv), ]
boxplot(boa.numeric[, c(1:10)])

# remove outliers for the loan_to_value_ratio variable
outliers.ltv <- boxplot(boa.numeric[, 5])$out
boa.numeric <- boa.numeric[-which(boa.numeric[, 5] %in% outliers.ltv), ]
boxplot(boa.numeric[, c(1:10)])


# remove applications with negative values for income
boa.numeric <- filter(boa.numeric, income > 0)

# inspect summary statistics
summary(boa.numeric[, c(1:10)])


# count missing values
sum(is.na(boa.numeric)) # 285,454 NAs
# remove missing values
# notce that becuase we remove observations containing missing values for the interest_rate variable
# we are indirectly keeping loan applications that were accepted (denied applications don't have interest rate values)
boa.numeric <- na.omit(boa.numeric)

# inspect the structure of boa.numeric
str(boa.numeric) # we now have 32,055 observations x 11 variables that we can work with

# inspect summary statistics once more
summary(boa.numeric[, c(1:10)])


# visualize all data in boa.numeric using the ggpair() function from the GGally package
plot.boa.numeric <- ggpairs(boa.numeric, mapping = aes(col = race)) +
  theme_bw()
# see the plot
plot.boa.numeric

# save plot with specified dimensions for easy viewing
ggsave("Bank_of_America_Asian_vs_African_American_2018_to_2020_nationWide.jpeg",
       width = 8000, height = 4000, units = "px")

# count number of applications for each racaial group
table(boa.numeric$race) # 20,488 Asian & 11,567 African American applicants

################################################################################

# IMPLEMENT PCA

# select the number of components to extract
fa.parallel(boa.numeric[, -11], fa = "pc", n.iter = 1000, show.legend = TRUE, 
            main = "Scree plot with parallel analysis") # the plot suggest I extract 4 components

# scale & center data because variables have different units of measurements (US Dollars, percentages, years) and ranges
# boa.numeric.centerd.and.scaled as "boncas"
boncas <- scale(boa.numeric[, -11], center = TRUE, scale = TRUE)

# extract principal components
pc.boa.num <- principal(boncas, nfactors = 4, rotate = "none", scores = TRUE)
pc.boa.num


# plot home loan applications across the first 2 PCs
# Bank of America as boa
# add PCA scores associated to each loan application for the first 2 components
boa_pca <- boa.numeric %>%
  mutate(PCA1 = pc.boa.num$scores[, 1], PCA2 = pc.boa.num$scores[, 2])

# inspect boa_pca
head(boa_pca)

ggplot(boa_pca, aes(PCA1, PCA2, col = race)) +
  geom_point(size = 0.1) +
  theme_bw()

################################################################################

# IMPLEMENT UMAP ALGORITHM

# install and load the umap package
install.packages("umap", dependencies = TRUE)
library(umap)

# create embedding
boa_umap <- select(boa.numeric, -race) %>%
  as.matrix() %>%
  umap(n_neighbors = 20, min_dist = 0.5, metric = "manhattan", n_epochs = 600, verbose = TRUE)

# visualize results
boa_umap_2 <- boa.numeric %>%
  mutate(UMAP1 = boa_umap$layout[, 1], UMAP2 = boa_umap$layout[, 2])

# inspect boa_umap_2
head(boa_umap_2)

ggplot(boa_umap_2, aes(UMAP1, UMAP2, col = race)) +
  geom_point(size = 0.1) +
  theme_bw()

################################################################################

# work on boa21 dataset now

# select variables of interest (feature selection)
median_family_income21 <- boa21$ffiec_msa_md_median_family_income
income21 <- boa21$income*1000
loanAmount21 <- boa21$loan_amount
property_value21 <- boa21$property_value
ltv21 <- boa21$loan_to_value_ratio
house.age21 <- boa21$tract_median_age_of_housing_units
interest_rate21 <- boa21$interest_rate
loan_coast21 <- boa21$total_loan_costs
origination_charges21 <- boa21$origination_charges
loan_term21 <- boa21$loan_term
race21 <- boa21$derived_race

# because PCA & UMAP only works with numeric variables (continuous data),
# create a new data frame with numeric variables and leaving behind all categorical variables
# in the original dataset except race
boa21.numeric <- data.frame(income21,
                          median_family_income21,
                          loanAmount21,
                          property_value21,
                          ltv21,
                          house.age21,
                          loan_coast21,
                          origination_charges21,
                          loan_term21,
                          interest_rate21,
                          race21)

# for our imaginary example, let's focus on Asian & African American loan applicants only
boa21.numeric <- filter(boa21.numeric, race21 == "Asian" | race21 == "Black or African American")


# identify outliers and remove them from boa21.numeric

boxplot(boa21.numeric[, c(1:10)])

# remove outliers for the income variable
outliers.inc <- boxplot(boa21.numeric[, 1])$out
boa21.numeric <- boa21.numeric[-which(boa21.numeric[, 1] %in% outliers.inc), ]
boxplot(boa21.numeric[, c(1:10)])

# remove outliers for the loan amount variable
outliers.la <- boxplot(boa21.numeric[, 3])$out
boa21.numeric <- boa21.numeric[-which(boa21.numeric[, 3] %in% outliers.la), ]
boxplot(boa21.numeric[, c(1:10)])

# remove outliers for the property value variable
outliers.pv <- boxplot(boa21.numeric[, 4])$out
boa21.numeric <- boa21.numeric[-which(boa21.numeric[, 4] %in% outliers.pv), ]
boxplot(boa21.numeric[, c(1:10)])

# remove outliers for the loan_to_value_ratio variable
outliers.ltv <- boxplot(boa21.numeric[, 5])$out
boa21.numeric <- boa21.numeric[-which(boa21.numeric[, 5] %in% outliers.ltv), ]
boxplot(boa21.numeric[, c(1:10)])

# inspect boa21.numeric
summary(boa21.numeric)

# remove applications with negative values for income
boa21.numeric <- filter(boa21.numeric, income21 > 0)

# identify missing values and remove them
sum(is.na(boa21.numeric))
boa21.numeric <- na.omit(boa21.numeric)

table(boa21.numeric$race21) # 13,551 observations from Asians and 8,172 observations from African American


###############################################################################3

# FIT LOGISTIC REGRESSION MODELS USING PC COMPONENTS & UMAP EMBEDDINGS AS
# EXPLANATORY VARIABLES TO PREDICT THE RACE OF LOAN APPLICANTS IN 2021

# code response variable (race) as 0 and 1 to implement logistic regression
boa21.numeric$race21[boa21.numeric$race21 == "Asian"] <- 1
boa21.numeric$race21[boa21.numeric$race21 == "Black or African American"] <- 0
boa21.numeric$race21 <- factor(
  boa21.numeric$race21,
  levels = c(0, 1),
  labels = c(0, 1)
)

str(boa21.numeric)

# create a train/test split of dataset
rows <- sample(nrow(boa21.numeric))
train.boa21.numeric <- boa21.numeric[rows, ]
# train:test split is 70%:30%
split <- round(nrow(train.boa21.numeric) * 0.70)
train <- train.boa21.numeric[1:split, ]
test <- train.boa21.numeric[(split + 1):nrow(train.boa21.numeric), ]
nrow(train) / nrow(train.boa21.numeric) # training dataset is 70% of the entire dataset


########################################
# PC COMPONENTS AS EXPLANATORY VARIABLES

# use PCA model created with data from 2018-2019-2020 to predict component scores on train dataset
pca_data_modeled_21 <- predict(pc.boa.num, data = train[, -11])[, 1:4]
head(pca_data_modeled_21)
# use PCA model to predict component scores on test dataset
pca_test_modeled_21 <- predict(pc.boa.num, data = test[, -11])[, 1:4]
head(pca_test_modeled_21)

# convert from matrix to data frame to use in logistic regression
class(pca_data_modeled_21)
pca_data_modeled_21 <- as.data.frame(pca_data_modeled_21)
class(pca_test_modeled_21)
pca_test_modeled_21 <- as.data.frame(pca_test_modeled_21)


# add PC component scores to train dataset
train <- train %>%
  mutate(PCA1 = pca_data_modeled_21[, 1], PCA2 = pca_data_modeled_21[, 2])

View(train)

# add PC component scores to test dataset
test <- test %>%
  mutate(PCA1 = pca_test_modeled_21[, 1], PCA2 = pca_test_modeled_21[, 2])

View(test)


# fit logistic regression model with PC components as explanatory variables
fit.glm.pca.21 <- glm(race21 ~ PCA1 + PCA2, data = train, family = binomial())

# inspect model's coefficients
summary(fit.glm.pca.21)


# make predictions on the test dataset
pred.test <- predict(fit.glm.pca.21, newdata = test, type = "response")

# evaluate model perdormance
# Confusion Matrix: counts of outcomes
actual_response <- test$race21
predicted_response <- ifelse(pred.test > 0.50, "1", "0")
outcomes <- table(predicted_response, actual_response)
outcomes

# evaluate model performance using functions from the yardstick package
confusion <- conf_mat(outcomes)
autoplot(confusion)
# obtain model's performance metrics
summary(confusion, event_level = "second")

# accuracy is the proportion of correct predictions: 0.769 or 76.9%
# sensitivity is the proportion of true positives: 0.815 or 81.5%
# specificity is the proportion of true negatives: 0.691 or 69.1%

##########################################
# UMAP DIMENSIONS AS EXPLANATORY VARIABLES

# create a train/test split of dataset
rowsU <- sample(nrow(boa21.numeric))
train.boa21.numericU <- boa21.numeric[rowsU, ]
# train:test split is 70%:30%
splitU <- round(nrow(train.boa21.numericU) * 0.70)
trainU <- train.boa21.numericU[1:splitU, ]
testU <- train.boa21.numericU[(splitU + 1):nrow(train.boa21.numericU), ]
nrow(trainU) / nrow(train.boa21.numericU) # training dataset is 70% of the entire dataset

# compute UMAP embedding on train dataset
UMAP_data_modeled_21 <- predict(boa_umap, data = trainU[, -11])
# compute UMAP embedding on test dataset
UMAP_test <- predict(boa_umap, data = testU[, -11])

# convert matrices to data frames
class(UMAP_data_modeled_21)
UMAP_train <- as.data.frame(UMAP_data_modeled_21)
class(UMAP_test)
UMAP_test <- as.data.frame(UMAP_test)


# add UMAP dimensions to trainU dataset
trainU <- trainU %>%
  mutate(UMAP1 = UMAP_train[, 1], UMAP2 = UMAP_train[, 2])
# add UMAP dimensions to testU dataset
testU <- testU %>%
  mutate(UMAP1 = UMAP_test[, 1], UMAP2 = UMAP_test[, 2])

# fit logistic regression model with UMAP dimensions as explanatory variables
fit.glm.UMAP.21 <- glm(race21 ~ UMAP1 + UMAP2, data = trainU, family = binomial())

# inspect model's coefficients
summary(fit.glm.UMAP.21)


# make predictions on the test dataset
pred.test.U <- predict(fit.glm.UMAP.21, newdata = testU, type = "response")

# evaluate model perdormance
# Confusion Matrix: counts of outcomes
actual_responseU <- testU$race21
predicted_responseU <- ifelse(pred.test.U > 0.50, "1", "0")
outcomesU <- table(predicted_responseU, actual_responseU)
outcomesU

# evaluate model performance using functions from the yardstick package
confusionU <- conf_mat(outcomesU)
autoplot(confusionU)
# obtain model's performance metrics
summary(confusionU, event_level = "second")

# accuracy is the proportion of correct predictions: 0.731 or 73.1%
# sensitivity is the proportion of true positives: 0.776 or 77.6%
# specificity is the proportion of true negatives: 0.657 or 65.7%
