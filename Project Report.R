---
  title: "Movielens Edx Machine Learning Project"
author: "Eddwin Cheteni"
date: "`r format(Sys.Date())`"
output: word_document
---
  
 --------------------
knitr::opts_chunk$set(echo = TRUE, progress = TRUE, verbose = TRUE)
-----------

## Table of contents

#`1. [Project Overview](#1.-Project-Overview)<br>
  
  #` 2. [Data exploration and pre-processing](#2.-Data-explorations-and-pre-processing)<br>
    
    #`   3. [Model training](#3.-Model-training)<br>
      
      #`   4. [Model results](#4.-Model-results)<br>
        
        #`    5. [Conclusion](#5.-Conclusion)<br>
          
     #`     6. [References](#6.-References)<br>
            
            # 1.0 Project Overview
            
            ## Introduction
            
           #` Recommendation system provides suggestions to the users through a filtering process that is based on user preferences and browsing history. The information about the user is taken as an input. The information is taken from the input that is in the form of browsing data. This recommender system has the ability to predict whether a particular user would prefer an item or not based on the user's profile. Recommender systems are beneficial to both service providers and users. They reduce transaction costs of finding and selecting items in an online shopping environment.

## Problem Statement

#` For this project, a movie recommendation system is created using the MovieLens dataset. The version of `Movielens` included in the dslabs package (which was used for some of the exercises in PH125.8x: Data Science: Machine Learning) is just a small subset of a much larger dataset with millions of ratings. The entire latest `MovieLens` dataset can be found <https://grouplens.org/datasets/movielens/latest/>. Recommendation system is created using all the tools learnt throughout the courses in this series. `MovieLens` data is downloaded using the available code to generate the datasets.

#` First, the datasets will be used to answer a short quiz on the `MovieLens` data. This will give the researcher an opportunity to familiarize with the data in order to prepare for the project submission. Second, the dataset will then be used to train a machine learning algorithm using the inputs in one subset to predict movie ratings in the validation set. 

#` The value used to evaluate algorithm performance is the `Root Mean Square Error`, or `RMSE`. RMSE is one of the most used measure of the differences between values predicted by a model and the values observed. `RMSE` is a measure of accuracy, to compare forecasting errors of different models for a particular dataset, a lower `RMSE` is better than a higher one. The effect of each error on `RMSE` is proportional to the size of the squared error; thus larger errors have a disproportionately large effect on `RMSE`. Consequently, RMSE is sensitive to outliers.

#` A comparison of the models will be based on better accuracy. The evaluation criteria for these algorithms is a `RMSE` expected to be lower than `0.8649`.

#` The function that computes the `RMSE` for vectors of ratings and their corresponding predictors will be the following:

#` $$ RMSE = \sqrt{\frac{1}{N}\displaystyle\sum_{u,i} (\hat{y}_{u,i}-y_{u,i})^{2}} $$

### MovieLens

#` <https://movielens.org/> is a project developed by <https://grouplens.org/>, a research laboratory at the University of Minnesota. `MovieLens` provides an online movie recommender application that uses anonymously-collected data to improve recommender algorithms. To help people develop the best recommendation algorithms, `MovieLens` also released several data sets. In this article, latest data set will be used, which has two sizes.

#` The full data set consists of more than 24 million ratings across more than `40,000` movies by more than `250,000` users. The file size is kept under `1GB` by using indexes instead of full string names. 

## Load the data

#` DataFrames are one of the easiest and best performing ways of manipulating data with R, but they require structured data in formats or sources such as CSV.

### Download the data from MovieLens

#` To download the data:

#` 1. We download the small version of the `ml-latest.zip` file from <https://grouplens.org/datasets/movielens/latest/>.
#` 2. Unzip the file. The files to be used are `movies.csv` and `ratings.csv` to create `edx` and `validation` set as indicated by the code below.

#` Luckily, a code to download and create `edx` and `validation` set is readily available which is to be used in this project.


##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

#` Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
#movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
#                                            title = as.character(title),
#                                            genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                            title = as.character(title),
                                            genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
      semi_join(edx, by = "movieId") %>%
      semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

### Loading important packages for this project.

#` Let's load some important library packages we might need during the process:
              
           
            if(!require(readr)) install.packages("readr")
            if(!require(dplyr)) install.packages("dplyr")
            if(!require(tidyr)) install.packages("tidyr")
            if(!require(stringr)) install.packages("stringr")
            if(!require(ggplot2)) install.packages("ggplot2")
            if(!require(gridExtra)) install.packages("gridExtra")
            if(!require(dslabs)) install.packages("dslabs")
            if(!require(ggrepel)) install.packages("ggrepel")
            if(!require(ggthemes)) install.packages("ggthemes")
            
            library(readr)
            library(dplyr)
            library(tidyr)
            library(stringr)
            library(ggplot2)
            library(gridExtra)
            library(dslabs)
            library(data.table)
            library(ggrepel)
            library(ggthemes)
            
            
            ## Data Exploration: Quiz questions
            
           #` This sub-section only covers the quizzes which is part of the grading of this project and it will also help to understand the structure of the dataset involved.
            
            
            #Q1: How many rows and columns are there in the edx dataset?
            dim(edx)
           
            #Q2a: How many zeros and threes were given as ratings in the edx dataset?
            #How many zeros were given as ratings in the edx dataset?
            edx %>% filter(rating == 0) %>% tally()
            
            #Q2b:How many threes were given as ratings in the edx dataset?
            #How many threes were given as ratings in the edx dataset?
            edx %>% filter(rating == 3) %>% tally()
           
            # Q3:How many different movies are in the edx dataset?
            n_distinct(edx$movieId)
           
            #Q4: How many different users are in the edx dataset?
            n_distinct(edx$userId)
          
            #Q5: How many movie ratings are in each of the following genres in the edx dataset?
            genres = c("Drama", "Comedy", "Thriller", "Romance")
            sapply(genres, function(g) {
              sum(str_detect(edx$genres, g))
            })
        
            #Q6: Which movie has the greatest number of ratings?
            edx %>% group_by(movieId, title) %>%
              summarize(count = n()) %>%
              arrange(desc(count))%>%head(5)
           
            #Q7: What are the five most given ratings in order from most to least?
            edx %>% group_by(rating) %>% summarize(count = n()) %>%
              arrange(desc(count))%>%head(5)
          
           
            #Q8 True or False: In general, half star ratings are less common than whole star ratings (e.g., there are fewer ratings of 3.5 than there are ratings of 3 or 4, etc.).
            edx %>% group_by(rating)%>%
              summarize(count=n())%>%
              arrange(desc(count))%>%
              head(10)
          
      
            #confirming on the frequency of ratings given by users
            edx %>% group_by(rating)%>%
              summarize(n=n())%>%
              arrange(desc(n))%>%
              ggplot()+
              #scale_y_log10() +
              geom_col(mapping = aes(x = rating, y = n), fill=I("brown"))
            
            
            # 2.0 Data Exploration and Pre-processing
            
            ## Data exploration (Continuation)
            
            ### Edx dataset
            
            #` Let have a look at the data types in our `edx` set and it shows that we have `int`,`num` and `chr` type of data of which it complies with the entries on each column.
            
           #` variables data types, echo=FALSE------------------
            #checking on the data types of the dataset columns
            str(edx)
           
            
           #` dimensions of edx set, echo=FALSE------------------
            #confirming the dimensions of the dataset
            dim(edx)
            
            ##There are `9000055` rows and `6` columns in `edx` dataset.
            
            ##Let have a look at the `edx` dataset again with a quick summary.
            
            #` ---summarizing edx dataset, echo=FALSE--------------
            #getting the summary of all variables
            summary(edx)
            
            
##Let's check if any missing values exist.

## -----check missing values, echo=TRUE----
anyNA(edx)#checking missing values

#` It seems there are no missing values in `edx` dataset.


#checking on the number of unique users and movies involved
edx %>% summarize(n_users = n_distinct(userId), n_movies = n_distinct(movieId))


#` As shown above, there are approximately `70000` unique users giving ratings to more than `10000` unique movies.

#` Movies were given ratings ranging from 0.5 as the minimum rating to 5 as the maximum rating.

#checking on unique ratings given by users
unique(edx$rating)


#` Some movies were given more ratings as compared to others. Lets find out which range of number of ratings were given to these movies.


#checking on the frequency of ratings given to movies
edx %>% 
  count(movieId) %>% #count the number of movies
  ggplot(aes(n)) + 
  geom_histogram(bins = 25, binwidth=0.2, color="black",show.legend = FALSE, aes(fill = cut(n, 100))) + 
  scale_x_log10() +   
  ggtitle("Rated Movies")


#` We can separate the year from the movie title in order to gain better understanding of some patterns, perhaps this could also affect viewers' preferences based on the year of production which in turn influence its recommendation system.
            

 #separating year from title and create new column named year
edx_yr = edx %>% mutate(year = as.numeric(str_extract(str_extract(title, "[/(]\\d{4}[/)]$"), regex("\\d{4}"))),title = str_remove(title, "[/(]\\d{4}[/)]$")) 
        
            
#` Let's have a look at the resulting dataset:


#taking a look at first 5 rows of the edx_yr dataset
edx_yr %>% head(5)

 #` Now the dataset clearly shows title and year of release separately.
 #` Lets create a dataset showing release year and numbers of movies that were released in that particular year.


n_movies_per_year <- edx_yr %>%
  select(movieId, year) %>% # select columns we need
  group_by(year) %>% # group by year
  summarize(count = n())  %>% # count movies per year
  arrange(year) #arrange by year starting from the early years going up
n_movies_per_year%>%head()#view first 5 rows 


#` We can make a review on the above `n_movies_per_year` dataset.


#make a plot of number of movies against year of release
n_movies_per_year %>%
  ggplot(aes(x = year, y = count)) +
  scale_x_log10() +
  geom_line(color="navy")


#` We can see that the data shows an exponential increase in movie business indicating a rapid rise from `1990` going to `2000` and started to drop thereafter.

#` Lets check the most popular movie genres year by year.


genres_by_year <- edx_yr %>% 
  separate_rows(genres, sep = "\\|") %>%    #separating genres of movies
  select(movieId, year, genres) %>%     #selecting required columns
  group_by(year, genres) %>%     #group by year and genres
  summarize(count = n()) %>%       #count movies per year and genre
  arrange(desc(year))

#` We can make a review on the above `genres_by_year` dataset:

genres_by_year %>% ggplot() + 
  geom_bar(aes(x=reorder(genres, genres,length), fill = I("brown"), show.legend=FALSE)) + 
  coord_flip() +
  xlab('Genres') + #vertical axis label
  ylab('Number of Movies') + #horizontal axis label
  ggtitle('Popularity by Genre')


#` Looking at the chart above, there seems to be no much variations in terms of the movie genre against reviews.

#` Lets' check the pattern of reviews with some plot to see the number of times each user has reviewed these movies.
            
            
            edx %>% count(userId) %>% 
              ggplot(aes(n)) + 
              geom_histogram(bins = 30, binwidth=0.2, color="black", show.legend = FALSE, aes(fill = cut(n, 100))) + 
              scale_x_log10() + 
              ylab('Number of Reviews') + #vertical axis label
              xlab('Number of Users') + #horizontal axis label
              ggtitle("User Reviews")
            
            
            #`  We can see that most users reviewed at less than 300 movies.
            
            #`   Let's make a review of movie ratings against the year of release:


#plot of rating vs release year
edx_yr %>% group_by(year) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(year, rating)) +
  geom_point(aes(color=rating)) +
  theme_hc() + 
  geom_smooth() +
  ggtitle("Release Year vs. Rating")

#` Looking at the above chart, we can see that older movies were given higher ratings compared to newer movies. This could indicate that these old movies are more important in terms of coming up with a recommender system for Movielens dataset.

## Data Partitioning

#` In order to build models that can be used to recommend movies to the viewers, we need to select important features needed and split the `edx` dataset into `train` and `test` set. For good results, we split the data into `80-20` percent ratio for `train-test` set.


#selecting important features
edxset <- edx_yr %>% select(userId,movieId,title,genres,year,rating) 
#view first 5 rows
head(edxset)


#` Now our data only include features that are important for model building phase.

#` We then split the dataset into train and test set.

set.seed(123)

#loading libraries for data partitioning process
library(caret)
library(lattice)

#creating an index to pick a sample for train set
My_Index <- createDataPartition(edxset$rating, p= 0.8, list = FALSE, times = 1) 
train <- edxset[My_Index, ] #creating the train set
test <- edxset[-My_Index, ] #creating the test set

# Make sure userId, year and movieId in test set are also in train set
test <- test %>% 
     semi_join(train, by = "movieId") %>%
     semi_join(train, by = "userId")%>%
     semi_join(train, by = "title")

#checking the dimensions of the train set
print("Dimensions of the train set are:")
dim(train)
#checking the dimension of the test set
print("Dimensions of the test set are:")
dim(test)


# 3.0 Model Training

## Naive-based model

#` Calculating the average rating of all the movies using train set. We can see that if we predict all unknown ratings with  $\mu$ we obtain the following RMSE.


#just the average of ratings
mu_hat <- mean(train$rating)
mu_hat

#` We observe that RMSE for the first model indicates that movie rating is `>3.5`.

#` We know from experience that some movies are just generally rated higher than others. We can use data to confirm this, lets say, if we consider movies with more than `1,000` ratings, the SE error for the average is at most `0.05`. Yet plotting these averages we see much greater variability than `0.05`:

train %>% group_by(movieId) %>%
  filter(n()>=1000)%>%
  summarize(avg_rating = mean(rating)) %>% 
  ggplot(aes(avg_rating)) + 
  geom_histogram(bins = 30, color="black", fill= "brown", show.legend = FALSE, aes(fill = cut(n, 100))) + 
  scale_x_log10() + 
  xlab("Ratings") +
  ggtitle("Average Ratings")

## Movie-based model

#` We know from experience that some movies are just generally rated higher than others.Lets see if there will an improvement by adding the term $b_i$ to represent the average rating for movie $i$.

#compute average rating
mu_hat <- mean(train$rating) 
#compute average rating based on movies
movie_avgs <- train %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu_hat)) 
#compute the predicted ratings on test set
predicted_ratings_movie_avg <- test %>% 
  left_join(movie_avgs, by='movieId') %>%
   mutate(pred = mu_hat + b_i) %>%
   pull(pred)


## Movie + User-effect model

#just the average
mu_hat <- mean(train$rating)
#compute average rating based on users
user_avgs <- train %>%
   left_join(movie_avgs, by='movieId') %>%
   group_by(userId) %>%
   summarize(b_u = mean(rating - mu_hat - b_i)) 
#compute the predicted ratings on test set
predicted_ratings_movie_user_avg <- test %>% 
   left_join(movie_avgs, by='movieId') %>%
   left_join(user_avgs, by='userId') %>%
   mutate(pred = mu_hat + b_i + b_u) %>%
   pull(pred)


## Movie + Title + User-effect model

#just the average of ratings
mu_hat <- mean(train$rating)
# Calculate the average by title
title_avgs <- train %>%
   left_join(movie_avgs, by='movieId') %>%
   left_join(user_avgs, by='userId') %>%
   group_by(title) %>%
   summarize(b_tt = mean(rating - mu_hat - b_i - b_u))
#compute the predicted ratings on test set
predicted_ratings_movie_title_user_avg <- test %>%  
   left_join(movie_avgs, by='movieId') %>%
   left_join(user_avgs, by='userId') %>%
   left_join(title_avgs, by='title') %>%
   mutate(pred = mu_hat + b_i + b_u + b_tt) %>%
   pull(pred)


## Movie + Title + User + Regularization

#` To better our models above and ensure we are converging to the best solution let us add a regularization constant for our movie rating, title effect and user-specific model. The purpose of this is to penalize large estimates that come from small sample sizes. Therefore, our estimates will try its best to guess the correct rating while being punished if the movie rating, user-specific, or title effect is too large. 

#` This method is a little more complicated than the prior two methods. That is we want to also validate how much we want to regularize the movie-rating, title, and user-specific effects. We will try several regularization models with our regularization constant `(lambda)` at different values. We will define the function to obtain `RMSEs` here and apply it in our results section:

lambdas <- seq(0, 15, 0.25)
regularize <- function(l){
    #calculate just the average of ratings
     mu <- mean(train$rating)
     #calculate the average by movie
     b_i <- train %>%
          group_by(movieId) %>%
          summarize(b_i = sum(rating - mu_hat)/(n()+l))
     #calculate the average by user
     b_u <- train %>% 
          left_join(b_i, by="movieId") %>%
          group_by(userId) %>%
          summarize(b_u = sum(rating - b_i - mu_hat)/(n()+l))
     #calculate the average by title
     b_tt <- train %>% 
          left_join(b_i, by="movieId") %>%
          left_join(b_u, by="userId") %>%
          group_by(title) %>%
          summarize(b_tt = sum(rating - b_i - b_u - mu_hat)/(n()+l))
     #calculate prediction on test set
     predicted_ratings <- test %>% 
          left_join(b_i, by = "movieId") %>%
          left_join(b_u, by = "userId") %>%
          left_join(b_tt, by = "title") %>%
          mutate(pred = mu_hat + b_i + b_u + b_tt) %>%
          .$pred
     
     return(RMSE(predicted_ratings, test$rating))
}


# 4.0 Model Results and Validation


#` In this section we will see how well our models worked to our test set and then we validate the best model to unseen data in the `Validation` set.

#` Before model validation, here is the defined `RMSE function`:


#the RMSE function
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}


## RMSE for Naive-based model 

#calculate the RMSE of the first model
naive_rmse <- RMSE(test$rating, mu_hat)
naive_rmse

## RMSE for Movie effect model

#calculate RMSE
movie_based_rmse <- RMSE(test$rating, predicted_ratings_movie_avg) 
movie_based_rmse

#` Based on the RMSE result obtained, `0.9442583` indicates an improvement. But can we make it much better than this?

#` We can see that these estimates vary as indicated on the plot below:

qplot(b_i, geom = "histogram", color = I("black"), fill=I("brown"), bins=40, data = movie_avgs)

## RMSE for Movie + user-specific model

#calculate RMSE
movie_user_based_rmse <- RMSE(test$rating, predicted_ratings_movie_user_avg)
movie_user_based_rmse

#` This seems to be a better model with an RMSE value of `0.867`. 

#` We can see that these estimates vary as indicated on the plot below:

qplot(b_u, geom = "histogram", color = I("black"), fill=I("brown"), bins=40, data = user_avgs)

##RMSE for Movie + Title effect + user-specific model

#calculate RMSE
movie_title_user_based_rmse <- RMSE(test$rating, predicted_ratings_movie_title_user_avg) 
movie_title_user_based_rmse

#` This seems to be a better model with an RMSE value of `0.865`. 

#`We can see that these estimates vary as indicated on the plot below:

qplot(b_tt, geom = "histogram", color = I("black"), fill=I("brown"), bins=40, data = title_avgs)

## RMSE for Movie + user-specific + title effect + regularization model

#` This result is a little more complicated than the prior two methods. That is we want to also validate how much we want to regularize the movie-rating and user-specific effects. We will try several regularization models with our regularization constant (lambda) at different values:

rmses <- sapply(lambdas, regularize)
#plotting RMSEs against lambdas
qplot(lambdas, rmses)  

#` Our minimum error is found with a lambda value of 4.5:


#calculate the minimum value of lambda
lambda <- lambdas[which.min(rmses)]
lambda
#returning RMSE for the regularized model on test set
movie_title_user_reg_rmse <- min(rmses)
movie_title_user_reg_rmse

## All Models RMSE Summary

#` The table below summarizes our RMSE values given our models explored:

#summarize all models RMSE values against the test-set
#table generated for RMSE results
results <- data_frame(Model=c("1. Naive-based","2. Movie-based","3. Movie+User-based","4. Movie+Title+User-based","5. Regularized Movie+Title+User-based"),  
                           RMSE = c(naive_rmse,movie_based_rmse, movie_user_based_rmse,movie_title_user_based_rmse,min(rmses)))  
results

#` As we can see the regularization model performed best when lambda was set to `4.5` with an RMSE of `0.865`. Let's proceed to perform model validation based on this model.

## Final Model Validation

#` Before we begin model validation, we need to separate year from the title and make sure that title in validation set are also in the train set.

#separating year from title and create new column named year
valid_yr = validation %>% mutate(year = as.numeric(str_extract(str_extract(title, "[/(]\\d{4}[/)]$"), regex("\\d{4}"))),title = str_remove(title, "[/(]\\d{4}[/)]$")) 

# Make sure userId and movieId in validation set are also in train set
valid_yr <- valid_yr %>% 
  semi_join(train, by = "movieId") %>%
  semi_join(train, by = "userId") %>%
  semi_join(train, by = "title")
# keep this for checking RMSE on model evaluation
model_scoring <- valid_yr$rating


#` As mentioned we will train our regularized model (which takes into account (1) average movie rating, (2) movie-rating, (3) user-rating, and (4) title-rating effects) with a regularization parameter, `lambda`, of `4.5`.

#calculate just the average of ratings
mu_reg <- mean(train$rating)
#calculate the average by movie
b_i_reg <- train %>%
  group_by(movieId) %>%
  summarize(b_i_reg = sum(rating - mu_reg)/(n()+lambda))
#calculate the average by user
b_u_reg <- train %>% 
  left_join(b_i_reg, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u_reg = sum(rating - b_i_reg - mu_reg)/(n()+lambda))
#calculate the average by title
b_t_reg <- train %>% 
  left_join(b_i_reg, by="movieId") %>%
  left_join(b_u_reg, by="userId") %>%
  group_by(title) %>%
  summarize(b_t_reg = sum(rating - b_i_reg - b_u_reg - mu_reg)/(n()+lambda))
#calculate prediction on test set
predicted_ratings_b_i_u_t <- valid_yr %>% 
  left_join(b_i_reg, by = "movieId") %>%
  left_join(b_u_reg, by = "userId") %>%
  left_join(b_t_reg, by = "title") %>%
  mutate(pred = mu_hat + b_i_reg + b_u_reg + b_t_reg) %>%
  .$pred

# returning RMSE using validation set with year and title separated  
rmse_validation <- RMSE(predicted_ratings_b_i_u_t, model_scoring)
rmse_validation
#` As shown above our RMSE value against unseen data (our validation dataset) is `0.865` which meets our project objective of an `RMSE <= 0.865`.

# 5.0 Conclusion

#` In conclusion to this project, we have achieved an exceptional RMSE of `0.865` against our validation dataset. This RMSE result support the `Regularized-Movie+Title+User-effect model` which gives an RMSE of `0.865` on the test set. This is very impressive of our model since it gives a better accuracy in terms of predicting movie ratings of new dataset.

#` Given more time I would like to try to incorporate time factor as well as genre to see if this can also influence our model positively by improving the model accuracy.

# 6.0 References

#` All material in this project is credited to Professor Rafael Irizarry and his team at HarvardX's Data Science course. Most material was learned through his course, book, and github: 

#` 1.Irizarry,R 2021 ***Introduction to Data Science: Data Analysis and Prediction Algorithms with R*** ,github page,accessed 22 January 2021, https://rafalab.github.io/dsbook/

#` Another great resource was prior movie recommendation projects. Specifically, one project was referenced:

#` 2. The work done by Brandon Rufino (how I was inspired to try data exploration techniques such as similarity measures),
#` https://github.com/brufino/MovieRecommendationSystem/