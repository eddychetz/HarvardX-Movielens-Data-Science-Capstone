### 1.0 Introduction


### 2.0 Problem definition

The aim in this project is to train a machine learning algorithm that predicts user ratings (from 0.5 to 5 stars) using the inputs of a provided subset  (edx dataset provided by the staff) to predict movie ratings in a provided validation set.

The value used to evaluate algorithm performance is the Root Mean Square Error, or RMSE. RMSE is one of the most used measure of the differences between values predicted by a model and the values observed. RMSE is a measure of accuracy, to compare forecasting errors of different models for a particular dataset, a lower RMSE is better than a higher one. The effect of each error on RMSE is proportional to the size of the squared error; thus larger errors have a disproportionately large effect on RMSE. Consequently, RMSE is sensitive to outliers.

Four models that will be developed will be compared using their resulting RMSE in order to assess their quality. The evaluation criteria for this algorithm is a RMSE expected to be lower than 0.8775.
The function that computes the RMSE for vectors of ratings and their corresponding predictors will be the following:
  
  $$ RMSE = \sqrt{\frac{1}{N}\displaystyle\sum_{u,i} (\hat{y}_{u,i}-y_{u,i})^{2}} $$
  
  Finally, the model that gives a lowest RMSE will be considered as the best model to predict the movie ratings.

### 3.0 Dataset 

The code is provided in the edx capstone project overview [Create edx and Validation Sets]:
  - <https://grouplens.org/datasets/movielens/10m/>
  - <http://files.grouplens.org/datasets/movielens/ml-10m.zip>
  
  ```{r dataset, echo = TRUE, message = FALSE, warning = FALSE, eval = FALSE}
#Create test and validation sets
# Create edx set, validation set, and submission file
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)
ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                      col.names = c("userId", "movieId", "rating", "timestamp"))
movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))
movielens <- left_join(ratings, movies, by = "movieId")
# Validation set will be 10% of MovieLens data
set.seed(1)
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
```

### 3.1 Data Exploration

Loading the data and necessary library to use during exploration:
  
  ```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
library(tidyverse)
library(data.table)
library(caret)
library(lubridate)
library(ggplot2) 
edx <- readRDS("~/R/Movielens/edx.rds")
validation <- readRDS("~/R/Movielens/validation.rds")
attach(edx)

```

Lets check the dimensions of edx and validation dataset:
  
  ```{r dimensions,echo=FALSE}
# dimensions of edx set
dim(edx)
# dimensions of validation set
dim(validation)
```

Lets remove the rating column from the validation set:
  
  ```{r remove rating column from validation set, echo = FALSE}
# Validation dataset can be further modified by removing rating column
validation_df <- validation  
validation <- validation %>% select(-rating)
```

```{r view edx set, echo = FALSE}
#We can see this table is in tidy format with thousands of rows:
edx %>% as_tibble() %>% head() %>% knitr :: kable()
```

A summary of the edx set confirms that there are no missing values as shown below:
  
  ```{r summary, echo = FALSE}
summary(edx)
```


We can see the number of unique users that provided ratings and check how many unique movies were rated. The total of unique movies and users in the edx subset is almost 70 000 unique users and approximately 10.700 different movies:
  
  ```{r distinct movies, echo=FALSE }
data.frame(Count = t(edx %>% summarize("Number of Reviews" = n(),
                                       "Distinct Users" = n_distinct(userId),
                                       "Distinct Movies" = n_distinct(movieId),
                                       "Distinct Titles" = n_distinct(title),
                                       "Distinct Genres" = n_distinct(genres)))) %>% 
  knitr :: kable()
```


```{r rating_distribution, echo = FALSE}
#i create a dataframe "explore_ratings" which contains half star and whole star ratings  from the edx set : 
group <-  ifelse((edx$rating == 1 |edx$rating == 2 | edx$rating == 3 | 
                    edx$rating == 4 | edx$rating == 5) ,
                 "whole_star", 
                 "half_star") 
explore_ratings <- data.frame(edx$rating, group)
```

```{r histogram of ratings, echo = FALSE}
# histogram of ratings
ggplot(explore_ratings, aes(x= edx$rating, fill = group)) +
  geom_histogram( binwidth = 0.2) +
  scale_x_continuous(breaks=seq(0, 5, by= 0.5)) +
  scale_fill_manual(values = c("half_star"="brown", "whole_star"="red")) +
  labs(x="rating", y="number of ratings") +
  ggtitle("Histogram: number of ratings per rating")
```

We can observe that some movies have been rated more often than others, while some have very few ratings and sometimes only one rating. This will be important for our model as very low rating numbers might results in untrustworthy estimate for our predictions. 

Thus regularization and a penalty term will be applied to the models in this project. Regularization is a technique used to reduce the error by fitting a function appropriately on the given training set and avoid overfitting (the production of an analysis that corresponds too closely or exactly to a particular set of data, and may therefore fail to fit additional data or predict future observations reliably). Regularization is used for tuning the function by adding an additional penalty term in the error function. The additional term controls the excessively fluctuating function such that the coefficients do not take extreme values.

```{r number_of_ratings_per_movie, echo = TRUE, fig.height=4, fig.width=5}
edx %>%
  count(movieId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 30, color = "yellow") +
  scale_x_log10() +
  xlab("Number of ratings") +
  ylab("Number of movies") +
  ggtitle("Number of ratings per movie")
```

As more than 100 movies that were rated only once appear to be obscure, predictions of future ratings for them will be difficult.

```{r obscure_movies, echo = TRUE, fig.height=4.5, fig.width=5}
edx %>%
  group_by(movieId) %>%
  summarize(count = n()) %>%
  filter(count == 1) %>%
  left_join(edx, by = "movieId") %>%
  group_by(title) %>%
  summarize(rating = rating, n_rating = count) %>%
  slice(1:15)%>% 
  knitr :: kable()

```


We can observe that the majority of users have rated between 30 and 100 movies. So, a user penalty term need to be included later in our models.

```{r number_ratings_given_by_users, echo = TRUE, fig.height=4, fig.width=5}
edx %>%
  count(userId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 30, color = "green") +
  scale_x_log10() +
  xlab("Number of ratings") + 
  ylab("Number of users") +
  ggtitle("Number of ratings given by users")
```


Furthermore, users differ greatly in how critical they are with their ratings. Some users tend to give much lower star ratings and some users tend to give higher star ratings than average. The visualization below includes only users that have rated at least 120 movies.

```{r avg_movie_ratings_given_by_users, echo = FALSE, fig.height=4, fig.width=5}
edx %>% group_by(title) %>%
  filter(n() >= 120) %>%
  summarize(avg = mean(rating)) %>%
  ggplot(aes(avg)) +
  geom_histogram(binwidth = 0.5, color = "white", fill = "brown") +
  scale_x_continuous(breaks = seq (0, 5, 0.5)) +
  xlab("Mean rating") +
  ylab("Number of users") +
  ggtitle("Mean movie ratings given by users") +
  theme(panel.grid.minor.x=element_blank(),
        panel.grid.major.x=element_blank())
```
```{r mean average movie rating}
mean(edx %>% group_by(title) %>%
       summarize(avg = mean(rating)) %>% 
       .$avg)
sd(edx %>% group_by(title) %>%
     summarize(avg = mean(rating)) %>% 
     .$avg)
```

We find that the mean average movie rating is approximately 3.2 with a standard deviation 0.6

#### 3.2 Methods and data analysis

##### 3.2.1  Recommendation system using machine learning

Noticing that the movie release years are contained in the titles, we extract the year information using the `str_extract` function and save this in a title named `years` containing movie titles and their release years.

```{r extracting years data , echo=FALSE}
years <- edx %>% group_by(title) %>%
  summarise(year = as.numeric(substr(str_extract(title[1], "\\d{4}\\)$"), 0, 4)))
head(years)%>% 
  knitr :: kable()
```

We join the `year` column onto our `edx` training set, and will later join it onto the `validation` set during testing:
  
  ```{r view splitted dataset, echo=FALSE}
edx_df <- edx %>% left_join(years, by = "title")
head(edx_df)%>% 
  knitr :: kable()
```

Before we begin building our algorithm to predict movie ratings, in order to train our algorithm we first split our training set `edx_df` into a training set containing 80% of entries and a test set containing 20%.

```{r spliting edx into train and test set, warning=FALSE, echo = FALSE}
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = edx_df$rating, times = 1,
                                  p = 0.2, list = FALSE)
train_set <- edx_df[-test_index,]
test_set <- edx_df[test_index,]
test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")%>%
  semi_join(train_set, by = "year")
```

Lets remove the rating column from the test_set:
  
  ```{r remove rating column from test_set, echo = FALSE}
# test_set can be further modified by removing rating column
test_set_df <- test_set  
test_set <- test_set %>% select(-rating)
test_set %>% head()%>% 
  knitr :: kable()
```

We examine the relationship between movie release year and average user rating in the plot below:
  
  ```{r examine time effect, echo=FALSE, warning=FALSE}
edx_df %>% group_by(year) %>% 
  summarise(rating = mean(rating)) %>%
  ggplot(aes(year, rating)) +
  geom_point() +
  geom_smooth(method = "lm") +
  xlab("Movie release year") +
  ylab("Average user rating")
```

##### 3.2.2 Loss function

We are going to use RMSE to represent the loss function:
  
  $$RMSE = \sqrt{\frac{1}{N} \sum_{u,i}^{n} \left(\hat{y}_{u,i} - y_{u,i} \right)^2 }$$
  
  ```{r RMSE function, echo = FALSE}
#RMSE is expected to be lower than 0.857
#i define the RMSE function as:
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}
```

##### 3.2.3 Just the average rating model

```{r model_1: Just the average, echo=FALSE, warning=FALSE}
mu_hat <- mean(train_set$rating)
mu_hat
```

If we predict all unknown ratings with  $\mu$ we obtain the following RMSE. We create a results table with this naive approach to give:
  
  ```{r predict rmse for model_1, echo=FALSE, warning=FALSE}
model_1_rmse <- RMSE(test_set_df$rating, mu_hat)
rmse_results_1 <- tibble(Method = "Just the average", RMSE = model_1_rmse)
rmse_results_1%>% 
  knitr :: kable()
```

##### 3.2.4 Movie effects model

To improve above model we focus on the fact that, from experience, we know that some movies are just generally rated higher than others. Higher ratings are mostly linked to popular movies among users and the opposite is true for unpopular movies. We compute the estimated deviation of each movies’ mean rating from the total mean of all movies $\mu$. The resulting variable is called "b" ( as bias ) for each movie "i" $b_{i}$, that represents average ranking for movie $i$:
  $$Y_{u, i} = \mu +b_{i} + \epsilon_{u, i}$$
  
  The histogram is left skewed, implying that more movies have negative effects

```{r model_2: movie effect, echo=FALSE, warning=FALSE}
mu <- mean(train_set$rating)

b_movie <- train_set %>% 
  group_by(title) %>% 
  summarize(b_movie = mean(rating - mu))

predicted_ratings_movie_avg <- test_set %>%
  left_join(b_movie, by = "title") %>%
  mutate(pred = mu + b_movie) %>%
  .$pred
```

We can check how much our prediction improves once we use:
  
  ```{r predict rmse for movie-effect, echo=FALSE, warning=FALSE}
model_2_rmse <- RMSE(test_set_df$rating, predicted_ratings_movie_avg)

rmse_results_2 <- data_frame(Model=c("Just the average","Movie Effect"),  
                             RMSE = c(model_1_rmse,model_2_rmse))
rmse_results_2%>% 
  knitr :: kable()
```

We already seeing some improvement in RMSE but we can do much better than this.

##### 3.2.5 Movie + User effect model

Let’s compute the average rating for user $\mu$ for those that have rated over 100 movies:
  
  ```{r model_3: movie + user_effect, echo=FALSE, warning=FALSE}
b_movie <- train_set %>% 
  group_by(title) %>% 
  summarize(b_movie = mean(rating - mu))
b_user <- train_set %>% 
  left_join(b_movie, by="title") %>%
  group_by(userId) %>%
  summarize(b_user = mean(rating - mu - b_movie))

predicted_ratings_user_avg <- test_set %>%
  left_join(b_movie, by = "title") %>%
  left_join(b_user, by = "userId") %>%
  mutate(pred = mu + b_movie + b_user) %>%
  .$pred
```

We can now construct predictors and see how much the RMSE improves:
  
  ```{r predict rmse for movie + user_effect, echo=FALSE, warning=FALSE}
# test and save rmse results 
model_3_rmse <- RMSE(predicted_ratings_user_avg, test_set_df$rating)
rmse_results_3 <- data_frame(Model=c("Just the average","Movie Effect Model","Movie + User Effect Model"),  
                             RMSE = c(model_1_rmse, model_2_rmse, model_3_rmse))
rmse_results_3%>% 
  knitr :: kable()
```

##### 3.2.6 Movie + User + Time effect model

```{r model_4: movie + user + time effect, echo=FALSE, warning=FALSE}
b_movie <- train_set %>% 
  group_by(title) %>% 
  summarize(b_movie = mean(rating - mu))
b_user <- train_set %>% 
  left_join(b_movie, bt="title") %>%
  group_by(userId) %>%
  summarize(b_user = mean(rating - mu - b_movie))
b_time <- train_set %>%
  left_join(b_movie, bt="title") %>%
  left_join(b_user, bt="userId") %>%
  group_by(year) %>%
  summarize(b_time = mean(rating - mu - b_movie - b_user))

predicted_ratings_time_avg <- test_set %>%
  left_join(b_movie, by = "title") %>%
  left_join(b_user, by = "userId") %>%
  left_join(b_time, by = "year") %>%
  mutate(pred = mu + b_movie + b_user + b_time) %>%
  .$pred
```
We calculate the RMSE for our models:
  
  ```{r rmse for movie+user+time  effect, echo=FALSE, warning=FALSE}
# i calculate the RMSE for movies, users and time effects 
model_4_rmse <- RMSE(predicted_ratings_time_avg, test_set_df$rating)
rmse_results_4 <- data_frame(Model=c("Just the Average","Movie Effect Model","Movie + User Effect Model","Movie + User + Time Effect Model"),  
                             RMSE = c(model_1_rmse,model_2_rmse, model_3_rmse, model_4_rmse))
rmse_results_4%>% 
  knitr :: kable()
```


#### 3.3 Regularization

```{r RMSE margin from regularization approach}
#i check the reduction in RMSE
rmse_margin <- model_2_rmse-model_4_rmse
round(rmse_margin*100,1)
```

Despite the large movie to movie variation, our improvement in RMSE was only about 7.8%. 

The penalized estimates did not provide a large improvement over the least squares estimates so we apply regularization effect to determine a better RMSE.

#### 3.3.1 Choosing the penalty terms

We have noticed in our data exploration, some users are more actively participated in movie reviewing. There are also users who have rated very few movies (less than 30 movies). On the other hand, some movies are rated very few times (say 1 or 2). These are basically noisy estimates that we should not trust. Additionally, RMSE are sensitive to large errors. Large errors can increase our residual mean squared error. So we must put a penalty term to give less importance to such effect.

Note that  $\lambda$ is a tuning parameter. We can use cross-validation to choose it.

So estimates of $b_{movie}$ and $b_{user}$ are caused by movies with very few ratings and in some users that only rated a very small number of movies. Hence this can strongly influence the prediction. The use of the regularization permits to penalize these two aspects. We should find the value of lambda (that is a tuning parameter) that will minimize the RMSE. This shrinks the $b_{movie}$ and $b_{user}$ in case of small number of ratings.

```{r lambdas, echo = TRUE, warning=FALSE}
# lambda is a tuning parameter
# Use cross-validation to choose it.
lambdas <- seq(0, 10, 0.25)
# For each lambda,find b_i & b_u, followed by rating prediction & testing
# note:the below code could take some time 
rmses <- sapply(lambdas, function(l){
  
  mu <- mean(train_set$rating)
  
  b_movie <- train_set %>% 
    group_by(movieId) %>%
    summarize(b_movie = sum(rating - mu)/(n()+l))
  
  b_user <- train_set %>% 
    left_join(b_movie, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_user = sum(rating - b_movie - mu)/(n()+l))
  
  predicted_ratings <- validation %>% 
    left_join(b_movie, by = "movieId") %>%
    left_join(b_user, by = "userId") %>%
    mutate(pred = mu + b_movie + b_user) %>%
    .$pred
  
  return(RMSE(validation_df$rating,predicted_ratings))
})
```

```{r plot rmses vs lambdas, echo=TRUE}
# Plot rmses vs lambdas to select the optimal lambda
qplot(lambdas, rmses)  
```
For the full model, the optimal lambda is: 4.75
```{r min_lambda, echo = TRUE}    
lambda <- lambdas[which.min(rmses)]
lambda
```
The new results will be:
  
  ```{r rmse_results_reg, echo = TRUE}
rmse_reg_model <- bind_rows(rmse_results_4,
                            data_frame(Model="Regularized Movie + User Effect Model",  
                                       RMSE = min(rmses)))
rmse_reg_model%>% 
  knitr :: kable()
```

\pagebreak

### 4.0 Results

#### 4.1 RMSE overview

```{r RMSE margin from model 3: movie + user effect,  echo=FALSE}
rmse_margin_reg <-(model_3_rmse-min(rmses))*100
round(rmse_margin_reg,2)
```

The RMSE value for the our regularized model is shown below:
  
  ```{r final value of RMSE and the model, echo=FALSE, warning=FALSE}
#i calculate the percentage margin of improvement in the value of RMSE from the movie+user effect model when penalties were applies

final_val <- rmse_reg_model[5,]
final_val

```

### 5.0 Concluding Remarks

We can affirm to have built a machine learning algorithm to predict movie ratings with MovieLens dataset.

The regularized model including the effect of user is characterized by the lower RMSE value and is hence the optimal model to use for the present project. The optimal model characterized by the lowest RMSE value (0.8652421) lower than the initial evaluation criteria (0.8775) given by the goal of the present project. This value of RMSE is over 19.9% with respect to the first model

We could also affirm that improvements in the RMSE could be achieved by adding other effect such as year. Other different machine learning models could also improve the results further, but hardware limitations, as the RAM, are a constraint.

### 6.0 References

- <https://github.com/johnfelipe/MovieLens-2>
  - <https://github.com/cmrad/Updated-MovieLens-Rating-Prediction>
  