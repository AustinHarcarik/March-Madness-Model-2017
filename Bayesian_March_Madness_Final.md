March Madness Model Report
================
Austin Harcarik
3/15/2017

Introduction:
=============

Rather than filling out my bracket with my gut this year, I decided to make a model and see how it does. The goal of this study is to model the score difference of any game (team a - team b) using data from the 2016-2017 NCAA basketball season. The data used in this study is from a Kaggle competition. There are about 5400 games and 351 teams in the dataset. Because of my strange obsession with Andrew Gelman, I had to make a bayesian hierarchical model.

Model:
======

This model has two main assumptions:
- each team has some 'true talent level'
- each team has some 'true home court advantage'

The score differential in game *i*, denoted as *y*<sub>*i*</sub> is modeled as a student-t distribution with 7 degrees of freedom to allow for outliers (per Andrew Gelman’s recommendation): *y*<sub>*i*</sub> *t*<sub>7</sub>(*a*<sub>*h**o**m**e*</sub> − *a*<sub>*a**w**a**y*</sub> + *b*<sub>*h**o**m**e*</sub>, *σ*<sub>*y*</sub>) where *a*<sub>*h**o**m**e*</sub> and *a*<sub>*a**w**a**y*</sub> are the true talent levels of the home and away teams, respectively, and *b*<sub>*h**o**m**e*</sub> is the home court advantage for the home team. I assume a hierarchical structure for both the true talents and true home court advantages: *a*<sub>*h**o**m**e*</sub>, *a*<sub>*a**w**a**y*</sub> *N*(0, *τ*<sub>*a*</sub>) *τ*<sub>*a*</sub> *U**n**i**f*(0, 14) *b*<sub>*h**o**m**e*</sub> *N*(0, *τ*<sub>*b*</sub>) *τ*<sub>*b*</sub> *U**n**i**f*(0, 7) *σ*<sub>*y*</sub> *U**n**i**f*(0, 20)

I tried to stick with noninformative priors, but I did limit the parameter space pretty significantly. I believe this to be justified since we have a pretty good idea of how basketball scores are distributed. Moreover, we expect the home court advantage distribution from which each team's home court advantage is drawn to be noticeably less variable than the true talent distribution.

Because about 10% of all games are played on a neutral court, I only included *b*<sub>*h**o**m**e*</sub> for games in which the home team was actually on its home court. Otherwise, this parameter is zero'd out.

``` r
### Cleaning and Munging Data ### 

# load libraries
library(plyr)
library(dplyr)
library(rstan)

# read in data (from Kaggle)
mydata <- read.csv('/users/austinharcarik/Desktop/March_Madness_Data/RegularSeasonCompactResults.csv')
teams <- read.csv('/users/austinharcarik/Desktop/March_Madness_Data/Teams.csv')
mm <- c();

# filter to 2017
mydata <- mydata %>%
  filter(Season == 2017)

# functions to map home and away teams 
map_home <- function(x) {
  home_team <- x[3]
  if (x[7] == 'H') {
    home_team <- x[3]
  }
  if (x[7] == 'A') {
    home_team <- x[5]
  }
  return(as.numeric(home_team))
}
map_away <- function(x) {
  away_team <- x[5]
  if (x[7] == 'H') {
    away_team <- x[5]
  }
  if (x[7] == 'A') {
    away_team <- x[3]
  }
  return(as.numeric(away_team))
}
is_home <- function(x) {
  if (x[7] != 'N') {
    home <- 1
  } else {
    home <- 0
  }
  return(home)
}

# data cleaning 
mydata$team_1 <- apply(X = mydata, FUN = map_home, MARGIN=1)
mydata$team_2 <- apply(X = mydata, FUN = map_away, MARGIN=1)
mydata$is_home <- apply(X = mydata, FUN = is_home, MARGIN=1)
idxs <- which(mydata$Wloc == 'A')
mydata$diff <- mydata$Wscore - mydata$Lscore
mydata[idxs,12] <- mydata[idxs,12] * -1
ids <- sort(unique(mydata$team_1))
new_ids <- 1:length(ids)
mydata$team_1 <- mapvalues(mydata$team_1, from=ids, to=new_ids)
mydata$team_2 <- mapvalues(mydata$team_2, from=ids, to=new_ids)
mydata <- mydata[,c(9:12)]

# create list 
mm$team1 <- mydata$team_1
mm$team2 <- mydata$team_2
mm$is_home <- mydata$is_home
mm$score_diff <- mydata$diff
mm$nteams <- length(unique(mydata$team_1))
mm$ngames <- nrow(mydata)
saveRDS(mm, 'mm_data.rds')

head(mydata)
```

    ##   team_1 team_2 is_home diff
    ## 1      4     51       1   17
    ## 2    226      7       1   -6
    ## 3     11    170       0    2
    ## 4     12    230       1   18
    ## 5     15    129       1    9
    ## 6     18    207       1   17

``` r
### fitting the model ###

# stan code 
write(x="
      data {
      int nteams; // number of teams 
      int ngames; // number of games
      int<lower=1, upper=nteams> team1[ngames]; // team 1 ID (1, ..., 351)
      int<lower=1, upper=nteams> team2[ngames]; // team 2 ID (1, ..., 351)
      vector[ngames] is_home; // binary variable for home court advantage 
      vector[ngames] score_diff; // team 1 points - team 2 points 
      }
      
      parameters {
      real<lower=0> tau_a; // talent level variation hyperparameter
      real<lower=0> tau_b; // home court advantage variation hyperparameter
      real<lower=0> sigma_y; // score difference variation 
      vector[nteams] a; // team talent levels 
      vector[nteams] b; // team home court advantages
      }
      
      model {
      tau_a ~ uniform(0,14);
      tau_b ~ uniform(0,7);
      sigma_y ~ uniform(0,20);
      a ~ normal(0,tau_a);
      b ~ normal(0,tau_b);
      for (i in 1:ngames) {
      if (is_home[i] == 1) {
      score_diff[i] ~ student_t(7,a[team1[i]]-a[team2[i]] + b[team1[i]],sigma_y);
      } else {
      score_diff[i] ~ student_t(7,a[team1[i]]-a[team2[i]],sigma_y);
      }
      } 
      } 
      
      ", 
      file='mm_model_2.stan')

# fit the model 
mm <- readRDS("mm_data.rds")
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())
fit <- stan(file = 'mm_model_2.stan', 
            data = mm, 
            iter = 2000, 
            chains = 4)

# examine fit
options(max.print=100)
print(fit, digits=1)
```

    ## Inference for Stan model: mm_model_2.
    ## 4 chains, each with iter=2000; warmup=1000; thin=1; 
    ## post-warmup draws per chain=1000, total post-warmup draws=4000.
    ## 
    ##             mean se_mean   sd     2.5%      25%      50%      75%    97.5%
    ## tau_a        9.1     0.0  0.4      8.3      8.8      9.1      9.3      9.9
    ## tau_b        4.1     0.0  0.3      3.6      3.9      4.1      4.3      4.6
    ## sigma_y      9.1     0.0  0.1      8.8      9.0      9.1      9.1      9.3
    ## a[1]        -8.0     0.0  2.5    -12.8     -9.6     -8.0     -6.3     -3.1
    ## a[2]        -4.2     0.0  2.4     -8.8     -5.8     -4.2     -2.6      0.7
    ## a[3]         4.8     0.0  2.2      0.5      3.3      4.8      6.2      9.1
    ## a[4]        10.0     0.0  2.3      5.7      8.5     10.0     11.6     14.6
    ## a[5]       -22.3     0.0  2.3    -26.9    -23.8    -22.3    -20.7    -17.7
    ## a[6]       -16.0     0.0  2.1    -20.1    -17.4    -16.0    -14.7    -11.8
    ## a[7]         1.9     0.0  2.2     -2.5      0.5      1.9      3.3      6.2
    ##         n_eff Rhat
    ## tau_a    4000    1
    ## tau_b    2343    1
    ## sigma_y  4000    1
    ## a[1]     4000    1
    ## a[2]     4000    1
    ## a[3]     4000    1
    ## a[4]     4000    1
    ## a[5]     4000    1
    ## a[6]     4000    1
    ## a[7]     4000    1
    ##  [ reached getOption("max.print") -- omitted 696 rows ]
    ## 
    ## Samples were drawn using NUTS(diag_e) at Wed Mar 15 19:01:48 2017.
    ## For each parameter, n_eff is a crude measure of effective sample size,
    ## and Rhat is the potential scale reduction factor on split chains (at 
    ## convergence, Rhat=1).

``` r
# extract parameters 
sims <- extract(fit)
a_sims <- sims$a
a <- colMeans(a_sims)
b_sims <- sims$b
b <- colMeans(b_sims)
sigma_y_sims <- sims$sigma_y
sigma_y <- mean(sigma_y_sims)

# create final dataframe
results <- data.frame(ids, new_ids, a, b)
names(results) <- c('Team_Id', 'Team_Num', 'True_Talent', 'Home_Advantage')
final_results <- inner_join(results, teams, by='Team_Id')

# who are the best teams? 
final_results <- final_results %>%
  arrange(desc(True_Talent))
head(final_results, 10)
```

    ##    Team_Id Team_Num True_Talent Home_Advantage      Team_Name
    ## 1     1211      105    23.92243       2.588192        Gonzaga
    ## 2     1437      326    21.86853       1.780859      Villanova
    ## 3     1246      139    20.05714       3.712186       Kentucky
    ## 4     1196       90    19.79564       1.426618        Florida
    ## 5     1452      339    19.77969       4.911934  West Virginia
    ## 6     1438      327    19.02029       3.490241       Virginia
    ## 7     1181       75    18.98760       1.871375           Duke
    ## 8     1314      205    18.89915       7.449634 North Carolina
    ## 9     1242      135    18.80910       3.545610         Kansas
    ## 10    1257      150    18.22033       4.876177     Louisville

``` r
# who has the best home court advantage?
final_results <- final_results %>%
  arrange(desc(Home_Advantage))
head(final_results, 10)
```

    ##    Team_Id Team_Num True_Talent Home_Advantage       Team_Name
    ## 1     1393      283    7.941484      10.461703        Syracuse
    ## 2     1199       93   14.552697       8.264464      Florida St
    ## 3     1427      317   -8.386025       7.934709  UT San Antonio
    ## 4     1456      343   -1.116030       7.767891  William & Mary
    ## 5     1314      205   18.899146       7.449634  North Carolina
    ## 6     1249      142   -5.961792       6.417195           Lamar
    ## 7     1288      181  -13.638479       6.348573       Morgan St
    ## 8     1433      322    8.578758       6.248512 VA Commonwealth
    ## 9     1411      301   -3.826920       6.174610     TX Southern
    ## 10    1153       47   14.641938       6.131684      Cincinnati

``` r
# function to make probabilistic predictions
predict_matchup <- function(high_seed_team, low_seed_team) {
  first_team_idx <- which(final_results$Team_Name == high_seed_team)
  second_team_idx <- which(final_results$Team_Name == low_seed_team)
  first_team_talent <- final_results[first_team_idx, 3]
  second_team_talent <- final_results[second_team_idx, 3]
  diff <- first_team_talent - second_team_talent
  prob <- 1 - pnorm(0, mean=diff, sd=sigma_y)
  cat('probability that', high_seed_team, 'beats', low_seed_team, ':', prob)
}

predict_matchup("Duke", "Troy")
```

    ## probability that Duke beats Troy : 0.9697481

``` r
predict_matchup("Arkansas", "Seton Hall")
```

    ## probability that Arkansas beats Seton Hall : 0.5451734

``` r
predict_matchup("Notre Dame", "Princeton")
```

    ## probability that Notre Dame beats Princeton : 0.8174829

Vallidation:
I will now make predictions on this season's data and look at the residuals.

``` r
# function to predict score differential 
predict_score_diff <- function(x) {
  team_1 <- x[1]
  team_2 <- x[2]
  team_1_idx <- which(final_results$Team_Num == team_1)
  team_2_idx <- which(final_results$Team_Num == team_2)
  team_1_talent <- final_results[team_1_idx, 3]
  team_2_talent <- final_results[team_2_idx, 3]
  team_1_home_adv <- final_results[team_1_idx, 4]
  diff <- team_1_talent - team_2_talent
  if (x[3] == 1) {
    diff <- team_1_talent - team_2_talent + team_1_home_adv
  }
  return(diff)
}

mydata$preds <- apply(X=mydata, FUN=predict_score_diff, MARGIN=1)
mydata$residual <- mydata$diff - mydata$preds

# examine residuals 
p1 <- ggplot(mydata, aes(residual)) 
p1 + geom_histogram(bandwidth=5, fill='red')
```

![](Bayesian_March_Madness_Final_files/figure-markdown_github/validation-1.png)

``` r
p2 <- ggplot(mydata, aes(preds, residual))
p2 + geom_point() + geom_abline(slope=0, intercept=0, color='red')
```

![](Bayesian_March_Madness_Final_files/figure-markdown_github/validation-2.png)

``` r
rmse <- sqrt(mean(mydata$residual^2))
cat('Root Mean Squared Error:', rmse)
```

    ## Root Mean Squared Error: 9.948494
    
![Residuals](/validation-1.png)
![Residual_Plot](/validation-2.png)
