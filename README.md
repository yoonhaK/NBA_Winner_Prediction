# NBA_Winner_Prediction
 Can we predict individual win shares of NBA players using other basketball metrics?
 According to Baketball-Reference, win shares is a metric that estimates the number of wins a player produces 
 for his team throughout the season
 
 For example, Michael Jordan is both the single season leader in win shares with 20.4 win shares and all-time career 
 leader in win shares with 214 win shares. 
 
 It hints that win shares could potentially be a good indicator of how much success a player contributes to their team

## Code used 
- Python version: 3.7
- Packages: pandas, numpy

## Web Scraping
Used pandas read_html to scrape Individual Player's Advanced Stats per season from year 2001 to 2021
from https://www.basketball-reference.com/. With each player, the stats collected are following:
- Rank(Rk), Player, Position (Pos), Age, Team(Tm), Games(G), Minutes Played (MP), 
Player Efficiency Rating (PER), True Shooting Percentage (TS%)
- 3-Point Attempt Rate (3PAr), Free Throw Attempt Rate (FTr),Offensive Rebound Percentage (ORB%),Defensive Rebound Percentage (DRB%)
,Total Rebound Percentage (TRB%), Assist Percentage(AST%), Steal Percentage(STL%), Block Percentage(BLK%), Turnover Percentage(TOV%)
, Usage Percentage(USG%)
- Offensive Win Shares(OWS), Defensive Win Shares(DWS), Win Shares(WS), Win Shares Per 48 Minutes (WS/48)
- Offensive Box Plus/Minus (OBPM), Defensive Box Plus/Minus(DBPM), Box Plus/Minus(BPM), Value over Replacement Player(VORP)

Also, since read_html function returns with list of dataframe, I converted into one big dataframe

## Data Cleaning
After scraping the data, I needed to clean it up so that it was usable for our model. I made the following changes:
- Removed multiple headers that were created when I scraped the data
- Removed empty columns 

## Explanatory Data Analysis

I looked at the distributions of the data with histograms and the value counts for the various categorical variables using pivot tables. 
I also looked pair graphs and correlation matrix to see the correlation. With these graphs, I selected some predict variables.
Below are a few highlights.

<p float="left">
<img src="https://github.com/yoonhaK/NBA_Winner_Prediction/blob/master/graph_images/pivot_table.png" width="300"/>
<img src="https://github.com/yoonhaK/NBA_Winner_Prediction/blob/master/graph_images/WS_histogram.png" width="500"/>
<img src="https://github.com/yoonhaK/NBA_Winner_Prediction/blob/master/graph_images/barchart.png" width="500"/>
<img src="https://github.com/yoonhaK/NBA_Winner_Prediction/blob/master/graph_images/heatmap.png" width="400"/>
<img src="https://github.com/yoonhaK/NBA_Winner_Prediction/blob/master/graph_images/pair_plots.png" width="500"/>
</p>

## Model Building


First, I transformed the categorical variables into dummy variables. I also split the data into train and tests sets with a test size of 20%.


I tried four different models and evaluated them using Mean Absolute Error. I chose MAE because it is relatively easy to interpret.

Three models I tried:
- Multiple Linear Regression: Since the WS is not categorical, it is a good model for the baseline.
- Decision Tree: Because of the sparse data from the many categorical variables, I thought a Decision Tree would be effective.
- Random Forest Regression: Because of the sparse data from the many categorical variables, I thought Random Forest would be a good fit. 

## Model Performance
The Random Forest model outperformed the other approaches on the test and validation sets.
- Random Forest : MAE = 0.4519528606965175
- Linear Regression: MAE = 0.5021073044170522
- Decision Tree: MAE = 0.5933855985540939

<img src="https://github.com/yoonhaK/NBA_Winner_Prediction/blob/master/graph_images/2021_top_ten.png" width="500"/>
