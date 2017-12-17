# Parallelized NLP Market Prediction
Stock market prediction through parallel processing of news stories and basic machine learning.

## Abstract
Many consider the stock market to be efficient, however many of these efficiencies are based on numerical data (ie. past values and financials). There is, however, a wealth of textual data which influences investors’ decisions and may lead to inefficiencies as it is hard to analyze quickly. This project explores how basic Natural Language Processing (NLP) algorithms might be parallelized to take advantage of these inefficiencies. It explores the challenges of memory management of and limited string operations in parallelized NLP as well as its feasibility. Finally, we present a possible solution to these considerations and show that parallelization has the potential to speed up NLP by 1.5 to 10 times.

## File Structure
```
./
./data/
./data/articles/*                                       <= Folders for each day of loaded articles
./data/articles/stock_market_prediction-10-28-2017/*    <= Articles for each stock by ticker stored in text files for the day specified by the folder
./data/stock_price_data.txt                             <= Text file containing stock price data. This file serves as the database storage on disk for all the pulled stock prices.
./data/(word_weight_data_opt1.txt)                      <= Not initially in the folder but will be created when the update command is run for the first time. 
                                                           This file contains the learned word weights for the basic weighting scheme.
./data/(word_weight_data_opt2.txt)                      <= Not initially in the folder but will be created when the update command is run for the first time. 
                                                           This file contains the learned word weights for the Bayesian weighting scheme.
./output/*                                              <= Empty until a prediction is made but will eventually contain every prediction made by the program indicated by the name of the text file.
./gpu_device_query.out                                  <= Output of a device query for the device testing was completed on.
./prediction_accuracy.png                               <= Generated graph of prediction acuracy from late October to early December based on the data available in the repository.
./README.md                                             <= This file.
./stock_market_prediction.py                            <= Python file containing all the code for the project. How to run and test is listed below. Examine comments and function names for clarity.                                                                                                                   
```

## How To Run
### GPU vs. CPU
This code is designed to be run on a general purpose graphics processing unit or a normal computer (CPU). If you wish to use a GPU, timing analysis and output comparisons will also be run so it will take considerably longer. To indicate which type of operation you would like to use, set the flag at the top of the file. Use `GPU = True` to enable the GPU code and `GPU = False` to run on a CPU. Some commands such as pulling articles and stock prices are not parallelized due to network constraints and cannot be run on GPU (or at least the class server we are testing on due to missing librarys). A library of articles and a database of stock prices is included so these functions are not necessary for testing.

### A Full Test of Parallelizable Aspects
The parallelizable aspects of this project can be tested on the class GPU by doing the following. First, clone the git repository to the server. It contains a collection of news articles and a database of stock prices that can be used in testing. The folder structure is explained above. The root of the directory contains `stock_market_prediction.py`. This is the file that contains all the functions and code. 

#### Train Two Models Over 3 Days
The downloaded repository does not have any models trained so the first step is to train a model for both the basic weighting scheme and the Bayesian weighting scheme. Running the following commands from the root directory will train the model over 3 days. You can adjust the end date to train over more days, however, it each day added will take longer as everything is being calculated by the GPU and CPU for comparison. Word weights for both weighting schemes will be updated using the GPU. The first command is for the basic word weights, the second is for the Bayesian ones.

```
sbatch --gres=gpu:1 --time=20 --wrap="python stock_market_prediction.py -u -b 11-7-2017 -e 11-9-2017"
sbatch --gres=gpu:1 --time=20 --wrap="python stock_market_prediction.py -u -b 11-7-2017 -e 11-9-2017 -o opt2"
```
The `-u` in the commands is for update word weights, the `-b` indicates the start day and `-e` indicates the end day. The `-o` indicates the type of weight to be updated. The default is `opt1` for the basic weighting scheme which is why it is left out of the first command.

#### Make A Prediction
Once trained, a prediction can be made. A prediction for the next day using both weighting types can be done through the following commands. The first is for making predictions with the basic weighting scheme and the second is for making predictions using the Bayesian classifier.
```
sbatch --gres=gpu:1 --time=20 --wrap="python stock_market_prediction.py -p -d 11-10-2017"
sbatch --gres=gpu:1 --time=20 --wrap="python stock_market_prediction.py -u -d 11-10-2017 -o opt2"
```
The `-p` indicates to make a prediction, the `-d` is to specify the day to make the prediction. The update command from the previous section can also use the `-d` option to update weights for a specific day. Once again `-o` is used to signal the Bayesian classifier in the second command.

#### Output
By "cat"-ing the slurm files, you will be able to see the output of the commands. The update commands will just list the functions being run, the accuracy between of the GPU and the time difference. The listed speedup will likely be fairly low due to it starting from an uninitialized database. The report goes into more depth on why this is the case.The output of the prediction commands will show the predictions for November 10th for each of the 20 stocks with some stats about the prediction. At the bottom of both outputs will be the accuracy and timing. The output of the prediction using the basic weights will also include weight analysis accuracy and timing. The Bayesian classifier prediction does not analyze the weights. If you wish, you can run:
```
./stock_market_prediction.py -v
```
This will indicate how correct the predictions were.

#### Next Testing Steps
If more tests are desired, you can continue to run the update and predict commands on different days. If you run them on days without stock price data or articles, you will recieve an error. You can determine which days have downloaded articles by examining the ./data/articles/ folder. Each day there is an article has a price in the database except December 5th which has, so far, only been used for predicting.

### Available Commands
#### PULLING ARTICLES (No GPU):

For the current day:
```
./stock_market_prediction.py -a
```

Supported Options:
```
none
```

#### PREDICTIONS:
(articles for the requested day must already be pulled)

For the current day with basic weights:
```
./stock_market_prediction.py -p
```

Supported Options:
```
-o <weight options>
-d <specified date>
```

#### PULLING STOCK PRICES:

For the current day:
```
./stock_market_prediction.py -s
```

Supported Options:
```
none
```

#### UPDATE WEIGHTS:
(articles and stock prices for the requested days must already be pulled)

For the update basic weights for the current day:
```
./stock_market_prediction.py -u
```

Supported Options:
```
-o <weight options>
-d <specified date>
-b <start date>
-e <end date>
```
If a start date is given without an end date, the end date will default to the current date. The specified date option does not work with the start or end date options. If there are dates within the date range (from -b date to -e date or current date) that do not have articles or stock data (such as weekends), they are skipped.

#### HELP:

To get this list printed:
```
./stock_market_prediction.py -h
```

#### WEIGHT ANALYSIS (helper function that just prints weight statistics):
(must have weights calculated for the standard weight option)

Print the weight statistics:
```
./stock_market_prediction.py -z
```

#### DETERMINE ACCURACY (helper function for analyzing predictions over time):

Determine the accuracy of any predictions in the ./output/ folder:
```
./stock_market_prediction.py -v
```

This command also generates the graph of accuracy relative to the days the predictions were made.



