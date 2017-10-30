#!/usr/bin/env python
####################################################################################################
## Created 10/27/2017 by Sam Beaulieu
##
##
####################################################################################################

import requests
import os
import datetime
import time
import logging
import sys, getopt
import numpy as np
from bs4 import BeautifulSoup as BS
from yahoo_finance import Share
import struct
import binascii
import math
from scipy.stats import norm

########### Global Variables and Configurations ###########
# Global Constants
GPU = False
STOCK_TAGS = [	'amzn',
				'amat',
				'agn',
				'goog',
				'hd',
				'lmt',
				'data',
				'nflx',
				'aapl',
				'ge',
				'tsla',
				'bac',
				'nvda',
				'baba',
				'wmt']

ARTICLES_PER_STOCK = 10
SUCCESS_THREASHOLD = 5

# Global Stock Data
stock_data = {}
stock_prices = {}

# Global Word Data
words_by_letter = []
num_words_by_letter = []

# Global Prediction Data
weights_all = []
weight_average = 0
weight_stdev = 0
weight_sum = 0
weight_max = 0
weight_min = 1
weight_count = 0

# Global Configurations
if not os.path.exists('./data/'):
	os.makedirs('./data/')
	os.makedirs('./data/articles/')
elif not os.path.exists('./data/articles/'):
	os.makedirs('./data/articles/')

if not os.path.exists('./output/'):
	os.makedirs('./output/')

logging.basicConfig(filename="./data/stock_market_prediction.log", level=logging.DEBUG, format="%(asctime)s: %(levelname)s>\t%(message)s")
logging.info('RUNNING STOCK MARKET PREDICTION')

'''
Morning Prediction Step
Step 1 is to iterate through stock symbols and collect their article text. This step is not meant to be done in parallel but
to simply get recent articles about the stock. For my analysis, I try to scrape 10 of the most recent articles from google news
using the stocks ticker symbol as the search key. 

The following are done for each stock ticker:
a) Make the request to google news and pull the recent article links, reformat them to work right
b) Take the first 10 articles and pull all of their content
c) For each article, try to find the article text or content text
d) If article text was found, pull all the paragraph tag text
e) Append the text to one big string for the article and add it to the article dictionary
'''
def pull_recent_articles():

	logging.info('Pulling new articles')

	today = datetime.datetime.now()
	today_str = str(today.month) + '-' + str(today.day) + '-' + str(today.year)
	directory = './data/articles/stock_market_prediction-' + today_str + '/'

	if not os.path.exists(directory):
		os.makedirs(directory)

	print 'Pulling articles '

	# Iterate through the stock tags
	for ticker in STOCK_TAGS:
		start = time.time()

		logging.debug('Starting to pull articles for: ' + ticker)

		file = open(directory + ticker + '.txt', 'w')

		# Make the request to google news for the ticker and get its raw html
		try:
			r = requests.get('https://www.google.com/search?biw=1366&bih=671&tbm=nws&ei=HRDyWcDlJoa-jwSuwZCYBg&q=' + ticker)
		except requests.exceptions.RequestException, error:
			logging.warning('- Could not pull articles for: ' + ticker + ', Error: ' + str(error))
			continue

		raw_html = BS(r.text, 'html.parser')

		# Add the ticker to the dictionary and set its value to an empty array
		# - This is where the article data will be stored
		stock_data[ticker] = []

		# Search the raw html for the article urls
		counter = 0
		for link in raw_html.find_all('a'):
			all_raw_link = link.get('href')

			# Find all the links indicating news articles and format them
			if all_raw_link[:7] == '/url?q=':
				relative_raw_links = all_raw_link[7:]
				link = relative_raw_links.split('&')[0]

				logging.debug('- Pulling article from: ' + link)

				# Once the link as been found, make a request for its html content
				try:
					r = requests.get(link)
				except requests.exceptions.RequestException, error:
					logging.warning('-- Could not pull article for: ' + link + ', Error: ' + str(error))
					continue

				raw_html = BS(r.text, 'html.parser')

				# Search the raw html for the content or article section 
				# - if there is none, skip this article and try the next one
				raw_text = raw_html.find('div', attrs={'id' : 'content'})

				if raw_text == None:
					raw_text = raw_html.find('div', attrs={'class' : 'article-content'})

				if raw_text == None:
					raw_text = raw_html.find('div', attrs={'class' : 'article'} )

				if raw_text == None:
					raw_text = raw_html.find('div', attrs={'class' : 'content'} )

				if raw_text == None:
					raw_text = raw_html.find('div', attrs={'id' : 'article-content'})

				if raw_text == None:
					raw_text = raw_html.find('div', attrs={'id' : 'main-content'})

				if raw_text == None:
					raw_text = raw_html.find('div', attrs={'id' : 'main_content'})

				if raw_text == None:
					raw_text = raw_html.find('div', attrs={'id' : 'article'} )

				if raw_text == None:
					raw_text = raw_html.find('article')

				if raw_text == None:
					logging.warning('-- Could not find articles in: ' + link + ', trying next link')
					continue

				# Get all the paragraph parts of the article's content
				article_parts = raw_text.find_all('p')
				if len(article_parts) == 0:
					logging.warning('-- Could not find a article text in: ' + link + ', trying next link')
					continue

				# Combine all the paragraphs into a single text block
				article = ''
				for paragraphs in article_parts:
					article += paragraphs.text
					article += ' '

				# Store the url and article as a tuple into the stock data and write them to the stock's file
				stock_data[ticker].append((link, article))
				file.write(link.encode("utf-8") + '\n\n')
				file.write(article.encode("utf-8") + '\n')
				file.write('\n')
				logging.debug('-- Article ' + ticker + ':' + str(counter) + ' from ' + link + ' has been pulled and is ' + str(len(article)) + ' characters')

				# Update the counter (only want a static number of articles per stock)
				counter += 1

			if counter == ARTICLES_PER_STOCK:
				file.close()
				logging.debug('Finished pulling articles for: ' + ticker + ', time: ' + str(time.time() - start))
				break

	# Check all tags for number of articles loaded
	for ticker in STOCK_TAGS:
		if len(stock_data[ticker]) < SUCCESS_THREASHOLD:
			logging.error('Could not pull the threshold (' + str(SUCCESS_THREASHOLD) + ') number of articles. Either not saved or another error occured.')
			return False
	return True

'''
Evening Update Step
Step 5 if articles have already been pulled, then instad of pulling them again and wasting the time that it takes, we can instead
load the articles from that directory. Articles for each morning are kept in their directory so we open that up and load them into
the dictionary.
'''
def load_articles(day):
	
	directory = './data/articles/stock_market_prediction-' + day + '/'
	logging.info('Loading articles from directory: ' + directory)

	print 'Loading articles'

	# Iterate through the stock tags
	for ticker in STOCK_TAGS:
		start = time.time()

		# Try to open the file for that stock from the given directory
		logging.debug('Starting to load articles for: ' + ticker)
		try:
			file = open(directory + ticker + '.txt', 'r+')
		except IOError, error:
			logging.warning('- Could not load articles for: ' + ticker + ', Error: ' + str(error))
			continue

		# Prepare variables to save the article data
		stock_data[ticker] = []
		current_link = ''
		current_article = ''

		# Read in lines from the file
		for lines in file:

			# If the line is the url, save the old data and update the current link
			if lines[:4] == 'http':

					# Save current article
					if current_link != '':
						stock_data[ticker].append((current_link[:-1], current_article))
						logging.debug('-- Article for ' + ticker + ' from ' + current_link[:-1] + ' has been pulled and is ' + str(len(current_article)) + ' characters')

					# Set to prepare for the next link
					current_link = lines
					current_article = ''
					logging.debug('- Loading article from: ' + current_link[:-1])

			# If the line is part of an article, add it to the current article
			else:
				current_article += lines

		# Load the last article
		if current_article != '' and current_link != '':
			stock_data[ticker].append((current_link[:-1], current_article))
			logging.debug('-- Article for ' + ticker + ' from ' + current_link[:-1]+ ' has been pulled and is ' + str(len(current_article)) + ' characters')

		logging.debug('Finished pulling ' + str(len(stock_data[ticker])) + ' articles for: ' + ticker + ', time: ' + str(time.time() - start))

	# Check all tags for number of articles loaded
	for ticker in STOCK_TAGS:
		if len(stock_data[ticker]) < SUCCESS_THREASHOLD:
			logging.error('Could not load the threshold (' + str(SUCCESS_THREASHOLD) + ') number of articles. Either not saved or another error occured.')
			return False
	return True

'''
Evening Update Step
Step 6 is to load past stock data into the data structure. This will be used before pulling the data after the first run. This pulls
data from the stock price data file and loads it up so that new prices can be added to it.
'''
def load_stock_prices():

	logging.info('Loading past stock prices')
	print 'Loading stock prices'

	try:
		file = open('./data/stock_price_data.txt', 'r+')
	except IOError, error:
		logging.warning('- Could not load stock prices, Error: ' + str(error))
		return

	# Iterate through the lines in the file
	current_stock = ''
	for lines in file:

		# If the first character is not a '-', is a ticker, add it to the dictionary (w/o the newline)
		if lines[:1] != '-':
			stock_prices[lines[:-1]] = {}
			current_stock = lines[:-1]
			logging.debug('- Loading price data for: ' + current_stock)

		# If the first character is a '-', it has data so split it up and add it
		else:
			data = lines.split()
			stock_prices[current_stock][data[1]] = (float(data[2]), float(data[3]))

	file.close()

'''
Evening Update Step
Step 8 is to pull all stock data for the current day and add it to the stock prices data structure. The yahoo finance api is used
for this and open and close prices are all thats kept.
'''
def pull_stock_prices():

	today = datetime.datetime.now()
	today_str = str(today.month) + '-' + str(today.day) + '-' + str(today.year)

	logging.info('Pulling stock prices for: ' + today_str)
	print 'Pulling stock prices'

	# For each stock, get todays value
	for tickers in STOCK_TAGS:

		logging.debug('- Pulling price data for: ' + tickers)

		# If the stock is not in the stock prices, add it
		if tickers not in stock_prices:	
			stock_prices[tickers] = {}

		# Get todays stock open and close
		open_p = Share(tickers).get_open()
		close_p = Share(tickers).get_price()

		logging.debug('-- Price for: ' + tickers + ' is open: ' + open_p + ', close: ' + close_p)

		# Add todays prices to the dictionary for today
		stock_prices[tickers][today_str] = (float(open_p), float(close_p))

'''
Evening Update Step
Step 10 is to save the stock price data structure to a file so it can be loaded on subsequent days. 
'''
def save_stock_prices():

	logging.info('Saving stock price data structure')
	print 'Saving stock prices'

	file = open('./data/stock_price_data.txt', 'w')

	# Iterate through all the stocks and write each to the file in the following manner:
	# amzn
	# - 10/27/2017 1053.23 1100.34
	# - 10/28/2017 ...
	# agn
	# - 10/27...

	# Iterate through the stocks to write them to the file
	for ticker in STOCK_TAGS:
		
		logging.debug('- Saving data for: ' + ticker)

		# Write the ticker to the file
		file.write(ticker + '\n')

		# Write each day to the file
		for days, price in stock_prices[ticker].iteritems():
			file.write('- ' + days + ' ' + str(price[0]) + ' ' + str(price[1]) + '\n')

	file.close()

'''
Morning Prediction Step & Evening Update Step
Step 2 & 7 is to load all the word weights into the right data structures so it can be used in analyzing the new data. The strange array structure 
is to enable implementation in pyCUDA in the near future. Otherwise, a hash function would have ben used for constant time access. The words are
stored by letter which will hopefully make it more efficient.
'''
def load_all_word_weights(option):
	
	logging.info('Loading word weights for weighting option: ' + option)
	print 'Loading word weights'

	# For each letter, add a 1000 word array of 28 characters each with the last 8 characters for a float (weight of the word) and int (number of occurences)
	# - abcdefghijklmnopq0.32########
	for letters in range(0, 26):
		letter_words = bytearray(28*1000)
		words_by_letter.append(letter_words)
		num_words_by_letter.append(0)


	# Open the file to be loaded
	try:
		file = open('./data/word_weight_data_' + option + '.txt', 'r+')
	except IOError, error:
		logging.warning('- Could not load word weights, Error: ' + str(error))
		return

	# Iterate over all the lines in the file
	letter_index = 0
	for lines in file:

		# If the first character is not a '-', it indicates a letter change
		if lines[:1] != '-':
			letter_index = ord(lines[:1]) - 97
			logging.debug('Loading words beginning with: ' + lines[:1])

		# If not a '-', pack up the data, store it, and update the number of words
		else:
			# Get the data
			data = lines.split()

			# Store the data
			struct.pack_into('16s f i i', words_by_letter[letter_index], num_words_by_letter[letter_index]*28, data[1], float(data[2]), int(data[3]), int(data[4]))
			num_words_by_letter[letter_index] += 1

	file.close()

'''
Evening Update Step
Step 9 is to update all the word weights. This function goes through each of the articles and finds every word in them. It then
calls the update_word function which either adds or updates the word in the weight array. 
'''
def update_all_word_weights(option, day):
	'''
	| |
	|_|
	|_|_______________
	|3|_0_|__1__|__2__
	|_|         |'hey'|
	| |         | 0.8  |
	| |         | 80  |

	'''

	logging.info('Updating word weights for: ' + day + ' with option: ' + option)
	print 'Updating word weights'

	# If the weighting arrays are empty, create them 
	if len(words_by_letter) == 0 or words_by_letter == 0:

		logging.debug('- Could not find data structure for word weights so creating it')

		# For each letter, add a 1000 word array of 28 characters each with the last 8 characters for a float (weight of the word) and int (number of occurences)
		# - abcdefghijklmnopq0.32####
		for letters in range(0, 26):
			letter_words = bytearray(28*1000)
			words_by_letter.append(letter_words)
			num_words_by_letter.append(0)

	# At this point, the weighting arrays are initialized or loaded
	# - Next step is to iterate through articles and get the words
	for ticker in STOCK_TAGS:
		
		logging.debug('- Updating word weights for: ' + ticker)

		# For each stock, iterate through the articles
		for articles in stock_data[ticker]:
		
			# Get the text (ignore link)
			text = articles[1]

			# Variables to keep track of words
			word_in_progress = False
			word_number = 0
			word_start_index = 0

			# Iterate through the characters to find words
			for ii, chars in enumerate(text):

				# If there is a word being found and non-character pops up, word is over
				if word_in_progress and (not chars.isalpha() or ii == len(text)):

					# Reset word variables
					word_in_progress = False
					word_number += 1

					# Get the found word
					found_word = text[word_start_index:ii]

					# Add the word to the word arrays or update its current value
					if len(found_word) > 1:
						update_word(ticker, option, found_word, day)

				# If a word is not being found and letter pops up, start the word
				elif not word_in_progress and chars.isalpha():

					# Start the word
					word_in_progress = True
					word_start_index = ii

'''
Evening Update Step Helper
Step 9.5 is to update the specific word. This function goes through all the other words starting with that letter and either updates
their weights or adds them to the list.
'''
def update_word(ticker, option, word_upper, day):
	
	# Make the word lowercase and get the length of the word
	word = word_upper.lower()
	len_word = len(word)

	# Find the letter index for the words array
	index = ord(word[:1]) - 97
	if index < 0 or index > 25:
		logging.warning('-- Could not find the following word in the database: ' + word)
		return

	# Get the array containing words of the right letter
	letter_words = words_by_letter[index]
	num_letter_words = num_words_by_letter[index]

	# Search that array for the current word
	found = False
	for ii in range(0, num_letter_words):
		
		# Get the current word data to be compared
		test_data = struct.unpack_from('16s f i i', letter_words, ii * 28)

		# Check if the word is the same
		if test_data[0][:len(test_data[0].split('\0', 1)[0])] == word :
			
			# If it is the same, mark it as found and update its values
			# weight = (weight * value + increase)/(value + increase)
			found = True
			change = stock_prices[ticker][day][1] - stock_prices[ticker][day][0]

			if option == 'opt1':
				if change > 0:
					weight = (test_data[1] * test_data[2] + 1) / (test_data[2] + 1)
					extra1 = test_data[2] + 1
					extra2 = test_data[3]

				else:
					weight = (test_data[1] * test_data[2]) / (test_data[2] + 1)
					extra1 = test_data[2] + 1
					extra2 = test_data[3]

			struct.pack_into('16s f i i', letter_words, ii * 28, word, weight, extra1, extra2)

			logging.debug('-- Updated ' + word + ' using ' + option + ' with weight of ' + str(weight) + ' and occurences of ' + str(extra1) + ', ' + str(extra2))
			break

	if not found:
		# Get whether the stock went up or down
		# weight is automatically 1 or 0 for the first one
		change = stock_prices[ticker][day][1] - stock_prices[ticker][day][0]

		if option == 'opt1':
			if change > 0:
				weight = 1
				extra1 = 1
				extra2 = 0
			else:
				weight = 1
				extra1 = 1
				extra2 = 0

		# Pack the data into the array
		struct.pack_into('16s f i i', letter_words, num_letter_words*28, word, weight, extra1, extra2)
		num_words_by_letter[index] += 1

		logging.debug('-- Added ' + word + ' with weight of ' + str(weight) + ' and occurences of ' + str(extra1) + ', ' + str(extra2))

'''
Evening Update Step
Step 11 is to save all the word weights to a file so it can be loaded in later to be used in subsequent days.
'''
def save_all_word_weights(option):

	logging.info('Saving word weights for weighting option ' + option)
	print 'Saving word weights'

	file = open('./data/word_weight_data_' + option + '.txt', 'w')

	# Iterate through all letters and words in each letter and write them to a file
	for first_letter in range(0, 26):

		# Write the letter to the file
		file.write(chr(first_letter + 97) + '\n')

		logging.debug('- Saving word weights for words starting with: ' + chr(first_letter+97))

		# For each letter, iterate through the words saved for that letter
		for words in range(0, num_words_by_letter[first_letter]):

			# For each word, unpack the word from the buffer
			raw_data = struct.unpack_from('16s f i i', words_by_letter[first_letter], words * 28)

			# Write the data to the file
			file.write('- ' + raw_data[0].split('\0', 1)[0] + ' ' + str(raw_data[1]) + ' ' + str(raw_data[2]) + ' ' + str(raw_data[3]) + '\n')

	file.close()

'''
Morning Prediction Step
Step 3 is to analyze the weights and find the distribution. That is, find the average and standard deviation which will be used to determine the confidence
that a stock will go up or down. Ideally this helps against biasing the training that the algorithm goes through. If stocks go up 7/10 days, then almost all 
the words will be biased to predict an upwards trend. Though this has some elements of validity, an efficient market should not be dependent on past days. 
By finding some statistics about the weights, we can hopefully bias against these predispositions in the training.

weight_average
weight_stdev
weight_sum
weight_max
weight_min
weight_count
'''
def analyze_weights():

	logging.info('Analyzing weights for distribution')
	print 'Analyzing weights'

	global weight_average
	global weight_stdev
	global weight_sum
	global weight_max
	global weight_min
	global weight_count

	# Iterate through each letter
	for letter in range(0, 26):

		logging.debug('- Analyzing word weights for letter: ' + chr(letter+97))

		# Iterate through each word for that letter
		for elements in range(0, num_words_by_letter[letter]):

			# For each word, unpack the word from the buffer
			raw_data = struct.unpack_from('16s f i i', words_by_letter[letter], elements * 28)
			weight = float(raw_data[1])

			# Add it to the list of weights to be analyzed later (for standard deviation)
			weights_all.append(weight)

			# Update sum, max, min, and count
			weight_count += 1
			weight_sum += weight

			if weight > weight_max:
				weight_max = weight

			if weight < weight_min:
				weight_min = weight

	# Once all weights have been iterated through, calculate the average
	weight_average = weight_sum / weight_count

	# Calculate the standard deviation
	running_sum = 0
	for weights in weights_all:
		running_sum += ((weights - weight_average) * (weights - weight_average))
	weight_stdev = math.sqrt(running_sum / (weight_count - 1))

	logging.debug('- Analysis finished with:')
	logging.debug('-- avg: ' + str(weight_average))
	logging.debug('-- std: ' + str(weight_stdev))
	logging.debug('-- sum: ' + str(weight_sum))
	logging.debug('-- cnt: ' + str(weight_count))
	logging.debug('-- max: ' + str(weight_max))
	logging.debug('-- min: ' + str(weight_min))

'''
Morning Prediction Step Helper
Step 4.5 is to get the weighting of a word. This is used for evaluating the stock based on the morning's articles.
'''
def get_word_weight(word_upper):

	# Make the word lowercase and get the length of the word
	word = word_upper.lower()
	len_word = len(word)

	# Find the letter index for the words array
	index = ord(word[:1]) - 97
	if index < 0 or index > 25:
		logging.warning('-- Could not find the following word in the database: ' + word)
		return

	# Get the array containing words of the right letter
	letter_words = words_by_letter[index]
	num_letter_words = num_words_by_letter[index]

	# Search that array for the current word
	found = False
	for ii in range(0, num_letter_words):
		
		# Get the current word data to be compared
		test_data = struct.unpack_from('16s f i i', letter_words, ii * 28)

		# Check if the word is the same
		if test_data[0][:len(test_data[0].split('\0', 1)[0])] == word :
			
			# If it is the same, return its value
			return test_data[1]

	# Could not find the word so returning the current average
	return weight_average


'''
Morning Prediction Step
Step 4 is to run through the words in the mornings articles and come up with a prediction for what they will do later in the day. The data is recorded so it 
can be looked at later to see how the algorithm is improving (if at all).
'''
def predict_movement():

	global weight_average
	global weight_stdev
	global weight_sum
	global weight_max
	global weight_min
	global weight_count

	logging.info('Prediction stock movement')
	print 'PREDICTIONS BASED ON:'
	print '\t- AVG: ', weight_average
	print '\t- STD: ', weight_stdev

	# Open file to store todays predictions in
	today = datetime.datetime.now()
	today_str = str(today.month) + '-' + str(today.day) + '-' + str(today.year)
	file = open('./output/prediction-' + today_str + '.txt', 'w')

	file.write('Predictions Based On Weighting Stats: \n')
	file.write('- Avg: ' + str(weight_average) + '\n')
	file.write('- Std: ' + str(weight_stdev) + '\n')
	file.write('- Sum: ' + str(weight_sum) + '\n')
	file.write('- Cnt: ' + str(weight_count) + '\n')
	file.write('- Max: ' + str(weight_max) + '\n')
	file.write('- Min: ' + str(weight_min) + '\n\n')


	# Iterate through stocks as predictions are seperate for each
	for tickers in STOCK_TAGS:

		logging.debug('- Finding prediction for: ' + tickers)

		stock_rating_sum = 0
		stock_rating_cnt = 0

		# Iterate through each article for the stock
		for articles in stock_data[tickers]:

			# Get the text (ignore link)
			text = articles[1]

			# Variables to keep track of words
			word_in_progress = False
			word_number = 0
			word_start_index = 0

			# Iterate through the characters to find words
			for ii, chars in enumerate(text):

				# If there is a word being found and non-character pops up, word is over
				if word_in_progress and (not chars.isalpha() or ii == len(text)):

					# Reset word variables
					word_in_progress = False
					word_number += 1

					# Get the found word
					found_word = text[word_start_index:ii]

					# Add the word to the word arrays or update its current value
					if len(found_word) > 1:
						
						stock_rating_sum += get_word_weight(found_word)
						stock_rating_cnt += 1


				# If a word is not being found and letter pops up, start the word
				elif not word_in_progress and chars.isalpha():

					# Start the word
					word_in_progress = True
					word_start_index = ii

		# After each word in every article has been examined for that stock, find the average rating
		stock_rating = stock_rating_sum / stock_rating_cnt

		# Calculate the number of standard deviations above the mean and find the probability of that for a 'normal' distribution 
		# - Assuming normal because as the word library increases, it should be able to be modeled as normal
		std_above_avg = (stock_rating - weight_average) / weight_stdev
		probability = norm(weight_average, weight_stdev).cdf(stock_rating)

		if std_above_avg > 0.5:
			rating = 'buy'
		elif std_above_avg < -0.5:
			rating = 'sell'
		else:
			rating = 'undecided'

		print 'RATING FOR: ', tickers
		print '\t- STD ABOVE MEAN: ', std_above_avg
		print '\t- RAW VAL RATING: ', stock_rating
		print '\t- PROBABILITY IS: ', probability
		print '\t- CORRESPONDS TO: ', rating

		file.write('Prediction for: ' + tickers + ' \n')
		file.write('- Std above mean: ' + str(std_above_avg) + '\n')
		file.write('- Raw val rating: ' + str(stock_rating) + '\n')
		file.write('- probability is: ' + str(probability) + '\n')
		file.write('- Corresponds to: ' + str(rating) + '\n\n')

	file.close()

def predict_movement2():

	global weight_average
	global weight_stdev
	global weight_sum
	global weight_max
	global weight_min
	global weight_count

	logging.info('Prediction stock movements with method 2 -> do not use stock weights within a standard deviation of the mean')
	print 'PREDICTIONS BASED ON:'
	print '\t- AVG: ', weight_average
	print '\t- STD: ', weight_stdev

	# Open file to store todays predictions in
	today = datetime.datetime.now()
	today_str = str(today.month) + '-' + str(today.day) + '-' + str(today.year)
	file = open('./output/prediction2-' + today_str + '.txt', 'w')

	file.write('Predictions Based On Weighting Stats: \n')
	file.write('- Avg: ' + str(weight_average) + '\n')
	file.write('- Std: ' + str(weight_stdev) + '\n')
	file.write('- Sum: ' + str(weight_sum) + '\n')
	file.write('- Cnt: ' + str(weight_count) + '\n')
	file.write('- Max: ' + str(weight_max) + '\n')
	file.write('- Min: ' + str(weight_min) + '\n\n')


	# Iterate through stocks as predictions are seperate for each
	for tickers in STOCK_TAGS:

		logging.debug('- Finding prediction for: ' + tickers)

		stock_rating_sum = 0
		stock_rating_cnt = 0

		# Iterate through each article for the stock
		for articles in stock_data[tickers]:

			# Get the text (ignore link)
			text = articles[1]

			# Variables to keep track of words
			word_in_progress = False
			word_number = 0
			word_start_index = 0

			# Iterate through the characters to find words
			for ii, chars in enumerate(text):

				# If there is a word being found and non-character pops up, word is over
				if word_in_progress and (not chars.isalpha() or ii == len(text)):

					# Reset word variables
					word_in_progress = False
					word_number += 1

					# Get the found word
					found_word = text[word_start_index:ii]

					# Add the word to the word arrays or update its current value
					if len(found_word) > 1:
						
						weight = get_word_weight(found_word)
						if weight > weight_stdev + weight_average or weight < weight_average - weight_stdev:
							stock_rating_sum += weight
							stock_rating_cnt += 1


				# If a word is not being found and letter pops up, start the word
				elif not word_in_progress and chars.isalpha():

					# Start the word
					word_in_progress = True
					word_start_index = ii

		# After each word in every article has been examined for that stock, find the average rating
		stock_rating = stock_rating_sum / stock_rating_cnt

		# Calculate the number of standard deviations above the mean and find the probability of that for a 'normal' distribution 
		# - Assuming normal because as the word library increases, it should be able to be modeled as normal
		std_above_avg = (stock_rating - weight_average) / weight_stdev
		probability = norm(weight_average, weight_stdev).cdf(stock_rating)

		if std_above_avg > 0.5:
			rating = 'buy'
		elif std_above_avg < -0.5:
			rating = 'sell'
		else:
			rating = 'undecided'

		print 'RATING FOR: ', tickers
		print '\t- STD ABOVE MEAN: ', std_above_avg
		print '\t- RAW VAL RATING: ', stock_rating
		print '\t- PROBABILITY IS: ', probability
		print '\t- CORRESPONDS TO: ', rating

		file.write('Prediction for: ' + tickers + ' \n')
		file.write('- Std above mean: ' + str(std_above_avg) + '\n')
		file.write('- Raw val rating: ' + str(stock_rating) + '\n')
		file.write('- probability is: ' + str(probability) + '\n')
		file.write('- Corresponds to: ' + str(rating) + '\n\n')

	file.close()

'''
Display help and exit
'''
def print_help():

	logging.debug('Displaying help and exiting')

	print 'To pull articles for the current day, process them, and return a rating, use:'
	print '\t ./stock_market_prediction.py -p\n'
	print 'To update word weights and stock prices for the current day, use:'
	print '\t ./stock_market_prediction.py -u\n'
	print 'To specify a day to update word weights for, use (stock prices must have already been updated):'
	print '\t ./stock_market_prediction.py -u -d 10-25-2017\n'
	print 'To specify a date range to update word weights for, use (stock prices must have already been updated):'
	print '\t ./stock_market_prediction.py -u -s 10-22-2017 -e 10-26-2017\n'

	sys.exit(2)

'''
Verifies that a date is valid and in the right format
'''
def verify_date(date):

	today = datetime.datetime.now()
	date_parts = date.split('-')

	if len(date_parts) < 3:
		return False

	try:
		test_date = datetime.datetime(int(date_parts[2]), int(date_parts[0]), int(date_parts[1]))
	except:
		return False

	if test_date > today:
		return False

	return True

'''
Main Execution
- Part 1: Parsing Inputs and Pulling, Storing, and Loading Articles
'''
def main():

	update = False
	day = 0  # 0 for today, -1 for range, 1 for a specific day
	specified_day = ''
	start_day = ''
	end_day = ''

	# First get any command line arguments to edit actions
	try:
		opts, args = getopt.getopt(sys.argv[1:], 'hpud:s:e:')
	except getopt.GetoptError:
		print_help()
	for opt, arg in opts:
		# Help message
		if opt == '-h':
			print_help()
		# Command line args request an update
		elif opt == '-u':
			update = True
		# Command line args request a prediction (can not have prediction and update together)
		elif opt == '-p' and not update:
			update = False
		# Command line args request a specific day
		elif opt == '-d' and day == 0:
			day = 1
			specified_day = arg
		# Command line args request a date range, this is the start
		elif opt == '-s' and day == 0:
			day = -1
			start_day = arg
		# Command line args specify an end to the date range
		elif opt == '-e' and day == -1:
			end_day = arg


	# Start of actual execution, begin by loading all the data
	logging.info('Loading all data')
	load_all_word_weights('opt1')

	# If pulling new stock data, do so
	if not update:
		logging.info('Pulling new articles and creating a prediction')

		today = datetime.datetime.now()
		load_articles(str(today.month) + '-' + str(today.day) + '-' + str(today.year))

		#pull_recent_articles()
		analyze_weights()
		predict_movement()
		predict_movement2()

	# If updating
	else:
		# Load all stock prices
		load_stock_prices()

		# Then for each of the days, update the word weights
		days = []

		# Today's date, get the day, and add it
		today = datetime.datetime.now()
		if day == 0:

			# First update the stock prices
			pull_stock_prices()
			
			days.append(str(today.month) + '-' + str(today.day) + '-' + str(today.year))

		# A specified date
		elif day == 1 and verify_date(specified_day):
			days.append(specified_day)

		# A range of dates
		elif day == -1:
			print 'A range... to be implemented.'

		for each in days:

			logging.info('Updating word weights for: ' + each)

			load_articles(each)
			update_all_word_weights('opt1', each)
			stock_data.clear()

		save_stock_prices()

	# After execution, save data
	logging.info('Saving all data')
	save_all_word_weights('opt1')

	logging.info('Finished running, closing')
	print 'Done.'



'''
Analysis Function Failures: Get the frequency of all words and generate two histograms. The first is a histogram of word occurances ignoring 
the most common words (500+ occurences). The second is a histogram in order from greatest to least with all words.
'''
def generate_histograms(directory):

	if not os.path.exists(directory):
		os.makedirs(directory)
		logging.debug('Created histogram directory: ' + directory)

	logging.info('Generating word frequencies and creating histograms')

	words = {}

	# Iterate through all the stocks
	for ticker in STOCK_TAGS:

		# For each stock iterate through the articles
		for articles in stock_data[ticker]:

			# For each article, we first set the splice beginning to 0
			start = 0
			word_in_progress = False

			# For the article we iterate through the characters 
			for ii, chars in enumerate(articles[1]):

				# If there is no word in progress and a letter is found, start a word
				if chars.isalpha() and not word_in_progress:
					word_in_progress = True
					start = ii
				# If there is a word in progress and a letter is not found, end the word
				elif word_in_progress and not chars.isalpha():
					new_word = articles[1][start:ii]
					word_in_progress = False
					
					# If it is longer than one character, add the word to the word array or update the count
					if len(new_word) > 1:

						if new_word in words:
							words[new_word] += 1
						else:
							words[new_word] = 1

	# Print the word frequency to a file as a histogram
	file = open(directory + 'word_frequencies.txt', 'w')

	words_total = 0
	words_unique = 0

	for key, value in words.iteritems():

		if value > 500:
			print 'Word: ', key, ' has value: ', value

		else:
			# Write the key so it can be seen easily and all are in line
			file.write(key)
			if len(key) < 8:
				file.write('\t\t\t|')
			elif len(key) < 16:
				file.write('\t\t|')
			else:
				file.write('\t|')

			# For each time it is seen, write one dot
			for ii in range(0, value):
				file.write('.')
				words_total += 1

			# At the end of the word, write a newline
			file.write('\n')
			words_unique += 1

	file.close()

	print 'Avg occurance: ', words_total/words_unique
	print 'Total words: ', words_unique


	# Print the word frequency to a file as a histogram
	file = open(directory + 'word_frequencies_ordered.txt', 'w')

	max_value = 0
	max_key = ''

	for items in words.iteritems():
		for key, value in words.iteritems():
			if value > max_value:
				max_value = value
				max_key = key

		file.write(max_key)
		if len(max_key) < 8:
			file.write('\t\t\t|')
		elif len(key) < 16:
			file.write('\t\t|')
		else:
			file.write('\t|')

		# For each time it is seen, write one dot
		for ii in range(0, max_value):
			file.write('.')
		file.write('\n')

		words[max_key] = -1
		max_value = 0
		max_key = ''

	file.close()

	logging.info('Histograms have been generated')

def find_sentences(article, sentences):

	sentence_started = False
	index = 0
	num = 0
	for ii, chars in enumerate(article):

		if chars.isalpha() and not sentence_started:
			sentence_started = True
			sentences[index] = ii
			index += 1

		if (chars == '.' or chars == '\n') and sentence_started:
			sentence_started = False
			sentences[index] = ii
			index += 1
			num += 1

	return num

def compare_sentences(article, sentences, num):

	compare_sentence = article[sentences[0]:sentences[1]]
	word_indices = []
	num_compare = 0
	index = 0
	word_in_progress = False
	for ii, chars in enumerate(compare_sentence):

		if not word_in_progress and chars.isalpha():
			word_in_progress = True
			word_indices.append(ii)

		if not chars.isalpha() and word_in_progress:
			word_in_progress = False
			word_indices.append(ii)
			num_compare += 1

		if ii == len(compare_sentence) and word_in_progress:
			word_indices.append(len(compare_sentence))
			num_compare += 1
	print word_indices
	print compare_sentence

	ss = 2
	for ii in range(0, num):
		similar = 0
		total = 0

		current = article[sentences[ss]:sentences[ss+1]]
		ss += 2
		#print current

		new_indices = []
		new_index = 0
		num_test = 0
		new_word_in_progress = False
		for jj, chars in enumerate(current):

			if not new_word_in_progress and chars.isalpha():
				new_word_in_progress = True
				new_indices.append(jj)

			if not chars.isalpha() and new_word_in_progress:
				new_word_in_progress = False
				new_indices.append(jj)
				num_test += 1

			if jj == len(current) and new_word_in_progress:
				new_indices.append(len(current))
				num_test += 1

		ww = 0
		ww2 = 0
		#print new_indices
		for each in range(0, num_test):
			word = current[new_indices[ww]:new_indices[ww+1]]
			#print '----- ', new_indices[ww], ', ', new_indices[ww+1], ' - ', word
			for comp in range(0, num_compare):
				#print '-------- ', comp, ', ', word_indices[ww2], ', ', word_indices[ww2+1], ' - ', compare_sentence[word_indices[ww2]:word_indices[ww2+1]]
				if word == compare_sentence[word_indices[ww2]:word_indices[ww2 + 1]]:
					similar += 1
					#print word, ' is in ', compare_sentence
					break
				ww2 += 2
			ww2 = 0
			ww+=2
			total += 1

		print 'Sentence ', ii, ' has ', similar, '/', total








main()