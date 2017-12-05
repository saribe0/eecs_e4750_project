#!/usr/bin/env python3
####################################################################################################
## Created 10/27/2017 by Sam Beaulieu
##
##
####################################################################################################

import pyopencl as cl

import requests
import os
import datetime
import time
import logging
import sys, getopt
import numpy as np
#from bs4 import BeautifulSoup as BS
import struct
import binascii
import math
from scipy.stats import norm
import re

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
				'wmt',
				'jpm',
				'mu',
				'uaa',
				'gild',
				'xom'  ]

ARTICLES_PER_STOCK = 10
SUCCESS_THREASHOLD = 5
MAX_WORDS_PER_LETTER = 2500

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

weights_all_o = []
weight_average_o = 0
weight_stdev_o = 0
weight_sum_o = 0
weight_count_o = 0

# Global Variables for Bayesion Prediction
total_up = 0
total_words_up = 0
total_down = 0
total_words_down = 0
c = 0.01

# Global Configurations
if not os.path.exists('./data/'):
	os.makedirs('./data/')
	os.makedirs('./data/articles/')
elif not os.path.exists('./data/articles/'):
	os.makedirs('./data/articles/')

if not os.path.exists('./output/'):
	os.makedirs('./output/')


# Get the correct opencl platform (taken from instructor sample code)
NAME = 'NVIDIA CUDA'
platforms = cl.get_platforms()
devs = None
for platform in platforms:
	if platform.name == NAME:
		devs = platform.get_devices()

# Set up command queue
ctx = cl.Context(devs)
queue = cl.CommandQueue(ctx)

'''
# Create array that will be used for output, structure looks like:
	# [sum of weights, weighted sum of weights, max, min, count, weighted count]
	# Length is 6
	out_stats = np.zeros((6,), dtype = np.uint32)

	# First part is to calculate these 6 outputs 

	#Prepare GPU buffers
	mf = cl.mem_flags
	words_by_letter_buff = cl.Buffer(ctx, mf.READ_ONLY, | mf.COPY_HOST_PTR, hostbuf = words_by_letter.flatten())
	num_words_by_letter_buff = cl.Buffer(ctx, mf.READ_ONLY, | mf.COPY_HOST_PTR, hostbuf = num_words_by_letter)
	out_stats_buff = cl.Buffer(ctx, mf.WRITE_ONLY, out_stats.nbytes)

	# Call the kernel
	prg.analyze_weights_1(queue, words_by_letter.shape, (512, 1), words_by_letter_buff, num_words_by_letter_buff, out_stats_buff)

'''

analysis_kernel = """

__kernel void analyze_weights_1(__global float* words_by_letter, __global int* num_words_by_letter, volatile __global float* out_stats, int max_words_per_letter) {
	
	// Get the word for the current work-item to focus on

	unsigned int word_id = get_global_id(0);
	unsigned int letter_id = get_global_id(1);
unsigned int groupx = get_group_id(0);
unsigned int groupy = get_group_id(1);
	// Create local arrays to store the data in

	volatile __local float local_out[6 * 512];

	// Prepare the indices for the reduction

	unsigned int work_item_id = get_local_id(0);

	// Get the weight and frequency for the current thread
	float weight = 0;	
	int frequency = 0;
	if (word_id < num_words_by_letter[letter_id]) {
		weight = words_by_letter[letter_id * max_words_per_letter * 7 + word_id * 7 + 4];
		frequency = as_int(words_by_letter[letter_id * max_words_per_letter * 7 + word_id * 7 + 5]);
	}

	if (word_id < 8 && letter_id == 1) { printf("[ word: %d, has weight %f, freq %d ]", word_id, weight, frequency); }
//	if (word_id == 0) { printf("[ letter %d, num %d ]", letter_id, num_words_by_letter[letter_id]); }
	// Each thread loads initial data into its own space in local memory

	local_out[512 * 0 + work_item_id] =  weight;
	local_out[512 * 1 + work_item_id] =  frequency * weight;
	local_out[512 * 2 + work_item_id] =  weight;
	local_out[512 * 3 + work_item_id] =  (word_id < num_words_by_letter[letter_id]) ? weight : 1;
	local_out[512 * 4 + work_item_id] =  (word_id < num_words_by_letter[letter_id]) ? 1 : 0;
	local_out[512 * 5 + work_item_id] =  frequency;

	for (unsigned int stride = 1; stride < 512; stride *= 2) {

		barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

		if ( work_item_id % stride == 0 && work_item_id < 256) {

			local_out[512 * 0 + work_item_id * 2] +=  local_out[512 * 0 + work_item_id * 2 + stride];
			local_out[512 * 1 + work_item_id * 2] +=  local_out[512 * 1 + work_item_id * 2 + stride];
			local_out[512 * 2 + work_item_id * 2] =  local_out[512 * 2 + work_item_id * 2 + stride] > local_out[512 * 2 + work_item_id * 2] ? local_out[512 * 2 + work_item_id * 2 + stride] : local_out[512 * 2 + work_item_id * 2];
			local_out[512 * 3 + work_item_id * 2] =  local_out[512 * 3 + work_item_id * 2 + stride] < local_out[512 * 3 + work_item_id * 2] ? local_out[512 * 3 + work_item_id * 2 + stride] : local_out[512 * 3 + work_item_id * 2];
			local_out[512 * 4 + work_item_id * 2] +=  local_out[512 * 4 + work_item_id * 2 + stride];
			local_out[512 * 5 + work_item_id * 2] +=  local_out[512 * 5 + work_item_id * 2 + stride];
		

			if (word_id < 8 && letter_id == 1) { printf("[ stride: %d, item: %d, value: %f ]", stride, word_id, local_out[512 * 0 + work_item_id * 2]); }
		}
	}
//	if (word_id == 0 && letter_id == 0) { printf("Final group sum: %f", local_out[0] );}
	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
//if (word_id == 0 && letter_id == 0) { printf("Final group sum: %f", local_out[0] );}
	if (work_item_id < 6) {
//		if( letter_id < 1) { printf("ID: %d -> %f", work_item_id, local_out[512 * work_item_id]);}
		out_stats[ (groupy * 5 + groupx)*6 + work_item_id] = local_out[512 * work_item_id];
//		if (letter_id < 1) { printf("[ work item %d, %u, %f ]", word_id, (groupy*5+groupx)*6, local_out[512 * work_item_id]); }
	}
}

__kernel void analyze_weights_2(__global float* words_by_letter, __global int* num_words_by_letter, volatile __global float* out_stats, int max_words_per_letter, int average, int weighted_average) {
	
	// Get the word for the current work-item to focus on

	unsigned int word_id = get_global_id(0);
	unsigned int letter_id = get_global_id(1);

	// Create local arrays to store the data in

	volatile __local float local_out[2 * 512];

	// Prepare the indices for the reduction

	unsigned int work_item_id = get_local_id(0);

	// Get the weight and frequency for the current thread	
	float weight = 0;	
	int frequency = 0;
	if (word_id < max_words_per_letter) {
		weight = words_by_letter[letter_id * max_words_per_letter + word_id * 7 + 4];
		frequency = as_int(words_by_letter[letter_id * max_words_per_letter + word_id * 7 + 5]);
	}

	// Each thread loads initial data into its own space in local memory

	local_out[512 * 0 + work_item_id] =  (weight - average) * (weight - average);
	local_out[512 * 1 + work_item_id] =  (weight - weighted_average) * (weight - weighted_average) * frequency;

	for (unsigned int stride = 1; stride <= 512; stride *= 2) {

		barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

		if ( work_item_id % stride == 0 && work_item_id < 256 && word_id < max_words_per_letter) {

			local_out[512 * 0 + work_item_id * 2] +=  local_out[512 * 0 + work_item_id * 2 + stride];
			local_out[512 * 1 + work_item_id * 2] +=  local_out[512 * 1 + work_item_id * 2 + stride];
		}
	}

	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

	if (work_item_id < 2) {

		out_stats[ (get_group_id(1) * get_num_groups(1) + get_group_id(0)) + work_item_id] = local_out[512 * work_item_id];
	}
}

"""

# Build the kernel
prg = cl.Program(ctx, analysis_kernel).build()


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

	print('Pulling articles')

	# Iterate through the stock tags
	for ticker in STOCK_TAGS:
		start = time.time()

		logging.debug('Starting to pull articles for: ' + ticker)

		file = open(directory + ticker + '.txt', 'w')

		# Make the request to google news for the ticker and get its raw html
		try:
			r = requests.get('https://www.google.com/search?biw=1366&bih=671&tbm=nws&ei=HRDyWcDlJoa-jwSuwZCYBg&q=' + ticker, timeout=2)
		except requests.exceptions.RequestException as error:
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
					r = requests.get(link, timeout=2)
				except requests.exceptions.RequestException as error:
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
					raw_text = raw_html.find('div', attrs={'id' : 'page-content'} )

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
					logging.warning('-- Could not find article text in: ' + link + ', trying next link')
					continue

				# Combine all the paragraphs into a single text block
				article = ''
				for paragraphs in article_parts:
					article += paragraphs.text
					article += ' '

				# Store the url and article as a tuple into the stock data and write them to the stock's file
				stock_data[ticker].append((link, article))
				file.write(link + '\n\n')
				file.write(article + '\n')
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
	if not os.path.exists(directory):
		logging.error('Could not find articles to load for: ' + day)
		return False

	logging.info('Loading articles from directory: ' + directory)
	print('Loading articles')

	# Iterate through the stock tags
	for ticker in STOCK_TAGS:
		start = time.time()

		# Try to open the file for that stock from the given directory
		logging.debug('Starting to load articles for: ' + ticker)
		try:
			file = open(directory + ticker + '.txt', 'r+')
		except IOError as error:
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

	global total_up
	global total_down

	logging.info('Loading past stock prices')
	print('Loading stock prices')

	try:
		file = open('./data/stock_price_data.txt', 'r+')
	except IOError as error:
		logging.warning('- Could not load stock prices, Error: ' + str(error))
		return

	# Iterate through the lines in the file
	current_stock = ''
	first = 0
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

			# Update up or down occurences
			if float(data[2]) > float(data[3]):
				total_down += 1
			elif float(data[2]) < float(data[3]):
				total_up += 1

	file.close()

'''
Evening Update Step
Step 8 is to pull all stock data for the current day and add it to the stock prices data structure. The yahoo finance api is used
for this and open and close prices are all thats kept.
'''
def pull_stock_prices():

	global total_up
	global total_down

	today = datetime.datetime.now()
	today_str = str(today.month) + '-' + str(today.day) + '-' + str(today.year)

	logging.info('Pulling stock prices for: ' + today_str)
	print('Pulling stock prices')

	# For each stock, get todays value
	for tickers in STOCK_TAGS:

		logging.debug('- Pulling price data for: ' + tickers)

		# If the stock is not in the stock prices, add it
		if tickers not in stock_prices:	
			stock_price[tickers] = {}

		# Get todays stock open and close
		open_p, close_p = get_price(tickers)

		# Update up and down occurences
		if open_p > close_p:
			total_down += 1
		elif open_p < close_p:
			total_up += 1

		logging.debug('-- Price for: ' + tickers + ' is open: ' + str(open_p) + ', close: ' + str(close_p))

		# Add todays prices to the dictionary for today
		stock_prices[tickers][today_str] = (open_p, close_p)

def get_price(ticker):

	# Make the request to google finance for the ticker and get its raw html
	try:
		r = requests.get('https://finance.google.com/finance?q=' + ticker)
	except requests.exceptions.RequestException as error:
		logging.warning('- Could not pull price for: ' + ticker + ', Error: ' + str(error))
		return -1, 1

	raw_html = BS(r.text, 'html.parser')

	open_html = raw_html.find('table', attrs={'class' : 'snap-data'})
	open_raw = open_html.find_all('tr')
	open_raw2 = open_raw[2].find('td', attrs={'class' : 'val'})
	try:
		open_p = float(open_raw2.text[:-1].replace(',',''))
	except:
		logging.warning('- Could not pull price for: ' + ticker)
		return -1, 2

	close_raw = raw_html.find('span', attrs={'id' : re.compile(r'ref_*')})
	try:
		close_p = float(close_raw.text.replace(',',''))
	except:
		logging.warning('- Could not pull price for: ' + ticker)
		return -1, 3

	return open_p, close_p

'''
Evening Update Step
Step 10 is to save the stock price data structure to a file so it can be loaded on subsequent days. 
'''
def save_stock_prices():

	logging.info('Saving stock price data structure')
	print('Saving stock prices')

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
		for days, price in stock_prices[ticker].items():
			file.write('- ' + days + ' ' + str(price[0]) + ' ' + str(price[1]) + '\n')

	file.close()

'''
Morning Prediction Step & Evening Update Step
Step 2 & 7 is to load all the word weights into the right data structures so it can be used in analyzing the new data. The strange array structure 
is to enable implementation in pyCUDA in the near future. Otherwise, a hash function would have ben used for constant time access. The words are
stored by letter which will hopefully make it more efficient.
'''
def load_all_word_weights(option):
	
	global total_up
	global total_down
	global total_words_up
	global total_words_down

	logging.info('Loading word weights for weighting option: ' + option)
	print('Loading word weights')

	# For each letter, add a MAX_WORDS_PER_LETTER word array of 28 characters each with the last 12 characters for a float (weight of the word) and int (number of occurences)
	# - abcdefghijklmnopq0.32########
	for letters in range(0, 26):
		letter_words = bytearray(28*MAX_WORDS_PER_LETTER)
		words_by_letter.append(letter_words)
		num_words_by_letter.append(0)


	# Open the file to be loaded
	try:
		file = open('./data/word_weight_data_' + option + '.txt', 'r+')
	except IOError as error:
		logging.warning('- Could not load word weights, Error: ' + str(error))
		return

	# Iterate over all the lines in the file
	letter_index = 0
	first = 0
	for lines in file:

		# If its the first line and option 2, load the total words up or down
		if first == 0 and option == 'opt2':
			temp = lines.split()
			total_up = int(temp[0])
			total_down = int(temp[1])
			first = 1
		elif first == 1 and option == 'opt2':
			temp = lines.split()
			total_words_up = int(temp[0])
			total_words_down = int(temp[1])
			first = 2

		# If the first character is not a '-', it indicates a letter change
		if lines[:1] != '-':
			letter_index = ord(lines[:1]) - 97
			logging.debug('Loading words beginning with: ' + lines[:1])

		# If not a '-', pack up the data, store it, and update the number of words
		else:
			# Get the data
			data = lines.split()

			# Store the data
			struct.pack_into('16s f i i', words_by_letter[letter_index], num_words_by_letter[letter_index]*28, data[1].decode('ascii', 'ignore').encode('utf-8'), float(data[2]), int(data[3]), int(data[4]))
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
	print('Updating word weights')

	# If the weighting arrays are empty, create them 
	if len(words_by_letter) == 0 or words_by_letter == 0:

		logging.debug('- Could not find data structure for word weights so creating it')

		# For each letter, add a MAX_WORDS_PER_LETTER word array of 28 characters each with the last 8 characters for a float (weight of the word) and int (number of occurences)
		# - abcdefghijklmnopq0.32####
		for letters in range(0, 26):
			letter_words = bytearray(28*MAX_WORDS_PER_LETTER)
			words_by_letter.append(letter_words)
			num_words_by_letter.append(0)

	# At this point, the weighting arrays are initialized or loaded
	# - Next step is to iterate through articles and get the words
	for ticker in STOCK_TAGS:
		
		logging.debug('- Updating word weights for: ' + ticker)

		if not ticker in stock_data:
			logging.warning('- Could not find articles loaded for ' + ticker)
			continue

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
	
	global total_words_up
	global total_words_down

	# Make the word lowercase and get the length of the word
	word = word_upper.lower()
	len_word = len(word)
	if len_word > 16:
		len_word = 16

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
		temp_word = test_data[0].decode('utf_8')

		# Check if the word is the same
		if temp_word[:len(temp_word.split('\0', 1)[0])] == word :
			
			# If it is the same, mark it as found and update its values
			# weight = (weight * value + increase)/(value + increase)
			found = True

			if not ticker in stock_prices:
				logging.error('-- Could not find stock ' + ticker + ' in stock price data')
				return
			if not day in stock_prices[ticker]:
				logging.error('-- Could not find stock price data for ' + ticker + ' on ' + day)
				return 

			change = stock_prices[ticker][day][1] - stock_prices[ticker][day][0]

			# Option 1: 1 for up, 0 for down, average the ups and downs for weights
			# weight = num_up / total
			# extra1 = total
			# extra2 => unused
			if option == 'opt1':
				if change > 0:
					weight = (test_data[1] * test_data[2] + 1) / (test_data[2] + 1)
					extra1 = test_data[2] + 1
					extra2 = test_data[3]

				else:
					weight = (test_data[1] * test_data[2]) / (test_data[2] + 1)
					extra1 = test_data[2] + 1
					extra2 = test_data[3]

			# Option 2: Bayesian classifier, probability of word given a label
			# weight => Unused, calculated seperately
			# extra1 = num_up
			# extra2 = num_down
			elif option == 'opt2':
				if change > 0:
					weight = test_data[1]
					extra1 = test_data[2] + 1
					extra2 = test_data[3]
					total_words_up += 1
				else:
					weight = test_data[1]
					extra1 = test_data[2]
					extra2 = test_data[3] + 1
					total_words_down += 1

			struct.pack_into('16s f i i', letter_words, ii * 28, word.encode('utf-8'), weight, extra1, extra2)

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
		
		elif option == 'opt2':
			if change > 0:
				weight = 0
				extra1 = 1
				extra2 = 0
				total_words_up += 1
			else:
				weight = 0
				extra1 = 0
				extra2 = 1
				total_words_down += 1

		# Pack the data into the array
		struct.pack_into('16s f i i', letter_words, num_letter_words*28, word.encode('utf-8'), weight, extra1, extra2)
		num_words_by_letter[index] += 1

		logging.debug('-- Added ' + word + ' with weight of ' + str(weight) + ' and occurences of ' + str(extra1) + ', ' + str(extra2))

'''
Evening Update Step
Step 11 is to save all the word weights to a file so it can be loaded in later to be used in subsequent days.
'''
def save_all_word_weights(option):

	logging.info('Saving word weights for weighting option ' + option)
	print('Saving word weights')

	file = open('./data/word_weight_data_' + option + '.txt', 'w')

	# First write the global data to the file
	if option == 'opt2':
		file.write(str(total_up) + ' ' + str(total_down) + '\n')
		file.write(str(total_words_up) + ' ' + str(total_words_down) + '\n')

	# Iterate through all letters and words in each letter and write them to a file
	for first_letter in range(0, 26):

		# Write the letter to the file
		file.write(chr(first_letter + 97) + '\n')

		logging.debug('- Saving word weights for words starting with: ' + chr(first_letter+97))

		# For each letter, iterate through the words saved for that letter
		for words in range(0, num_words_by_letter[first_letter]):

			# For each word, unpack the word from the buffer
			raw_data = struct.unpack_from('16s f i i', words_by_letter[first_letter], words * 28)
			temp_word = raw_data[0].decode('utf-8')

			# Write the data to the file
			file.write('- ' + temp_word.split('\0', 1)[0] + ' ' + str(raw_data[1]) + ' ' + str(raw_data[2]) + ' ' + str(raw_data[3]) + '\n')

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

_o -> Based on occurences, not frequencies
'''
def analyze_weights():

	logging.info('Analyzing weights for distribution')
	print('Analyzing weights')

	global weight_average
	global weight_stdev
	global weight_sum
	global weight_max
	global weight_min
	global weight_count

	global weight_average_o
	global weight_stdev_o
	global weight_sum_o
	global weight_count_o

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
			for n in range(0, raw_data[2]):
				weights_all_o.append(weight)

			# Update sum, max, min, and count
			weight_count += 1
			weight_sum += weight

			weight_count_o += raw_data[2]
			weight_sum_o += weight * raw_data[2]

			if weight > weight_max:
				weight_max = weight

			if weight < weight_min:
				weight_min = weight

	# If no words have been found with weights return false
	if weight_count == 0:
		logging.error('Could not find any words with weights to analyze')
		return False

	# Once all weights have been iterated through, calculate the average
	weight_average = weight_sum / weight_count
	weight_average_o = weight_sum_o / weight_count_o

	# Calculate the standard deviation
	running_sum = 0
	for weights in weights_all:
		running_sum += ((weights - weight_average) * (weights - weight_average))
	weight_stdev = math.sqrt(running_sum / (weight_count - 1))

	running_sum_o = 0
	for weights in weights_all_o:
		running_sum_o += ((weights - weight_average_o) * (weights - weight_average_o))
	weight_stdev_o = math.sqrt(running_sum_o / (weight_count_o - 1))

	logging.debug('- Analysis finished with:')
	logging.debug('-- avg: ' + str(weight_average))
	logging.debug('-- std: ' + str(weight_stdev))
	logging.debug('-- sum: ' + str(weight_sum))
	logging.debug('-- cnt: ' + str(weight_count))
	logging.debug('-- max: ' + str(weight_max))
	logging.debug('-- min: ' + str(weight_min))
	logging.debug('-- avg_o: ' + str(weight_average_o))
	logging.debug('-- std_o: ' + str(weight_stdev_o))
	logging.debug('-- sum_o: ' + str(weight_sum_o))
	logging.debug('-- cnt_o: ' + str(weight_count_o))

	return True


def analyze_weights_gpu():

	logging.info('Analyzing weights for distribution using the gpu')
	print('Analyzing weights with gpu')

	global weight_average
	global weight_stdev
	global weight_sum
	global weight_max
	global weight_min
	global weight_count

	global weight_average_o
	global weight_stdev_o
	global weight_sum_o
	global weight_count_o

	# Create array that will be used for output, structure looks like:
	# [sum of weights, weighted sum of weights, max, min, count, weighted count]
	# Length is 6
	out_stats = np.zeros((130, 6), dtype = np.float32)

	# First part is to calculate these 6 outputs 

	#Prepare GPU buffers
	mf = cl.mem_flags
	words_by_letter_buff = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = np.asarray(words_by_letter))
	num_words_by_letter_buff = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = np.asarray(num_words_by_letter, dtype = np.uint32))
	out_stats_buff = cl.Buffer(ctx, mf.WRITE_ONLY, out_stats.nbytes)

	print(num_words_by_letter)
	print(np.asarray(num_words_by_letter))
	# Call the kernel
	prg.analyze_weights_1(queue, (2560, 26), (512, 1), words_by_letter_buff, num_words_by_letter_buff, out_stats_buff, np.uint32(MAX_WORDS_PER_LETTER))


	# Pull results from the GPU
	cl.enqueue_copy(queue, out_stats, out_stats_buff)

	# Set all the global variabels for the function
	for each in out_stats:
		print(each[0])
		weight_sum += each[0]
		weight_sum_o += each[1]
		weight_max = each[2] if each[2] > weight_max else weight_max
		weight_min = each[3] if each[3] < weight_min else weight_min
		weight_count += each[4]
		weight_count_o += each[5]

	# Calculate the averages
	weight_average = weight_sum / weight_count
	weight_average_o = weight_sum_o / weight_count_o

	# Prepare the GPU buffers for the standard deviation calculation
	# [sum of avg-weight, weighted sum of avg-weight]
	out_std_sum = np.zeros((128,2), dtype = np.float32)
	out_std_sum_buff = cl.Buffer(ctx, mf.WRITE_ONLY, out_std_sum.nbytes)

	# Call the kernel
	prg.analyze_weights_2(queue, (2560, 26), (512, 1), words_by_letter_buff, num_words_by_letter_buff, out_std_sum_buff, np.uint32(MAX_WORDS_PER_LETTER), np.uint32(weight_average), np.uint32(weight_average_o))

	# Pull resutls from the GPU
	cl.enqueue_copy(queue, out_std_sum, out_std_sum_buff)

	# Set the std deviation global variables
	out_std = 0
	out_std_w = 0
	for each in out_std_sum:
		out_std += each[0]
		out_std_w += each[1]

	weight_stdev = math.sqrt(out_std / (weight_count - 1))
	weight_stdev_o = math.sqrt(out_std_w / (weight_count_o - 1))

	logging.debug('- Analysis finished with:')
	logging.debug('-- avg: ' + str(weight_average))
	logging.debug('-- std: ' + str(weight_stdev))
	logging.debug('-- sum: ' + str(weight_sum))
	logging.debug('-- cnt: ' + str(weight_count))
	logging.debug('-- max: ' + str(weight_max))
	logging.debug('-- min: ' + str(weight_min))
	logging.debug('-- avg_o: ' + str(weight_average_o))
	logging.debug('-- std_o: ' + str(weight_stdev_o))
	logging.debug('-- sum_o: ' + str(weight_sum_o))
	logging.debug('-- cnt_o: ' + str(weight_count_o))

	return True


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
		temp_word = test_data[0].decode('utf_8')

		# Check if the word is the same
		if temp_word[:len(temp_word.split('\0', 1)[0])] == word:
			
			# If it is the same, return its value
			return test_data[1]

	# Could not find the word so returning the current average
	return weight_average

'''
Morning Prediction Step Helper: Naive Bayes
Step 4.6 is to find the probability of a word given the stock going up and down. Uses Laplacian smoothing using value c.
'''
def get_word_probability_given_label(word_upper, c):

	global total_words_up
	global total_words_down

	# Make the word lowercase and get the length of the word
	word = word_upper.lower()
	len_word = len(word)

	# Calculate the total number of words
	num_words = 0
	for ii in range(0, 26):
		num_words += num_words_by_letter[ii]

	# Find the letter index for the words array
	index = ord(word[:1]) - 97
	if index < 0 or index > 25:
		logging.warning('-- Could not find the following word in the database: ' + word)
		return None, None

	# Get the array containing words of the right letter
	letter_words = words_by_letter[index]
	num_letter_words = num_words_by_letter[index]

	# Search that array for the current word
	found = False
	for ii in range(0, num_letter_words):
		
		# Get the current word data to be compared
		test_data = struct.unpack_from('16s f i i', letter_words, ii * 28)
		temp_word = test_data[0].decode('utf_8')

		# Check if the word is the same
		if temp_word[:len(temp_word.split('\0', 1)[0])] == word:
			
			# If it is the same, calculate and return the probabilities
			up = (float(test_data[2]) + c) / (total_words_up + c * num_words)
			down = (float(test_data[3]) + c) / (total_words_down + c * num_words)
			return up, down

	# Could not find the word so returning None
	return None, None

'''
Morning Prediction Step
Step 4 is to run through the words in the mornings articles and come up with a prediction for what they will do later in the day. The data is recorded so it 
can be looked at later to see how the algorithm is improving (if at all).
'''
def predict_movement(day):

	global weight_average
	global weight_stdev
	global weight_sum
	global weight_max
	global weight_min
	global weight_count

	logging.info('Prediction stock movement')
	print('PREDICTIONS BASED ON:')
	print('\t- AVG: ', weight_average)
	print('\t- STD: ', weight_stdev)

	# Open file to store todays predictions in
	file = open('./output/prediction-' + day + '.txt', 'w')

	file.write('Prediction Method 3: \n')
	file.write('Using all weights in prediciton.\n')
	file.write('Buy if 0.5 std above mean, sell if 0.5 std below mean. Otherwise undecided.\n')
	file.write('Weighting stats based on unique words. \n\n')

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

		if not tickers in stock_data:
			logging.warning('- Could not find articles loaded for ' + tickers)
			continue

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

		print('RATING FOR: ', tickers)
		print('\t- STD ABOVE MEAN: ', std_above_avg)
		print('\t- RAW VAL RATING: ', stock_rating)
		print('\t- PROBABILITY IS: ', probability)
		print('\t- CORRESPONDS TO: ', rating)

		file.write('Prediction for: ' + tickers + ' \n')
		file.write('- Std above mean: ' + str(std_above_avg) + '\n')
		file.write('- Raw val rating: ' + str(stock_rating) + '\n')
		file.write('- probability is: ' + str(probability) + '\n')
		file.write('- Corresponds to: ' + str(rating) + '\n\n')

	file.close()

def predict_movement2(day):

	global weight_average
	global weight_stdev
	global weight_sum
	global weight_max
	global weight_min
	global weight_count

	logging.info('Prediction stock movements with method 2 -> do not use stock weights within a standard deviation of the mean, buy/sell if more than 0.5 std away from mean')
	print('PREDICTIONS BASED ON:')
	print('\t- AVG: ', weight_average)
	print('\t- STD: ', weight_stdev)

	# Open file to store todays predictions in
	file = open('./output/prediction2-' + day + '.txt', 'w')

	file.write('Prediction Method 2: \n')
	file.write('Not using weights within one standard deviation of the mean in prediciton.\n')
	file.write('Buy if 0.5 std above mean, sell if 0.5 std below mean. Otherwise undecided.\n')
	file.write('Weighting stats based on unique words. \n\n')

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

		if not tickers in stock_data:
			logging.warning('- Could not find articles loaded for ' + tickers)
			continue

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

		print('RATING FOR: ', tickers)
		print('\t- STD ABOVE MEAN: ', std_above_avg)
		print('\t- RAW VAL RATING: ', stock_rating)
		print('\t- PROBABILITY IS: ', probability)
		print('\t- CORRESPONDS TO: ', rating)

		file.write('Prediction for: ' + tickers + ' \n')
		file.write('- Std above mean: ' + str(std_above_avg) + '\n')
		file.write('- Raw val rating: ' + str(stock_rating) + '\n')
		file.write('- probability is: ' + str(probability) + '\n')
		file.write('- Corresponds to: ' + str(rating) + '\n\n')

	file.close()

def predict_movement3(day):

	global weight_average
	global weight_stdev
	global weight_sum
	global weight_max
	global weight_min
	global weight_count

	logging.info('Prediction stock movements with method 3 -> If rating is above mean, buy, below mean, sell')
	print('PREDICTIONS BASED ON:')
	print('\t- AVG: ', weight_average)
	print('\t- STD: ', weight_stdev)

	# Open file to store todays predictions in
	file = open('./output/prediction3-' + day + '.txt', 'w')

	file.write('Prediction Method 3: \n')
	file.write('Using all weights in prediciton.\n')
	file.write('Buy if above mean, sell if below mean. \n')
	file.write('Weighting stats based on unique words.\n\n')

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

		if not tickers in stock_data:
			logging.warning('- Could not find articles loaded for ' + tickers)
			continue

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

		if stock_rating > weight_average:
			rating = 'buy'
		elif stock_rating < weight_average:
			rating = 'sell'
		else:
			rating = 'undecided'

		print('RATING FOR: ', tickers)
		print('\t- STD ABOVE MEAN: ', std_above_avg)
		print('\t- RAW VAL RATING: ', stock_rating)
		print('\t- PROBABILITY IS: ', probability)
		print('\t- CORRESPONDS TO: ', rating)

		file.write('Prediction for: ' + tickers + ' \n')
		file.write('- Std above mean: ' + str(std_above_avg) + '\n')
		file.write('- Raw val rating: ' + str(stock_rating) + '\n')
		file.write('- probability is: ' + str(probability) + '\n')
		file.write('- Corresponds to: ' + str(rating) + '\n\n')

	file.close()

def predict_movement4(day):

	global weight_average_o
	global weight_stdev_o
	global weight_sum_o
	global weight_max
	global weight_min
	global weight_count_o

	logging.info('Prediction stock movement,  weight stats on each occurence')
	print('PREDICTIONS BASED ON:')
	print('\t- AVG: ', weight_average_o)
	print('\t- STD: ', weight_stdev_o)

	# Open file to store todays predictions in
	file = open('./output/prediction4-' + day + '.txt', 'w')

	file.write('Prediction Method 4: \n')
	file.write('Using all weights in prediciton.\n')
	file.write('Buy if 0.5 std above mean, sell if 0.5 std below mean. Otherwise undecided.\n')
	file.write('Weighting stats based on each occurence of each words. \n\n')

	file.write('Predictions Based On Weighting Stats: \n')
	file.write('- Avg: ' + str(weight_average_o) + '\n')
	file.write('- Std: ' + str(weight_stdev_o) + '\n')
	file.write('- Sum: ' + str(weight_sum_o) + '\n')
	file.write('- Cnt: ' + str(weight_count_o) + '\n')
	file.write('- Max: ' + str(weight_max) + '\n')
	file.write('- Min: ' + str(weight_min) + '\n\n')


	# Iterate through stocks as predictions are seperate for each
	for tickers in STOCK_TAGS:

		logging.debug('- Finding prediction for: ' + tickers)

		stock_rating_sum = 0
		stock_rating_cnt = 0

		if not tickers in stock_data:
			logging.warning('- Could not find articles loaded for ' + tickers)
			continue

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
		std_above_avg = (stock_rating - weight_average_o) / weight_stdev_o
		probability = norm(weight_average_o, weight_stdev_o).cdf(stock_rating)

		if std_above_avg > 0.5:
			rating = 'buy'
		elif std_above_avg < -0.5:
			rating = 'sell'
		else:
			rating = 'undecided'

		print('RATING FOR: ', tickers)
		print('\t- STD ABOVE MEAN: ', std_above_avg)
		print('\t- RAW VAL RATING: ', stock_rating)
		print('\t- PROBABILITY IS: ', probability)
		print('\t- CORRESPONDS TO: ', rating)

		file.write('Prediction for: ' + tickers + ' \n')
		file.write('- Std above mean: ' + str(std_above_avg) + '\n')
		file.write('- Raw val rating: ' + str(stock_rating) + '\n')
		file.write('- probability is: ' + str(probability) + '\n')
		file.write('- Corresponds to: ' + str(rating) + '\n\n')

	file.close()

def predict_movement5(day):

	global weight_average_o
	global weight_stdev_o
	global weight_sum_o
	global weight_max
	global weight_min
	global weight_count_o

	logging.info('Prediction stock movements with method 2 -> do not use stock weights within a standard deviation of the mean, buy/sell if more than 0.5 std away from mean, weight stats on each occurence')
	print('PREDICTIONS BASED ON:')
	print('\t- AVG: ', weight_average_o)
	print('\t- STD: ', weight_stdev_o)

	# Open file to store todays predictions in
	file = open('./output/prediction5-' + day + '.txt', 'w')

	file.write('Prediction Method 5: \n')
	file.write('Not using weights within one standard deviation of the mean in prediciton.\n')
	file.write('Buy if 0.5 std above mean, sell if 0.5 std below mean. Otherwise undecided.\n')
	file.write('Weighting stats based on each occurence of each words. \n\n')

	file.write('Predictions Based On Weighting Stats: \n')
	file.write('- Avg: ' + str(weight_average_o) + '\n')
	file.write('- Std: ' + str(weight_stdev_o) + '\n')
	file.write('- Sum: ' + str(weight_sum_o) + '\n')
	file.write('- Cnt: ' + str(weight_count_o) + '\n')
	file.write('- Max: ' + str(weight_max) + '\n')
	file.write('- Min: ' + str(weight_min) + '\n\n')


	# Iterate through stocks as predictions are seperate for each
	for tickers in STOCK_TAGS:

		logging.debug('- Finding prediction for: ' + tickers)

		stock_rating_sum = 0
		stock_rating_cnt = 0

		if not tickers in stock_data:
			logging.warning('- Could not find articles loaded for ' + tickers)
			continue

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
						if weight > weight_stdev_o + weight_average_o or weight < weight_average_o - weight_stdev_o:
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
		std_above_avg = (stock_rating - weight_average_o) / weight_stdev_o
		probability = norm(weight_average_o, weight_stdev_o).cdf(stock_rating)

		if std_above_avg > 0.5:
			rating = 'buy'
		elif std_above_avg < -0.5:
			rating = 'sell'
		else:
			rating = 'undecided'

		print('RATING FOR: ', tickers)
		print('\t- STD ABOVE MEAN: ', std_above_avg)
		print('\t- RAW VAL RATING: ', stock_rating)
		print('\t- PROBABILITY IS: ', probability)
		print('\t- CORRESPONDS TO: ', rating)

		file.write('Prediction for: ' + tickers + ' \n')
		file.write('- Std above mean: ' + str(std_above_avg) + '\n')
		file.write('- Raw val rating: ' + str(stock_rating) + '\n')
		file.write('- probability is: ' + str(probability) + '\n')
		file.write('- Corresponds to: ' + str(rating) + '\n\n')

	file.close()

def predict_movement6(day):

	global weight_average_o
	global weight_stdev_o
	global weight_sum_o
	global weight_max
	global weight_min
	global weight_count_o

	logging.info('Prediction stock movements with method 6 -> If rating is above mean, buy, below mean, sell, weight stats on each occurence')
	print('PREDICTIONS BASED ON:')
	print('\t- AVG: ', weight_average_o)
	print('\t- STD: ', weight_stdev_o)

	# Open file to store todays predictions in
	file = open('./output/prediction6-' + day + '.txt', 'w')

	file.write('Prediction Method 6: \n')
	file.write('Using all weights in prediciton.\n')
	file.write('Buy if above mean, sell if below mean. \n')
	file.write('Weighting stats based on each occurence of each words. \n\n')

	file.write('Predictions Based On Weighting Stats: \n')
	file.write('- Avg: ' + str(weight_average_o) + '\n')
	file.write('- Std: ' + str(weight_stdev_o) + '\n')
	file.write('- Sum: ' + str(weight_sum_o) + '\n')
	file.write('- Cnt: ' + str(weight_count_o) + '\n')
	file.write('- Max: ' + str(weight_max) + '\n')
	file.write('- Min: ' + str(weight_min) + '\n\n')


	# Iterate through stocks as predictions are seperate for each
	for tickers in STOCK_TAGS:

		logging.debug('- Finding prediction for: ' + tickers)

		stock_rating_sum = 0
		stock_rating_cnt = 0

		if not tickers in stock_data:
			logging.warning('- Could not find articles loaded for ' + tickers)
			continue

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
		std_above_avg = (stock_rating - weight_average_o) / weight_stdev_o
		probability = norm(weight_average_o, weight_stdev_o).cdf(stock_rating)

		if stock_rating > weight_average_o:
			rating = 'buy'
		elif stock_rating < weight_average_o:
			rating = 'sell'
		else:
			rating = 'undecided'

		print('RATING FOR: ', tickers)
		print('\t- STD ABOVE MEAN: ', std_above_avg)
		print('\t- RAW VAL RATING: ', stock_rating)
		print('\t- PROBABILITY IS: ', probability)
		print('\t- CORRESPONDS TO: ', rating)

		file.write('Prediction for: ' + tickers + ' \n')
		file.write('- Std above mean: ' + str(std_above_avg) + '\n')
		file.write('- Raw val rating: ' + str(stock_rating) + '\n')
		file.write('- probability is: ' + str(probability) + '\n')
		file.write('- Corresponds to: ' + str(rating) + '\n\n')

	file.close()

def predict_movement7(day):

	global total_up
	global total_down
	global total_words_up
	global total_words_down
	global c

	logging.info('Prediction stock movements with method 7 -> Naive Bayes Classifier')

	# Open file to store todays predictions in
	file = open('./output/prediction7-' + day + '.txt', 'w')

	file.write('Prediction Method 7: \n')
	file.write('Naive Bayes Classifier -> Requires word weight option 2 is loaded.\n')
	file.write('P(Word=w|Label=y) = (count(w,y)+c) / ([total number of words with label y] + c * vocabulary size). \n')
	file.write('Value for label: log(P(y, word1, word2, ... wordn)) = log(P(y)) + log(P(word1 | y)) + log(P(word2 | y)) + ... + log(P(wordn | y)). \n')
	file.write('Label (Up or Down) with the greatest value is chosen.\n\n')

	file.write('Predictions Based On: \n')
	file.write('- Total Up Days    : ' + str(total_up) + '\n')
	file.write('- Total Down Days  : ' + str(total_down) + '\n')
	file.write('- Total Up Words   : ' + str(total_words_up) + '\n')
	file.write('- Total Down Words : ' + str(total_words_down) + '\n')
	file.write('- Value for C      : ' + str(c) + '\n\n')


	# Iterate through stocks as predictions are seperate for each
	for tickers in STOCK_TAGS:

		logging.debug('- Finding prediction for: ' + tickers)

		# Initialize the up and down probabilities with the probability of it happening
		stock_rating_up = math.log(float(total_up) / (total_up + total_down))
		stock_rating_down = math.log(float(total_down) / (total_up + total_down))

		if not tickers in stock_data:
			logging.warning('- Could not find articles loaded for ' + tickers)
			continue

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

						up, down = get_word_probability_given_label(found_word, c)
						
						if up != None and down != None:
							stock_rating_up += math.log(up)
							stock_rating_down += math.log(down)


				# If a word is not being found and letter pops up, start the word
				elif not word_in_progress and chars.isalpha():

					# Start the word
					word_in_progress = True
					word_start_index = ii

		# If the up value is greater, rating is a buy, else sell
		if stock_rating_up > stock_rating_down:
			rating = 'buy'
		elif stock_rating_up < stock_rating_down:
			rating = 'sell'
		else:
			rating = 'undecided'

		print('RATING FOR: ', tickers)
		print('\t- UP RATING IS  : ', stock_rating_up)
		print('\t- DOWN RATING IS: ', stock_rating_down)
		print('\t- CORRESPONDS TO: ', rating)

		file.write('Prediction for: ' + tickers + ' \n')
		file.write('- Up rating is  : ' + str(stock_rating_up) + '\n')
		file.write('- Down rating is: ' + str(stock_rating_down) + '\n')
		file.write('- Corresponds to: ' + str(rating) + '\n\n')

	file.close()

'''
Display help and exit
'''
def print_help():

	logging.debug('Displaying help and exiting')

	print('To predict stock price based on articles for the current day, use (must have already pulled articles): ')
	print('\t ./stock_market_prediction.py -p\n')
	print('To predict stock price based on articles for the current day with a specific weighting option, use (must have already pulled articles): ')
	print('\t ./stock_market_prediction.py -p -o option\n')
	print('To predict stock price based on articles for a specific day, use (must have already pulled articles): ')
	print('\t ./stock_market_prediction.py -p -d mm-d-yyyy\n')
	print('To predict stock price based on articles for a specific day with a specific weighting option, use (must have already pulled articles): ')
	print('\t ./stock_market_prediction.py -p -d mm-d-yyyy -o option\n')
	print('To pull articles for the current day, use (currently no support for pulling articles for previous days):')
	print('\t ./stock_market_prediction.py -a\n')
	print('To pull stock prices for the current day, use (currently no support for pulling prices for previous days):')
	print('\t ./stock_market_prediction.py -s\n')
	print('To update word weights for articles and prices from the current day, use (must have already pulled weights and prices):')
	print('\t ./stock_market_prediction.py -u\n')
	print('To update word weights for articles and prices from the current day with a specific weighting option, use (must have already pulled weights and prices):')
	print('\t ./stock_market_prediction.py -u -o option\n')
	print('To update word weights for articles and prices from a specifc day, use (must have already pulled weights and prices):')
	print('\t ./stock_market_prediction.py -u -d mm-d-yyyy\n')
	print('To update word weights for articles and prices from a specifc day with a specific weighting option, use (must have already pulled weights and prices):')
	print('\t ./stock_market_prediction.py -u -d mm-d-yyyy -o option\n')
	print('To update word weights for articles and prices from a date range, use (must have already pulled weights and prices):')
	print('\t ./stock_market_prediction.py -u -b mm-d-yyyy -e mm-d-yyyy\n')
	print('To update word weights for articles and prices from a date range with a specific weighting option, use (must have already pulled weights and prices):')
	print('\t ./stock_market_prediction.py -u -b mm-d-yyyy -e mm-d-yyyy -o option\n')

	print('\nCurrently available weighting options are opt1 and opt2. opt1 uses average with 1 for a word seen with up and 0 with a word seen with down.\n opt2 uses a Naive Bayes classifier.\n')
	sys.exit(2)

'''
Verifies that a date is valid and in the right format
'''
def verify_date(date):

	today = datetime.date.today()
	date_parts = date.split('-')

	if len(date_parts) < 3:
		return False

	try:
		test_date = datetime.date(int(date_parts[2]), int(date_parts[0]), int(date_parts[1]))
	except:
		return False

	if test_date > today:
		return False

	return True

'''
Simply prints the analysis of the weights (does not analyze them, just prints)
- Only for weighting option opt1
'''
def print_weight_analysis():
	print('')
	print('Weight Stats (Based on number of words):')
	print('- Avg: ' + str(weight_average))
	print('- Std: ' + str(weight_stdev))
	print('- Sum: ' + str(weight_sum))
	print('- Cnt: ' + str(weight_count))
	print('- Max: ' + str(weight_max))
	print('- Min: ' + str(weight_min) + '\n')

	print('Weight Stats (Based on occurences of words):')
	print('- Avg: ' + str(weight_average_o))
	print('- Std: ' + str(weight_stdev_o))
	print('- Sum: ' + str(weight_sum_o))
	print('- Cnt: ' + str(weight_count_o))
	print('- Max: ' + str(weight_max))
	print('- Min: ' + str(weight_min) + '\n')

'''
Looks over Predictions and determines accuracy
'''
def determine_accuracy():

	# Loop over all prediction files
	for filename in os.listdir('./output/'):

		# Open the file
		try:
			file = open('./output/' + filename, 'r+')
		except IOError as error:
			print('Could not open: ' + filename)
			continue

		num_correct = 0
		num_wrong = 0
		num_undecided = 0
		current_stock = ''
		found = False

		# Get the day the prediction was made
		try:
			ii = filename.index('-')
			prediction_date = filename[ii+1:-4]
		except:
			continue

		# Iterate through the file
		for lines in file:

			# Find a prediction
			if lines[:len('Prediction for:')] == 'Prediction for:' and found == False:

				# Record the stock
				current_stock = lines[len('Prediction for:') + 1:-2]

				# Set found to true to look for the rating
				found = True

			# Find a rating
			if lines[:len('- Corresponds to:')] == '- Corresponds to:' and found == True:

				# Get the rating
				rating = lines[len('- Corresponds to:') + 1:-1]

				# Get the chane in the stock price for that day
				change = stock_prices[current_stock][prediction_date][1] - stock_prices[current_stock][prediction_date][0]

				# Check if the prediction was correct
				if change > 0 and rating == 'buy':
					num_correct += 1
				elif change < 0 and rating == 'sell':
					num_correct += 1
				elif rating == 'undecided':
					num_undecided += 1
				else:
					num_wrong += 1

				# Set found back to false to find evaluate the next stock
				found = False

		# At the end of the file, state the statistics
		if num_correct + num_wrong != 0:
			print(filename + '\t: Correct: ' + str(float(num_correct) / (num_wrong + num_correct) * 100) + '%')
		else:
			print(filename + '\t: All undecided')

'''
Main Execution
- Part 1: Parsing Inputs and Pulling, Storing, and Loading Articles
'''
def main():

	execution_type = -1
	specified_day = ''
	start_day = ''
	end_day = ''
	weight_opt = 'opt1'

	today = datetime.date.today()
	today_str = str(today.month) + '-' + str(today.day) + '-' + str(today.year)

	# First get any command line arguments to edit actions
	try:
		opts, args = getopt.getopt(sys.argv[1:], 'hpsauzvd:b:e:o:')
	except getopt.GetoptError:
		print_help()
	for opt, arg in opts:
		# Help message
		if opt == '-h':
			print_help()
		# Command line argument for Predict
		elif opt == '-p':
			execution_type = 0
		# Command line argument for pull new Articles
		elif opt == '-a':
			execution_type = 1
		# Command line argument for pull new Stock prices
		elif opt == '-s':
			execution_type = 2
		# Command line argument for Update word weights
		elif opt == '-u':
			execution_type = 3
		# Command line argument for adding a specfic Day (will only matter for predict and update)
		elif opt == '-d':
			specified_day = arg
		# Command line argument for specifying the Beginning of a date range (only for update)
		elif opt == '-b':
			start_day = arg
		# Command line argument for specifying the End of a date range (only for update)
		elif opt == '-e':
			end_day = arg
		# Command for specifying which weighting system to use
		elif opt == '-o':
			if arg == 'opt1' or arg == 'opt2':
				weight_opt = arg
		# Command line argument for printing current weight stats
		elif opt == '-z':
			load_all_word_weights('opt1')
			if not analyze_weights_gpu():
				print('Error: Unable to analyze weights')
				sys.exit(-1)
			print_weight_analysis()
			sys.exit(0)
		elif opt == '-v':
			load_stock_prices()
			determine_accuracy()
			sys.exit(0)

	# Depending on the input type, preform the proper action
	# Predict
	if execution_type == 0:

		# First prepare the day to predict on
		if not verify_date(specified_day):
			specified_day = today_str

		logging.info('Predicting price movement for: ' + specified_day)

		# Call the proper functions
		load_all_word_weights(weight_opt)
		if not load_articles(specified_day):
			print('Error: Could not load articles for: ', specified_day)
			sys.exit(-1)
		if not analyze_weights():
			print('Error: Unable to analyze weights')
			sys.exit(-1)

		if weight_opt == 'opt1':
			predict_movement(specified_day)
			predict_movement2(specified_day)
			predict_movement3(specified_day)
			predict_movement4(specified_day)
			predict_movement5(specified_day)
			predict_movement6(specified_day)
		
		elif weight_opt == 'opt2':
			predict_movement7(specified_day)

		logging.info('Predicting complete')

	# Pull new articles
	elif execution_type == 1:

		logging.info('Pulling articles for: ' + today_str)

		# Call the proper functions
		pull_recent_articles()

		logging.info('Pulling articles complete')

	# Pull new stock prices
	elif execution_type == 2:

		logging.info('Pulling stock prices for: ' + today_str)

		# Call the proper functions
		load_stock_prices()
		pull_stock_prices()
		save_stock_prices()

		logging.info('Pulling stock prices complete')

	# Update word weights
	elif execution_type == 3:

		# Load non-day specific values
		logging.info('Loading non-day specific data before updating word weights')

		# Prepare the days to update word weights for
		days = []

		# First check to see if a day is specified, if it is, set it to the day
		# - If not, use the current day as the specified day
		if not verify_date(specified_day):
			specified_day = today_str

		# If there is no start day specified, then it is assumed that the user wants
		# - a specified day or the current day. This is added to the days array.
		if not verify_date(start_day):
			days.append(specified_day)

		# If there is a start day specified, then the it checks to see if there is an end day. If not, 
		# - it sets the current day to the end date. The difference between the start data and the and date
		# - is calculated and the dates inbetween are added to the days array.
		else:
			date_parts1 = start_day.split('-')
			date1 = datetime.date(int(date_parts1[2]), int(date_parts1[0]), int(date_parts1[1]))
			date2 = datetime.date.today()
			if verify_date(end_day):
				date_parts2 = end_day.split('-')
				date2 = datetime.date(int(date_parts2[2]), int(date_parts2[0]), int(date_parts2[1]))

			delta = date2-date1
			for each_day in range(0, delta.days + 1):
				temp_day = date1 + datetime.timedelta(days = each_day)
				days.append(str(temp_day.month) + '-' + str(temp_day.day) + '-' + str(temp_day.year))

		load_all_word_weights(weight_opt)
		load_stock_prices()

		for each in days:

			logging.info('Updating word weights for: ' + each)

			# Call the proper functions
			if load_articles(each):
				update_all_word_weights(weight_opt, each)
			stock_data.clear() # To prepare for the next set of articles

		# Save data
		logging.info('Saving word weights')
		save_all_word_weights(weight_opt)
		
		logging.info('Updating word weights complete')

	print('Done')




main()
