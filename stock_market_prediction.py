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
from bs4 import BeautifulSoup as BS

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

# Global Variables and Configurations
stock_data = {}
logging.basicConfig(filename="stock_market_prediction.log", level=logging.DEBUG, format="%(asctime)s: %(levelname)s>\t%(message)s")
logging.info('RUNNING STOCK MARKET PREDICTION')

'''
Step 1.1 is to iterate through stock symbols and collect their article text. This step is not meant to be done in parallel but
to simply get recent articles about the stock. For my analysis, I try to scrape 10 of the most recent articles from google news
using the stocks ticker symbol as the search key. 

The following are done for each stock ticker:
a) Make the request to google news and pull the recent article links, reformat them to work right
b) Take the first 10 articles and pull all of their content
c) For each article, try to find the article text or content text
d) If article text was found, pull all the paragraph tag text
e) Append the text to one big string for the article and add it to the article dictionary
'''
def pull_recent_articles(directory):

	# Iterate through the stock tags
	for ticker in STOCK_TAGS:
		start = time.time()

		logging.debug('Starting to pull articles for: ' + ticker)
		print 'Pulling articles for: ', ticker
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
		if len(stock_data[ticker]) < ARTICLES_PER_STOCK:
			return False
	return True

'''
Step 1.2 if articles have already been pulled, then instad of pulling them again and wasting the time that it takes, we can instead
load the articles from that directory. Articles for each morning are kept in their directory so we open that up and load them into
the dictionary.
'''
def load_articles(directory):
	
	# Iterate through the stock tags
	for ticker in STOCK_TAGS:
		start = time.time()
		print 'Loading articles for: ', ticker

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

		logging.debug('Finished pulling articles for: ' + ticker + ', time: ' + str(time.time() - start))

	# Check all tags for number of articles loaded
	for ticker in STOCK_TAGS:
		if len(stock_data[ticker]) < ARTICLES_PER_STOCK:
			return False
	return True

'''
Main Execution
- Part 1: Parsing Inputs and Pulling, Storing, and Loading Articles
'''
def main():

	'''
	The first part of the main function is to parse through the inputs and determine whether data should be loaded
	or stored. If it should be loaded, the directory it should be loaded from is also determined. This allows the 
	program to prepare itself to work with the data and/or to save data for future analysis.

	'''

	# First, prepare the folder to save all the articles for the day in
	today = datetime.datetime.now()
	directory = './stock_market_prediction-' + str(today.month) + '-' + str(today.day) + '-' + str(today.year) + '/'
	create_new = True

	# Get any command line arguments to edit actions
	try:
		opts, args = getopt.getopt(sys.argv[1:], 'hld:')
	except getopt.GetoptError:
		print './stock_market_prediction.py \n./stock_market_prediction.py -l [-d <directory_to_load>]'
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print './stock_market_prediction.py \n./stock_market_prediction.py -l [-d <directory_to_load>]'
			logging.debug('Displaying help and exiting')
			sys.exit(2)
		elif opt == '-l':
			create_new = False
		elif opt == '-d':
			directory = arg

	# Log the running configuration
	if create_new:
		logging.info('Pulling new articles and storing them in: ' + directory)
	else:
		logging.info('Loading articles from: ' + directory)

	# Check to see if the directory exists
	# - Create if if creating new, exit if does exist but trying to load
	if not os.path.exists(directory) and create_new:
		os.makedirs(directory)
		logging.debug('Created directory: ' + directory)
	elif not os.path.exists(directory):
		logging.error(directory + ' does not exist, no articles to load, exiting')
		print 'Error: ', directory, ' does not exist'
		sys.exit(-1)

	# Either pull or load the articles
	if create_new:
		# Pull new articles if creating new
		if not pull_recent_articles(directory):
			logging.error('A fatal error occured while pulling articles, exiting')
			sys.exit(-1)
		else:
			logging.info('Successfully pulled new articles')
	else:
		# Load articles if loading existing ones
		if not load_articles(directory):
			logging.error('A fatal error occured while loading articles, exiting')
			sys.exit(-1)
		else:
			logging.info('Successfully loaded articles')


	'''
	Now that data has been loaded, the second part of the program is to analyze the data. For now, that is restricted
	to seeing which words are common so that a proper analysis scheme can be determined.

	'''
	print ''
	print 'Finding words in articles'

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
	file = open('word_frequencies.txt', 'w')

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















main()