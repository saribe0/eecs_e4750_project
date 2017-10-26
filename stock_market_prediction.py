#!/usr/bin/env python

import requests
from bs4 import BeautifulSoup as BS

STOCK_TAGS = [	'amzn',
				'amat',
				'agn',
				'goog',
				'hd',
				'lmt',
				'data']

ARTICLES_PER_STOCK = 3
stock_data = {}

'''
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
	# Iterate through the stock tags
	for ticker in STOCK_TAGS:

		# Make the request to google news for the ticker and get its raw html
		r = requests.get('https://www.google.com/search?biw=1366&bih=671&tbm=nws&ei=HRDyWcDlJoa-jwSuwZCYBg&q=' + ticker)
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

				# Once the link as been found, make a request for its html content
				r = requests.get(link)
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
					raw_text = raw_html.find('div', attrs={'id' : 'article'} )

				if raw_text == None:
					continue

				# Get all the paragraph parts of the article's content
				article_parts = raw_text.find_all('p')

				# Combine all the paragraphs into a single text block
				article = ''
				for paragraphs in article_parts:
					article += paragraphs.text

				# Store the url and article as a touple into the stock data dictionary
				print article
				print ''
				print '=================== NEXT =================='
				stock_data[ticker].append((link, article))

				# Update the counter (only want 10 articles per stock)
				counter += 1

			if counter == ARTICLES_PER_STOCK:
				break

pull_recent_articles()