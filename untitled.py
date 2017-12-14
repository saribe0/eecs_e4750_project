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

		start_all = time.time()
		cpu = 0

		# Iterate through each article for the stock
		for articles in stock_data[tickers]:

			# Get the text (ignore link)
			text = articles[1]

			# Get an array of words with two or more characters for the text
			words_in_text = re.compile('[A-Za-z][A-Za-z][A-Za-z]+').findall(text)

			start = time.time()

			# Update the word's info
			for words in words_in_text:

				stock_rating_sum += get_word_weight(words)
				stock_rating_cnt += 1

			end = time.time()
			cpu += end - start

		# After each word in every article has been examined for that stock, find the average rating
		stock_rating = stock_rating_sum / stock_rating_cnt

		# Calculate the number of standard deviations above the mean and find the probability of that for a 'normal' distribution 
		# - Assuming normal because as the word library increases, it should be able to be modeled as normal
		std_above_avg = (stock_rating - weight_average) / weight_stdev
		probability = norm(weight_average, weight_stdev).cdf(stock_rating)

		end_all = time.time()

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
		print('========================> Time All: ', end_all - start_all)
		print('========================> Time Cpu: ', cpu)

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

			# Get an array of words with two or more characters for the text
			words_in_text = re.compile('[A-Za-z][A-Za-z][A-Za-z]+').findall(text)

			# Update the word's info
			for words in words_in_text:

				weight = get_word_weight(words)
				if weight > weight_stdev + weight_average or weight < weight_average - weight_stdev:
					stock_rating_sum += weight
					stock_rating_cnt += 1

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

			# Get an array of words with two or more characters for the text
			words_in_text = re.compile('[A-Za-z][A-Za-z][A-Za-z]+').findall(text)

			# Update the word's info
			for words in words_in_text:

				stock_rating_sum += get_word_weight(words)
				stock_rating_cnt += 1

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

			# Get an array of words with two or more characters for the text
			words_in_text = re.compile('[A-Za-z][A-Za-z][A-Za-z]+').findall(text)

			# Update the word's info
			for words in words_in_text:

				stock_rating_sum += get_word_weight(words)
				stock_rating_cnt += 1

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

			# Get an array of words with two or more characters for the text
			words_in_text = re.compile('[A-Za-z][A-Za-z][A-Za-z]+').findall(text)

			# Update the word's info
			for words in words_in_text:

				weight = get_word_weight(words)
				if weight > weight_stdev_o + weight_average_o or weight < weight_average_o - weight_stdev_o:
					stock_rating_sum += weight
					stock_rating_cnt += 1

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

			# Get an array of words with two or more characters for the text
			words_in_text = re.compile('[A-Za-z][A-Za-z][A-Za-z]+').findall(text)

			# Update the word's info
			for words in words_in_text:

				stock_rating_sum += get_word_weight(words)
				stock_rating_cnt += 1

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

def predict_movement(day):

	global weight_average
	global weight_stdev
	global weight_sum
	global weight_max
	global weight_min
	global weight_count

	global weight_average_o
	global weight_stdev_o
	global weight_sum_o
	global weight_max
	global weight_min
	global weight_count_o

	logging.info('Predicting stock movements')

	all_predictions = {}
	all_std_devs = {}
	all_probabilities = {}
	all_raw_ratings = {}

	# Iterate through stocks as predictions are seperate for each
	for tickers in STOCK_TAGS:

		logging.debug('- Finding prediction for: ' + tickers)

		all_predictions[tickers] = []
		all_std_devs[tickers] = []
		all_probabilities[tickers] = []
		all_raw_ratings[tickers] = []

		stock_rating_sum_p1 = 0
		stock_rating_cnt_p1 = 0

		stock_rating_sum_p2 = 0
		stock_rating_cnt_p2 = 0

		stock_rating_sum_p5 = 0
		stock_rating_cnt_p5 = 0

		if not tickers in stock_data:
			logging.warning('- Could not find articles loaded for ' + tickers)
			continue

		start_all = time.time()
		cpu = 0

		# Iterate through each article for the stock
		for articles in stock_data[tickers]:

			# Get the text (ignore link)
			text = articles[1]

			# Get an array of words with two or more characters for the text
			words_in_text = re.compile('[A-Za-z][A-Za-z][A-Za-z]+').findall(text)

			start = time.time()

			# Update the word's info
			for words in words_in_text:

				weight = get_word_weight(words)

				# Prediction Method 1, 3, 4, 6
				stock_rating_sum_p1 += weight
				stock_rating_cnt_p1 += 1

				# Prediction Method 2
				if weight > weight_stdev + weight_average or weight < weight_average - weight_stdev:
					stock_rating_sum_p2 += weight
					stock_rating_cnt_p2 += 1

				# Prediction Method 5
				if weight > weight_stdev_o + weight_average_o or weight < weight_average_o - weight_stdev_o:
					stock_rating_sum_p5 += weight
					stock_rating_cnt_p5 += 1

			end = time.time()
			cpu += end - start

		# After each word in every article has been examined for that stock, find the average rating
		stock_rating_p1 = stock_rating_sum_p1 / stock_rating_cnt_p1
		stock_rating_p2 = stock_rating_sum_p2 / stock_rating_cnt_p2
		stock_rating_p5 = stock_rating_sum_p5 / stock_rating_cnt_p5

		# Calculate the number of standard deviations above the mean and find the probability of that for a 'normal' distribution 
		# - Assuming normal because as the word library increases, it should be able to be modeled as normal
		std_above_avg_p1 = (stock_rating_p1 - weight_average) / weight_stdev
		probability_p1 = norm(weight_average, weight_stdev).cdf(stock_rating_p1)

		std_above_avg_p2 = (stock_rating_p2 - weight_average) / weight_stdev
		probability_p2 = norm(weight_average, weight_stdev).cdf(stock_rating_p2)

		std_above_avg_p4 = (stock_rating_p1 - weight_average_o) / weight_stdev_o
		probability_p4 = norm(weight_average_o, weight_stdev_o).cdf(stock_rating_p1)

		std_above_avg_p5 = (stock_rating_p5 - weight_average_o) / weight_stdev_o
		probability_p5 = norm(weight_average_o, weight_stdev_o).cdf(stock_rating_p5)

		end_all = time.time()

		# Update the variables for prediction 1 in slot 0
		all_std_devs[tickers].append(std_above_avg_p1)
		all_probabilities[tickers].append(probability_p1)
		all_raw_ratings[tickers].append(stock_rating_p1)

		if std_above_avg_p1 > 0.5:
			all_predictions[tickers].append(1)
		elif std_above_avg_p1 < -0.5:
			all_predictions[tickers].append(-1)
		else:
			all_predictions[tickers].append(0)

		# Update the variables for prediction 2 in slot 1
		all_std_devs[tickers].append(std_above_avg_p2)
		all_probabilities[tickers].append(probability_p2)
		all_raw_ratings[tickers].append(stock_rating_p2)

		if std_above_avg_p2 > 0.5:
			all_predictions[tickers].append(1)
		elif std_above_avg_p2 < -0.5:
			all_predictions[tickers].append(-1)
		else:
			all_predictions[tickers].append(0)

		# Update the variables for prediction 3 (Which are based off the same stats as prediction 1) in slot 2
		all_std_devs[tickers].append(std_above_avg_p1)
		all_probabilities[tickers].append(probability_p1)
		all_raw_ratings[tickers].append(stock_rating_p1)

		if std_above_avg_p1 > weight_average:
			all_predictions[tickers].append(1)
		elif std_above_avg_p1 < weight_average:
			all_predictions[tickers].append(-1)
		else:
			all_predictions[tickers].append(0)

		# Update the variables for prediction 4 (uses p1's weighting with different weight analysis) in slot 3
		all_std_devs[tickers].append(std_above_avg_p4)
		all_probabilities[tickers].append(probability_p4)
		all_raw_ratings[tickers].append(stock_rating_p1)

		if std_above_avg_p4 > 0.5:
			all_predictions[tickers].append(1)
		elif std_above_avg_p4 < -0.5:
			all_predictions[tickers].append(-1)
		else:
			all_predictions[tickers].append(0)

		# Update the variables for prediction 5 in slot 4
		all_std_devs[tickers].append(std_above_avg_p5)
		all_probabilities[tickers].append(probability_p5)
		all_raw_ratings[tickers].append(stock_rating_p5)

		if std_above_avg_p5 > 0.5:
			all_predictions[tickers].append(1)
		elif std_above_avg_p5 < -0.5:
			all_predictions[tickers].append(-1)
		else:
			all_predictions[tickers].append(0)

		# Update the variables for prediction 6 (Which are based off the same stats as prediction 4) in slot 5
		all_std_devs[tickers].append(std_above_avg_p4)
		all_probabilities[tickers].append(probability_p4)
		all_raw_ratings[tickers].append(stock_rating_p1)

		if std_above_avg_p4 > weight_average_o:
			all_predictions[tickers].append(1)
		elif std_above_avg_p4 < weight_average_o:
			all_predictions[tickers].append(-1)
		else:
			all_predictions[tickers].append(0)

	# Iterate through the predictions, print them, and write them to files
	for ii in range(0, 6)

		# Open file to store todays predictions in
		file = open('./output/prediction' + str(ii) +'-' + day + '.txt', 'w')

		# Print the header info and open the file
		# When less than 3, uses normal weight analysis 
		if ii < 3:
			print('PREDICTIONS BASED ON:')
			print('\t- AVG: ', weight_average)
			print('\t- STD: ', weight_stdev)

			if ii == 0:
				file.write('Prediction Method 1: \n')
				file.write('Using all weights in prediciton.\n')
				file.write('Buy if 0.5 std above mean, sell if 0.5 std below mean. Otherwise undecided.\n')
				file.write('Weighting stats based on unique words. \n\n')
			if ii == 1:
				file.write('Prediction Method 2: \n')
				file.write('Not using weights within one standard deviation of the mean in prediciton.\n')
				file.write('Buy if 0.5 std above mean, sell if 0.5 std below mean. Otherwise undecided.\n')
				file.write('Weighting stats based on unique words. \n\n')
			if ii == 2:
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
		# When greater than 2, uses the weighted weights
		else:
			print('PREDICTIONS BASED ON:')
			print('\t- AVG: ', weight_average_o)
			print('\t- STD: ', weight_stdev_o)

			if ii == 3:
				file.write('Prediction Method 4: \n')
				file.write('Using all weights in prediciton.\n')
				file.write('Buy if 0.5 std above mean, sell if 0.5 std below mean. Otherwise undecided.\n')
				file.write('Weighting stats based on each occurence of each words. \n\n')
			if ii == 4:
				file.write('Prediction Method 5: \n')
				file.write('Not using weights within one standard deviation of the mean in prediciton.\n')
				file.write('Buy if 0.5 std above mean, sell if 0.5 std below mean. Otherwise undecided.\n')
				file.write('Weighting stats based on each occurence of each words. \n\n')
			if ii == 5:
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

		# For each prediction, iterate through the stocks
		for tickers in STOCK_TAGS:

			if all_predictions[tickers][ii] == 1:
				rating = 'buy'
			elif all_predictions[tickers][ii] == -1:
				rating = 'sell'
			else:
				rating = 'undecided'

			# For each stock, print and write the rating
			print('RATING FOR: ', tickers)
			print('\t- STD ABOVE MEAN: ', all_std_devs[tickers][ii])
			print('\t- RAW VAL RATING: ', all_raw_ratings[tickers][ii])
			print('\t- PROBABILITY IS: ', all_probabilities[tickers][ii])
			print('\t- CORRESPONDS TO: ', rating)

			file.write('Prediction for: ' + tickers + ' \n')
			file.write('- Std above mean: ' + str(all_std_devs[tickers][ii]) + '\n')
			file.write('- Raw val rating: ' + str(all_raw_ratings[tickers][ii]) + '\n')
			file.write('- probability is: ' + str(all_probabilities[tickers][ii]) + '\n')
			file.write('- Corresponds to: ' + str(rating) + '\n\n')

		file.close()


# If it is the same, calculate and return the probabilities
			up = (float(test_data[2]) + c) / (total_words_up + c * num_words)
			down = (float(test_data[3]) + c) / (total_words_down + c * num_words)
			return up, down

__kernel void predict_bayes(__global char* words, __global int* weights, __global char* weights_char, __global int* num_weights_letter, volatile __global float* out_probabilities_up, volatile __global float* out_probabilities_down, int total_weights_up, int total_weights_down, int total_weights, int c, int max_words_per_letter, int word_max)


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

			# Get an array of words with two or more characters for the text
			words_in_text = re.compile('[A-Za-z][A-Za-z][A-Za-z]+').findall(text)

			# Update the word's info
			for words in words_in_text:

				up, down = get_word_probability_given_label(words, c)
						
				if up != None and down != None:
					stock_rating_up += math.log(up)
					stock_rating_down += math.log(down)

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



	def predict_movement_gpu(day):

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

	logging.info('Predicting stock movements with the GPU')

	all_predictions = {}
	all_std_devs = {}
	all_probabilities = {}
	all_raw_ratings = {}

	# Iterate through stocks as predictions are seperate for each
	for tickers in STOCK_TAGS:

		logging.debug('- Finding prediction for: ' + tickers)

		all_predictions[tickers] = []
		all_std_devs[tickers] = []
		all_probabilities[tickers] = []
		all_raw_ratings[tickers] = []

		words_in_text = []

		if not tickers in stock_data:
			logging.warning('- Could not find articles loaded for ' + tickers)
			continue

		start_all = time.time()

		# Iterate through each article for the stock
		for articles in stock_data[tickers]:

			# Get the text (ignore link)
			text = articles[1]

			# Get an array of words with two or more characters for the text
			words_in_text += re.compile('[A-Za-z][A-Za-z][A-Za-z]+').findall(text)

		# Store the words to be read by the kernel
		word_data = bytearray(len(words_in_text)*16)
		for ii, word in enumerate(words_in_text):
			struct.pack_into('16s', word_data, ii * 16, word.lower().encode('utf-8'))

		# Prepare an output buffer
		out_weights = np.zeros((len(words_in_text), ), dtype = np.float32)

		# Create the buffers for the GPU
		mf = cl.mem_flags
		word_data_buff = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = word_data)
		weights_buff = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = np.asarray(words_by_letter))
		weights_char_buff = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = np.asarray(words_by_letter))
		num_weights_buff = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = np.asarray(num_words_by_letter, dtype = np.int32))
		out_weights_buff = cl.Buffer(ctx, mf.WRITE_ONLY, out_weights.nbytes)

		# Determine the grid size
		groups, extra = divmod(len(words_in_text), 256)
		grid = ((groups + (extra>0))*256, 2560)

		start = time.time()

		# Call the kernel
		prg_2.predict(queue, grid, None, word_data_buff, weights_buff, weights_char_buff, num_weights_buff, out_weights_buff, np.uint32(MAX_WORDS_PER_LETTER), np.uint32(len(words_in_text)))

		# Collect the output
		cl.enqueue_copy(queue, out_weights, out_weights_buff)

		stock_rating_sum_p1 = 0

		stock_rating_sum_p2 = 0
		stock_rating_cnt_p2 = 0

		stock_rating_sum_p5 = 0
		stock_rating_cnt_p5 = 0

		for w in out_weights:

			# Prediction Method 1, 3, 4, 6
			stock_rating_sum_p1 += w

			# Prediction Method 2
			if weight > weight_stdev + weight_average or weight < weight_average - weight_stdev:
				stock_rating_sum_p2 += weight
				stock_rating_cnt_p2 += 1

			# Prediction Method 5
			if weight > weight_stdev_o + weight_average_o or weight < weight_average_o - weight_stdev_o:
				stock_rating_sum_p5 += weight
				stock_rating_cnt_p5 += 1

		stock_rating_cnt_p1 = len(words_in_text)

		end = time.time()

		# After each word in every article has been examined for that stock, find the average rating
		stock_rating_p1 = stock_rating_sum_p1 / stock_rating_cnt_p1
		stock_rating_p2 = stock_rating_sum_p2 / stock_rating_cnt_p2
		stock_rating_p5 = stock_rating_sum_p5 / stock_rating_cnt_p5

		# Calculate the number of standard deviations above the mean and find the probability of that for a 'normal' distribution 
		# - Assuming normal because as the word library increases, it should be able to be modeled as normal
		std_above_avg_p1 = (stock_rating_p1 - weight_average) / weight_stdev
		probability_p1 = norm(weight_average, weight_stdev).cdf(stock_rating_p1)

		std_above_avg_p2 = (stock_rating_p2 - weight_average) / weight_stdev
		probability_p2 = norm(weight_average, weight_stdev).cdf(stock_rating_p2)

		std_above_avg_p4 = (stock_rating_p1 - weight_average_o) / weight_stdev_o
		probability_p4 = norm(weight_average_o, weight_stdev_o).cdf(stock_rating_p1)

		std_above_avg_p5 = (stock_rating_p5 - weight_average_o) / weight_stdev_o
		probability_p5 = norm(weight_average_o, weight_stdev_o).cdf(stock_rating_p5)

		end_all = time.time()

		gpu_kernel_time.append(end - start)
		gpu_function_time.append(end_all - start_all)