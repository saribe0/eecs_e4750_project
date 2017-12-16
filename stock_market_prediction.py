#!/usr/bin/env python
####################################################################################################
## Created 10/27/2017 by Sam Beaulieu
##
##
####################################################################################################

# Specifies GPU/CPU calculations will be prepformed
GPU = True

if GPU:
	import pyopencl as cl
else:
	from bs4 import BeautifulSoup as BS
	import requests
import os
import datetime
import time
import logging
import sys, getopt
import numpy as np
import struct
import binascii
import math
from scipy.stats import norm
from matplotlib import pyplot as plt
from matplotlib import dates as plt_dates
import re

########### Global Variables and Configurations ###########
# Global Constants
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

# Timing variables for CPU and GPU
predict_cpu_kernel_time = []
predict_gpu_kernel_time = []
predict_cpu_function_time = []
predict_gpu_function_time = []

analysis_cpu_kernel_time = []
analysis_gpu_kernel_time = []
analysis_cpu_function_time = []
analysis_gpu_function_time = []

update_cpu_kernel_time = []
update_gpu_kernel_time = []
update_cpu_function_time = []
update_gpu_function_time = []

prediction_outputs_gpu = []
prediction_outputs_cpu = []

analysis_outputs_gpu = []
analysis_outputs_cpu = []

update_outputs_gpu = []
update_outputs_cpu = []

# Global Configurations
if not os.path.exists('./data/'):
	os.makedirs('./data/')
	os.makedirs('./data/articles/')
elif not os.path.exists('./data/articles/'):
	os.makedirs('./data/articles/')

if not os.path.exists('./output/'):
	os.makedirs('./output/')


# Get the correct opencl platform (taken from instructor sample code)
if GPU:
	NAME = 'NVIDIA CUDA'
	platforms = cl.get_platforms()
	devs = None
	for platform in platforms:
		if platform.name == NAME:
			devs = platform.get_devices()

	# Set up command queue
	ctx = cl.Context(devs)
	queue = cl.CommandQueue(ctx)


analysis_kernel = """

__kernel void analyze_weights_1(__global int* words_by_letter, __global int* num_words_by_letter, volatile __global float* out_stats, int max_words_per_letter) {
	
	// Get the word for the current work-item to focus on

	unsigned int word_id = get_global_id(0);
	unsigned int letter_id = get_global_id(1);

	// Prepare the indices for the reduction

	unsigned int work_item_id = get_local_id(0);
	unsigned int work_group_x = get_group_id(0);
	unsigned int work_group_y = get_group_id(1);
	unsigned int group_size = get_local_size(0);

	// Create local arrays to store the data in
	// The value 512 must be equal to the group size. This is the only value in the kernel
		// - that must be updated when the group size is changed.

	volatile __local float local_out[6 * 512];

	// Get the weight and frequency for the current thread

	float weight = 0;	
	int frequency = 0;
	if (word_id < num_words_by_letter[letter_id]) {
		frequency = words_by_letter[letter_id * max_words_per_letter * 7 + word_id * 7 + 5];
		weight = (float)words_by_letter[letter_id * max_words_per_letter * 7 + word_id * 7 + 6] / frequency;
	}

	// Each thread loads initial data into its own space in local memory

	local_out[group_size * 0 + work_item_id] =  weight;
	local_out[group_size * 1 + work_item_id] =  frequency * weight;
	local_out[group_size * 2 + work_item_id] =  weight;
	local_out[group_size * 3 + work_item_id] =  (word_id < num_words_by_letter[letter_id]) ? weight : 1;
	local_out[group_size * 4 + work_item_id] =  (word_id < num_words_by_letter[letter_id]) ? 1 : 0;
	local_out[group_size * 5 + work_item_id] =  frequency;


	// Preform reduction

	for (unsigned int stride = 1; stride < group_size; stride *= 2) {

		barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

		if ( work_item_id % stride == 0 && work_item_id < group_size / 2) {

			local_out[group_size * 0 + work_item_id * 2] +=  local_out[group_size * 0 + work_item_id * 2 + stride];
			local_out[group_size * 1 + work_item_id * 2] +=  local_out[group_size * 1 + work_item_id * 2 + stride];
			local_out[group_size * 2 + work_item_id * 2] =   local_out[group_size * 2 + work_item_id * 2 + stride] > local_out[group_size * 2 + work_item_id * 2] ? local_out[group_size * 2 + work_item_id * 2 + stride] : local_out[group_size * 2 + work_item_id * 2];
			local_out[group_size * 3 + work_item_id * 2] =   local_out[group_size * 3 + work_item_id * 2 + stride] < local_out[group_size * 3 + work_item_id * 2] ? local_out[group_size * 3 + work_item_id * 2 + stride] : local_out[group_size * 3 + work_item_id * 2];
			local_out[group_size * 4 + work_item_id * 2] +=  local_out[group_size * 4 + work_item_id * 2 + stride];
			local_out[group_size * 5 + work_item_id * 2] +=  local_out[group_size * 5 + work_item_id * 2 + stride];
		}
	}

	// Synchronize work items again to ensure all are done reduction before writeback

	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

	// Writeback to output

	if (work_item_id < 6) {
		out_stats[ (work_group_y * 5 + work_group_x) * 6 + work_item_id] = local_out[group_size * work_item_id];
	}
}

__kernel void analyze_weights_2(__global int* words_by_letter, __global int* num_words_by_letter, volatile __global float* out_stats, int max_words_per_letter, float average, float weighted_average) {
	
	// Get the word for the current work-item to focus on

	unsigned int word_id = get_global_id(0);
	unsigned int letter_id = get_global_id(1);

	// Prepare the indices for the reduction

	unsigned int work_item_id = get_local_id(0);
	unsigned int work_group_x = get_group_id(0);
	unsigned int work_group_y = get_group_id(1);
	unsigned int group_size = get_local_size(0);

	// Create local arrays to store the data in
	// The value 512 must be equal to the group size. This is the only value in the kernel
	// - that must be updated when the group size is changed.

	volatile __local float local_out[2 * 512];

	// Get the weight and frequency for the current thread

	float weight = 0;	
	int frequency = 0;
	if (word_id < num_words_by_letter[letter_id]) {
		frequency = words_by_letter[letter_id * max_words_per_letter * 7 + word_id * 7 + 5];
		weight = (float)words_by_letter[letter_id * max_words_per_letter * 7 + word_id * 7 + 6] / frequency;
	}

	// Each thread loads initial data into its own space in local memory - initialize the normal average to zero first to avoid incorrect initialization for out of bounds values

	local_out[group_size * 0 + work_item_id] = 0;

	if (word_id < num_words_by_letter[letter_id])
		local_out[group_size * 0 + work_item_id] =  (weight - average) * (weight - average);

	local_out[group_size * 1 + work_item_id] =  (weight - weighted_average) * (weight - weighted_average) * frequency;

	// Preform reduction

	for (unsigned int stride = 1; stride < group_size; stride *= 2) {

		barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

		if ( work_item_id % stride == 0 && work_item_id < group_size / 2) {

			local_out[group_size * 0 + work_item_id * 2] +=  local_out[group_size * 0 + work_item_id * 2 + stride];
			local_out[group_size * 1 + work_item_id * 2] +=  local_out[group_size * 1 + work_item_id * 2 + stride];
		}
	}

	// Synchronize work items again to ensure all are done reduction before writeback

	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

	// Writeback to output

	if (work_item_id < 2) {
		out_stats[ (work_group_y * 5 + work_group_x) * 2 + work_item_id] = local_out[group_size * work_item_id];
	}
}

"""

predict_kernel = """

__kernel void predict(__global char* words, __global int* weights, __global char* weights_char, __global int* num_weights_letter, volatile __global float* out_weights, int max_words_per_letter, int word_max) {
	
	// Get the word for the current work-item to focus on

	unsigned int word_id = get_global_id(0);

	unsigned int word_index = word_id * 16;

	// Get the weight forthe current work item to focus on

	unsigned int weight_id = get_global_id(1);

	if ( word_id < word_max ) 
	{
		unsigned int letter_index;
		if (words[word_index] > 96) 
			letter_index = words[word_index] - 'a';
		else
			letter_index = words[word_index] - 'A';

		unsigned int weight_index = letter_index * max_words_per_letter * 28 + weight_id * 28;
		unsigned int weight_index_int = letter_index * max_words_per_letter * 7 + weight_id * 7;
		unsigned int weight_max = letter_index * max_words_per_letter * 28 + num_weights_letter[letter_index] * 28;

		// Get the inputs and outputs to be compared

		char word_0 = words[word_index + 0];
	 	char word_1 = words[word_index + 1];
		char word_2 = words[word_index + 2];
		char word_3 = words[word_index + 3];
		char word_4 = words[word_index + 4];
	 	char word_5 = words[word_index + 5];
		char word_6 = words[word_index + 6];
		char word_7 = words[word_index + 7];
		char word_8 = words[word_index + 8];
	 	char word_9 = words[word_index + 9];
		char word_10 = words[word_index + 10];
		char word_11 = words[word_index + 11];
		char word_12 = words[word_index + 12];
	 	char word_13 = words[word_index + 13];
		char word_14 = words[word_index + 14];
		char word_15 = words[word_index + 15];

		if ( weight_index < weight_max ) 
		{
			char word_w_0 = weights_char[weight_index + 0];
			char word_w_1 = weights_char[weight_index + 1];
			char word_w_2 = weights_char[weight_index + 2];
			char word_w_3 = weights_char[weight_index + 3];
			char word_w_4 = weights_char[weight_index + 4];
			char word_w_5 = weights_char[weight_index + 5];
			char word_w_6 = weights_char[weight_index + 6];
			char word_w_7 = weights_char[weight_index + 7];
			char word_w_8 = weights_char[weight_index + 8];
			char word_w_9 = weights_char[weight_index + 9];
			char word_w_10 = weights_char[weight_index + 10];
			char word_w_11 = weights_char[weight_index + 11];
			char word_w_12 = weights_char[weight_index + 12];
			char word_w_13 = weights_char[weight_index + 13];
			char word_w_14 = weights_char[weight_index + 14];
			char word_w_15 = weights_char[weight_index + 15];

			// Compare them and update the output if necessary

			if ( word_0 == word_w_0 && word_1 == word_w_1 && word_2 == word_w_2 && word_3 == word_w_3 &&
				 word_4 == word_w_4 && word_5 == word_w_5 && word_6 == word_w_6 && word_7 == word_w_7 &&
				 word_8 == word_w_8 && word_9 == word_w_9 && word_10 == word_w_10 && word_11 == word_w_11 &&
				 word_12 == word_w_12 && word_13 == word_w_13 && word_14 == word_w_14 && word_15 == word_w_15)
			{
				int frequency = weights[weight_index_int + 5];
				float weight = (float)weights[weight_index_int + 6] / frequency;

				out_weights[word_id] = weight;
			}
		}
	}
}

__kernel void predict_bayes(__global char* words, __global int* weights, __global char* weights_char, __global int* num_weights_letter, volatile __global float* out_probabilities_up, volatile __global float* out_probabilities_down, int total_weights_up, int total_weights_down, int total_weights, float c, int max_words_per_letter, int word_max) {
	
	// Get the word for the current work-item to focus on

	unsigned int word_id = get_global_id(0);

	unsigned int word_index = word_id * 16;

	// Get the weight forthe current work item to focus on

	unsigned int weight_id = get_global_id(1);

	if ( word_id < word_max ) 
	{
		unsigned int letter_index;
		if (words[word_index] > 96) 
			letter_index = words[word_index] - 'a';
		else
			letter_index = words[word_index] - 'A';

		unsigned int weight_index = letter_index * max_words_per_letter * 28 + weight_id * 28;
		unsigned int weight_index_int = letter_index * max_words_per_letter * 7 + weight_id * 7;
		unsigned int weight_max = letter_index * max_words_per_letter * 28 + num_weights_letter[letter_index] * 28;

		// Get the inputs and outputs to be compared

		char word_0 = words[word_index + 0];
	 	char word_1 = words[word_index + 1];
		char word_2 = words[word_index + 2];
		char word_3 = words[word_index + 3];
		char word_4 = words[word_index + 4];
	 	char word_5 = words[word_index + 5];
		char word_6 = words[word_index + 6];
		char word_7 = words[word_index + 7];
		char word_8 = words[word_index + 8];
	 	char word_9 = words[word_index + 9];
		char word_10 = words[word_index + 10];
		char word_11 = words[word_index + 11];
		char word_12 = words[word_index + 12];
	 	char word_13 = words[word_index + 13];
		char word_14 = words[word_index + 14];
		char word_15 = words[word_index + 15];

		if ( weight_index < weight_max ) 
		{
			char word_w_0 = weights_char[weight_index + 0];
			char word_w_1 = weights_char[weight_index + 1];
			char word_w_2 = weights_char[weight_index + 2];
			char word_w_3 = weights_char[weight_index + 3];
			char word_w_4 = weights_char[weight_index + 4];
			char word_w_5 = weights_char[weight_index + 5];
			char word_w_6 = weights_char[weight_index + 6];
			char word_w_7 = weights_char[weight_index + 7];
			char word_w_8 = weights_char[weight_index + 8];
			char word_w_9 = weights_char[weight_index + 9];
			char word_w_10 = weights_char[weight_index + 10];
			char word_w_11 = weights_char[weight_index + 11];
			char word_w_12 = weights_char[weight_index + 12];
			char word_w_13 = weights_char[weight_index + 13];
			char word_w_14 = weights_char[weight_index + 14];
			char word_w_15 = weights_char[weight_index + 15];

			// Compare them and update the output if necessary

			if ( word_0 == word_w_0 && word_1 == word_w_1 && word_2 == word_w_2 && word_3 == word_w_3 &&
				 word_4 == word_w_4 && word_5 == word_w_5 && word_6 == word_w_6 && word_7 == word_w_7 &&
				 word_8 == word_w_8 && word_9 == word_w_9 && word_10 == word_w_10 && word_11 == word_w_11 &&
				 word_12 == word_w_12 && word_13 == word_w_13 && word_14 == word_w_14 && word_15 == word_w_15)
			{
				float up = ((float)weights[weight_index_int + 5] + c) / (total_weights_up + c * total_weights);
				float down = ((float)weights[weight_index_int + 6] + c) / (total_weights_down + c * total_weights);

				out_probabilities_up[word_id] = up;
				out_probabilities_down[word_id] = down;
			}
		}
	}
}
"""

update_kernel = """

__kernel void update(__global char* words, volatile __global int* word_bitmap, volatile __global int* weights, __global char* weights_char, __global int* num_weights_letter, int max_words_per_letter, int word_max, int direction) {
	
	// Get the word for the current work-item to focus on

	unsigned int word_id = get_global_id(0);

	unsigned int word_index = word_id * 16;

	// Get the weight for the current work item to focus on

	unsigned int weight_id = get_global_id(1);

	if ( word_id < word_max ) 
	{
		unsigned int letter_index;
		if (words[word_index] > 96) 
			letter_index = words[word_index] - 'a';
		else
			letter_index = words[word_index] - 'A';

		unsigned int weight_index = letter_index * max_words_per_letter * 28 + weight_id * 28;
		unsigned int weight_index_int = letter_index * max_words_per_letter * 7 + weight_id * 7;
		unsigned int weight_max = letter_index * max_words_per_letter * 28 + num_weights_letter[letter_index] * 28;

		// Get the inputs and outputs to be compared

		char word_0 = words[word_index + 0];
	 	char word_1 = words[word_index + 1];
		char word_2 = words[word_index + 2];
		char word_3 = words[word_index + 3];
		char word_4 = words[word_index + 4];
	 	char word_5 = words[word_index + 5];
		char word_6 = words[word_index + 6];
		char word_7 = words[word_index + 7];
		char word_8 = words[word_index + 8];
	 	char word_9 = words[word_index + 9];
		char word_10 = words[word_index + 10];
		char word_11 = words[word_index + 11];
		char word_12 = words[word_index + 12];
	 	char word_13 = words[word_index + 13];
		char word_14 = words[word_index + 14];
		char word_15 = words[word_index + 15];

		if ( weight_index < weight_max ) 
		{
			char word_w_0 = weights_char[weight_index + 0];
			char word_w_1 = weights_char[weight_index + 1];
			char word_w_2 = weights_char[weight_index + 2];
			char word_w_3 = weights_char[weight_index + 3];
			char word_w_4 = weights_char[weight_index + 4];
			char word_w_5 = weights_char[weight_index + 5];
			char word_w_6 = weights_char[weight_index + 6];
			char word_w_7 = weights_char[weight_index + 7];
			char word_w_8 = weights_char[weight_index + 8];
			char word_w_9 = weights_char[weight_index + 9];
			char word_w_10 = weights_char[weight_index + 10];
			char word_w_11 = weights_char[weight_index + 11];
			char word_w_12 = weights_char[weight_index + 12];
			char word_w_13 = weights_char[weight_index + 13];
			char word_w_14 = weights_char[weight_index + 14];
			char word_w_15 = weights_char[weight_index + 15];

			// Compare them and update the output if necessary

			if ( word_0 == word_w_0 && word_1 == word_w_1 && word_2 == word_w_2 && word_3 == word_w_3 &&
				 word_4 == word_w_4 && word_5 == word_w_5 && word_6 == word_w_6 && word_7 == word_w_7 &&
				 word_8 == word_w_8 && word_9 == word_w_9 && word_10 == word_w_10 && word_11 == word_w_11 &&
				 word_12 == word_w_12 && word_13 == word_w_13 && word_14 == word_w_14 && word_15 == word_w_15)
			{
				if (direction == 1)
				{
					atomic_inc(weights + weight_index_int + 5);
					atomic_inc(weights + weight_index_int + 6);
				}
				else
				{
					atomic_inc(weights + weight_index_int + 5);
				}

				word_bitmap[word_id] = 1;
			}
		}
	}
}

__kernel void update_bayes(__global char* words, volatile __global int* word_bitmap, volatile __global int* weights, __global char* weights_char, __global int* num_weights_letter, int max_words_per_letter, int word_max, int direction) {
	
	// Get the word for the current work-item to focus on

	unsigned int word_id = get_global_id(0);

	unsigned int word_index = word_id * 16;

	// Get the weight for the current work item to focus on

	unsigned int weight_id = get_global_id(1);

	if ( word_id < word_max ) 
	{
		unsigned int letter_index;
		if (words[word_index] > 96) 
			letter_index = words[word_index] - 'a';
		else
			letter_index = words[word_index] - 'A';

		unsigned int weight_index = letter_index * max_words_per_letter * 28 + weight_id * 28;
		unsigned int weight_index_int = letter_index * max_words_per_letter * 7 + weight_id * 7;
		unsigned int weight_max = letter_index * max_words_per_letter * 28 + num_weights_letter[letter_index] * 28;

		// Get the inputs and outputs to be compared

		char word_0 = words[word_index + 0];
	 	char word_1 = words[word_index + 1];
		char word_2 = words[word_index + 2];
		char word_3 = words[word_index + 3];
		char word_4 = words[word_index + 4];
	 	char word_5 = words[word_index + 5];
		char word_6 = words[word_index + 6];
		char word_7 = words[word_index + 7];
		char word_8 = words[word_index + 8];
	 	char word_9 = words[word_index + 9];
		char word_10 = words[word_index + 10];
		char word_11 = words[word_index + 11];
		char word_12 = words[word_index + 12];
	 	char word_13 = words[word_index + 13];
		char word_14 = words[word_index + 14];
		char word_15 = words[word_index + 15];

		if ( weight_index < weight_max ) 
		{
			char word_w_0 = weights_char[weight_index + 0];
			char word_w_1 = weights_char[weight_index + 1];
			char word_w_2 = weights_char[weight_index + 2];
			char word_w_3 = weights_char[weight_index + 3];
			char word_w_4 = weights_char[weight_index + 4];
			char word_w_5 = weights_char[weight_index + 5];
			char word_w_6 = weights_char[weight_index + 6];
			char word_w_7 = weights_char[weight_index + 7];
			char word_w_8 = weights_char[weight_index + 8];
			char word_w_9 = weights_char[weight_index + 9];
			char word_w_10 = weights_char[weight_index + 10];
			char word_w_11 = weights_char[weight_index + 11];
			char word_w_12 = weights_char[weight_index + 12];
			char word_w_13 = weights_char[weight_index + 13];
			char word_w_14 = weights_char[weight_index + 14];
			char word_w_15 = weights_char[weight_index + 15];

			// Compare them and update the output if necessary

			if ( word_0 == word_w_0 && word_1 == word_w_1 && word_2 == word_w_2 && word_3 == word_w_3 &&
				 word_4 == word_w_4 && word_5 == word_w_5 && word_6 == word_w_6 && word_7 == word_w_7 &&
				 word_8 == word_w_8 && word_9 == word_w_9 && word_10 == word_w_10 && word_11 == word_w_11 &&
				 word_12 == word_w_12 && word_13 == word_w_13 && word_14 == word_w_14 && word_15 == word_w_15)
			{
				if (direction == 1)
				{
					atomic_inc(weights + weight_index_int + 5);
				}
				else
				{
					atomic_inc(weights + weight_index_int + 6);
				}

				word_bitmap[word_id] = 1;
			}
		}
	}
}
"""


# Build the kernel
if GPU:
	analysis_prg = cl.Program(ctx, analysis_kernel).build()
	predict_prg = cl.Program(ctx, predict_kernel).build()
	update_prg = cl.Program(ctx, update_kernel).build()


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
			if GPU: 
				struct.pack_into('16s f i i', words_by_letter[letter_index], num_words_by_letter[letter_index]*28, data[1].decode('ascii', 'ignore').encode('utf-8'), float(data[2]), int(data[3]), int(data[4]))
			else:
				struct.pack_into('16s f i i', words_by_letter[letter_index], num_words_by_letter[letter_index]*28, data[1].encode('utf-8'), float(data[2]), int(data[3]), int(data[4]))
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

		start_all = time.time()
		cpu = 0

		# For each stock, iterate through the articles
		for articles in stock_data[ticker]:
		
			# Get the text (ignore link)
			text = articles[1]

			# Get an array of words with two or more characters for the text
			words_in_text = re.compile('[A-Za-z][A-Za-z][A-Za-z]+').findall(text)

			start = time.time()

			# Update each word
			for each_word in words_in_text:
				update_word(ticker, option, each_word, day)

			end = time.time()
			cpu += end - start

		end_all = time.time()

		update_cpu_kernel_time.append(cpu)
		update_cpu_function_time.append(end_all - start_all)

	# Add all the weights to the cpu weight array
	# NOTE: This is ONLY for comparison of outputs between GPU and CPU. It has nothing to do with actual computation
	for letter in range(0, 26):

		letter_words = words_by_letter[letter]

		for each_word in range(0, num_words_by_letter[letter]):

			test_data = struct.unpack_from('16s f i i', letter_words, each_word * 28)

			update_outputs_cpu.append(test_data[2])
			update_outputs_cpu.append(test_data[3])


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
		if temp_word[:len(temp_word.split('\0', 1)[0])] == word[:len(word.split('\0', 1)[0])]:
			
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
					weight = test_data[1]
					extra1 = test_data[2] + 1
					extra2 = test_data[3] + 1

				else:
					weight = test_data[1]
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
				weight = 0
				extra1 = 1
				extra2 = 1
			else:
				weight = 0
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


def update_all_word_weights_gpu(option, day):
	'''
	| |
	|_|
	|_|_______________
	|3|_0_|__1__|__2__
	|_|         |'hey'|
	| |         | 0.8  |
	| |         | 80  |

	'''

	logging.info('Updating word weights for: ' + day + ' with option: ' + option + ' using the gpu')
	print('Updating word weights with gpu')

	# If the weighting arrays are empty, create them 
	if len(words_by_letter) == 0 or words_by_letter == 0:

		logging.debug('- Could not find data structure for word weights so creating it')

		# For each letter, add a MAX_WORDS_PER_LETTER word array of 28 characters each with the last 8 characters for a float (weight of the word) and int (number of occurences)
		# - abcdefghijklmnopq0.32####
		for letters in range(0, 26):
			letter_words = bytearray(28*MAX_WORDS_PER_LETTER)
			words_by_letter.append(letter_words)
			num_words_by_letter.append(0)

	for ticker in STOCK_TAGS:
		
		logging.debug('- Updating word weights for: ' + ticker)

		if ticker not in stock_data:
			logging.warning('- Could not find articles loaded for ' + ticker)
			continue

		start_all = time.time()

		# Determine the direction the stock took on the day in question
		change = stock_prices[ticker][day][1] - stock_prices[ticker][day][0]
		if change > 0:
			direction = 1
		else:
			direction = 0

		words_in_text = []

		# For each stock, iterate through the articles
		for articles in stock_data[ticker]:
	
			# Get the text (ignore link)
			text = articles[1]

			# Get an array of words with three or more characters for the text
			words_in_text += re.compile('[A-Za-z][A-Za-z][A-Za-z]+').findall(text)

		# Store the words to be read by the kernel
		word_data = bytearray(len(words_in_text)*16)
		for ii, word in enumerate(words_in_text):
			struct.pack_into('16s', word_data, ii * 16, word.lower().encode('utf-8'))

		# Prepare the bitmap output buffer
		word_bitmap = np.zeros((len(words_in_text), ), dtype = np.uint32)

		# Create the buffers for the GPU
		mf = cl.mem_flags
		word_data_buff = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = word_data)
		word_bitmap_buff = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf = word_bitmap)
		weights_buff = cl.Buffer(ctx, mf.COPY_HOST_PTR, hostbuf = np.asarray(words_by_letter))
		weights_char_buff = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = np.asarray(words_by_letter))
		num_weights_buff = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = np.asarray(num_words_by_letter, dtype = np.int32))

		# Determine the grid size
		groups, extra = divmod(len(words_in_text), 256)
		grid = ((groups + (extra>0))*256, 2560)

		start = time.time()

		# Call the right kernel
		if option == 'opt1':
			update_prg.update(queue, grid, None, word_data_buff, word_bitmap_buff, weights_buff, weights_char_buff, num_weights_buff, np.uint32(MAX_WORDS_PER_LETTER), np.uint32(len(words_in_text)), np.uint32(direction))
		elif option == 'opt2':
			update_prg.update_bayes(queue, grid, None, word_data_buff, word_bitmap_buff, weights_buff, weights_char_buff, num_weights_buff, np.uint32(MAX_WORDS_PER_LETTER), np.uint32(len(words_in_text)), np.uint32(direction))
		else:
			continue

		# Collect the output
		cl.enqueue_copy(queue, word_bitmap, word_bitmap_buff)
		
		for ii in range(0, 26):
			cl.enqueue_copy(queue, words_by_letter[ii], weights_buff, device_offset = 2500 * 28 * ii)

		# Update the words that couldn't be updated by the kernel
		for ii, bit in enumerate(word_bitmap):

			# Only update words that couldn't be updated by the kernel
			# - A full update is still needed (full search) because if one word was not found, then every other identical one will not be found either
			# - we only want it added once so the other times it will have to be searched.
			if bit == 0:
				update_word(ticker, option, words_in_text[ii], day)

		end = time.time()
		end_all = time.time()

		update_gpu_kernel_time.append(end - start)
		update_gpu_function_time.append(end_all - start_all)


	# Add all the weights to the gpu weight array
	# NOTE: This is ONLY for comparison of outputs between GPU and CPU. It has nothing to do with actual computation
	for letter in range(0, 26):

		letter_words = words_by_letter[letter]

		for each_word in range(0, num_words_by_letter[letter]):

			test_data = struct.unpack_from('16s f i i', letter_words, each_word * 28)

			update_outputs_gpu.append(test_data[2])
			update_outputs_gpu.append(test_data[3])


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

	start_all = time.time()

	cpu = 0
	start = time.time()

	# Iterate through each letter
	for letter in range(0, 26):

		logging.debug('- Analyzing word weights for letter: ' + chr(letter+97))

		# Iterate through each word for that letter
		for elements in range(0, num_words_by_letter[letter]):

			# For each word, unpack the word from the buffer
			raw_data = struct.unpack_from('16s f i i', words_by_letter[letter], elements * 28)
			weight = float(raw_data[3]) / raw_data[2]

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

	end = time.time()
	cpu += end - start

	# Once all weights have been iterated through, calculate the average
	weight_average = weight_sum / weight_count
	weight_average_o = weight_sum_o / weight_count_o

	# Calculate the standard deviation
	start = time.time()

	running_sum = 0
	for weights in weights_all:
		running_sum += ((weights - weight_average) * (weights - weight_average))

	running_sum_o = 0
	for weights in weights_all_o:
		running_sum_o += ((weights - weight_average_o) * (weights - weight_average_o))

	end = time.time()
	cpu += end - start

	weight_stdev = math.sqrt(running_sum / (weight_count - 1))
	weight_stdev_o = math.sqrt(running_sum_o / (weight_count_o - 1))

	end_all = time.time()

	analysis_cpu_kernel_time.append(cpu)
	analysis_cpu_function_time.append(end_all - start_all)

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

	analysis_outputs_cpu.append(weight_average)
	analysis_outputs_cpu.append(weight_stdev)
	analysis_outputs_cpu.append(weight_sum)
	analysis_outputs_cpu.append(weight_count)
	analysis_outputs_cpu.append(weight_max)
	analysis_outputs_cpu.append(weight_min)
	analysis_outputs_cpu.append(weight_average_o)
	analysis_outputs_cpu.append(weight_stdev_o)
	analysis_outputs_cpu.append(weight_sum_o)
	analysis_outputs_cpu.append(weight_count_o)

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

	start_all = time.time()

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

	# Call the kernel
	gpu = 0
	start = time.time()
	analysis_prg.analyze_weights_1(queue, (2560, 26), (512, 1), words_by_letter_buff, num_words_by_letter_buff, out_stats_buff, np.uint32(MAX_WORDS_PER_LETTER))

	# Pull results from the GPU
	cl.enqueue_copy(queue, out_stats, out_stats_buff)

	# Set all the global variabels for the function
	for each in out_stats:
		weight_sum += each[0]
		weight_sum_o += each[1]
		weight_max = each[2] if each[2] > weight_max else weight_max
		weight_min = each[3] if each[3] < weight_min else weight_min
		weight_count += each[4]
		weight_count_o += each[5]

	end = time.time()
	gpu += end - start

	# Calculate the averages
	weight_average = weight_sum / weight_count
	weight_average_o = weight_sum_o / weight_count_o

	# Prepare the GPU buffers for the standard deviation calculation
	# [sum of avg-weight, weighted sum of avg-weight]
	mf = cl.mem_flags
	out_std_sum = np.zeros((130,2), dtype = np.float32)
	out_std_sum_buff = cl.Buffer(ctx, mf.WRITE_ONLY, out_std_sum.nbytes)
	words_by_letter_buff = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = np.asarray(words_by_letter))
	num_words_by_letter_buff = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = np.asarray(num_words_by_letter, dtype = np.uint32))

	# Call the kernel
	start = time.time()
	analysis_prg.analyze_weights_2(queue, (2560, 26), (512, 1), words_by_letter_buff, num_words_by_letter_buff, out_std_sum_buff, np.uint32(MAX_WORDS_PER_LETTER), np.float32(weight_average), np.float32(weight_average_o))
	end = time.time()
	gpu += end - start

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

	end_all = time.time()

	analysis_gpu_kernel_time.append(gpu)
	analysis_gpu_function_time.append(end_all - start_all)

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

	analysis_outputs_gpu.append(weight_average)
	analysis_outputs_gpu.append(weight_stdev)
	analysis_outputs_gpu.append(weight_sum)
	analysis_outputs_gpu.append(weight_count)
	analysis_outputs_gpu.append(weight_max)
	analysis_outputs_gpu.append(weight_min)
	analysis_outputs_gpu.append(weight_average_o)
	analysis_outputs_gpu.append(weight_stdev_o)
	analysis_outputs_gpu.append(weight_sum_o)
	analysis_outputs_gpu.append(weight_count_o)

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
			return np.float32(np.float32(test_data[3]) / test_data[2])

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

	global weight_average_o
	global weight_stdev_o
	global weight_sum_o
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
				
				prediction_outputs_cpu.append(weight)

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

		predict_cpu_kernel_time.append(cpu)
		predict_cpu_function_time.append(end_all - start_all)

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

		if stock_rating_p1 > weight_average:
			all_predictions[tickers].append(1)
		elif stock_rating_p1 < weight_average:
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

		if stock_rating_p1 > weight_average_o:
			all_predictions[tickers].append(1)
		elif stock_rating_p1 < weight_average_o:
			all_predictions[tickers].append(-1)
		else:
			all_predictions[tickers].append(0)

	write_predictions_to_file_and_print(day, all_predictions, all_std_devs, all_probabilities, all_raw_ratings)


def predict_movement7(day):

	global total_up
	global total_down
	global total_words_up
	global total_words_down
	global c

	logging.info('Prediction stock movements with method 7 -> Naive Bayes Classifier')

	print('')
	print('BAYES PREDICTIONS ')

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

				up, down = get_word_probability_given_label(words, c)
						
				if up != None and down != None:
					stock_rating_up += math.log(up)
					stock_rating_down += math.log(down)

			end = time.time()
			cpu += end - start

		end_all = time.time()

		predict_cpu_kernel_time.append(cpu)
		predict_cpu_function_time.append(end_all - start_all)

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
		predict_prg.predict(queue, grid, None, word_data_buff, weights_buff, weights_char_buff, num_weights_buff, out_weights_buff, np.uint32(MAX_WORDS_PER_LETTER), np.uint32(len(words_in_text)))

		# Collect the output
		cl.enqueue_copy(queue, out_weights, out_weights_buff)

		stock_rating_sum_p1 = 0

		stock_rating_sum_p2 = 0
		stock_rating_cnt_p2 = 0

		stock_rating_sum_p5 = 0
		stock_rating_cnt_p5 = 0

		for w in out_weights:

			prediction_outputs_gpu.append(w)

			# Prediction Method 1, 3, 4, 6
			stock_rating_sum_p1 += w

			# Prediction Method 2
			if w > weight_stdev + weight_average or w < weight_average - weight_stdev:
				stock_rating_sum_p2 += w
				stock_rating_cnt_p2 += 1

			# Prediction Method 5
			if w > weight_stdev_o + weight_average_o or w < weight_average_o - weight_stdev_o:
				stock_rating_sum_p5 += w
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

		predict_gpu_kernel_time.append(end - start)
		predict_gpu_function_time.append(end_all - start_all)

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

		if stock_rating_p1 > weight_average:
			all_predictions[tickers].append(1)
		elif stock_rating_p1 < weight_average:
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

		if stock_rating_p1 > weight_average_o:
			all_predictions[tickers].append(1)
		elif stock_rating_p1 < weight_average_o:
			all_predictions[tickers].append(-1)
		else:
			all_predictions[tickers].append(0)

	write_predictions_to_file_and_print(day, all_predictions, all_std_devs, all_probabilities, all_raw_ratings)


def predict_movement7_gpu(day):

	global total_up
	global total_down
	global total_words_up
	global total_words_down
	global c

	logging.info('Prediction stock movements with method 7 -> Naive Bayes Classifier with the GPU')

	print('')
	print('BAYES PREDICTIONS WITH GPU ')

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

		# Calculate the total number of words
		num_words = 0
		for ii in range(0, 26):
			num_words += num_words_by_letter[ii]

		# Prepare an output buffer
		out_up = np.zeros((len(words_in_text), ), dtype = np.float32)
		out_down = np.zeros((len(words_in_text), ), dtype = np.float32)

		# Create the buffers for the GPU
		mf = cl.mem_flags
		word_data_buff = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = word_data)
		weights_buff = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = np.asarray(words_by_letter))
		weights_char_buff = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = np.asarray(words_by_letter))
		num_weights_buff = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = np.asarray(num_words_by_letter, dtype = np.int32))
		out_up_buff = cl.Buffer(ctx, mf.WRITE_ONLY, out_up.nbytes)
		out_down_buff = cl.Buffer(ctx, mf.WRITE_ONLY, out_down.nbytes)

		# Determine the grid size
		groups, extra = divmod(len(words_in_text), 256)
		grid = ((groups + (extra>0))*256, 2560)

		start = time.time()

		# Call the kernel
		predict_prg.predict_bayes(queue, grid, None, word_data_buff, weights_buff, weights_char_buff, num_weights_buff, out_up_buff, out_down_buff, np.uint32(total_words_up), np.uint32(total_words_down), np.uint32(num_words), np.float32(c), np.uint32(MAX_WORDS_PER_LETTER), np.uint32(len(words_in_text)))

		# Collect the output
		cl.enqueue_copy(queue, out_up, out_up_buff)
		cl.enqueue_copy(queue, out_down, out_down_buff)

		# Get the prediction for the stock
		for w in out_up:
			if w > 0:
				stock_rating_up += math.log(w)

		for w in out_down:
			if w > 0:
				stock_rating_down += math.log(w)

		end = time.time()
		end_all = time.time()

		predict_gpu_kernel_time.append(end - start)
		predict_gpu_function_time.append(end_all - start_all)

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


def write_predictions_to_file_and_print(day, all_predictions, all_std_devs, all_probabilities, all_raw_ratings):

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

	# Iterate through the predictions, print them, and write them to files
	for ii in range(0, 6):

		# Open file to store todays predictions in
		file = open('./output/prediction' + str(ii) +'-' + day + '.txt', 'w')

		# Print the header info and open the file
		# When less than 3, uses normal weight analysis 
		if ii < 3:
			print('')
			print('PREDICTIONS FOR METHOD ' + str(ii + 1) + ' BASED ON:')
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
			print('')
			print('PREDICTIONS FOR METHOD ' + str(ii + 1) + ' BASED ON:')
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

	prediction_results = [[], [], [], [], [], [], []]

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

		# Get the prediction method type
		# 1-6 use the very basic weighting algorithim, 7 uses Naive Bayes
		try: 
			if (filename[10] == '-'):
				prediction_type = 1
			else:
				prediction_type = int(filename[10])
		except:
			continue

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

		# Add the values to the date set and prediction lists
		if num_correct + num_wrong != 0:
			date_parts = prediction_date.split('-')
			test_date = datetime.date(int(date_parts[2]), int(date_parts[0]), int(date_parts[1]))
			prediction_results[prediction_type - 1].append([ float(num_correct) / (num_wrong + num_correct) , test_date ])
	
	# Plot everything
	plt.figure()
	plt.gcf()
	legend = []
	for ii, types in enumerate(prediction_results):
		dates = []
		vals = []

		# Split the dates and values
		for each in types:
			dates.append(each[1])
			vals.append(each[0])

		# Order the two pairs and create the plot
		if(len(dates) > 0):
			new_dates, new_vals = zip(*sorted(zip(dates, vals)))
			plt.plot(new_dates, new_vals)

			legend.append('Prediction  ' + str(ii + 1))
	
	plt.legend(legend, loc = 'upper left')
	plt.xlabel('Date')
	plt.ylabel('Accuracy')
	plt.savefig('prediction_accuracy.png')



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
			if GPU:
				if not analyze_weights() or not analyze_weights_gpu():
					print('Error: Unable to analyze weights')
					sys.exit(-1)
			else:
				if not analyze_weights():
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

		if weight_opt == 'opt1':
			if GPU:
				if not analyze_weights() or not analyze_weights_gpu():
					print('Error: Unable to analyze weights')
					sys.exit(-1)

				predict_movement(specified_day)
				predict_movement_gpu(specified_day)
			else:
				if not analyze_weights():
					print('Error: Unable to analyze weights')
					sys.exit(-1)

				predict_movement(specified_day)
		
		elif weight_opt == 'opt2':
			if GPU:
				predict_movement7(specified_day)
				predict_movement7_gpu(specified_day)
			else:
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

		load_stock_prices()

		# Run everything on the CPU
		load_all_word_weights(weight_opt)
		for each in days:

			logging.info('Updating word weights for: ' + each)

			# Call the proper functions
			if load_articles(each):
				update_all_word_weights(weight_opt, each)
			stock_data.clear() # To prepare for the next set of articles

		if GPU:
			# Clear and run everything on the GPU
			del words_by_letter[:]
			del num_words_by_letter[:]

			load_all_word_weights(weight_opt)

			# Run everything on the GPU
			for each in days:

				logging.info('Updating word weights with the gpu for: ' + each)

				# Call the proper functions
				if load_articles(each):
					update_all_word_weights_gpu(weight_opt, each)
				stock_data.clear() # To prepare for the next set of articles

		# Save data
		logging.info('Saving word weights')
		save_all_word_weights(weight_opt)
		
		logging.info('Updating word weights complete')

	# Determine speedup from GPU and comparison between GPU and CPU outputs
	if GPU:
		print('') 
		if len(analysis_outputs_gpu) != len(analysis_outputs_cpu):
			print('Mismatch in number of analysis outputs between GPU and CPU.')
		else:
			max_percentage = 0
			sum_percentage = 0
			for ii in range(0, len(analysis_outputs_cpu)):

				if (analysis_outputs_cpu[ii] + analysis_outputs_gpu[ii]) != 0:

					percentage = 2 * abs(analysis_outputs_cpu[ii] - analysis_outputs_gpu[ii]) / (analysis_outputs_cpu[ii] + analysis_outputs_gpu[ii])

					sum_percentage += percentage
					if percentage > max_percentage:
						max_percentage = sum_percentage

			print('Analysis Percent Difference: Max = ' + str(max_percentage * 100) + ', Avg = ' + str(sum_percentage*100 / len(analysis_outputs_cpu)))

		if len(prediction_outputs_gpu) != len(prediction_outputs_cpu):
			print('Mismatch in number of prediction outputs between GPU and CPU.')

		else:
			max_percentage = 0
			sum_percentage = 0
			for ii in range(0, len(prediction_outputs_cpu)):

				if (prediction_outputs_cpu[ii] + prediction_outputs_gpu[ii]) != 0:
					percentage = 2 * abs(prediction_outputs_cpu[ii] - prediction_outputs_gpu[ii]) / (prediction_outputs_cpu[ii] + prediction_outputs_gpu[ii])

					sum_percentage += percentage
					if percentage > max_percentage:
						max_percentage = sum_percentage

			print('Analysis Percent Difference: Max = ' + str(max_percentage * 100) + ', Avg = ' + str(sum_percentage*100 / len(prediction_outputs_cpu)))

		if len(update_outputs_gpu) != len(update_outputs_cpu):
			print('Mismatch in number of update outputs between GPU and CPU.')

		else:
			max_percentage = 0
			sum_percentage = 0
			for ii in range(0, len(update_outputs_cpu)):

				if (update_outputs_cpu[ii] + update_outputs_gpu[ii]) != 0:
					percentage = 2 * abs(update_outputs_cpu[ii] - update_outputs_gpu[ii]) / (update_outputs_cpu[ii] + update_outputs_gpu[ii])

					sum_percentage += percentage
					if percentage > max_percentage:
						max_percentage = sum_percentage

			print('Analysis Percent Difference: Max = ' + str(max_percentage * 100) + ', Avg = ' + str(sum_percentage*100 / len(update_outputs_cpu)))

		print('')
		if len(analysis_cpu_kernel_time) != 0 and len(analysis_cpu_function_time) != 0 and len(analysis_gpu_kernel_time) != 0 and len(analysis_gpu_function_time) != 0:
			sum_cpu_function = 0
			sum_cpu_kernel = 0
			for times in analysis_cpu_function_time:
				sum_cpu_function += times
			for times in analysis_cpu_kernel_time:
				sum_cpu_kernel += times

			sum_gpu_function = 0
			sum_gpu_kernel = 0
			for times in analysis_gpu_function_time:
				sum_gpu_function += times
			for times in analysis_gpu_kernel_time:
				sum_gpu_kernel += times

			print('Analysis Speedup: Kernel = ' + str(sum_cpu_kernel / sum_gpu_kernel) + ', Function = ' + str(sum_cpu_function / sum_gpu_function))

		if len(predict_cpu_kernel_time) != 0 and len(predict_cpu_function_time) != 0 and len(predict_gpu_kernel_time) != 0 and len(predict_gpu_function_time) != 0:
			sum_cpu_function = 0
			sum_cpu_kernel = 0
			for times in predict_cpu_function_time:
				sum_cpu_function += times
			for times in predict_cpu_kernel_time:
				sum_cpu_kernel += times

			sum_gpu_function = 0
			sum_gpu_kernel = 0
			for times in predict_gpu_function_time:
				sum_gpu_function += times
			for times in predict_gpu_kernel_time:
				sum_gpu_kernel += times

			print('Prediction Speedup: Kernel = ' + str(sum_cpu_kernel / sum_gpu_kernel) + ', Function = ' + str(sum_cpu_function / sum_gpu_function))

		if len(update_cpu_kernel_time) != 0 and len(update_cpu_function_time) != 0 and len(update_gpu_kernel_time) != 0 and len(update_gpu_function_time) != 0:
			sum_cpu_function = 0
			sum_cpu_kernel = 0
			for times in update_cpu_function_time:
				sum_cpu_function += times
			for times in update_cpu_kernel_time:
				sum_cpu_kernel += times

			sum_gpu_function = 0
			sum_gpu_kernel = 0
			for times in update_gpu_function_time:
				sum_gpu_function += times
			for times in update_gpu_kernel_time:
				sum_gpu_kernel += times

			print('Update Speedup: Kernel = ' + str(sum_cpu_kernel / sum_gpu_kernel) + ', Function = ' + str(sum_cpu_function / sum_gpu_function))


	print('')
	print('Done')




main()
