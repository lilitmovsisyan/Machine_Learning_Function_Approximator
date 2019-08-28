"""
LILIT_MOVSISYAN_run_mlp.py

This script runs functions defined in LILIT_MOVSISYAN_FunctionApproximator.py
in order to create and compare various MLPs.

  ----------------------------------------------------------------------
  author:       Lilit Movsisyan
  Date:         02/01/2019
  ----------------------------------------------------------------------

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import LILIT_MOVSISYAN_FunctionApproximator_FIXED as mlp

# this is required to permit multiple copies of the OpenMP runtime to be linked
# to the programme.  Failure to include the following two lines will result in
# an error that Spyder will not report.  On PyCharm the error provided will be
#    OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.
#    ...
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

############################# CONTROL TEST PARAMETERS################################

# Network training parameters
learning_rate    = 0.01
training_epochs  = 100
min_x = -10
max_x = 10
Ngen  = 10000

# Network architecture parameters
n_input    = 1   # 1D function, so there is one input feature per example
n_classes  = 1   # Regression output is single valued
n_hidden_1 = 50  # 1st layer num features
n_hidden_2 = 50  # 2nd layer num features

#Flags for plotting and printing out.  If plotting is set to True then in addition to the 
# output plots, the script will make plots of the test and train data.  The verbose printout
# option set to True will result in detailed output being printed.
plotting = False
verbose = False


######################################################################################

# Generate training data and test data
data_sets = mlp.create_datasets(Ngen, min_x, max_x, my_function=mlp.myFunctionTF)

######################################################################################

# CREATE MLP NETWORKS:

# # TESTING - two layer perceptron, no dropout, no batch learning. 
# # Why doesn't this work the way it did before I added the ability to use batch learning when desired?
# network_0 = mlp.run_mlp(
#     data_sets, 
#     learning_rate, 
#     training_epochs, 
#     n_input, 
#     n_classes, 
#     n_hidden_1,
#     n_hidden_2=n_hidden_2, 
#     dropout_keep_prob=1.0, 
#     n_batches=1,
#     plotting=plotting, 
#     verbose=verbose
# )

# #TESTING
# mlp.plot_network_response(data_sets, network_0, "network_0_test.pdf")



# Single layer perceptron
network_1 = mlp.run_mlp(
    data_sets, 
    learning_rate, 
    training_epochs, 
    n_input, 
    n_classes, 
    n_hidden_1,
    n_hidden_2=None, 
    dropout_keep_prob=1.0, 
    n_batches=1,
    plotting=plotting, 
    verbose=verbose,
)

# Two-layer MLP
network_2 = mlp.run_mlp(
    data_sets, 
    learning_rate, 
    training_epochs,
    n_input, 
    n_classes, 
    n_hidden_1,
    n_hidden_2=n_hidden_2, 
    dropout_keep_prob=1.0, 
    n_batches=1,
    plotting=plotting, 
    verbose=verbose,    
)

# Two-layer MLP with dropout (keep_prob = 0.5)
network_3 = mlp.run_mlp(
  data_sets,
  learning_rate,
  training_epochs,
  n_input,
  n_classes,
  n_hidden_1,
  n_hidden_2=n_hidden_2,
  dropout_keep_prob=0.5,
  n_batches=1,
  plotting=plotting,
  verbose=verbose,
)


# Two-layer MLP with mini-batch learning (10 batches)
network_4 = mlp.run_mlp(
  data_sets,
  learning_rate,
  training_epochs,
  n_input,
  n_classes,
  n_hidden_1,
  n_hidden_2=n_hidden_2,
  dropout_keep_prob=1.0,
  n_batches=10,
  plotting=plotting,
  verbose=verbose,
)

# Two-layer MLP with dropout AND mini-batch learning (10 batches)
network_5 = mlp.run_mlp(
  data_sets,
  learning_rate,
  training_epochs,
  n_input,
  n_classes,
  n_hidden_1,
  n_hidden_2=n_hidden_2,
  dropout_keep_prob=0.5,
  n_batches=10,
  plotting=plotting,
  verbose=verbose,
)

# Two-layer MLP with dropout (keep_prob = 0.9)
network_6 = mlp.run_mlp(
  data_sets,
  learning_rate,
  training_epochs,
  n_input,
  n_classes,
  n_hidden_1,
  n_hidden_2=n_hidden_2,
  dropout_keep_prob=0.9,
  n_batches=1,
  plotting=plotting,
  verbose=verbose,
)


########################################################################################
#
##plot comparison plots -
## 1. plot validation for each MLP individually 
## 2. plot NETWORK RESPONSE FUNCTION (x vs y) for two networks against test data (3 lines)
## 3. plot LOSS vs EPOCH (loss measured against TEST DATA) for two networks (2 lines)
#
#
##---------------------------
# single layer perceptron:
mlp.plot_network_response(data_sets, network_1, "Single-Layer Network", "network_1_single_layer.pdf", 'b*')

# two layer MLP:
mlp.plot_network_response(data_sets, network_2, "Two-Layer MLP", "network_2_mlp.pdf", 'r*')

## compare single layer and two layer perceptron (MLP):
#mlp.plot_loss_comparison(data_sets, network_1, network_2, "Single-Layer network", "Two-Layer MLP", "compare_1_or_2_layer.pdf", 'b*', 'r*')
#
##---------------------------
# MLP with dropout (keep_prob = 0.5):
mlp.plot_network_response(data_sets, network_3, "MLP with Dropout (keep_prob=0.5)", "network_3_dropout.pdf", 'g*')

# MLP with dropout (keep_prob = 0.9):
mlp.plot_network_response(data_sets, network_6, "MLP with Dropout (keep_prob=0.9)", "network_6_dropout.pdf", 'g*')

# compare MLP with and without dropout (keep_prob = 0.5):
mlp.plot_loss_comparison(data_sets, network_2, network_3, "MLP without dropout", "MLP with dropout", "compare_dropout.pdf", 'r*', 'g*')

# compare MLP with dropout at keep_prob=0.5 and 0.9:
mlp.plot_loss_comparison(data_sets, network_3, network_6, "MLP without keep_prob=0.5", "MLP with keep_prob=0.9", "compare_keep_prob.pdf", 'g*', 'm*')


##---------------------------
# MLP trained with mini-batch learning (batch size = 10):
mlp.plot_network_response(data_sets, network_4, "MLP with Batch Training (10 batches)", "network_4_batch_learning.pdf", 'c*')

# compare MLP trained with and without mini-batch learning:
mlp.plot_loss_comparison(data_sets, network_2, network_4, "MLP trained on all data", "MLP trained on batches", "compare_batch.pdf", 'r*', 'c*')

##---------------------------
# MLP with dropout AND trained with mini-batch learning (batch size = 10):
mlp.plot_network_response(data_sets, network_5, "MLP with both batch training and Dropout (keep_prob=0.5)", "network_5_batch_and_dropout.pdf", 'm*')

### # compare MLP trained with dropout, or mini-batch learning, or both:
### #### THIS IS NOT GONNA WORK - NEED THREE PLOTS mlp.plot_loss_comparison(data_sets, network_4, network_5, "compare_combined_batch_dropout.pdf")

###---------------------------

# PLOT A GRAPH TO COMPARE DROPOUT NETWORKS:

plt.subplot(2, 1, 1)
plt.plot(data_sets["testdata"], data_sets["test_value"], 'k.', label='Test data')
plt.ylabel('f(x) = sqr(x)')
plt.xlabel('x')
plt.title('Test data')
#plt.plot(network_1["input_value"], network_1["prediction_value"], 'b*', label='Single layer perceptron')
plt.plot(network_2["input_value"], network_2["prediction_value"], 'r*', label='Two layer perceptron (MLP)')
plt.plot(network_3["input_value"], network_3["prediction_value"], 'g*', label='MLP with dropout (keep_prob=0.5)')
#plt.plot(network_4["input_value"], network_4["prediction_value"], 'c*', label='MLP with batch learning')
#plt.plot(network_5["input_value"], network_5["prediction_value"], 'm*', label='MLP with dropout and batch learning')
plt.plot(network_6["input_value"], network_6["prediction_value"], 'm*', label='MLP with dropout (keep_prob=0.9)')

plt.ylabel('f(x) = sqr(x)')
plt.xlabel('x')
plt.legend()
plt.title('Network Response Function')

ax = plt.subplot(2, 1, 2)
#plt.plot(network_1["epoch_set"], network_1["loss_test_value"], 'bx', label='Single layer perceptron')
plt.plot(network_2["epoch_set"], network_2["loss_test_value"], 'rx', label='Two layer perceptron (MLP)')
plt.plot(network_3["epoch_set"], network_3["loss_test_value"], 'gx', label='MLP with dropout (keep_prob=0.5)')
#plt.plot(network_4["epoch_set"], network_4["loss_test_value"], 'cx', label='MLP with batch learning')
#plt.plot(network_5["epoch_set"], network_5["loss_test_value"], 'mx', label='MLP with dropout and batch learning')
plt.plot(network_6["epoch_set"], network_6["loss_test_value"], 'mx', label='MLP with dropout (keep_prob=0.9)')

plt.ylabel('loss')
ax.set_yscale('log')
plt.xlabel('epoch')
#plt.legend()
#plt.grid()


plt.savefig("compare_all_dropout.pdf")
plt.show()
#
###---------------------------
##
# PLOT A GRAPH TO COMPARE BACHING NETWORKS:

plt.subplot(2, 1, 1)
plt.plot(data_sets["testdata"], data_sets["test_value"], 'k.', label='Test data')
plt.ylabel('f(x) = sqr(x)')
plt.xlabel('x')
plt.title('Test data')
#plt.plot(network_1["input_value"], network_1["prediction_value"], 'b*', label='Single layer perceptron')
plt.plot(network_2["input_value"], network_2["prediction_value"], 'r*', label='Two layer perceptron (MLP)')
plt.plot(network_3["input_value"], network_3["prediction_value"], 'g*', label='MLP with dropout (keep_prob=0.5)')
plt.plot(network_4["input_value"], network_4["prediction_value"], 'c*', label='MLP with batch learning')
plt.plot(network_5["input_value"], network_5["prediction_value"], 'm*', label='MLP with dropout and batch learning')
#plt.plot(network_6["input_value"], network_6["prediction_value"], 'm*', label='MLP with dropout (keep_prob=0.9)')

plt.ylabel('f(x) = sqr(x)')
plt.xlabel('x')
plt.legend()
plt.title('Network Response Function')

ax = plt.subplot(2, 1, 2)
#plt.plot(network_1["epoch_set"], network_1["loss_test_value"], 'bx', label='Single layer perceptron')
plt.plot(network_2["epoch_set"], network_2["loss_test_value"], 'rx', label='Two layer perceptron (MLP)')
plt.plot(network_3["epoch_set"], network_3["loss_test_value"], 'gx', label='MLP with dropout (keep_prob=0.5)')
plt.plot(network_4["epoch_set"], network_4["loss_test_value"], 'cx', label='MLP with batch learning')
plt.plot(network_5["epoch_set"], network_5["loss_test_value"], 'mx', label='MLP with dropout and batch learning')
#plt.plot(network_6["epoch_set"], network_6["loss_test_value"], 'mx', label='MLP with dropout (keep_prob=0.9)')

plt.ylabel('loss')
ax.set_yscale('log')
plt.xlabel('epoch')
#plt.legend()
#plt.grid()


plt.savefig("compare_all_batch.pdf")
plt.show()

###---------------------------

# PLOT A GRAPH TO COMPARE ALL (FIVE MAIN) NETWORKS:

plt.subplot(2, 1, 1)
plt.plot(data_sets["testdata"], data_sets["test_value"], 'k.', label='Test data')
plt.ylabel('f(x) = sqr(x)')
plt.xlabel('x')
plt.title('Test data')
plt.plot(network_1["input_value"], network_1["prediction_value"], 'b*', label='Single layer perceptron')
plt.plot(network_2["input_value"], network_2["prediction_value"], 'r*', label='Two layer perceptron (MLP)')
plt.plot(network_3["input_value"], network_3["prediction_value"], 'g*', label='MLP with dropout (keep_prob=0.5)')
plt.plot(network_4["input_value"], network_4["prediction_value"], 'c*', label='MLP with batch learning')
plt.plot(network_5["input_value"], network_5["prediction_value"], 'm*', label='MLP with dropout and batch learning')

plt.ylabel('f(x) = sqr(x)')
plt.xlabel('x')
plt.legend()
plt.title('Network Response Function')

ax = plt.subplot(2, 1, 2)
plt.plot(network_1["epoch_set"], network_1["loss_test_value"], 'bx', label='Single layer perceptron')
plt.plot(network_2["epoch_set"], network_2["loss_test_value"], 'rx', label='Two layer perceptron (MLP)')
plt.plot(network_3["epoch_set"], network_3["loss_test_value"], 'gx', label='MLP with dropout (keep_prob=0.5)')
plt.plot(network_4["epoch_set"], network_4["loss_test_value"], 'cx', label='MLP with batch learning')
plt.plot(network_5["epoch_set"], network_5["loss_test_value"], 'mx', label='MLP with dropout and batch learning')

plt.ylabel('loss')
ax.set_yscale('log')
plt.xlabel('epoch')
#plt.legend()
#plt.grid()


plt.savefig("compare_all.pdf")
plt.show()

