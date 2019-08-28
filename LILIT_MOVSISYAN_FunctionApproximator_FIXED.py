"""
LILIT_MOVSISYAN_FunctionApproximator.py

This script contains functions for generating and comparing Multi-Layer Perceptrons
which can be used as function approximators. It includes:

 - a function for creating single-layer or two-layer perceptrons
   with the option of using dropout and mini-batch training;
 - functions to generate the data required for training and testing the MLPs;
 - functions to plot the network response function and loss function of the MLPs.

To run this script, please see LILIT_MOVSISYAN_run_mlp.py

  ----------------------------------------------------------------------
  author:       Lilit Movsisyan
  Date:         01/08/2019
  ----------------------------------------------------------------------

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# this is required to permit multiple copies of the OpenMP runtime to be linked
# to the programme.  Failure to include the following two lines will result in
# an error that Spyder will not report.  On PyCharm the error provided will be
#    OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.
#    ...
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


##############################################################################
def myFunctionTF(arg):
    """
    User defined function for the MLP to learn.  The default example is 
    the square root function.
    """
    return tf.square(arg)

##############################################################################
def create_datasets(Ngen, min_x=1, max_x=10, my_function=myFunctionTF):
  """
  This function generates a set of training data and a set of test data of size Ngen
  to be used in training and validation of a multi-layer perceptron. 
  Two sets of random uniform distribution is generated with values between min_x and max_x,
  representing x values for a training set and a testing set of data. Then, a user-specified function 
  (default is the tf.square(), myFunctionTF, above) is applied to these x values to generate corresponding y values.
  The function returns a dictionary containing 4 NumPy arrays corresponding to the 
  x and y values for the trianing set and the testing set.
  """

  # generate data, the input data is a random number betwen min_x and max_x,
  # and the corresponding label value is the square of that number
  print("Generating the test and training sets.  There are ", Ngen, " examples in each")
  tftraindata = tf.random_uniform([Ngen, 1], min_x, max_x)  # training set
  tftestdata  = tf.random_uniform([Ngen, 1], min_x, max_x)  # test set

  # Initializing the variables
  init  = tf.global_variables_initializer()

  # Start the session to embark on the training cycle
  sess = tf.Session()
  sess.run(init)

  # convert the training data to np arrays so that these can be used with the feed_dict when training
  traindata  = sess.run(tftraindata) 
  target_value = sess.run(my_function(traindata))

  # convert the test data to np arrays so that these can be used with the feed_dict when training
  testdata  = sess.run(tftestdata) 
  test_value = sess.run(my_function(testdata))

  return {
    "Ngen"        : Ngen,
    "min_x"       : min_x,
    "max_x"       : max_x,
    "traindata"   : traindata,
    "target_value": target_value,
    "testdata"    : testdata,
    "test_value"  : test_value,
  }



##############################################################################
##############################################################################

def run_mlp(
    data_sets, 
    learning_rate, 
    training_epochs,
    n_input, 
    n_classes, 
    n_hidden_1,
    n_hidden_2=None,
    dropout_keep_prob=1.0,
    n_batches=1,
    plotting=True, 
    verbose=True
):

  """
  This function creates a Multi-Layer Perceptron using the parameters entered as 
  arguments, and returns a dictionary of values which can be used to plot the 
  loss and the network response function of the MLP.

  This function takes the following arguments:
    - data_sets,              -> the training and test data sets, as produced by the output of the create_datasets() function above.
    - learning_rate,          -> parameter adjustment step size.
    - training_epochs,        -> number of training epochs to optimise parameters over.
    - n_input,                -> the number of input features.
    - n_classes,              -> the number of output classes.
    - n_hidden_1,             -> the number of nodes in the first hidden layer.
    - n_hidden_2=None,        -> number of nodes in the second hidden layer (if None, no second layer is added)
    - dropout_keep_prob=None, -> takes a value between 0 and 1. Each node will be dropped with a probability of (1 - dropout_keep_prob).
    - n_batches=1,            -> number of mini-batches. 
    - plotting=True,          -> if True, will plot the values for the training data and the test data.
    - verbose=True            -> if True, will display verbose print values.

  This function will return a dictionary with the following values:

  {
    "input_value"      : input_value,      -> an array of x values.
    "epoch_set"        : epoch_set,        -> an array of training epoch indices.
    "loss_test_value"  : loss_test_value,  -> an array of the loss compared to test values, for each epoch.
    "loss_set"         : loss_set,         -> an array of the loss compared to training values, for each epoch.
    "prediction_value" : prediction_value, -> an array of preiction values generated by the network for each x input value.
    "probabilities"    : probabilities,    -> a Tensor object containing the final optimised parameters (the Network Response Function).
  }
  """

  ##############################################################################
  # NETWORK PARAMETERS
  ##############################################################################

  # Grab the following parapemeters from the data_sets argument 
  # (data_sets contains arrays produced by the create_datasets() function above)
  Ngen         = data_sets["Ngen"]
  min_x        = data_sets["min_x"]
  max_x        = data_sets["max_x"]
  traindata    = data_sets["traindata"]
  target_value = data_sets["target_value"]
  testdata     = data_sets["testdata"]
  test_value   = data_sets["test_value"]


  print("--------------------------------------------")
  print("Number of input features           = ", n_input)
  print("Number of output classes           = ", n_classes)
  print("Number of examples to generate     = ", Ngen)
  print("Learning rate                alpha = ", learning_rate)
  print("Number of training epochs          = ", training_epochs)
  print("Number of batches of training data = ", n_batches)
  if dropout_keep_prob < 1.0:
    print("Dropout keep probability         = ", dropout_keep_prob)
  print("--------------------------------------------")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#### ADD PRINTOUTS FOR NUMBER OF LAYERS< NODES, ETC???????????????????????????????

  ##############################################################################
  # DEFINE GRAPH 
  ##############################################################################

  # tf Graph input:
  #  x_: is the tensor for the input data (the placeholder entry None is used for that;
  #     and the number of features input (n_input = 1).
  #
  #  y_: is the tensor for the output value of the function that is being approximated by 
  #     the MLP.
  #
  x_ = tf.placeholder(tf.float32, [None, n_input], name="x_")
  y_ = tf.placeholder(tf.float32, [None, n_classes], name="y_")
  ### You need to have a placeholder for the keep_prob so that you can disable dropout when you test your model, i.e. when you make predictions you want to use all nodes.
  keep_prob_ = tf.placeholder(tf.float32, name="dropout") 


  # We construct layer 1 from a weight set, a bias set and the activiation function used
  # to process the impulse set of features for a given example in order to produce a 
  # predictive output for that example.
  #
  #  w_layer_1:    the weights for layer 1.  The first index is the input feature (pixel)
  #                and the second index is the node index for the perceptron in the first
  #                layer.
  #  bias_layer_1: the biases for layer 1.  There is a single bias for each node in the 
  #                layer.
  #  layer_1:      the activation functions for layer 1
  #

  print("Creating a hidden layer with ", n_hidden_1, " nodes")
  w_layer_1      = tf.Variable(tf.random_normal([n_input, n_hidden_1]), name="w_layer_1")
  bias_layer_1   = tf.Variable(tf.random_normal([n_hidden_1]), name="bias_layer_1")
  layer_1        = tf.nn.relu(tf.add(tf.matmul(x_, w_layer_1), bias_layer_1), name="layer_1")

  if dropout_keep_prob < 1:
    print("Applying dropout on first hidden layer with keep probability of ", dropout_keep_prob)
  dlayer_1       = tf.nn.dropout(layer_1, keep_prob_)

  final_layer     = dlayer_1
  final_nodes     = n_hidden_1

  if n_hidden_2 != None:
    print("Creating a second hidden layer with ", n_hidden_2, " nodes")
    w_layer_2     = tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name="w_layer_2") ### shape for w_layer_2 is now [n_hidden_1, n_hidden_2] (i.e. [50,50]), since now our input to layer_2 is the output of layer_1, NOT x_
    bias_layer_2  = tf.Variable(tf.random_normal([n_hidden_2]), name="bias_layer2")
    layer_2       = tf.nn.relu(tf.add(tf.matmul(dlayer_1, w_layer_2), bias_layer_2), name="layer_2") ### input is weighted layer_1, not weighted x_, which is what we did in layer_1 above.

    if dropout_keep_prob < 1:
        print("Applying dropout on second hidden layer with keep probability of ", dropout_keep_prob)
    dlayer_2   = tf.nn.dropout(layer_2, keep_prob_) 

    final_layer   = dlayer_2
    final_nodes   = n_hidden_2

  # Similarly we now construct the output of the network, where the output layer
  # combines the information down into a space of evidences for the possible
  # classes in the problem (n_classes=1 for this regression problem).
  print("Creating the output layer, ", n_classes, " output values")
  output       = tf.Variable(tf.random_normal([final_nodes, n_classes]), name="output_layer") ### input is now n_hidden_2, not n_hidden_1, which is what we had before.
  bias_output  = tf.Variable(tf.random_normal([n_classes]), name="bias_output")
  # define operation for computing the regression output - this is our model prediction
  probabilities = tf.matmul(final_layer, output) + bias_output

  #optimise with l2 loss function or mean squared error loss function
  print("Using the L2 loss function implemented in tf.nn")
  loss = tf.nn.l2_loss(y_ - probabilities)
#  print("Using the mean squared error function implemented in tf.compat.v1.losses")
#  loss = tf.compat.v1.losses.mean_squared_error(y_, probabilities)

  # Alternative way to write the loss function
  #loss = tf.reduce_sum((y_ - probabilities)*(y_ - probabilities))

  # optimizer: take the Adam optimiser, see https://arxiv.org/pdf/1412.6980v8.pdf for
  # details of this algorithm.
  print("Using the Adam optimiser to train the network")
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

  
  ##############################################################################
  ##############################################################################
  # plot the test and training data

  if plotting:
    # plot the function f(x)=sqr(x)
    plt.subplot(2, 1, 1)
    plt.plot(traindata, target_value, 'b.', label="Training data")
    plt.ylabel('f(x) = sqr(x)')
    plt.xlabel('x')
    plt.title('Training data')

    plt.subplot(2, 1, 2)
    plt.plot(testdata, test_value, 'k.', label="Test data")
    plt.ylabel('f(x) = sqr(x)')
    plt.xlabel('x')
    plt.title('Test data')

    plt.tight_layout()
    plt.show()




  ##############################################################################
  # SPLIT DATA INTO BATCHES
  ##############################################################################
  
  batch_size = (Ngen // n_batches)

  batch_data_list = []
  batch_value_list = []

  for i in range(n_batches):
      start_index = i*batch_size
      end_index   = (i*batch_size) + batch_size
      batch_data_list.append(traindata[start_index:end_index])
      batch_value_list.append(target_value[start_index:end_index])
  
  batch_data_array = np.array(batch_data_list)
  batch_value_array = np.array(batch_value_list)

  if verbose:
      
      print("-------------------TESTING BATCHES------------------")
      print("Number of batches                         = ", n_batches)
      print("Batch size                                = ", batch_size)
      print("length of batch_data_list                 = ", len(batch_data_list)==n_batches)
      print("length of batch_value_list                = ", len(batch_value_list)==n_batches)
      print("length of batch_data_array                = ", len(batch_data_array)==n_batches)
      print("length of batch_value_array               = ", len(batch_value_array)==n_batches)
      print("length of first item in batch_data_array  = ", len(batch_data_array[0])==batch_size)
      print("length of first item in batch_value_array = ", len(batch_value_array[0])==batch_size)
      print("------------------------------------------------------")



  ##############################################################################
  # TRAINING AND PREDICTION
  ##############################################################################

  # initialize Session
  sess = tf.Session()
  init = tf.global_variables_initializer()
  sess.run(init)


  # arrays to compare the input value vs the prediction
  input_value      = []
  epoch_set        = []
  loss_test_value  = []
  loss_set         = []
  prediction_value = []

  ###############################################################################################
  # THE FOLLOWING SECTION OF CODE ALLOWS BATCH LEARNING.
  # BUT IT GIVES WEIRD RESULTS, EVEN WHEN BATCH SIZE IS SET TO 1 (that means no batches - just one set of data)
  # COMMENT THIS SECTION OUT AND UNCOMMENT THE NEXT SECTION BELOW IT TO REVERT TO PREVIOUS VERSION THAT WORKED.
  ###############################################################################################

  for epoch in range(training_epochs):
     
     batch = epoch % n_batches
     training_x_values = batch_data_array[batch]
     training_y_values = batch_value_array[batch]
     
     if verbose:
         print("TESTING CORRECT BATCH NUMBER: ", batch)
         print("TESTING CORRECT NUMBER OF DATA: ", len(training_x_values)==batch_size)
         print("TESTING CORRECT NUMBER OF DATA: ", len(training_y_values)==batch_size)

     the_loss = 0.
    
     if verbose:
         print("Training epoch number ", epoch)
    

     sess.run(optimizer, feed_dict={x_: training_x_values, y_: training_y_values, keep_prob_: dropout_keep_prob})

     the_loss = sess.run(loss, feed_dict={x_: training_x_values, y_: training_y_values, keep_prob_: 1.0})
#     mse = the_loss / batch_size
#     loss_set.append(mse)
     loss_set.append(the_loss)
     epoch_set.append(epoch+1)
    
     the_loss = sess.run(loss, feed_dict={x_: testdata, y_: test_value, keep_prob_: 1.0})
#     mse = the_loss / Ngen
#     loss_test_value.append(mse)
     loss_test_value.append(the_loss)


     # # RESET OPTIMISED VALUES TO ZERO? - otherwise it's just the same as running over all your data bit by bit. 
     # # We don't want one batch to influence another until the end......... 
     # print("CHECKING SHAPE OF probabilities: ", probabilities)
     # #for parameter in probabilities:
     # tf.zeros_like(probabilities)
     # print("CHECKING SHAPE OF probabilities: ", probabilities)
     # And then would need to optimise over the full set of epochs' training. 


     #
     # This is a regression analysis problem, so we want to evaluate and display the output_layer
     # value (model response function), and not an output prediction (which would have been appropraite
     # for a classification problem)
     #
     if epoch == training_epochs-1:
         step = (max_x - min_x)/100
         for i in range(100):
             thisx = min_x + i*step
             pred = probabilities.eval(feed_dict={x_: [[thisx]], keep_prob_: 1.0}, session=sess)
             if verbose:
                 print ("x = ", thisx, ", prediction =", pred)
             input_value.append(thisx)
             prediction_value.append(pred[0])

             pred = probabilities.eval(feed_dict={x_: [[-thisx]], keep_prob_: 1.0}, session=sess)
             if verbose:
                 print ("x = ", -thisx, ", prediction =", pred)
             input_value.append(-thisx)
             prediction_value.append(pred[0])




  ##############################################################################
              
  # check the loss function for the last epoch vs the number of data
  print("Loss function for the final epoch = ", loss_test_value[-1], " (test data)")
  print("Pseudo chi^2 = loss/Ndata         = {0:4f} (test data)".format( loss_test_value[-1]/Ngen ))
  # Result summary
  print ("Training phase finished")

  ##############################################################################
  # RETURN RESULTS
  ##############################################################################
  return {
    "input_value"      : input_value,
    "epoch_set"        : epoch_set,
    "loss_test_value"  : loss_test_value,
    "loss_set"         : loss_set,
    "prediction_value" : prediction_value,
    "probabilities"    : probabilities,
  }


##############################################################################
##############################################################################
##############################################################################
##############################################################################

def plot_network_response(data_sets, network, network_name, filename, line_colour='r*'):
  """
  This function plots a graph showing, first, the network response function 
  compared to the test data set, and second, the loss as a function of training epoch 
  for both the training dataset and the test data set. This latter graph can be 
  used for cross validation against overfitting of the network.
  """
  title = 'Network Response Function for ' + network_name
  
  
  plt.subplot(2, 1, 1)
  plt.plot(data_sets["testdata"], data_sets["test_value"], 'k.', label='Test data')
  plt.ylabel('f(x) = sqr(x)')
  plt.xlabel('x')
  plt.title('Test data')
  plt.plot(network["input_value"], network["prediction_value"], line_colour, label='Network Response Function')
  plt.ylabel('f(x) = sqr(x)')
  plt.xlabel('x')
  plt.legend()
#  plt.grid()
  plt.title(title)

  ax = plt.subplot(2, 1, 2)
  plt.plot(network["epoch_set"], network["loss_set"], 'bo', label='MLP Training phase loss')
  plt.plot(network["epoch_set"], network["loss_test_value"], 'rx', label='MLP Testing phase loss')
  plt.ylabel('loss')
  ax.set_yscale('log')
  plt.xlabel('epoch')
  plt.legend()
#  plt.grid()


  plt.savefig(filename)
  plt.show()

##############################################################################

def plot_loss_comparison(data_sets, network_a, network_b, network_a_name, network_b_name, filename, a_linecolour='b*', b_linecolour='r*'):
  """
  This function plots a graph comparing the network response functions of two networks
  and the loss of these two networks as a function of training epoch for the test data.
  """
  a_loss_label = network_a_name + "Training phase"
  b_loss_label = network_b_name + "Testing phase"
  
#  plt.grid()
  
  plt.subplot(2, 1, 1)
  plt.plot(data_sets["testdata"], data_sets["test_value"], 'k.', label='Test data')
  plt.ylabel('f(x) = sqr(x)')
  plt.xlabel('x')
  plt.title('Test data')
  plt.plot(network_a["input_value"], network_a["prediction_value"], a_linecolour, label=network_a_name)
  plt.plot(network_b["input_value"], network_b["prediction_value"], b_linecolour, label=network_b_name)
  plt.ylabel('f(x) = sqr(x)')
  plt.xlabel('x')
  plt.legend()
#  plt.grid()
  plt.title('Network Response Function Comparison')

  ax = plt.subplot(2, 1, 2)
  plt.plot(network_a["epoch_set"], network_a["loss_test_value"], a_linecolour, label=a_loss_label)
  plt.plot(network_b["epoch_set"], network_b["loss_test_value"], b_linecolour, label=b_loss_label)

  plt.ylabel('loss')
  ax.set_yscale('log')
  plt.xlabel('epoch')
  #plt.legend()
#  plt.grid()


  plt.savefig(filename)
  plt.show()



##############################################################################
##############################################################################
