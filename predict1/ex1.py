import numpy as np

# features of the data
toes = np.array([8.5, 9.5, 9.9, 9.0])
wlrec = np.array([0.65, 0.8, 0.8, 0.9])
nfans = np.array([1.2, 1.3, 0.5, 1.0])
# weights of the neural network
weights = np.array([0.5, 0.2, 0.8])

def w_sum(input, weights): # input is a vector of features, weights is a vector of weights for each neuron
  assert(len(input) == len(weights))
  output = 0
  for i in range(len(input)): # for each neuron in the input
    output += input[i] * weights[i] # multiply the input by the corresponding weight and add it to the output
  return output

def neural_network(input, weights): # input is a vector of features
  pre = w_sum(input, weights)
  return pre

input = np.array([toes[0], wlrec[0], nfans[0]])
pre1 = neural_network(input, weights) # calculate the output of the first neuron
print("single neuron's output:",pre1)

# we can change the input and weights of neural network, which it transforms np.array
def neural_network2(input, weights): # input is a matrix of features, mutiple inputs of different dimensions
  pre = 0
  for i in range(input.shape[0]):
    for j in range(input.shape[1]):
      pre += input[i][j] * weights[i] # multiply the input by the corresponding weight and add it to the output
  return pre
input2 = np.array([toes, wlrec, nfans])
pre2 = neural_network2(input2, weights) # calculate the output of the entire neural network
print("entire neural network's output:",pre2)
