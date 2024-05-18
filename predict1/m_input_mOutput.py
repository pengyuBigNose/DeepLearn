import numpy as np

toes = np.array([8.5, 9.5, 9.9, 9.0])
wlrec = np.array([0.65, 0.8, 0.8, 0.9])
nfans = np.array([1.2, 1.3, 0.5, 1.0])

input = wlrec[0]

weights = np.array([0.8, 0.2, 0.3])

# Single input multiple output
def ele_mul(input, weights):  # element-wise multiplication
  output = [0,0,0]
  assert(len(output) == len(weights))
  for i in range(len(weights)):
    output[i] = input * weights[i]
  return output

def neural_network(input, weights):
  pred = ele_mul(input, weights)
  return pred

pred = neural_network(input, weights)
print(pred)

# multiple input multiple output
weights2 = [[0.1,0.1,-0.3],[0.1,0.2,0.0],[0.0,1.3,0.1]]
input2 = [toes[0],wlrec[0],nfans[0]]

def w_sum(input, weights): # input is a vector of features, weights is a vector of weights for each neuron
  assert(len(input) == len(weights))
  output = 0
  for i in range(len(input)): # for each neuron in the input
    output += input[i] * weights[i] # multiply the input by the corresponding weight and add it to the output
  return output

def ele_mul_matrix(vect, matrix):  # element-wise multiplication
  assert(len(vect)==len(matrix))
  output = [0,0,0]
  for i in range(len(matrix)):
    output[i] = w_sum(vect, matrix[i])  # apply the w_sum function to each row of the matrix
  return output

def neural_network_matrix(input, weights):
  pred = ele_mul_matrix(input, weights)
  return pred

pred2 = neural_network_matrix(input2, weights2)
print(pred2)