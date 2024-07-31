import numpy as np
import matplotlib.pyplot as plt

def data_matrix(file_path, rows, columns):
    data = np.zeros((rows, columns))
    l = 0
    with open(file_path, 'r') as file:
        for line in file:
            if l < rows:  
                values = line.strip().split()
                for column in range(columns):
                    value = float(values[column])
                    data[l, column] = value
                l += 1  

    return data


source_data ='flood_data_set.txt'
data = data_matrix(source_data,252,9) #training set fold1 not swap
data_normalized = data / 600
input_data = data_normalized[:,:8]
output_data = data_normalized[:,8:]

source_data2 ='flood_data_test.txt'
unsee_data = data_matrix(source_data2,63,9)
unsee_data_normalized = unsee_data / 600
unsee_input_data = unsee_data_normalized[:,:8]
unsee_output_data = unsee_data_normalized[:,8:]


#architecture
hidden_size = 3
input_size = input_data.shape[1]
output_size = output_data.shape[1]

np.random.seed(42) 
weights_hidden = np.random.rand(input_size, hidden_size)
biases_hidden = np.random.rand(hidden_size)
weights_output = np.random.rand(hidden_size, output_size)
biases_output = np.random.rand(output_size)

# Initialize previous weight changes with zeros
prev_weights_output_change = np.zeros_like(weights_output)
prev_biases_output_change = np.zeros_like(biases_output)
prev_weights_hidden_change = np.zeros_like(weights_hidden)
prev_biases_hidden_change = np.zeros_like(biases_hidden)


def relu(x): # try to use tanh()
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

epochs = 80000
learning_rate = 0.0001
momentum_rate = 0.1
i = 0
plot_epoch=[]
plot_loss=[]

print("started_bias_hidden :\n", biases_hidden)
print("started_weights_hidden : \n", weights_hidden)
print("started_bias_output :\n", biases_output)
print("started_weights_output : \n", weights_output)
print("-------------------------------")

for epoch in range(epochs) :
    #forward propagation
    hidden_layer_input = np.dot(input_data, weights_hidden) + biases_hidden  #V(i)
    hidden_layer_output = relu(hidden_layer_input)   #Y(i)
    output_layer_input = np.dot(hidden_layer_output, weights_output) + biases_output #O(i)
    predicted_output = output_layer_input  #keep output value to compare and find sum square error

    #calculate the loss (mean squared error)
    loss = np.mean((predicted_output - output_data) ** 2)
    loss = round(loss,8)
    
    # Backpropagation
    output_error = predicted_output - output_data
    output_gradient = output_error
    weights_output_change = (learning_rate * np.dot(hidden_layer_output.T, output_gradient)) + (momentum_rate * prev_weights_output_change)
    biases_output_change = (learning_rate * np.sum(output_gradient, axis=0)) + (momentum_rate * prev_biases_output_change)
    weights_output -= weights_output_change
    biases_output -= biases_output_change

    hidden_error = np.dot(output_gradient, weights_output.T) * relu_derivative(hidden_layer_input)
    weights_hidden_change = (learning_rate * np.dot(input_data.T, hidden_error)) + (momentum_rate * prev_weights_hidden_change)
    biases_hidden_change = (learning_rate * np.sum(hidden_error, axis=0)) + (momentum_rate * prev_biases_hidden_change)
    weights_hidden -= weights_hidden_change
    biases_hidden -= biases_hidden_change
  
    # Update the previous weight changes
    prev_weights_output_change = weights_output_change
    prev_biases_output_change = biases_output_change
    prev_weights_hidden_change = weights_hidden_change
    prev_biases_hidden_change = biases_hidden_change

    if(epoch % 100 == 0):
        plot_epoch.append(epoch)
        plot_loss.append(loss)
    
    #print("Loss >> ",round(loss,8))
    #print("Predicted output : \n",predicted_output) 

print("Loss >> ",round(loss,8)) 
predicted_output = predicted_output *600
print("Predicted output : \n",predicted_output)

final_bias_hidden = biases_hidden
final_weights_hidden =  weights_hidden
fianl_bias_output = biases_output
fianl_weights_output = weights_output

print("-------------------------------\n")
print("final_bias_hidden : \n", biases_hidden)
print("final_weights_hidden : \n", weights_hidden)
print("fianl_bias_output :\n", biases_output)
print("fianl_weights_output : \n", weights_output)

'''''
plt.figure(figsize=(8, 6))

plt.subplot(2, 1, 1)
plt.plot(plot_epoch,plot_loss)
plt.xlabel("Epoch")
plt.ylabel("LOSS")
plt.ylim(0.0001,0.009)
plt.title("at Lrate {} & Hidden {} & Momentum = {} ".format(learning_rate,hidden_size,momentum_rate))

plt.subplot(2, 1, 2)
plt.plot(output_data, label="Desired Output", marker='x',color ='yellow')
plt.plot(predicted_output/600, label="Processed Output", marker='x',color ='blue')
plt.title("Desired Output vs Predicted Output")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()'''

#forward propagation for test set
unsee_hidden_layer_input = np.dot(unsee_input_data, final_weights_hidden) + final_bias_hidden
unsee_hidden_layer_output = relu(unsee_hidden_layer_input)
unsee_output_layer_input = np.dot(unsee_hidden_layer_output, fianl_weights_output) + fianl_bias_output
unsee_predicted_output = unsee_output_layer_input

#calculate the loss (mean squared error)
loss = np.mean((unsee_predicted_output - unsee_output_data) ** 2)
loss = round(loss,8)

plt.figure(figsize=(8, 6))
plt.plot(unsee_output_data, label="Desired Output", marker='x',color ='yellow')
plt.plot(unsee_predicted_output, label="Processed Output", marker='x',color ='blue')
plt.title("Desired Output vs Predicted Output at loss = {}".format(loss))
plt.legend()
plt.grid(True)

plt.show()
