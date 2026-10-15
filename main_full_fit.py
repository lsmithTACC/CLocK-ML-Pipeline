# Import base python libraries
import glob
import re

# Import numpy for array-handling
import numpy as np

# Import matplotlib for plotting
from matplotlib import pyplot as plt

# Import Pytorch for neural network training
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.parametrizations as param

# Import sklearn libraries for convenient data segmentation
from sklearn.model_selection import train_test_split

# Define device 
print('Checking for GPU...')
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")



# Data Loading and Normalization --------------------------------------------------------

# Define number of input and output parameters (hard-coded for now)
# Features are convolved, channels are mixed
num_input_features = 360
num_input_channels = 3
num_output_features = 3


# List all data directories
directories = glob.glob("Simulated_Data_for_TACC/*/")

# Initialize input/output data
data_in = np.zeros((len(directories),num_input_channels,num_input_features))
data_out = np.zeros((len(directories),num_output_features))

# Loop through data directoties
for case_count, data_dir in enumerate(directories):
    numbers = [int(x) for x in re.findall(r'\d+', data_dir)]
    data_out[case_count,:] = numbers[0:num_output_features]

    data = np.genfromtxt(data_dir + 'Simulated_Dipole_Fit.txt', delimiter=',', dtype=float, skip_header=1)
    data_in[case_count,:,:] = np.transpose(data)[1:4,0:-1] # np.transpose(data)[-3:,0:-1]

# Normalization (min/max scaling for now)
# All numbers are positive so this just moves everything within [0,1]
#min_in = np.zeros((num_input_features,))
#max_in = np.zeros((num_input_features,))
min_out = np.zeros((num_output_features,))
max_out = np.zeros((num_output_features,))
mean_out = np.zeros((num_output_features,))
std_out = np.zeros((num_output_features,))

#data_in = np.sqrt(data_in)
data_in = data_in**(1/4)
#data_in = data_in/np.max(data_in)

for ind in range(0,num_input_channels):
#    min_in[ind] = np.min(data_in[:,ind])
#    max_in[ind] = np.max(data_in[:,ind])
#    data_in[:,ind,:] = (data_in[:,ind,:] - np.min(data_in[:,ind,:]))/(np.max(data_in[:,ind,:])-np.min(data_in[:,ind,:]))
    data_in[:,ind,:] = (data_in[:,ind,:] - np.mean(data_in[:,ind,:]))/(np.std(data_in[:,ind,:]))

for ind in range(0,num_output_features):	
	min_out[ind] = np.min(data_out[:,ind])
	max_out[ind] = np.max(data_out[:,ind])
	data_out[:,ind] = (data_out[:,ind] - np.min(data_out[:,ind]))/(np.max(data_out[:,ind])-np.min(data_out[:,ind]))
	#mean_out[ind] = np.mean(data_out[:,ind])
	#std_out[ind] = np.std(data_out[:,ind])
	#data_out[:,ind] = (data_out[:,ind] - np.mean(data_out[:,ind]))/(np.std(data_out[:,ind]))


# bins = 100
# fig, ax1 = plt.subplots(figsize=(4, 4))
# #ax1.hist(data_in[:,0,:].flatten(),bins=bins)
# ax1.hist(data_out[:,2],bins=bins)
# #ax1.set_xlim(0,1)
# ax1.set_xlabel("Value")
# ax1.set_ylabel("Counts")
# plt.show()


# Convert to pytorch tensor
data_in = torch.tensor(data_in,dtype=torch.float32)
data_out = torch.tensor(data_out,dtype=torch.float32)

# # Move to GPU
data_in = data_in.to(device)
data_out = data_out.to(device)

# # Display shapes of the input and output data
print('Shape of input data:')
print(data_in.shape)
print('Shape of output data:')
print(data_out.shape)



# # Data Splitting ------------------------------------------------------------------------

# Setup test/train split
test_size = 0.2
data_in_train, data_in_test, data_out_train, data_out_test = train_test_split(
	data_in, data_out, test_size=test_size, random_state=42
 	)

# Display shapes of the input and output data
print('Shape of training data:')
print(data_in_train.shape)
print(data_out_train.shape)

print('Shape of test data:')
print(data_in_test.shape)
print(data_out_test.shape)



# Architecture -----------------------------------------------------------------------

class CLOCK(torch.nn.Module):

    # Initialization
    def __init__(  
        self,
        hidden_channels: int,      # The 'width' or 'capacity' of the network
        num_input_features: int,         # The number of features in our input vector
        num_input_channels: int,          # The number of channels in our input vectoe
        num_output_features: int,         # The number of features in our output vector
        n_epoch: int,              # Training parameter: number of iterations or epochs
        batch_size: int,           # Training parameter: the number of samples per batch
        learning_rate: float,      # Training parameter: the initial learning rate
        ):
        super().__init__()

        # The following lines are a PyTorch quirk, we store the input parameters as class constants so they 
        # can be called within the training method
        self.hidden_channels = hidden_channels
        self.num_input_features = num_input_features
        self.num_input_channels = num_input_channels
        self.num_output_features = num_output_features
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.activation = nn.Tanh()

        # Define each layer of the feedforward architecture
        self.cnn1 = nn.Conv1d(self.num_input_channels, self.hidden_channels, kernel_size = 5, padding = 'same') 
        self.pool1 = nn.MaxPool1d(kernel_size = 2)
        self.cnn2 = nn.Conv1d(self.hidden_channels, self.hidden_channels, kernel_size = 5, padding = 'same') 
        self.pool2 = nn.MaxPool1d(kernel_size = 2)
        self.cnn3 = nn.Conv1d(self.hidden_channels, self.hidden_channels, kernel_size = 5, padding = 'same') 
        self.pool3 = nn.MaxPool1d(kernel_size = 3)
        self.cnn4 = nn.Conv1d(self.hidden_channels, self.hidden_channels, kernel_size = 5, padding = 'same')
        self.pool4 = nn.MaxPool1d(kernel_size = 3)

        self.layer1 = nn.Linear(int(self.num_input_features/(2*2*3*3)*self.hidden_channels),self.hidden_channels)
        self.layer2 = nn.Linear(self.hidden_channels,self.hidden_channels)
        self.layer3 = nn.Linear(self.hidden_channels,self.hidden_channels)
        self.layer4 = nn.Linear(self.hidden_channels,self.num_output_features)

        # Define loss
        self.loss = nn.MSELoss()

    # Forward pass function
    def forward(self, xin):

        # Extract current batch size
        batch_size = xin.shape[0]

        # Convolution steps
        x = self.pool1(self.activation(self.cnn1(xin)))
        x = self.pool2(self.activation(self.cnn2(x)))
        x = self.pool3(self.activation(self.cnn3(x)))
        x = self.pool4(self.activation(self.cnn4(x)))

        # Flatten
        x = torch.flatten(x, start_dim=-2)

        # MLP steps
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.activation(self.layer3(x))
        x = self.layer4(x)

        # Ensure outputs are within [0,1]
        xout = x #torch.sigmoid(x)
        return xout 

    # Define training loop as a function
    def train(self, data_in, data_out, data_in_test, data_out_test):

        # Select an optimizer (Adam is a good starting point for most surrogate optimization tasks) and add the relevant
        # model parameters
        print('Setting up optimizer...')
        optimizer = optim.Adam([
        	{'params': self.cnn1.parameters()},
        	{'params': self.cnn2.parameters()},
        	{'params': self.cnn3.parameters()},
        	{'params': self.cnn4.parameters()},
            {'params': self.layer1.parameters()},
            {'params': self.layer2.parameters()},
            {'params': self.layer3.parameters()},
            {'params': self.layer4.parameters()},
            ],lr = self.learning_rate)

        # Initialize loss tracking. Note that we also store the prediction for a "base case" at each epoch so we can 
        # visualize the machine's optimization route later on.
        loss_list_train = []
        loss_list_test = []

        # Begin iterating through epochs
        print('Starting training...')
        for it in range(0, self.n_epoch):

            # Shuffle the data
            ind_shuffle = torch.randperm(data_in.size()[0])
            data_in = data_in[ind_shuffle]
            data_out = data_out[ind_shuffle]

            # Loop through batches
            for ind in range(0,data_in.size()[0],self.batch_size):
                loss = 0.0
                optimizer.zero_grad()
                ind_batch = range(ind,ind+self.batch_size)
                if ind+self.batch_size > data_in.size()[0]:
                    ind_batch = range(ind,data_in.size()[0])

                # Compute predictions for each sample    
                data_pred = self(data_in[ind_batch])

                # Compute loss for each sample
                loss = self.loss(data_pred,data_out[ind_batch]) 

                # Backprogate and update the network weights
                loss.backward()
                optimizer.step()

            # Compute training data loss (for monitoring overfitting)
            data_pred = self(data_in)
            loss_train = self.loss(data_pred,data_out).item()
                
            # Compute test data loss (for monitoring overfitting)
            data_pred_test = self(data_in_test)
            loss_test = self.loss(data_pred_test,data_out_test).item()
            loss_list_train.append(loss_train) 
            loss_list_test.append(loss_test)

            # Display train + test loss at regular increments
            if it == 0 or (it+1)%100==0:
                print('Epoch: ' + str(it) + '  |  ' + 'Train Loss: ' + str(loss_train) + '  |  ' + 'Test Loss: ' + str(loss_test))

        # Store the test + train loss as a model parameter
        self.loss_list_train = loss_list_train
        self.loss_list_test = loss_list_test     


# Training -----------------------------------------------------------------------


# Initialize model architecture
print('Initializing model...')
clock = CLOCK(
	hidden_channels=16,# Change to 16 for a less noisy learning curve
	n_epoch= 1000,
	batch_size=32, # Change to 32 for a less noisy learning curve
	learning_rate = 3e-4,
    num_input_features = num_input_features,
    num_input_channels = num_input_channels,
    num_output_features = num_output_features
	)
clock = clock.to(device)
print('Model initialized!')

# Train model
clock.train(data_in_train,data_out_train,data_in_test,data_out_test)


# Evaluate finished model on training + testing data for plotting
data_pred_train = clock(data_in_train)
data_pred_test = clock(data_in_test)

# Undo normalization for plotting
for ind in range(0,num_output_features):	
    data_out_train[:,ind] = data_out_train[:,ind]*(max_out[ind]-min_out[ind]) + min_out[ind]
    data_out_test[:,ind] = data_out_test[:,ind]*(max_out[ind]-min_out[ind]) + min_out[ind]
    data_pred_train[:,ind] = data_pred_train[:,ind]*(max_out[ind]-min_out[ind]) + min_out[ind]
    data_pred_test[:,ind] = data_pred_test[:,ind]*(max_out[ind]-min_out[ind]) + min_out[ind]

    #data_out_train[:,ind] = data_out_train[:,ind]*(std_out[ind]) + mean_out[ind]
    #data_out_test[:,ind] = data_out_test[:,ind]*(std_out[ind]) + mean_out[ind]
    #data_pred_train[:,ind] = data_pred_train[:,ind]*(std_out[ind]) + mean_out[ind]
    #data_pred_test[:,ind] = data_pred_test[:,ind]*(std_out[ind]) + mean_out[ind]


#max_out[0] = np.max(data_out_train.cpu().detach().numpy()[:,0])
#min_out[0] = np.min(data_out_train.cpu().detach().numpy()[:,0])
#max_out[1] = np.max(data_out_train.cpu().detach().numpy()[:,1])
#min_out[1] = np.min(data_out_train.cpu().detach().numpy()[:,1])
#max_out[2] = np.max(data_out_train.cpu().detach().numpy()[:,2])
#min_out[2] = np.min(data_out_train.cpu().detach().numpy()[:,2])
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
ax1.plot(data_out_train[:,0].cpu().detach().numpy(),data_pred_train[:,0].cpu().detach().numpy(),'o')
ax1.plot([0,max_out[0]],[0,max_out[0]],'--k')
ax1.set_xlim(min_out[0],max_out[0])
ax1.set_ylim(min_out[0],max_out[0])
ax1.set_xlabel("True Length (nm)")
ax1.set_ylabel("Predicted Length (nm)")
ax1.set_title('Training Set')


ax2.plot(data_out_train[:,1].cpu().detach().numpy(),data_pred_train[:,1].cpu().detach().numpy(),'o')
ax2.plot([0,max_out[1]],[0,max_out[1]],'--k')
ax2.set_xlim(min_out[1],max_out[1])
ax2.set_ylim(min_out[1],max_out[1])
ax2.set_xlabel("True Width (nm)")
ax2.set_ylabel("Predicted Width (nm)")
ax2.set_title('Training Set')

ax3.plot(data_out_train[:,2].cpu().detach().numpy(),data_pred_train[:,2].cpu().detach().numpy(),'o')
ax3.plot([0,max_out[2]],[0,max_out[2]],'--k')
ax3.set_xlim(min_out[2],max_out[2])
ax3.set_ylim(min_out[2],max_out[2])
ax3.set_xlabel("True Orientation (deg.)")
ax3.set_ylabel("Predicted Orientation (deg.)")
ax3.set_title('Training Set')

ax4.plot(data_out_test[:,0].cpu().detach().numpy(),data_pred_test[:,0].cpu().detach().numpy(),'o')
ax4.plot([0,max_out[0]],[0,max_out[0]],'--k')
ax4.set_xlim(min_out[0],max_out[0])
ax4.set_ylim(min_out[0],max_out[0])
ax4.set_xlabel("True Length (nm)")
ax4.set_ylabel("Predicted Length (nm)")
ax4.set_title('Test Set')


ax5.plot(data_out_test[:,1].cpu().detach().numpy(),data_pred_test[:,1].cpu().detach().numpy(),'o')
ax5.plot([0,max_out[1]],[0,max_out[1]],'--k')
ax5.set_xlim(min_out[1],max_out[1])
ax5.set_ylim(min_out[1],max_out[1])
ax5.set_xlabel("True Width (nm)")
ax5.set_ylabel("Predicted Width (nm)")
ax5.set_title('Test Set')

ax6.plot(data_out_test[:,2].cpu().detach().numpy(),data_pred_test[:,2].cpu().detach().numpy(),'o')
ax6.plot([0,max_out[2]],[0,max_out[2]],'--k')
ax6.set_xlim(min_out[2],max_out[2])
ax6.set_ylim(min_out[2],max_out[2])
ax6.set_xlabel("True Orientation (deg.)")
ax6.set_ylabel("Predicted Orientation (deg.)")
ax6.set_title('Test Set')

# plt.savefig("truth_v_prediction.png")
# plt.show()
plt.tight_layout()
plt.savefig("truth_v_prediction_pipeline_2.png")

# Plot learning curve
fig, ax7 = plt.subplots(figsize=(6, 6))
ax7.semilogy(clock.loss_list_train, label='Train')
ax7.semilogy(clock.loss_list_test, label='Test')
ax7.set_xlabel("Epoch")
ax7.set_ylabel("Mean Squared Error (MSE)")
plt.legend()
plt.savefig("learning_curve_pipeline_2.png")

