import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pywt
import kymatio
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils import plot_model

# Import data from CSV
data = pd.read_csv("C:/Users/thang/Downloads/StockPAD-master/StockPAD-master/data_new/2BCG.csv")

# Extract specific columns
data = data.iloc[:, [0, 2]]  # Assuming columns 1 and 5 (0-based indexing)
# Extract the second column data and flip it
y = data.iloc[:, 1].values[::-1]

# Create array for time index
t = range(1, len(y) + 1)

# Plot histogram
plt.figure()
plt.hist(y, bins=20)  # Adjust the number of bins as needed
plt.xlabel('Price')
plt.ylabel('Number of Samples')
# plt.show()


# Calculate log-returns
a = np.zeros_like(y)
a[1:] = y[:-1]
log_returns = np.log(y/a)
log_returns[0] = 0
# Plot histogram of log-returns
plt.figure()
plt.hist(log_returns, bins=20)  # Adjust the number of bins as needed
plt.xlabel('Log return')
plt.ylabel('Number of Samples')
# plt.show()

allSignals = log_returns

# Thiết lập fs là một giá trị ngẫu nhiên
np.random.seed(0)  # Để đảm bảo kết quả tạo số ngẫu nhiên như MATLAB, bạn có thể sử dụng hàm np.random.seed()
fs = 120

tstart = 0

# Plot the cumulative wind-turbine vibration monitoring data
t = np.arange(len(allSignals)) / fs + tstart
plt.plot(t, allSignals)
plt.title("Cumulative Wind-Turbine Vibration Monitoring")
plt.xlabel("Time (sec) -- 6 seconds per day for 50 days")
plt.ylabel("Voltage")
# plt.show()

frameSize = 1 * fs
frameRate = 0.5 * fs
nframe = int(np.ceil((len(allSignals) - frameSize) / frameRate)) + 1
nday = 1

# Initialize XAll matrix
XAll = np.zeros((frameSize, nframe * nday))
colIdx = 0

for start in range(0, len(allSignals) - frameSize + 1, int(frameRate)):
    end = start + frameSize
    XAll[:, colIdx] = allSignals[start:end]
    colIdx += 1

XAll = XAll.astype(np.float32)
print(XAll.shape)

J = 3  # Number of scales
Q = 1  # Number of wavelets per octave
scattering = kymatio.Scattering1D(J=J, shape=frameSize)

SAll = scattering(XAll)

# define input sequenc
sequence = SAll
# reshape input into [samples, timesteps, features]
n_in = len(sequence)
sequence = sequence.reshape((1, n_in, 1))

# define model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_in,1)))
model.add(RepeatVector(n_in))
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer='adam', loss='mse')

model.fit(sequence, sequence, epochs=300, verbose=0)
plot_model(model, show_shapes=True, to_file='reconstruct_lstm_autoencoder.png')
# demonstrate recreation
yhat = model.predict(sequence, verbose=0)

model = model(inputs=model.inputs, outputs=model.layers[0].output)
plot_model(model, show_shapes=True, to_file='lstm_encoder.png')
# get the feature vector for the input sequence
yhat = model.predict(sequence)
print(yhat.shape)
print(yhat)