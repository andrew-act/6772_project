import pyshark
import pandas as pd
from datetime import datetime
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Read pcap file
pcap_path = r"C:\Users\chentian an\Desktop\leetcode\002.pcapng" # 替换为你的文件路径
cap = pyshark.FileCapture(pcap_path, use_json=True, include_raw=False)
# store (timestamp, length, protocol) information
records = []
for packet in cap:
    try:
         # Extraction time, length, protocol type
        timestamp = float(packet.sniff_timestamp)
        length = int(packet.length)
        if 'TCP' in packet:
            proto = 'TCP'
        elif 'UDP' in packet:
            proto = 'UDP'
        else:
            continue  # ignore none TCP/UDP

        records.append((timestamp, length, proto))
    except Exception as e:
        print(f"Error: {e}")
        continue

cap.close()

# convert to DataFrame
df = pd.DataFrame(records, columns=['timestamp', 'length', 'protocol'])

# Rounded to the nearest 0.1 second (time bucket)
df['time_bin'] = (df['timestamp'] * 10).astype(int) / 10.0

# Rounded to the nearest 0.1 second (time bucket)
grouped = df.groupby(['time_bin', 'protocol'])['length'].sum().unstack(fill_value=0)

# Supplementary columns
if 'TCP' not in grouped.columns:
    grouped['TCP'] = 0
if 'UDP' not in grouped.columns:
    grouped['UDP'] = 0

##Sort by time
grouped = grouped.sort_index()

# print first line
print(grouped.head())

## Plotting tcp and udp traffic, based on time of day.
# plt.subplot(3,1,1)
# plt.figure(figsize=(14, 6))
# plt.plot(grouped.index, grouped['TCP'], label='TCP Bytes', color='steelblue')
# plt.plot(grouped.index, grouped['UDP'], label='UDP Bytes', color='orange')
# plt.xlabel('Time (seconds)')
# plt.ylabel('Bytes per 0.1s')
# plt.title('TCP/UDP Network Traffic per 0.1 Second')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()


# Increase the total number of bytes in a column
grouped['total'] = grouped['TCP'] + grouped['UDP']

# Use ‘TCP’, ‘UDP’, ‘total’ as features
features = grouped[['TCP', 'UDP', 'total']].values

def create_lstm_dataset(data, look_back=10, predict_step=1):
    X, y = [], []
    for i in range(len(data) - look_back - predict_step + 1):
        X.append(data[i:i + look_back])                      # Input: 10 points in time TCP/UDP/Total
        y.append(data[i + look_back + predict_step - 1][-1]) # Goal: Predict total future (i.e. total bytes)
    return np.array(X), np.array(y)

X, y = create_lstm_dataset(features, look_back=10)
print(f"LSTM input shape: {X.shape}")  # Shape: (number of samples, 10, 3)/\


scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(grouped[['TCP', 'UDP', 'total']])
X, y = create_lstm_dataset(features_scaled, look_back=10)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# create model
model = Sequential()
model.add(LSTM(64, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dense(1))  # predict total netfolw

model.compile(optimizer='adam', loss='mse')

# stop eslier to reduce overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=5)

# train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)
# predict
y_pred = model.predict(X_test)

# Reduction of predictions to original units (bytes)
y_pred_inv = scaler.inverse_transform(np.hstack([np.zeros((len(y_pred), 2)), y_pred]))[:, -1]
y_test_inv = scaler.inverse_transform(np.hstack([np.zeros((len(y_test), 2)), y_test.reshape(-1, 1)]))[:, -1]

## plot prediction
# plt.subplot(3,1,2)
# plt.figure(figsize=(14,6))
# plt.plot(y_test_inv, label='Actual Total Bytes')
# plt.plot(y_pred_inv, label='Predicted Total Bytes', alpha=0.7)
# plt.title("LSTM Prediction of Network Traffic (Total Bytes per 0.1s)")
# plt.xlabel("Time Steps")
# plt.ylabel("Bytes")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# plot loss
#plt.subplot(3,1,3)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("LSTM Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()