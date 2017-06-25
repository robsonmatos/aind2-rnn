import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series,window_size):
    # Create containers for input/output pairs
    X = []
    y = []
    
    # Produce input/output pairs sliding a window of size window_size over the time series and spacing each window 
    # by one time step
    window_start = 0
    while (window_start + window_size) < len(series):
        X.append(series[window_start:window_start + window_size])
        y.append(series[window_start + window_size])
        window_start += 1
        
    # Reshape each input/output pair
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)
    
    return X, y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(step_size, window_size):
    # Create a sequential model
    model = Sequential()
    # Add a LSTM layer with 5 hidden units and input_shape=(window_size, 1)
    model.add(LSTM(5, input_shape=(window_size, 1)))
    # Add a Dense layer with linear activation
    model.add(Dense(1, activation='linear'))

    return model

### TODO: list all unique characters in the text and remove any non-english ones
def clean_text(text):
    # find all unique characters in the text
    print(sorted(list(set(text))))

    # remove as many non-english characters and character sequences as you can
    text = text.replace('$',' ')
    text = text.replace('"',' ')
    text = text.replace('%',' ')
    text = text.replace('*',' ')
    text = text.replace('/',' ')
    text = text.replace('@',' ')
    text = text.replace('à',' ')
    text = text.replace('â',' ')
    text = text.replace('é',' ')
    text = text.replace('è',' ')
    text = text.replace('-',' ')

    # shorten any extra dead space created above
    text = text.replace('  ',' ')

    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text,window_size,step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    
    window_start = 0
    while (window_start + window_size) < len(text):
        inputs.append(text[window_start:window_start + window_size])
        outputs.append(text[window_start + window_size])
        window_start += step_size
    
    return inputs, outputs
