def create_lstm_model(input_shape, output_shape, lstm_layers, lstm_units, dropout_rate=0.2, optimizer='adam'):
    model = Sequential()
    
    for i, units in enumerate(lstm_units):
        if i == 0:
            model.add(LSTM(units, input_shape=input_shape, return_sequences=True))
        else:
            model.add(LSTM(units, return_sequences=True))
        model.add(Dropout(dropout_rate))
        
    model.add(LSTM(lstm_units[-1], return_sequences=False))
    model.add(Dropout(dropout_rate))
    
    model.add(Dense(output_shape))
    
    model.compile(optimizer=optimizer, loss='mse')
    
    return model
