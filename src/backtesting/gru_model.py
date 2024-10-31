from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout

def create_gru_model(input_shape, output_shape, gru_layers, gru_units, dropout_rate=0.2, optimizer='adam', recurrent_dropout=0.0, bidirectional=False):
    model = Sequential()
    
    for i, units in enumerate(gru_units):
        if i == 0:
            # First GRU layer
            if bidirectional:
                model.add(Bidirectional(GRU(units, input_shape=input_shape, return_sequences=True, recurrent_dropout=recurrent_dropout)))
            else:
                model.add(GRU(units, input_shape=input_shape, return_sequences=True, recurrent_dropout=recurrent_dropout))
        else:
            # Subsequent GRU layers
            if bidirectional:
                model.add(Bidirectional(GRU(units, return_sequences=True, recurrent_dropout=recurrent_dropout)))
            else:
                model.add(GRU(units, return_sequences=True, recurrent_dropout=recurrent_dropout))
        
        model.add(Dropout(dropout_rate))
        
    # Final GRU layer
    if bidirectional:
        model.add(Bidirectional(GRU(gru_units[-1], return_sequences=False, recurrent_dropout=recurrent_dropout)))
    else:
        model.add(GRU(gru_units[-1], return_sequences=False, recurrent_dropout=recurrent_dropout))
    
    model.add(Dropout(dropout_rate))
    
    model.add(Dense(output_shape))
    
    model.compile(optimizer=optimizer, loss='mse')
    
    return model
