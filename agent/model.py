import tensorflow as tf
# from tensorflow.keras.layers import Input, Concatenate, Dense
from keras.layers import Input, Dense
from keras.models import Model
from agent.layers import RunningStatsNorm, Variable, Squeeze


def mlp_model(state_size,
        action_size,
        layer_size=(64, 64),
        activation='relu',
        state_shift=True,
        state_scale=True,
        value_separate=False,
        continuous=True):
    
    inputs = Input(shape=(state_size,), dtype='float32', name='state')
    norm_inputs = x = RunningStatsNorm(state_shift, state_scale, name="norm_state")(inputs)
    for i, size in enumerate(layer_size):
        x = Dense(size, activation=activation, name='%sfc%02d' % ('pi_', i+1))(x)

    logits = Dense(action_size, activation=None, name='logits')(x)

    if continuous:
        logits = Variable(name='logstd')(logits)

    if value_separate:
        x = norm_inputs
        for i, size in enumerate(layer_size):
            x = Dense(size, activation=activation, name='%sfc%02d' % ('value_', i+1))(x)

    value = Dense(1, activation=None, name='value_out')(x)
    value = Squeeze(axis=-1)(value)

    return Model(inputs, [logits, value])
