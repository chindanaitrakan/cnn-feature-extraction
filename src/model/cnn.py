from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Dense, Dropout, Flatten

def get_new_model(input_shape):
  '''
  This function returns a compiled CNN with specifications given above.
  '''

  input_layer = Input(shape=input_shape, name='input')
  h = Conv2D(filters=16, kernel_size=(3,3), 
             activation='relu', padding='same', name='conv2d_1')(input_layer)
  h = Conv2D(filters=16, kernel_size=(3,3), 
             activation='relu', padding='same', name='conv2d_2')(h)

  h = MaxPool2D(pool_size=(2,2), name='pool_1')(h)

  h = Conv2D(filters=16, kernel_size=(3,3), 
             activation='relu', padding='same', name='conv2d_3')(h)
  h = Conv2D(filters=16, kernel_size=(3,3), 
             activation='relu', padding='same', name='conv2d_4')(h)

  h = MaxPool2D(pool_size=(2,2), name='pool_2')(h)

  h = Conv2D(filters=16, kernel_size=(3,3), 
             activation='relu', padding='same', name='conv2d_5')(h)
  h = Conv2D(filters=16, kernel_size=(3,3), 
             activation='relu', padding='same', name='conv2d_6')(h)
    
  h = Dense(64, activation='relu', name='dense_1')(h)
  h = Dropout(0.5, name='dropout_1')(h)
  h = Flatten(name='flatten_1')(h)
  output_layer = Dense(10, activation='softmax', name='dense_2')(h)

  model = Model(inputs=input_layer, outputs=output_layer, name='model_CNN')

  model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  
  return model