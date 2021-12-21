from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation,Flatten
from keras.layers import Reshape, Lambda, BatchNormalization,Permute
from keras.layers.merge import add, concatenate
from keras.models import Model,Sequential
from keras.layers.recurrent import LSTM

train_path = r'Dataset_GTZAN\train' #r'/content/drive/MyDrive/images_original' for colab use this
test_path = r'Dataset_GTZAN\test'

model = Sequential()

# input_shape = (224,224,1)

# inputs = Input(name='the_input', shape=input_shape, dtype='float32')  


# inner = Conv2D(64, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(inputs)  # (None, 128, 64, 64)
# inner = BatchNormalization()(inner)
# inner = Activation('relu')(inner)
# inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner)  # (None,64, 32, 64)

model.add(Conv2D(64, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), name='max1'))


model.add(Conv2D(128, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), name='max2'))


model.add(Conv2D(256, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3), padding='same', name='conv4', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1, 2), name='max3'))



model.add(Conv2D(512, (3, 3), padding='same', name='conv5', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(512, (3, 3), padding='same', name='conv6')) 
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1, 2), name='max4'))


model.add(Conv2D(512, (2, 2), padding='same', kernel_initializer='he_normal', name='con7'))
model.add(BatchNormalization())
model.add(Activation('relu'))

# model.add(Flatten())
# model.add(Dense(256))


model.add(Reshape((512,-1))) # makes it (512,56*14) i think 
model.add(Permute((2, 1)))
model.add(LSTM(32,return_sequences=True))
model.add(LSTM(32))
model.add(Flatten())
model.add(Dense(10,activation = 'softmax'))



model.build(input_shape=(None,224,224,3))

model.summary()


model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
  )

# Use the Image Data Generator to import the images from the dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                shear_range = 0.2,
                                zoom_range = 0.2,
                                horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


# Make sure you provide the same target size as initialied for the image size
training_set = train_datagen.flow_from_directory(train_path,
                                                target_size = (224, 224),
                                                batch_size = 20,
                                                class_mode = 'categorical')


test_set = test_datagen.flow_from_directory(test_path,
                                            target_size = (224, 224),
                                            batch_size = 20,
                                            class_mode = 'categorical')

r = model.fit(
  training_set,
  validation_data=test_set,
  epochs=150,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)  
  )
