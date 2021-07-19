import os
from tensorflow.keras import backend as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam

# Clear the session to create a new model
keras.clear_session()


# Get Directories
root_dir = os.path.abspath("")
training_data = os.path.join(root_dir, "train/data/training")
validation_data = os.path.join(root_dir, "train/data/validation")


# Keras Parameters
epochs = 20
width, height = 150, 150
batch_size = 22
train_steps = 22
validation_steps = 22
conv1_filters = 32
conv2_filters = 64
filter1_size = (3, 3)
filter2_size = (2, 2)
pool_size = (2, 2)
classes = 2
lr = 0.0004

# Prepare the images

training_data_gen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_data_gen = ImageDataGenerator(rescale=1. / 255)

training_generator = training_data_gen.flow_from_directory(
    training_data,
    target_size=(height, width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_data_gen.flow_from_directory(
    validation_data,
    target_size=(height, width),
    batch_size=batch_size,
    class_mode='categorical')

cnn = Sequential()
cnn.add(Convolution2D(conv1_filters, filter1_size, padding="same", input_shape=(width, height, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=pool_size))

cnn.add(Convolution2D(conv2_filters, filter2_size, padding="same"))
cnn.add(MaxPooling2D(pool_size=pool_size))

cnn.add(Flatten())
cnn.add(Dense(256, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(classes, activation='softmax'))

cnn.compile(loss='categorical_crossentropy',
            optimizer=Adam(lr=lr),
            metrics=['accuracy'])

cnn.fit_generator(
    training_generator,
    steps_per_epoch=train_steps,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps)

# Generate the model

target_dir = root_dir + '/model/'

if not os.path.exists(target_dir):
    os.mkdir(target_dir)
cnn.save('./model/model.h5')
cnn.save_weights('./model/weights.h5')
