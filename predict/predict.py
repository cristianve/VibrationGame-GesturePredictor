import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# Prepare the model
root_dir = os.path.abspath("")

model = root_dir + '/model/model.h5'
model_weights = root_dir + '/model/weights.h5'
cnn = load_model(model)
cnn.load_weights(model_weights)

# Keras Properties
width, height = 150, 150


def predict(file):
    x = load_img(file, target_size=(width, height))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    array = cnn.predict(x)
    result = array[0]
    print(result)

    answer = np.argmax(result)
    if answer == 0:
        print("======= | Green | =======")
    elif answer == 1:
        print("======= | Red | =======")

    return answer


predict(root_dir + "/prepare/to_test/plots/prueba.png")

