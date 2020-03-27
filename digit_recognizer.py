# Digit Recognizer

# importing libraries
import pandas as pd
import numpy as np

# Importing datasets
dataset = pd.read_csv("train.csv")
ds = pd.read_csv("test.csv")
X = dataset.iloc[:, 1: ].values
y = dataset.iloc[:, 0].values
X_test = ds.iloc[:, : ].values

# Splitting into training & validation datasets
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                      test_size = 0.2,
                                                      random_state = 0)


# Reshaping X & X_test
X_train = X_train.reshape(-1, 28, 28, 1).astype("float32")
X_valid = X_valid.reshape(-1, 28, 28, 1).astype("float32")
X_test = X_test.reshape(-1, 28, 28, 1).astype("float32")

# Normalizing the inputs
X_train = X_train / 255
X_valid = X_valid / 255

# Encoding labels
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
lbl_y = LabelEncoder()
y_train = lbl_y.fit_transform(y_train)
y_train = y_train.reshape(-1, 1)
ohe = OneHotEncoder(categorical_features = [0])
y_train = ohe.fit_transform(y_train).toarray()

y_valid = lbl_y.fit_transform(y_valid)
y_valid = y_valid.reshape(-1, 1)
ohe = OneHotEncoder(categorical_features = [0])
y_valid = ohe.fit_transform(y_valid).toarray()

'''
# Splitting into training & validation datasets
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                      test_size = 0.2,
  '''                                                    random_state = 0)

# Building the CNN Model
# Importing libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
#from keras.optimizers import RMSprop

# Bulding the network
# Step 1 - Adding the convolutional layers
classifier = Sequential()

classifier.add(Convolution2D(filters = 30,
                     activation = "relu",
                     kernel_size = (5,5),
                     input_shape = (28, 28, 1)))

classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Convolution2D(filters = 15,
                     activation = "relu",
                     kernel_size = (3,3)))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Dropout(0.2))
classifier.add(Flatten())

classifier.add(Dense(units = 128, activation = "relu"))
classifier.add(Dense(units = 50, activation = "relu"))
classifier.add(Dense(units = 10, activation = "softmax"))

classifier.compile(optimizer = "adam",
                   loss = "categorical_crossentropy",
                   metrics = ["accuracy"])

# Fitting the model to the dataset and validating its performance with test set
classifier.fit(X_train,
               y_train,
               batch_size = 200,
               epochs = 80,
               verbose = 1,
               validation_data = (X_valid, y_valid))
'''# Data Augmentation
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale = 1./255,
                                   rotation_range = 1,
                                   zoom_range = 0.2,
                                   width_shift_range = 0.05,
                                   height_shift_range = 0.05)

datagen.fit(X_train)

classifier.fit_generator(datagen.flow(X_train, y_train, batch_size = 512),
                         steps_per_epoch = X_train.shape[0] // 512,
                         epochs = 10,
                         verbose = 2,
                         validation_data = (X_valid, y_valid))
'''
# Prediction
y_pred = classifier.predict_classes(X_test)

# Kaggle Submission CSV
pd.DataFrame({"ImageId":list(range(1,len(y_pred)+1)),
              "Label":y_pred}).to_csv("complex_cnn_2.csv",
                                           index=False,
                                           header=True)