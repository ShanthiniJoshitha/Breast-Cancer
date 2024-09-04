import os
import random
import shutil
from sklearn.model_selection import train_test_split

# Set the path to your dataset directory
dataset_dir = '/kaggle/input/breast-ultrasound-images-dataset/Dataset_BUSI_with_GT'

# Set the path to the directory where you want to store the train and test splits
output_dir = '/kaggle/working/'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Split ratio for train and test sets
train_ratio = 0.8
test_ratio = 0.2

# Get the list of subdirectories (classes) in the dataset directory
class_dirs = [subdir for subdir in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, subdir))]

# Iterate over the class subdirectories
for class_dir in class_dirs:
    # Get the path to the current class directory
    current_class_dir = os.path.join(dataset_dir, class_dir)
    
    # Get the list of image files in the current class directory
    image_files = os.listdir(current_class_dir)
    
    # Split the image files into train and test sets
    train_files, test_files = train_test_split(image_files, test_size=test_ratio, random_state=42)
    
    # Create separate directories for train and test splits within the output directory
    train_dir = os.path.join(output_dir, 'train', class_dir)
    os.makedirs(train_dir, exist_ok=True)
    
    test_dir = os.path.join(output_dir, 'test', class_dir)
    os.makedirs(test_dir, exist_ok=True)
    
    # Move the train files to the train split directory
    for train_file in train_files:
        src_path = os.path.join(current_class_dir, train_file)
        dst_path = os.path.join(train_dir, train_file)
        shutil.copy(src_path, dst_path)
    
    # Move the test files to the test split directory
    for test_file in test_files:
        src_path = os.path.join(current_class_dir, test_file)
        dst_path = os.path.join(test_dir, test_file)
        shutil.copy(src_path, dst_path)
import matplotlib.pyplot as plt 
import seaborn as sns
import keras 
from keras.models import Sequential 
from keras.layers import Conv2D,Flatten,Dense,MaxPooling2D,Dropout
from sklearn.metrics import accuracy_score
train_dir='/kaggle/working/train'
test_dir='/kaggle/working/test'
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_directory(train_dir,target_size=(224,224),batch_size=20,class_mode='categorical')
test_generator=test_datagen.flow_from_directory(test_dir,target_size=(224,224),batch_size=20,class_mode='categorical')
CONVOLUTION NEURAL NETWORK

model=Sequential()
print(train_generator[0][0].shape)
MODEL

from keras.layers import Input

input_shape = (224, 224, 3)

model = Sequential()
model.add(Input(shape=input_shape))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.summary()
from tensorflow.keras import optimizers 
model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
history = model.fit(train_generator, steps_per_epoch=train_generator.samples // train_generator.batch_size,
                    epochs=10, validation_data=test_generator, validation_steps=test_generator.samples // test_generator.batch_size)
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(1,len(acc)+1)
plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.legend()

plt.show()
plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.legend()

plt.show()
model.save('breastcancer.h5')
TESTING BY GIVING ONE KNOWN IMAGE AS INPUT

from tensorflow.keras.preprocessing import image 
path='/kaggle/working/test/malignant/malignant (153).png'
img=image.load_img(path,target_size=(224,224))
plt.imshow(img,interpolation='nearest')
plt.show()
img_array=np.array(img)
img_array.shape
img_array=img_array.reshape(1,224,224,3)
a=model.predict(img_array)
if a[0][0]==1.:
    print('benign')
elif a[0][1]==1.:
    print('malignant')
else:
    print('normal')
