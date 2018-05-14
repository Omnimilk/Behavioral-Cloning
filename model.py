import csv
import cv2
import numpy as np
from PIL import Image
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Cropping2D,Dropout,ELU, Flatten, Dense, Lambda, Conv2D,MaxPooling2D
from keras.regularizers import l2
import matplotlib.pyplot as plt

def compensate_steering(image,steering,trans_range):
	"""
	compensate for the horizontal translation on the steering angle
	"""	
	rows, cols, ch = image.shape
	# horizontal translation with 0.008 steering compensation per pixel
	tr_x = trans_range * np.random.uniform()-trans_range/2
	steer_ang = steering + tr_x/trans_range*.4
	
	translate_matrix = np.float32([[1,0,tr_x],
                                    [0,1,0]])
	image_tr = cv2.warpAffine(image,translate_matrix,(cols,rows))
	
	return image_tr,steer_ang

def data_augmentation(image, steer_ang, trans_range = 50):
    """Augment the image and steering angle"""    
    rows, cols, chs = image.shape
  
    # translate image and compensate for steering angle 
    image, steer_ang = compensate_steering(image, steer_ang, trans_range) 
    
    # crop image region of interest
    image = image[70:136, 0+trans_range:cols-trans_range]#66*220
    
    # flip image and steering with 50/50 chance
    if np.random.uniform()>= 0.5: 
        image = cv2.flip(image, 1)
        steer_ang = -steer_ang
    
    # augment brightness
    image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image[:,:,2] = image[:,:,2] * np.random.uniform(0.25,1.25) 
    # print(image.shape) 
    return image, steer_ang

def generate_batch_samples(path, batch_size=128):
    while True:
        batch_images, batch_steerings = [],[]
        with open(path + 'driving_log.csv') as f:
            reader = csv.reader(f)
            for line in reader:
				#skip the title line
                if line[3] =="steering":
                    continue
                steering = float(line[3])
				#use multiple cameras, add correction steerings to side cameras
                correction =0.25
                camera = np.random.randint(3)#choose a random camera, 0 for center; 1 for left; 2 for right
                if camera ==0:
                    img = np.asarray(Image.open(path+line[0]))
                elif camera ==1:
                    img = np.asarray(Image.open(path+line[1][1:]))
                    steering += correction
                else:
                    img = np.asarray(Image.open(path+line[2][1:]))
                    steering -= correction
                #augment data
                img, steering = data_augmentation(img,steering)
                # print("img shape {}".format(img.shape))
                batch_images.append(img)
                batch_steerings.append(steering)
				
                if len(batch_images) >= batch_size/2.0:
                    #convert to nparray
                    batch_images = np.asarray(batch_images)
                    batch_steerings = np.asarray(batch_steerings).squeeze()   
					# shuffle batch
                    batch_images, batch_steerings, = shuffle(batch_images, batch_steerings, random_state=0)
					#may waste some data
                    # print("batch images shape: {}, batch steerings shape: {}".format(batch_images.shape, batch_steerings.shape))
                    yield (batch_images[0:batch_size,:,:,:], batch_steerings[0:batch_size])
                    batch_images, batch_steerings = [], []

def leNetModel():
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape = (160,320,3)))
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Conv2D(6,3,3,activation='relu'))
    model.add(Conv2D(6,3,3,activation='relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(12,3,3,activation='relu'))
    model.add(Conv2D(12,3,3,activation='relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    model.compile(loss="mse",optimizer = "adam")
    return model

def nvidiaModel(keep_prob=0.2,reg_val=0.01):
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1, input_shape = (66,220,3)))
    # model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Conv2D(24, 5, 5, subsample=(2, 2), border_mode="valid", init="glorot_uniform", W_regularizer=l2(reg_val)))
    model.add(ELU())
    model.add(Dropout(keep_prob))
	
    model.add(Conv2D(36, 5, 5, subsample=(2, 2), border_mode="valid", init="glorot_uniform"))
    model.add(ELU())
    model.add(Dropout(keep_prob))
	
    model.add(Conv2D(48, 5, 5, subsample=(2, 2), border_mode="valid", init="glorot_uniform"))
    model.add(ELU())
    model.add(Dropout(keep_prob))

    model.add(Conv2D(64, 3, 3, subsample=(1, 1), border_mode="valid", init="glorot_uniform"))
    model.add(ELU())
    model.add(Dropout(keep_prob))

    model.add(Conv2D(64, 3, 3, subsample=(1, 1), border_mode="valid", init="glorot_uniform"))
    model.add(ELU())
    model.add(Dropout(keep_prob))
	
    model.add(Flatten())

    model.add(Dense(100))
    model.add(ELU())
    model.add(Dropout(keep_prob))
 
    model.add(Dense(50))
    model.add(ELU())
    model.add(Dropout(keep_prob))
	
    model.add(Dense(10))
    model.add(ELU())

    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse") 	
    return model

def main():
	# initialize generators
    training_gen = generate_batch_samples("data/", batch_size=128)
    model = nvidiaModel()
    model.fit_generator(training_gen,nb_epoch=8,samples_per_epoch=128*300)
    model.save("nvidiaModel.h5")

    # img = np.asarray(Image.open("data/IMG/center_2016_12_01_13_32_54_976.jpg"))
    # trans_range  =50
    # img_cut = img[70:135, 0+trans_range:320-trans_range]
    # plt.imshow(img)
    # plt.show()
    # plt.imshow(img_cut)
    # plt.show()



if __name__ == '__main__':
    main()
	



