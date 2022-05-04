#!/usr/bin/env python
# coding: utf-8
#  ____________________________
# /\                           \
# \_| Created by: Nikin Baidar |
#   |                          |
#   |  ________________________|_
#   \_/_________________________/

# Import modules and specify project root directory

# # Py general
# import numpy
# import pandas
# import os
# import pathlib
# import random

# # Images/figures
# from matplotlib import pyplot
# from matplotlib import figure
# from cv2 import imread
# from cv2 import resize
# from cv2 import IMREAD_GRAYSCALE

# # Data perping
# from sklearn.utils import shuffle
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.utils import to_categorical

# # Models
# from keras.models import Sequential
# from keras.models import Model

# # Layers
# from keras.layers import Conv2D
# from keras.layers import MaxPooling2D
# from keras.layers import Flatten
# from keras.layers import Dense
# from keras.layers import Dropout

# # Activation functions
# from keras.layers import ReLU
# from keras.layers import LeakyReLU
# from keras.layers import PReLU
# from keras.layers import Softmax

# # Callbacks
# from keras.callbacks import EarlyStopping

# # Optimizers
# from tensorflow.keras.optimizers import SGD
# from tensorflow.keras.optimizers import Adam

# # HP tuners
# from tensorflow.keras.optimizers.schedules import ExponentialDecay

# # Performancec evaluation
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import classification_report

# Define the project root directory
project_root = pathlib.Path.home().joinpath('projects/tumor_classification')

# ### Class definitions

class Images:
    """ A class with methods to import and prepare images and also 
    learn the class distribution.
    
    # Args:
      * img_size
      * validation_ratio
      * random_state
    
    # Class methods:
      * getClassDistribution
      * shuffleData 
      * getRandomImage
    """
    
    ####################
    # Class attributes #
    ####################

    labels =  {
        'glioma_tumor'    : 0,
        'meningioma_tumor': 1,
        'no_tumor'        : 2,
        'pituitary_tumor' : 3
    }
    
    # Specify paths
    training_set_dir = project_root.joinpath('mri_dataset/Training')
    testing_set_dir  = project_root.joinpath('mri_dataset/Testing')
    
    def __init__(self, img_size, validation_ratio=None, random_state=101):
        
        global img_shape
    
        img_shape = (img_size, img_size, 1)
        
        self.random_state = random_state
        self.img_size = (img_size, img_size)
        self.X_train  = []
        self.Y_train  = []
        self.X_test   = []
        self.Y_test   = []
        
        # Import the training set.
        for label in self.labels.keys():
            input_dir = self.training_set_dir.joinpath(label)
            for item in os.listdir(input_dir):
                img_path  = input_dir.joinpath(item).as_posix()
                input_img = imread(img_path, IMREAD_GRAYSCALE) * (1/255.0)
                input_img = resize(input_img, self.img_size)
                self.X_train.append(input_img)
                self.Y_train.append(self.labels[label])
                
        # Import the test set.
        for label in self.labels.keys():
            input_dir = self.testing_set_dir.joinpath(label)
            for item in os.listdir(input_dir):
                path = input_dir.joinpath(item).as_posix()
                input_img = imread(path, IMREAD_GRAYSCALE) * (1/255.0)
                input_img = resize(input_img, self.img_size)
                self.X_test.append(input_img)
                self.Y_test.append(self.labels[label])
                
        # Change things into numpy arrays.
        self.x_train = numpy.array(self.X_train)
        self.y_train = numpy.array(self.Y_train)
        self.x_test  = numpy.array(self.X_test)
        self.y_test  = numpy.array(self.Y_test)
        # Reshape inputs
        self.x_train = self.x_train.reshape(-1, img_size, img_size, 1)
        self.x_test  = self.x_test.reshape(-1, img_size, img_size, 1)
        # Reshape plus encode labels/outputs
        self.y_train = self.y_train.reshape(-1, 1)
        self.y_train = to_categorical(self.y_train)
        self.y_test  = self.y_test.reshape(-1)
        
        # If validation_ratio is provided create a validation set.
        if validation_ratio:
            self.x_train, self.x_val , self.y_train, self.y_val =  \
            train_test_split(
                self.x_train,
                self.y_train,
                test_size=validation_ratio,
                random_state=self.random_state
            )
            self.Y_val   = list(numpy.argmax(self.y_val, axis=1))
            self.Y_train = list(numpy.argmax(self.y_train, axis=1))
        
    #######################################
    # Class methods pertaining to phase 1 #
    #######################################
    
    def getClassDistribution(self, mode='training', visualize=False):
        """Return a pandas dataframe with class distribution information."""
        if mode in 'training':
            classes = self.Y_train
        elif mode in 'test':
            classes = self.Y_test
        elif mode in 'validation':
            classes = self.Y_val
        else: 
            return None
        self.class_frequencies  = [classes.count(self.labels[label])
                                   for label in self.labels]
        self.class_distribution = [count/len(classes)
                                   for count in self.class_frequencies]
        data = {
            'classes'     : self.labels.keys(), 
            'value_count' : self.class_frequencies,
            'distribution': self.class_distribution
        }
        table = pandas.DataFrame(data)
        if visualize:
            pyplot.bar(self.labels.keys(), self.class_frequencies, width=0.4,
                       color=['#7EA8B4', '#A17544', '#D1D1CC', '#7F8C83'])
            pyplot.grid(linewidth=0.5, axis='y', linestyle='--' )
            pyplot.title(f'Class distribution in the {mode} set.')
        return table
        
    def shuffleData(self, mode='training'):
        """Shuffles the data in the training (or the specified) dataset."""
        if mode == 'training':
            self.x_train, self.y_train = shuffle(self.x_train, self.y_train)
        elif mode == 'test':
            self.x_test, self.y_test = shuffle(self.x_test, self.y_test)
        elif mode == 'validation':
            self.x_val, self.y_val = shuffle(self.x_val, self.y_val)
    
    def getRandomImage(self, mode='training', label=None, printLabel=False):
        """Fetches random images from the training (or the specified)
        dataset. If label is given, random images are selected from that
        class."""
        if mode == 'training':
            x, y = self.x_train, list(numpy.argmax(self.y_train, axis=1))
        elif mode == 'test':
            x, y = self.x_test, self.y_test
        elif mode == 'validation':
            try:
                x, y = self.x_val, list(numpy.argmax(self.y_val, axis=1))
            except:
                print('Validation set not defined.')
                return None
            
        label = self.labels.get(label, 4)
        if label < 4:
            tumour_class_index = 4
            while not tumour_class_index == label:
                fig = random.randrange(0, len(y))
                tumour_class_index = y[fig]
        else:
            fig = random.randrange(len(y))
            tumour_class_index = y[fig]
        tumour_class = list(self.labels.keys())[tumour_class_index]
        img = x[fig]
        if printLabel:
            print(tumour_class)
        return img    

    ########################################
    # Class methods  pertaining to phase 2 #
    ########################################
    
    ## Maybe add methos for preprocessing images later.

class CNNModel:
    """CNNModel class represents a convolutional neural net and defines
    methods to train, test, and validate the model. Other methods are
    also defined to assess the model performance and improve upon it.

    Create a sequential Convolutional Neural Network model. Add layers
    to the neural net and then compile it. The layers are added in the
    following pattern:
    
    %======================================================%
    % CONV -> [CONV -> DROPOUT? -> POOL]*M -> FLAT -> FC*2 %
    %======================================================%
    where,
      - -> represents the flow of the neural net.
      - * represents repetition.
      - CONV means a convolution layer.
      - DROPOUT? represents an optional dropout layer.
      - POOL represents a pooling layer.
      - M is the depth i.e. the number of hidden layers.
      - FLAT represents a flat layer (2D -> 1D)
      - FC means a fully connected dense layer.
    > NB: 1 CONV, 1 POOL plus a DROPoUT? collecitvely make a single
    > hidden layer unit.
  
    # Args:
      * depth (default 3): Number of hidden layers in the neural net.
      * filters (default 8):
      * kernel_size (default 3):
      * optimizer (default Adam): The optimizing algorithm.
      * loss_function (default 'categorical_crossentropy').
      * activation (default ReLU): Activation used for the 
        outputs of the input and the hidden layers.
      * addDropouts (default False): Add dropouts if model overfits.
  
    # Class Methods:
      * __init__
      * describe
      * train
      * getDataPoints
      * plotLosses
      * makePrediction
      * getConfusionMatrix
      * getClassificationReport
      * evaluate
      * finalTest
      * getConvLayers
      * extractFilters
      * displayFeatures
    """
    
    ####################
    # Class attributes #
    ####################
  
    depth       = 3
    filters     = 8
    kernel_size = (3,3)
    strides     = 1
    padding     = 'same'
    optimizer   = Adam()
    activation  = ReLU()
    loss        = 'categorical_crossentropy'
  
    def __init__(self, depth=depth, filters=filters, kernel_size=kernel_size,
                 strides=strides, padding=padding, optimizer=optimizer,
                 activation=activation, addDropouts=False):
        """Compiles the neural net."""
        self.classifier  = Sequential()
        self.input_shape = img_shape
        self.depth       = depth
        self.filters     = filters
        self.kernel_size = kernel_size
        self.strides     = strides
        self.padding     = padding
        self.optimizer   = optimizer
        self.activation  = activation

        # Start adding layers to the classifier.
        self.classifier.add(
            Conv2D(
                filters=self.filters,
                kernel_size=self.kernel_size,
                padding=self.padding,
                strides=1,
                activation=self.activation,
                input_shape=self.input_shape,
                name='input'
            )
        )
        
        for count in range(depth):
            self.filters *= 2 # Linearly increase #filters as you go deeper. 
            self.classifier.add(
                Conv2D(
                    filters=self.filters,
                    kernel_size=self.kernel_size,
                    strides=2,
                    padding='same',
                    activation=self.activation,
                    name=f'conv{count + 1}'
                )
            )
            if addDropouts:
                self.classifier.add(
                    Dropout(0.3, name=f'drop{count + 1}')
                )
            self.classifier.add(
                MaxPooling2D(
                    pool_size=(2,2),
                    strides=2,
                    name=f'pool{count + 1}'
                )
            )

        self.classifier.add(Flatten(name='flat{}'.format(count:=count+2)))
        
        self.classifier.add(
            Dense(
                units=8,
                activation='sigmoid',
                name='dense{}'.format(count := count+1)
            )
        )
        
        self.classifier.add(Dense(units=4, activation=Softmax(), name='output'))
    
        self.classifier.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=['accuracy']
        )
        
        # For the 2nd phase.
        self.convLayers = [index for (index, layer) 
                           in enumerate(self.classifier.layers)
                           if isinstance(layer, Conv2D)]
        
        self.poolLayers = [index for (index, layer) 
                           in enumerate(self.classifier.layers)
                           if isinstance(layer, MaxPooling2D)]
        
    #######################################
    # Class methods pertaining to phase 1 #
    #######################################
    
    def describe(self):
        """Print the classifier summary."""
        return self.classifier.summary()
        
    def train (self, x, y, epochs=10, batch_size=32, class_weight=None, 
               validation_data=None, showProgress=False):
        """Trains an instance of class CNNclassifier on a dataset and
        returns a history object with the training history.
    
        ## Args:
           * x, y: input and labels 
           * epochs (default 10): Number of epochs to train the
             classifier. An epoch is an iteration over the entire
             dataset. 
           * class_weight: A dict object that maps labels to a weight.
             Tells the classifier to "pay more attention" to samples 
             from an under-represented class.
           * validation_data: Dataset on which to evaluate the loss and
             any classifier metrics at the end of each epoch.
           * patience (default 10): If validation_data is provided,
             the number of epochs after which training will stop if
             there isn't any progress.
           * showProgress: Enable/Disable verbosity.
        """
        callback_list=[]

        if validation_data:
            callback_list.append(EarlyStopping(
                monitor='val_loss', patience=0.4*epochs)
            )
        self.history = self.classifier.fit(
            x, y,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            verbose=showProgress,
            validation_data=validation_data,
            callbacks=callback_list
        )
        
        self.history = self.history.history
    
    def getDataPoints(self, mode='loss'):
        """ Returns the losses during the training. Mode can be any one of 
        the keys of the History created during fitting."""
        return self.history[mode]
  
    def plotLosses(self):
        pyplot.plot(self.history['loss'], 'o-g', label="Training Losses")
        if 'val_loss' in self.history.keys():
            pyplot.plot(self.history['val_loss'], 'o--',
                        label="Validation Losses")
        pyplot.xlabel("Epochs")
        pyplot.ylabel("Losses")
        pyplot.title("Losses over Epochs")
        pyplot.legend()
        pyplot.grid(linewidth=0.5)
        pyplot.tight_layout()
  
    def makePrediction(self, x):
        """Predicts the classes of a the input x."""
        self.predicted_classes = self.classifier.predict(x)
  
    def getConfusionMatrix(self, true_classes):
        """Takes true classes and returns a confusion matrix as a pandas
        dataframe."""
        labels = Images.labels.keys()
        try:
            true_classes.shape[1]
            true_classes = numpy.argmax(true_classes, axis=1)
        except:
            pass
        predicted_classes = numpy.argmax(self.predicted_classes, axis=1)
        confusionMatrix = confusion_matrix(true_classes, predicted_classes)
        confusionMatrix = pandas.DataFrame(confusionMatrix, columns=labels)
        confusionMatrix.insert(0, 'labels', labels)
        print('\nConfusion Matrix:')
        print('='*17, end='\n'*2)
        print(confusionMatrix)
  
    def getClassificationReport(self, true_classes):
        try:
            true_classes.shape[1]
            true_classes = numpy.argmax(true_classes, axis=1)
        except:
            pass
        predicted_classes = numpy.argmax(self.predicted_classes, axis=1)
        classificationReport = classification_report(
            true_classes, predicted_classes, zero_division=0
        )
        print('\nClassification Report:')
        print('='*22, end='\n'*2)
        print(classificationReport)
        
    def evaluate(self):
        self.makePrediction(tumours.x_val)
        self.getConfusionMatrix(tumours.y_val)
        self.getClassificationReport(tumours.y_val)
        
    def finalTest(self):
        self.makePrediction(tumours.x_test)
        self.getConfusionMatrix(tumours.y_test)
        self.getClassificationReport(tumours.y_test)
        
    #######################################
    # Class methods pertaining to phase 2 #
    #######################################
 
    def extractFilters(self):
        """Extract the filters for the convolutional layers."""
        print('#'*35)
        print('# Filters of convolutional layers #')
        print('#'*35, end='\n'*2)
        filters = [self.classifier.layers[i].get_weights()[0]
                   for i in self.convLayers]
        layer_count = 0
        for filter_ in filters:
            filter_count = filter_.shape[-1]
            filter_depth = filter_.shape[-2]
            fig = pyplot.figure(figsize=(35,35))
            fig.suptitle(f'Layer_{self.convLayers[layer_count]}', fontsize=35)
            layer_count += 1
            for i in range(filter_count):
                pyplot.subplot(8, 8, i+1)
                pyplot.imshow(filter_[:,:,0,i], cmap='gray')
                pyplot.axis('off')
                pyplot.tight_layout(pad=3.0)
                pyplot.gcf().set_facecolor('orangered')
                        
    def displayFeatureMaps(self, layer_instance, image):
        if layer_instance == 'conv':
            layers = self.convLayers
            print('#'*36)
            print('# Features of convolutional layers #')
            print('#'*36, end='\n'*2)
        elif layer_instance == 'pool':
            layers = self.poolLayers
            print('#'*30)
            print('# Features of pooling layers #')
            print('#'*30, end='\n'*2)
        else:
            raise ValueError(f'{layer_instance} is not defined.')
            
        inputs  = self.classifier.inputs
        outputs = [self.classifier.layers[i].output for i in layers]
        fmapper = Model(inputs=inputs, outputs=outputs)
        image   = numpy.expand_dims(image, axis=0)
        feature_maps = fmapper.predict(image)
        layer_count = 0
        for fmap in feature_maps:
            fmap_count = fmap.shape[-1]
            fmap_depth = fmap.shape[0]
            fig = pyplot.figure(figsize=(35,35))
            fig.suptitle(f'Layer_{layers[layer_count]}' , fontsize=40)
            layer_count += 1
            for i in range(fmap_count):
                pyplot.subplot(8, 8, i+1)
                pyplot.imshow(fmap[0,:,:,i] , cmap='gray')
                pyplot.axis('off')
                pyplot.tight_layout(pad=3.0)
