# Deep Learning Framework
import tensorflow as tf  
from tensorflow.keras.optimizers import *  
from tensorflow import keras 
from keras import layers  
from keras.layers import *  
from keras.models import Model  
from keras import backend as K 

# Define image-related constants
IMG_WIDTH = 256 
IMG_HEIGHT = 256  
IMG_CHANNELS = 4  
n_classes = 1  


####################################################################################################


def mk_simple_net(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS):
    # input_layer = keras.Input(shape=(None, None, 4), name='input')
    input_layer = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), name='input')

    x = keras.layers.Conv2D(filters=4, kernel_size=(3,3), padding='same', activation='relu')(input_layer)
    x = keras.layers.Conv2D(filters=8, kernel_size=(3,3), padding='same', activation='relu')(x)
    x = keras.layers.Conv2D(filters=8, kernel_size=(3,3), padding='same', activation='relu')(x)
    x = keras.layers.Conv2D(filters=16, kernel_size=(3,3), padding='same', activation='relu')(x)
    x = keras.layers.Conv2D(filters=32, kernel_size=(1,1), padding='same', activation='relu')(x)
    x = keras.layers.Conv2D(filters=n_classes, kernel_size=(1,1), padding='same', activation='sigmoid')(x)
    return keras.Model(input_layer, x)


####################################################################################################


def mk_multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = inputs

    #Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
     
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
     
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
     
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    #Expansive path 
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
     
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
     
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
     
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
     
    outputs = Conv2D(n_classes, (1, 1), activation='sigmoid')(c9)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model

#-----------------------------------------------

def down_block(x, filters, use_maxpool = True):
    x = Conv2D(filters, 3, padding= 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters, 3, padding= 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    if use_maxpool == True:
        return  MaxPooling2D(strides= (2,2))(x), x
    else:
        return x
    
def up_block(x,y, filters):
    x = UpSampling2D()(x)
    x = Concatenate(axis = 3)([x,y])
    x = Conv2D(filters, 3, padding= 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters, 3, padding= 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    return x

def mk_Unet(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS):
    dropout = 0.2
    filter = [64,128,256,512, 1024]
    
    # encode
    # input = Input(shape = input_size)
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    x, temp1 = down_block(inputs, filter[0])
    x, temp2 = down_block(x, filter[1])
    x, temp3 = down_block(x, filter[2])
    x, temp4 = down_block(x, filter[3])
    x = down_block(x, filter[4], use_maxpool= False)
    # decode 
    x = up_block(x, temp4, filter[3])
    x = up_block(x, temp3, filter[2])
    x = up_block(x, temp2, filter[1])
    x = up_block(x, temp1, filter[0])
    x = Dropout(dropout)(x)

    # output = Conv2D(classes, 1, activation= 'softmax')(x)
    # model = models.Model(input, output, name = 'unet')
    # model.summary()
    # return model

    outputs = Conv2D(n_classes, (1, 1), activation='sigmoid')(x)
    model = Model(inputs=[inputs], outputs=[outputs], name = 'unet')
    return model

# source: https://github.com/Nguyendat-bit/U-net


####################################################################################################


def mk__unet_plusplus(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS):
    nb_filter = [32,64,128,256,512]
    # Build U-Net++ model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    c1 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (inputs)
    c1 = Dropout(0.5) (c1)
    c1 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
    c1 = Dropout(0.5) (c1)
    p1 = MaxPooling2D((2, 2), strides=(2, 2)) (c1)

    c2 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = Dropout(0.5) (c2)
    c2 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
    c2 = Dropout(0.5) (c2)
    p2 = MaxPooling2D((2, 2), strides=(2, 2)) (c2)

    up1_2 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up12', padding='same')(c2)
    conv1_2 = concatenate([up1_2, c1], name='merge12', axis=3)
    c3 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv1_2)
    c3 = Dropout(0.5) (c3)
    c3 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
    c3 = Dropout(0.5) (c3)

    conv3_1 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
    conv3_1 = Dropout(0.5) (conv3_1)
    conv3_1 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv3_1)
    conv3_1 = Dropout(0.5) (conv3_1)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_1)

    up2_2 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up22', padding='same')(conv3_1)
    conv2_2 = concatenate([up2_2, c2], name='merge22', axis=3) #x10
    conv2_2 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv2_2)
    conv2_2 = Dropout(0.5) (conv2_2)
    conv2_2 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv2_2)
    conv2_2 = Dropout(0.5) (conv2_2)

    up1_3 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up13', padding='same')(conv2_2)
    conv1_3 = concatenate([up1_3, c1, c3], name='merge13', axis=3)
    conv1_3 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv1_3)
    conv1_3 = Dropout(0.5) (conv1_3)
    conv1_3 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv1_3)
    conv1_3 = Dropout(0.5) (conv1_3)

    conv4_1 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (pool3)
    conv4_1 = Dropout(0.5) (conv4_1)
    conv4_1 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv4_1)
    conv4_1 = Dropout(0.5) (conv4_1)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(conv4_1)

    up3_2 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up32', padding='same')(conv4_1)
    conv3_2 = concatenate([up3_2, conv3_1], name='merge32', axis=3) #x20
    conv3_2 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv3_2)
    conv3_2 = Dropout(0.5) (conv3_2)
    conv3_2 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv3_2)
    conv3_2 = Dropout(0.5) (conv3_2)

    up2_3 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up23', padding='same')(conv3_2)
    conv2_3 = concatenate([up2_3, c2, conv2_2], name='merge23', axis=3)
    conv2_3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv2_3)
    conv2_3 = Dropout(0.5) (conv2_3)
    conv2_3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv2_3)
    conv2_3 = Dropout(0.5) (conv2_3)

    up1_4 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up14', padding='same')(conv2_3)
    conv1_4 = concatenate([up1_4, c1, c3, conv1_3], name='merge14', axis=3)
    conv1_4 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv1_4)
    conv1_4 = Dropout(0.5) (conv1_4)
    conv1_4 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv1_4)
    conv1_4 = Dropout(0.5) (conv1_4)

    conv5_1 = Conv2D(512, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (pool4)
    conv5_1 = Dropout(0.5) (conv5_1)
    conv5_1 = Conv2D(512, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv5_1)
    conv5_1 = Dropout(0.5) (conv5_1)

    up4_2 = Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up42', padding='same')(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=3) #x30
    conv4_2 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv4_2)
    conv4_2 = Dropout(0.5) (conv4_2)
    conv4_2 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv4_2)
    conv4_2 = Dropout(0.5) (conv4_2)

    up3_3 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up33', padding='same')(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1, conv3_2], name='merge33', axis=3)
    conv3_3 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv3_3)
    conv3_3 = Dropout(0.5) (conv3_3)
    conv3_3 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv3_3)
    conv3_3 = Dropout(0.5) (conv3_3)

    up2_4 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up24', padding='same')(conv3_3)
    conv2_4 = concatenate([up2_4, c2, conv2_2, conv2_3], name='merge24', axis=3)
    conv2_4 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv2_4)
    conv2_4 = Dropout(0.5) (conv2_4)
    conv2_4 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv2_4)
    conv2_4 = Dropout(0.5) (conv2_4)

    up1_5 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up15', padding='same')(conv2_4)
    conv1_5 = concatenate([up1_5, c1, c3, conv1_3, conv1_4], name='merge15', axis=3)
    conv1_5 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv1_5)
    conv1_5 = Dropout(0.5) (conv1_5)
    conv1_5 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv1_5)
    conv1_5 = Dropout(0.5) (conv1_5)

    nestnet_output_4 = Conv2D(n_classes, (1, 1), activation='sigmoid', kernel_initializer = 'he_normal',  name='output_4', padding='same')(conv1_5)
    
    model = Model([inputs], [nestnet_output_4])
    
    return model


####################################################################################################


def mk_rs_net(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS, use_batch_norm=True, dropout_on_last_layer_only=True):
    input_layer = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), name='input')
    # -----------------------------------------------------------------------
    conv1 = keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same',
                      kernel_regularizer=tf.keras.regularizers.L2(1e-4))(input_layer)
    conv1 = keras.layers.BatchNormalization(momentum=0.70)(conv1) if use_batch_norm else conv1
    conv1 = keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same',
                      kernel_regularizer=tf.keras.regularizers.L2(1e-4))(conv1)
    conv1 = BatchNormalization(momentum=0.70)(conv1) if use_batch_norm else conv1
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # -----------------------------------------------------------------------
    conv2 = keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same',
                      kernel_regularizer=tf.keras.regularizers.L2(1e-4))(pool1)
    conv2 = keras.layers.BatchNormalization(momentum=0.70)(conv2) if use_batch_norm else conv2
    conv2 = keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same',
                      kernel_regularizer=tf.keras.regularizers.L2(1e-4))(conv2)
    conv2 = keras.layers.BatchNormalization(momentum=0.70)(conv2) if use_batch_norm else conv2
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    # -----------------------------------------------------------------------
    conv3 = keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same',
                      kernel_regularizer=tf.keras.regularizers.L2(1e-4))(pool2)
    conv3 = keras.layers.BatchNormalization(momentum=0.70)(conv3) if use_batch_norm else conv3
    conv3 = keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same',
                      kernel_regularizer=tf.keras.regularizers.L2(1e-4))(conv3)
    conv3 = keras.layers.BatchNormalization(momentum=0.70)(conv3) if use_batch_norm else conv3
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    # -----------------------------------------------------------------------
    conv4 = keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same',
                      kernel_regularizer=tf.keras.regularizers.L2(1e-4))(pool3)
    conv4 = keras.layers.BatchNormalization(momentum=0.70)(conv4) if use_batch_norm else conv4
    conv4 = keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same',
                      kernel_regularizer=tf.keras.regularizers.L2(1e-4))(conv4)
    conv4 = keras.layers.BatchNormalization(momentum=0.70)(conv4) if use_batch_norm else conv4
    pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)
    # -----------------------------------------------------------------------
    conv5 = keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same',
                      kernel_regularizer=tf.keras.regularizers.L2(1e-4))(pool4)
    conv5 = keras.layers.BatchNormalization(momentum=0.70)(conv5) if use_batch_norm else conv5
    conv5 = keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same',
                      kernel_regularizer=tf.keras.regularizers.L2(1e-4))(conv5)
    conv5 = keras.layers.BatchNormalization(momentum=0.70)(conv5) if use_batch_norm else conv5
    # -----------------------------------------------------------------------
    up6 = keras.layers.Concatenate(axis=3)([keras.layers.UpSampling2D(size=(2, 2))(conv5), conv4])
    conv6 = keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same',
                      kernel_regularizer=tf.keras.regularizers.L2(1e-4))(up6)
    conv6 = keras.layers.Dropout(0.5)(conv6) if not dropout_on_last_layer_only else conv6
    conv6 = keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same',
                      kernel_regularizer=tf.keras.regularizers.L2(1e-4))(conv6)
    conv6 = keras.layers.Dropout(0.5)(conv6) if not dropout_on_last_layer_only else conv6
    # -----------------------------------------------------------------------
    up7 = keras.layers.Concatenate(axis=3)([keras.layers.UpSampling2D(size=(2, 2))(conv6), conv3])
    conv7 = keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same',
                      kernel_regularizer=tf.keras.regularizers.L2(1e-4))(up7)
    conv7 = keras.layers.Dropout(0.5)(conv7) if not dropout_on_last_layer_only else conv7
    conv7 = keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same',
                      kernel_regularizer=tf.keras.regularizers.L2(1e-4))(conv7)
    conv7 = keras.layers.Dropout(0.5)(conv7) if not dropout_on_last_layer_only else conv7
    # -----------------------------------------------------------------------
    up8 = keras.layers.Concatenate(axis=3)([keras.layers.UpSampling2D(size=(2, 2))(conv7), conv2])
    conv8 = keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same',
                      kernel_regularizer=tf.keras.regularizers.L2(1e-4))(up8)
    conv8 = keras.layers.Dropout(0.5)(conv8) if not dropout_on_last_layer_only else conv8
    conv8 = keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same',
                      kernel_regularizer=tf.keras.regularizers.L2(1e-4))(conv8)
    conv8 = keras.layers.Dropout(0.5)(conv8) if not dropout_on_last_layer_only else conv8
    # -----------------------------------------------------------------------
    up9 = keras.layers.Concatenate(axis=3)([keras.layers.UpSampling2D(size=(2, 2))(conv8), conv1])
    conv9 = keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same',
                      kernel_regularizer=tf.keras.regularizers.L2(1e-4))(up9)
    conv9 = keras.layers.Dropout(0.5)(conv9) if not dropout_on_last_layer_only else conv9
    conv9 = keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same',
                      kernel_regularizer=tf.keras.regularizers.L2(1e-4))(conv9)
    conv9 = keras.layers.Dropout(0.5)(conv9)
    # -----------------------------------------------------------------------
    conv10 = keras.layers.Conv2D(filters=n_classes, kernel_size=(1,1), activation='sigmoid')(conv9)
    # -----------------------------------------------------------------------
    model = keras.Model(inputs=input_layer, outputs=conv10)

    return model


####################################################################################################


def mk_DeeplabV3Plus(num_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=3):

    def convolution_block(
        block_input,
        num_filters=256,
        kernel_size=3,
        dilation_rate=1,
        padding="same",
        use_bias=False,):
        x = layers.Conv2D(
            num_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding="same",
            use_bias=use_bias,
            kernel_initializer=keras.initializers.HeNormal(),)(block_input)
        x = layers.BatchNormalization()(x)
        return tf.nn.relu(x)


    def DilatedSpatialPyramidPooling(dspp_input):
        dims = dspp_input.shape
        x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)

        x = convolution_block(x, kernel_size=1, use_bias=True)
        out_pool = layers.UpSampling2D(
            size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",)(x)

        out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
        out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
        out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
        out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

        x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
        output = convolution_block(x, kernel_size=1)
        return output
    
    model_input = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    resnet50 = keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_tensor=model_input
    )
    x = resnet50.get_layer("conv4_block6_2_relu").output
    
    x = DilatedSpatialPyramidPooling(x)

    input_a = layers.UpSampling2D(size=(IMG_HEIGHT // 4 // x.shape[1], IMG_WIDTH // 4 // x.shape[2]), interpolation="bilinear",)(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = layers.UpSampling2D(
        size=(IMG_HEIGHT // x.shape[1], IMG_WIDTH // x.shape[2]),
        interpolation="bilinear",
    )(x)

    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same", activation='sigmoid')(x)

    return keras.Model(inputs=model_input, outputs=model_output)


####################################################################################################


def mk_cloudXnet(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=3):

    smooth = 0.0000001

    def aspp(x,out_shape):
        b0=SeparableConv2D(256,(1,1),padding="same",use_bias=False)(x)
        b0=BatchNormalization()(b0)
        b0=Activation("relu")(b0)

        #b5=DepthwiseConv2D((3,3),dilation_rate=(3,3),padding="same",use_bias=False)(x)
        #b5=BatchNormalization()(b5)
        #b5=Activation("relu")(b5)
        #b5=SeparableConv2D(256,(1,1),padding="same",use_bias=False)(b5)
        #b5=BatchNormalization()(b5)
        #b5=Activation("relu")(b5)

        b1=DepthwiseConv2D((3,3),dilation_rate=(6,6),padding="same",use_bias=False)(x)
        b1=BatchNormalization()(b1)
        b1=Activation("relu")(b1)
        b1=SeparableConv2D(256,(1,1),padding="same",use_bias=False)(b1)
        b1=BatchNormalization()(b1)
        b1=Activation("relu")(b1)

        b2=DepthwiseConv2D((3,3),dilation_rate=(12,12),padding="same",use_bias=False)(x)
        b2=BatchNormalization()(b2)
        b2=Activation("relu")(b2)
        b2=SeparableConv2D(256,(1,1),padding="same",use_bias=False)(b2)
        b2=BatchNormalization()(b2)
        b2=Activation("relu")(b2)	

        b3=DepthwiseConv2D((3,3),dilation_rate=(18,18),padding="same",use_bias=False)(x)
        b3=BatchNormalization()(b3)
        b3=Activation("relu")(b3)
        b3=SeparableConv2D(256,(1,1),padding="same",use_bias=False)(b3)
        b3=BatchNormalization()(b3)
        b3=Activation("relu")(b3)

        b4=AveragePooling2D(pool_size=(out_shape,out_shape))(x)
        b4=SeparableConv2D(256,(1,1),padding="same",use_bias=False)(b4)
        b4=BatchNormalization()(b4)
        b4=Activation("relu")(b4)
        b4=UpSampling2D((out_shape,out_shape), interpolation='bilinear')(b4)
        x=Concatenate()([b4,b0,b1,b2,b3])
        return x


    def jacc_coef(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return 1 - ((intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth))

    def bn_relu(input_tensor):
        """It adds a Batch_normalization layer before a Relu
        """
        input_tensor = BatchNormalization(axis=3)(input_tensor)
        return Activation("relu")(input_tensor)


    def contr_arm(input_tensor, filters, kernel_size):
        """It adds a feedforward signal to the output of two following conv layers in contracting path
        TO DO: remove keras.layers.add and replace it with add only
        """

        x = SeparableConv2D(filters, kernel_size, padding='same')(input_tensor)
        x = bn_relu(x)

        x = SeparableConv2D(filters, kernel_size, padding='same')(x)
        x = bn_relu(x)

        filters_b = filters // 2
        kernel_size_b = (kernel_size[0]-2, kernel_size[0]-2)  # creates a kernl size of (1,1) out of (3,3)

        x1 = SeparableConv2D(filters_b, kernel_size_b, padding='same')(input_tensor)
        x1 = bn_relu(x1)

        x1 = concatenate([input_tensor, x1], axis=3)
        x = keras.layers.add([x, x1])
        x = Activation("relu")(x)
        return x


    def imprv_contr_arm(input_tensor, filters, kernel_size ):
        """It adds a feedforward signal to the output of two following conv layers in contracting path
        """

        x = SeparableConv2D(filters, kernel_size, padding='same')(input_tensor)
        x = bn_relu(x)

        x0 = SeparableConv2D(filters, kernel_size, padding='same')(x)
        x0 = bn_relu(x0)

        x = SeparableConv2D(filters, kernel_size, padding='same')(x0)
        x = bn_relu(x)

        filters_b = filters // 2
        kernel_size_b = (kernel_size[0]-2, kernel_size[0]-2)  # creates a kernl size of (1,1) out of (3,3)

        x1 = SeparableConv2D(filters_b, kernel_size_b, padding='same')(input_tensor)
        x1 = bn_relu(x1)

        x1 = concatenate([input_tensor, x1], axis=3)

        x2 = SeparableConv2D(filters, kernel_size_b, padding='same')(x0)
        x2 = bn_relu(x2)

        x = keras.layers.add([x, x1, x2])
        x = Activation("relu")(x)
        return x


    def bridge(input_tensor, filters, kernel_size):
        """It is exactly like the identity_block plus a dropout layer. This block only uses in the valley of the UNet
        """

        x = SeparableConv2D(filters, kernel_size, padding='same')(input_tensor)
        x = bn_relu(x)

        x = SeparableConv2D(filters, kernel_size, padding='same')(x)
        x = Dropout(.15)(x)
        x = bn_relu(x)

        filters_b = filters // 2
        kernel_size_b = (kernel_size[0]-2, kernel_size[0]-2)  # creates a kernl size of (1,1) out of (3,3)

        x1 =SeparableConv2D(filters_b, kernel_size_b, padding='same')(input_tensor)
        x1 = bn_relu(x1)

        x1 = concatenate([input_tensor, x1], axis=3)
        x = keras.layers.add([x, x1])
        x = Activation("relu")(x)
        return x


    def conv_block_exp_path(input_tensor, filters, kernel_size):
        """It Is only the convolution part inside each expanding path's block
        """

        x = Conv2D(filters, kernel_size, padding='same')(input_tensor)
        x = bn_relu(x)

        x = Conv2D(filters, kernel_size, padding='same')(x)
        x = bn_relu(x)
        return x


    def conv_block_exp_path3(input_tensor, filters, kernel_size):
        """It Is only the convolution part inside each expanding path's block
        """

        x = Conv2D(filters, kernel_size, padding='same')(input_tensor)
        x = bn_relu(x)

        x = Conv2D(filters, kernel_size, padding='same')(x)
        x = bn_relu(x)

        x = Conv2D(filters, kernel_size, padding='same')(x)
        x = bn_relu(x)
        return x


    def add_block_exp_path(input_tensor1, input_tensor2, input_tensor3):
        """It is for adding two feed forwards to the output of the two following conv layers in expanding path
        """

        x = keras.layers.add([input_tensor1, input_tensor2, input_tensor3])
        x = Activation("relu")(x)
        return x


    def improve_ff_block4(input_tensor1, input_tensor2 ,input_tensor3, input_tensor4, pure_ff):
        """It improves the skip connection by using previous layers feature maps
        TO DO: shrink all of ff blocks in one function/class
        """

        for ix in range(1):
            if ix == 0:
                x1 = input_tensor1
            x1 = concatenate([x1, input_tensor1], axis=3)
            x1 = MaxPooling2D(pool_size=(2, 2))(x1)

        for ix in range(3):
            if ix == 0:
                x2 = input_tensor2
            x2 = concatenate([x2, input_tensor2], axis=3)
        x2 = MaxPooling2D(pool_size=(4, 4))(x2)

        for ix in range(7):
            if ix == 0:
                x3 = input_tensor3
            x3 = concatenate([x3, input_tensor3], axis=3)
        x3 = MaxPooling2D(pool_size=(8, 8))(x3)

        for ix in range(15):
            if ix == 0:
                x4 = input_tensor4
            x4 = concatenate([x4, input_tensor4], axis=3)
        x4 = MaxPooling2D(pool_size=(16, 16))(x4)

        x = keras.layers.add([x1, x2, x3, x4, pure_ff])
        x = Activation("relu")(x)
        return x


    def improve_ff_block3(input_tensor1, input_tensor2, input_tensor3, pure_ff):
        """It improves the skip connection by using previous layers feature maps
        """

        for ix in range(1):
            if ix == 0:
                x1 = input_tensor1
            x1 = concatenate([x1, input_tensor1], axis=3)
        x1 = MaxPooling2D(pool_size=(2, 2))(x1)

        for ix in range(3):
            if ix == 0:
                x2 = input_tensor2
            x2 = concatenate([x2, input_tensor2], axis=3)
        x2 = MaxPooling2D(pool_size=(4, 4))(x2)

        for ix in range(7):
            if ix == 0:
                x3 = input_tensor3
            x3 = concatenate([x3, input_tensor3], axis=3)
        x3 = MaxPooling2D(pool_size=(8, 8))(x3)

        x = keras.layers.add([x1, x2, x3, pure_ff])
        x = Activation("relu")(x)
        return x


    def improve_ff_block2(input_tensor1, input_tensor2, pure_ff):
        """It improves the skip connection by using previous layers feature maps
        """

        for ix in range(1):
            if ix == 0:
                x1 = input_tensor1
            x1 = concatenate([x1, input_tensor1], axis=3)
        x1 = MaxPooling2D(pool_size=(2, 2))(x1)

        for ix in range(3):
            if ix == 0:
                x2 = input_tensor2
            x2 = concatenate([x2, input_tensor2], axis=3)
        x2 = MaxPooling2D(pool_size=(4, 4))(x2)

        x = keras.layers.add([x1, x2, pure_ff])
        x = Activation("relu")(x)
        return x


    def improve_ff_block1(input_tensor1, pure_ff):
        """It improves the skip connection by using previous layers feature maps
        """

        for ix in range(1):
            if ix == 0:
                x1 = input_tensor1
            x1 = concatenate([x1, input_tensor1], axis=3)
        x1 = MaxPooling2D(pool_size=(2, 2))(x1)

        x = keras.layers.add([x1, pure_ff])
        x = Activation("relu")(x)
        return x

    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)

    conv1 = contr_arm(conv1, 32, (3, 3))
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = contr_arm(pool1, 64, (3, 3))
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = contr_arm(pool2, 128, (3, 3))
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = contr_arm(pool3, 256, (3, 3))
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = imprv_contr_arm(pool4, 512, (3, 3))
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

    conv6 = bridge(pool5, 1024, (3, 3))
    
    conv6  = aspp(conv6,IMG_HEIGHT/32)

    convT7 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv6)
    prevup7 = improve_ff_block4(input_tensor1=conv4, input_tensor2=conv3, input_tensor3=conv2, input_tensor4=conv1, pure_ff=conv5)
    up7 = concatenate([convT7, prevup7], axis=3)
    conv7 = conv_block_exp_path3(input_tensor=up7, filters=512, kernel_size=(3, 3))
    conv7 = add_block_exp_path(conv7, conv5, convT7)

    convT8 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv7)
    prevup8 = improve_ff_block3(input_tensor1=conv3, input_tensor2=conv2, input_tensor3=conv1, pure_ff=conv4)
    up8 = concatenate([convT8, prevup8], axis=3)
    conv8 = conv_block_exp_path(input_tensor=up8, filters=256, kernel_size=(3, 3))
    conv8 = add_block_exp_path(input_tensor1=conv8, input_tensor2=conv4, input_tensor3=convT8)

    convT9 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv8)
    prevup9 = improve_ff_block2(input_tensor1=conv2, input_tensor2=conv1, pure_ff=conv3)
    up9 = concatenate([convT9, prevup9], axis=3)
    conv9 = conv_block_exp_path(input_tensor=up9, filters=128, kernel_size=(3, 3))
    conv9 = add_block_exp_path(input_tensor1=conv9, input_tensor2=conv3, input_tensor3=convT9)

    convT10 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv9)
    prevup10 = improve_ff_block1(input_tensor1=conv1, pure_ff=conv2)
    up10 = concatenate([convT10, prevup10], axis=3)
    conv10 = conv_block_exp_path(input_tensor=up10, filters=64, kernel_size=(3, 3))
    conv10 = add_block_exp_path(input_tensor1=conv10, input_tensor2=conv2, input_tensor3=convT10)

    convT11 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv10)
    up11 = concatenate([convT11, conv1], axis=3)
    conv11 = conv_block_exp_path(input_tensor=up11, filters=32, kernel_size=(3, 3))
    conv11 = add_block_exp_path(input_tensor1=conv11, input_tensor2=conv1, input_tensor3=convT11)

    conv12 = Conv2D(n_classes, (1, 1), activation='sigmoid')(conv11)

    return Model(inputs=[inputs], outputs=[conv12])


####################################################################################################


def mk_cloud_net(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS):

    def bn_relu(input_tensor):
        """It adds a Batch_normalization layer before a Relu
        """
        input_tensor = BatchNormalization(axis=3)(input_tensor)
        return Activation("relu")(input_tensor)


    def contr_arm(input_tensor, filters, kernel_size):
        """It adds a feedforward signal to the output of two following conv layers in contracting path
        TO DO: remove keras.layers.add and replace it with add only
        """

        x = Conv2D(filters, kernel_size, padding='same')(input_tensor)
        x = bn_relu(x)

        x = Conv2D(filters, kernel_size, padding='same')(x)
        x = bn_relu(x)

        filters_b = filters // 2
        kernel_size_b = (kernel_size[0]-2, kernel_size[0]-2)  # creates a kernl size of (1,1) out of (3,3)

        x1 = Conv2D(filters_b, kernel_size_b, padding='same')(input_tensor)
        x1 = bn_relu(x1)

        x1 = concatenate([input_tensor, x1], axis=3)
        x = keras.layers.add([x, x1])
        x = Activation("relu")(x)
        return x


    def imprv_contr_arm(input_tensor, filters, kernel_size ):
        """It adds a feedforward signal to the output of two following conv layers in contracting path
        """

        x = Conv2D(filters, kernel_size, padding='same')(input_tensor)
        x = bn_relu(x)

        x0 = Conv2D(filters, kernel_size, padding='same')(x)
        x0 = bn_relu(x0)

        x = Conv2D(filters, kernel_size, padding='same')(x0)
        x = bn_relu(x)

        filters_b = filters // 2
        kernel_size_b = (kernel_size[0]-2, kernel_size[0]-2)  # creates a kernl size of (1,1) out of (3,3)

        x1 = Conv2D(filters_b, kernel_size_b, padding='same')(input_tensor)
        x1 = bn_relu(x1)

        x1 = concatenate([input_tensor, x1], axis=3)

        x2 = Conv2D(filters, kernel_size_b, padding='same')(x0)
        x2 = bn_relu(x2)

        x = keras.layers.add([x, x1, x2])
        x = Activation("relu")(x)
        return x


    def bridge(input_tensor, filters, kernel_size):
        """It is exactly like the identity_block plus a dropout layer. This block only uses in the valley of the UNet
        """

        x = Conv2D(filters, kernel_size, padding='same')(input_tensor)
        x = bn_relu(x)

        x = Conv2D(filters, kernel_size, padding='same')(x)
        x = Dropout(.15)(x)
        x = bn_relu(x)

        filters_b = filters // 2
        kernel_size_b = (kernel_size[0]-2, kernel_size[0]-2)  # creates a kernl size of (1,1) out of (3,3)

        x1 = Conv2D(filters_b, kernel_size_b, padding='same')(input_tensor)
        x1 = bn_relu(x1)

        x1 = concatenate([input_tensor, x1], axis=3)
        x = keras.layers.add([x, x1])
        x = Activation("relu")(x)
        return x


    def conv_block_exp_path(input_tensor, filters, kernel_size):
        """It Is only the convolution part inside each expanding path's block
        """

        x = Conv2D(filters, kernel_size, padding='same')(input_tensor)
        x = bn_relu(x)

        x = Conv2D(filters, kernel_size, padding='same')(x)
        x = bn_relu(x)
        return x


    def conv_block_exp_path3(input_tensor, filters, kernel_size):
        """It Is only the convolution part inside each expanding path's block
        """

        x = Conv2D(filters, kernel_size, padding='same')(input_tensor)
        x = bn_relu(x)

        x = Conv2D(filters, kernel_size, padding='same')(x)
        x = bn_relu(x)

        x = Conv2D(filters, kernel_size, padding='same')(x)
        x = bn_relu(x)
        return x


    def add_block_exp_path(input_tensor1, input_tensor2, input_tensor3):
        """It is for adding two feed forwards to the output of the two following conv layers in expanding path
        """

        x = keras.layers.add([input_tensor1, input_tensor2, input_tensor3])
        x = Activation("relu")(x)
        return x


    def improve_ff_block4(input_tensor1, input_tensor2 ,input_tensor3, input_tensor4, pure_ff):
        """It improves the skip connection by using previous layers feature maps
        TO DO: shrink all of ff blocks in one function/class
        """

        for ix in range(1):
            if ix == 0:
                x1 = input_tensor1
            x1 = concatenate([x1, input_tensor1], axis=3)
        x1 = MaxPooling2D(pool_size=(2, 2))(x1)

        for ix in range(3):
            if ix == 0:
                x2 = input_tensor2
            x2 = concatenate([x2, input_tensor2], axis=3)
        x2 = MaxPooling2D(pool_size=(4, 4))(x2)

        for ix in range(7):
            if ix == 0:
                x3 = input_tensor3
            x3 = concatenate([x3, input_tensor3], axis=3)
        x3 = MaxPooling2D(pool_size=(8, 8))(x3)

        for ix in range(15):
            if ix == 0:
                x4 = input_tensor4
            x4 = concatenate([x4, input_tensor4], axis=3)
        x4 = MaxPooling2D(pool_size=(16, 16))(x4)

        x = keras.layers.add([x1, x2, x3, x4, pure_ff])
        x = Activation("relu")(x)
        return x


    def improve_ff_block3(input_tensor1, input_tensor2, input_tensor3, pure_ff):
        """It improves the skip connection by using previous layers feature maps
        """

        for ix in range(1):
            if ix == 0:
                x1 = input_tensor1
            x1 = concatenate([x1, input_tensor1], axis=3)
        x1 = MaxPooling2D(pool_size=(2, 2))(x1)

        for ix in range(3):
            if ix == 0:
                x2 = input_tensor2
            x2 = concatenate([x2, input_tensor2], axis=3)
        x2 = MaxPooling2D(pool_size=(4, 4))(x2)

        for ix in range(7):
            if ix == 0:
                x3 = input_tensor3
            x3 = concatenate([x3, input_tensor3], axis=3)
        x3 = MaxPooling2D(pool_size=(8, 8))(x3)

        x = keras.layers.add([x1, x2, x3, pure_ff])
        x = Activation("relu")(x)
        return x


    def improve_ff_block2(input_tensor1, input_tensor2, pure_ff):
        """It improves the skip connection by using previous layers feature maps
        """

        for ix in range(1):
            if ix == 0:
                x1 = input_tensor1
            x1 = concatenate([x1, input_tensor1], axis=3)
        x1 = MaxPooling2D(pool_size=(2, 2))(x1)

        for ix in range(3):
            if ix == 0:
                x2 = input_tensor2
            x2 = concatenate([x2, input_tensor2], axis=3)
        x2 = MaxPooling2D(pool_size=(4, 4))(x2)

        x = keras.layers.add([x1, x2, pure_ff])
        x = Activation("relu")(x)
        return x


    def improve_ff_block1(input_tensor1, pure_ff):
        """It improves the skip connection by using previous layers feature maps
        """

        for ix in range(1):
            if ix == 0:
                x1 = input_tensor1
            x1 = concatenate([x1, input_tensor1], axis=3)
        x1 = MaxPooling2D(pool_size=(2, 2))(x1)

        x = keras.layers.add([x1, pure_ff])
        x = Activation("relu")(x)
        return x

    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)

    conv1 = contr_arm(conv1, 32, (3, 3))
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = contr_arm(pool1, 64, (3, 3))
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = contr_arm(pool2, 128, (3, 3))
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = contr_arm(pool3, 256, (3, 3))
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = imprv_contr_arm(pool4, 512, (3, 3))
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

    conv6 = bridge(pool5, 1024, (3, 3))

    convT7 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv6)
    prevup7 = improve_ff_block4(input_tensor1=conv4, input_tensor2=conv3, input_tensor3=conv2, input_tensor4=conv1, pure_ff=conv5)
    up7 = concatenate([convT7, prevup7], axis=3)
    conv7 = conv_block_exp_path3(input_tensor=up7, filters=512, kernel_size=(3, 3))
    conv7 = add_block_exp_path(conv7, conv5, convT7)

    convT8 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv7)
    prevup8 = improve_ff_block3(input_tensor1=conv3, input_tensor2=conv2, input_tensor3=conv1, pure_ff=conv4)
    up8 = concatenate([convT8, prevup8], axis=3)
    conv8 = conv_block_exp_path(input_tensor=up8, filters=256, kernel_size=(3, 3))
    conv8 = add_block_exp_path(input_tensor1=conv8, input_tensor2=conv4, input_tensor3=convT8)

    convT9 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv8)
    prevup9 = improve_ff_block2(input_tensor1=conv2, input_tensor2=conv1, pure_ff=conv3)
    up9 = concatenate([convT9, prevup9], axis=3)
    conv9 = conv_block_exp_path(input_tensor=up9, filters=128, kernel_size=(3, 3))
    conv9 = add_block_exp_path(input_tensor1=conv9, input_tensor2=conv3, input_tensor3=convT9)

    convT10 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv9)
    prevup10 = improve_ff_block1(input_tensor1=conv1, pure_ff=conv2)
    up10 = concatenate([convT10, prevup10], axis=3)
    conv10 = conv_block_exp_path(input_tensor=up10, filters=64, kernel_size=(3, 3))
    conv10 = add_block_exp_path(input_tensor1=conv10, input_tensor2=conv2, input_tensor3=convT10)

    convT11 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv10)
    up11 = concatenate([convT11, conv1], axis=3)
    conv11 = conv_block_exp_path(input_tensor=up11, filters=32, kernel_size=(3, 3))
    conv11 = add_block_exp_path(input_tensor1=conv11, input_tensor2=conv1, input_tensor3=convT11)

    conv12 = Conv2D(n_classes, (1, 1), activation='sigmoid')(conv11)

    return Model(inputs=[inputs], outputs=[conv12])
