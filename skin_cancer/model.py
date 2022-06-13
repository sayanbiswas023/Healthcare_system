import efficientnet.tfkeras as efn
import tensorflow as tf

EFNS = [efn.EfficientNetB0, efn.EfficientNetB1, efn.EfficientNetB2, efn.EfficientNetB3, 
        efn.EfficientNetB4, efn.EfficientNetB5, efn.EfficientNetB6, efn.EfficientNetB7]

IMG_SIZE = 256
EFF_NET = 6
def build_model(img_dims, ef):
    inp = tf.keras.layers.Input(shape=(img_dims, img_dims, 3))
    base = EFNS[ef](input_shape=(img_dims, img_dims, 3), weights='imagenet', include_top=False)
    x = base(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inp, outputs=x)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05), 
                  metrics=['AUC', 'binary_accuracy'])
    return model