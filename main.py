import os
from pathlib import Path

import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.layers.preprocessing.string_lookup import StringLookup

# download data from https://www.kaggle.com/fournierp/captcha-version-2-images
# and make sure the images are in the folder named captcha_images
# TODO comments to be added
data_dir = Path("./captcha_images/")

input_images = sorted(list(map(str, list(data_dir.glob("*.png")))))
ground_truth_labels = [img.split(os.path.sep)[-1].split(".png")[0] for img in input_images]
unique_characters = set(character for label in ground_truth_labels for character in label)

img_width = 200
img_height = 50

batch_size = 16

downsample_factor = 4

max_length = max([len(label) for label in ground_truth_labels])

convert_char_to_number = StringLookup(vocabulary=list(unique_characters), mask_token=None)

convert_number_to_char = StringLookup(
    vocabulary=convert_char_to_number.get_vocabulary(), mask_token=None, invert=True
)


def split_data(images, labels, train_size=0.9, shuffle=True):
    size = len(images)
    indices = np.arange(size)
    if shuffle:
        np.random.shuffle(indices)
    train_samples = int(size * train_size)
    splitted_x_train, splitted_y_train = images[indices[:train_samples]], labels[indices[:train_samples]]
    splitted_x_valid, splitted_y_valid = images[indices[train_samples:]], labels[indices[train_samples:]]
    return splitted_x_train, splitted_x_valid, splitted_y_train, splitted_y_valid


x_train, x_valid, y_train, y_valid = split_data(np.array(input_images), np.array(ground_truth_labels))


def encode_sample(img_path, gt_label):
    img = tf.io.read_file(img_path)

    img = tf.io.decode_png(img, channels=1)

    img = tf.image.convert_image_dtype(img, tf.float32)

    img = tf.image.resize(img, [img_height, img_width])

    reshaped_img = tf.transpose(img, perm=[1, 0, 2])

    encoded_gt_label = convert_char_to_number(tf.strings.unicode_split(gt_label, input_encoding="UTF-8"))

    return {"image": reshaped_img, "label": encoded_gt_label}


train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train = (
    train.map(encode_sample, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(
        buffer_size=tf.data.AUTOTUNE)
)

validation = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
validation = (validation.map(encode_sample, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(
    buffer_size=tf.data.AUTOTUNE)
)

_, ax = plt.subplots(4, 4, figsize=(10, 5))
for batch in train.take(1):
    input_images = batch["image"]
    ground_truth_labels = batch["label"]
    for i in range(16):
        img = (input_images[i] * 255).numpy().astype("uint8")
        label = tf.strings.reduce_join(convert_number_to_char(ground_truth_labels[i])).numpy().decode("utf-8")
        ax[i // 4, i % 4].imshow(img[:, :, 0].T, cmap="gray")
        ax[i // 4, i % 4].set_title(label)
        ax[i // 4, i % 4].axis("off")
plt.show()

"""
##MyDenseLayer 
"""


class MyDenseLayer(layers.Dense):

    def __init__(self, units, activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None, **kwargs):
        super().__init__(units, activation, use_bias, kernel_initializer, bias_initializer, kernel_regularizer,
                         bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, **kwargs)
        self.mean = tf.keras.metrics.Mean(name='metric_mean')

    def build(self, input_shape):
        return super().build(input_shape)

    def call(self, inputs):
        output = super().call(inputs)
        self.add_metric(self.mean(inputs))
        self.add_metric(tf.reduce_sum(inputs), name='metric_sum')
        # TODO other usefull metrics to be found

        return output


"""
## Model
"""


class CTCLayer(layers.Layer):

    def __call__(self, *args, **kwargs):
        print(args)
        y_true = args[0]
        y_pred = args[1]

        batch_length = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_length, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_length, 1), dtype="int64")

        loss = self.ctc_loss_function(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # self.add_metric(metrics.binary_accuracy(y_true,y_pred))
        return super().__call__(*args, **kwargs)

    def __init__(self, name=None):
        super().__init__(name=name)
        self.ctc_loss_function = keras.backend.ctc_batch_cost


def build_model():
    input_img = layers.Input(
        shape=(img_width, img_height, 1), name="image", dtype="float32"
    )
    labels = layers.Input(name="label", shape=(None,), dtype="float32")

    x = layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1",
    )(input_img)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    x = layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv2",
    )(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    new_shape = ((img_width // 4), (img_height // 4) * 64)
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

    x = MyDenseLayer(len(convert_char_to_number.get_vocabulary()) + 1, activation="softmax", name="dense2")(x)

    output = CTCLayer(name="ctc_loss")(labels, x)

    built_model = keras.models.Model(
        inputs=[input_img, labels], outputs=output, name="ocr_model_v1"
    )

    optimizer = keras.optimizers.Adam()

    built_model.compile(optimizer=optimizer)
    return built_model


model = build_model()
model.summary()
# time.sleep(100000)

# TODO better epoch number to be found
epochs = 100
stop_margin = 10

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=stop_margin, restore_best_weights=True
)

history = model.fit(
    train,
    validation_data=validation,
    epochs=epochs,
    callbacks=[early_stopping],
)

prediction_model = keras.models.Model(
    model.get_layer(name="image").input, model.get_layer(name="dense2").output
)
prediction_model.summary()
prediction_model.save('first_working_model')


def decode_batch_predictions(prediction):
    input_len = np.ones(prediction.shape[0]) * prediction.shape[1]

    results = keras.backend.ctc_decode(prediction, input_length=input_len, greedy=True)[0][0][
              :, :max_length
              ]

    predicted_text = []
    for result in results:
        result = tf.strings.reduce_join(convert_number_to_char(result)).numpy().decode("utf-8")
        predicted_text.append(result)
    return predicted_text


for batch in validation.take(1):
    batch_images = batch["image"]
    batch_labels = batch["label"]

    predictions = prediction_model.predict(batch_images)
    prediction_text = decode_batch_predictions(predictions)

    original_text = []
    for label in batch_labels:
        label = tf.strings.reduce_join(convert_number_to_char(label)).numpy().decode("utf-8")
        original_text.append(label)

    _, ax = plt.subplots(4, 4, figsize=(15, 5))
    for i in range(len(prediction_text)):
        img = (batch_images[i, :, :, 0] * 255).numpy().astype(np.uint8)
        img = img.T
        title = f"Prediction: {prediction_text[i]}"
        ax[i // 4, i % 4].imshow(img, cmap="gray")
        ax[i // 4, i % 4].set_title(title)
        ax[i // 4, i % 4].axis("off")
plt.show()

"""
plotting metrices
"""
N = np.arange(0, len(history.history["loss"]))
plt.style.use("ggplot")
plt.figure()
plt.plot(N, history.history["loss"], label="train_loss")
plt.plot(N, history.history["val_loss"], label="val_loss")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.show()

plt.style.use("ggplot")
plt.figure()
plt.plot(N, history.history["metric_mean"], label="train_metric_mean")
plt.plot(N, history.history["val_metric_mean"], label="val_metric_mean")
plt.title("Mean Metric")
plt.xlabel("Epochs")
plt.ylabel("Mean")
plt.legend(loc="lower left")
plt.show()

plt.style.use("ggplot")
plt.figure()
plt.plot(N, history.history["metric_sum"], label="train_metric_sum")
plt.plot(N, history.history["val_metric_sum"], label="val_metric_sum")
plt.title("Sumation Metric")
plt.xlabel("Epochs")
plt.ylabel("Sum")
plt.legend(loc="lower left")
plt.show()
