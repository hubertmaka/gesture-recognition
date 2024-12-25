import os
import logging
from dataclasses import dataclass
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt


JESTER_DATASET_DIR_PATH = os.path.join("/", "home","tfprofile","gesture-recognition")
TRAIN_INFO_PATH = os.path.join(JESTER_DATASET_DIR_PATH, "info", "jester-v1-train.csv")
VAL_INFO_PATH = os.path.join(JESTER_DATASET_DIR_PATH, "info", "jester-v1-validation.csv")
TEST_INFO_PATH = os.path.join(JESTER_DATASET_DIR_PATH, "info", "jester-original-test.csv")
LABELS_INFO_PATH = os.path.join(JESTER_DATASET_DIR_PATH, "info", 'jester-v1-labels.csv')
VIDEO_DIR_PATH = os.path.join(JESTER_DATASET_DIR_PATH, "data")


train_df = pd.read_csv(os.path.join(JESTER_DATASET_DIR_PATH, "info", "train_df.csv"))
val_df = pd.read_csv(os.path.join(JESTER_DATASET_DIR_PATH, "info", "val_df.csv"))
test_df = pd.read_csv(os.path.join(JESTER_DATASET_DIR_PATH, "info", "test_df.csv"))

columns = ["label"]
labels_info = pd.read_csv(os.path.join(JESTER_DATASET_DIR_PATH, "info", "jester-v1-labels.csv"), names=columns)


SET_SIZE = (121_500 + len(val_df) + len(test_df)) // 1
TRAIN_SET_SIZE = 121_500 // 1
VAL_SET_SIZE = len(val_df) // 1
TEST_SET_SIZE = len(test_df) // 1
MIN_SEQ_LEN = 30
CLASSES = [i for i in range(27)]
CLASSES_PERCENTAGE = [1/len(CLASSES) for _ in range(len(CLASSES))]
CLASSES_WITH_PERCENTAGE = {cls: percentage for cls, percentage in zip(CLASSES, CLASSES_PERCENTAGE)}
TRAIN_CLASSES_WITH_AMOUNT = {cls: int(percentage * TRAIN_SET_SIZE) for cls, percentage in zip(CLASSES, CLASSES_PERCENTAGE)}
TRAIN_CLASSES_WITH_AMOUNT[26] = TRAIN_CLASSES_WITH_AMOUNT.get(26) + (4500 // 1)
VAL_CLASSES_WITH_AMOUNT = {cls: int(percentage * VAL_SET_SIZE) for cls, percentage in zip(CLASSES, CLASSES_PERCENTAGE)}
VAL_CLASSES_WITH_AMOUNT[26] = VAL_CLASSES_WITH_AMOUNT.get(26) + (450 // 1)
TEST_CLASSES_WITH_AMOUNT = {cls: int(percentage * TEST_SET_SIZE) for cls, percentage in zip(CLASSES, CLASSES_PERCENTAGE)}
TEST_CLASSES_WITH_AMOUNT[26] = TEST_CLASSES_WITH_AMOUNT.get(26) + 450 // 1
CLASS_MAPPING = {cls: idx for idx, cls in enumerate(CLASSES)}


def preprocess_df(dataframe: pd.DataFrame, subset_info: dict[int, int], min_seq_len: int,
                  training: bool) -> pd.DataFrame:
    def _get_longer_seq_len_than(df: pd.DataFrame, seq_len: int) -> pd.DataFrame:
        return df[df["seq_len"] >= seq_len]

    def _get_df_subset(df: pd.DataFrame, classes_with_amount: dict[int, int], training: bool) -> pd.DataFrame:
        subset_df = pd.DataFrame()
        for key, val in classes_with_amount.items():
            class_df = df[df["label_id"] == key]
            current_count = len(class_df)

            if training:
                if current_count < val:
                    needed_count = val - current_count
                    sampled_df = class_df.sample(n=needed_count, replace=True)
                    class_df = pd.concat([class_df, sampled_df], ignore_index=True)
                subset_df = pd.concat([subset_df, class_df.sample(n=val, replace=False)], ignore_index=True)
            else:
                subset_df = pd.concat([subset_df, class_df.sample(n=min(val, current_count), replace=False)],
                                      ignore_index=True)

        return subset_df.sample(frac=1, ignore_index=True)

    # new_df = _get_longer_seq_len_than(dataframe, min_seq_len)
    new_df = _get_df_subset(dataframe, subset_info, training)
    return new_df


train_subset_df = preprocess_df(train_df, TRAIN_CLASSES_WITH_AMOUNT, MIN_SEQ_LEN, training=True)
val_subset_df = preprocess_df(val_df, VAL_CLASSES_WITH_AMOUNT, MIN_SEQ_LEN, training=False)
test_subset_df = preprocess_df(test_df, TEST_CLASSES_WITH_AMOUNT, MIN_SEQ_LEN, training=False)

labels = train_subset_df['label_id'].map(CLASS_MAPPING).values

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)

class_weight_dict = dict(zip(np.unique(labels), class_weights))


def create_dataset_from_dir(df: pd.DataFrame, seq_len: int, class_mapping: dict[int, int]):
    sequences = []
    labels = []

    for _, row in df.iterrows():
        video_path = row['path']
        label_id = row['label_id']
        image_paths = generate_image_paths(video_path, seq_len, row)
        sequences.append(image_paths)
        mapped_label_id = class_mapping[label_id]
        labels.append(mapped_label_id)

    dataset = tf.data.Dataset.from_tensor_slices((sequences, labels))
    return dataset


@dataclass
class InputShape:
    weight: int
    height: int
    channels: int

    def as_tuple(self) -> tuple[int, int, int]:
        return self.weight, self.height, self.channels


def generate_image_paths(video_dir_path: str, seq_len: int, row: pd.Series):
    image_paths = []
    current_frames: int = row["seq_len"]

    if current_frames >= seq_len:
        mid_point = current_frames // 2
        start_point = max(0, mid_point - seq_len // 2)
        end_point = start_point + seq_len

        for i in range(start_point + 1, end_point + 1):
            img_name = f"{i:05d}.jpg"
            img_path = os.path.join(video_dir_path, img_name)
            image_paths.append(img_path)

    else:
        padding_needed = seq_len - current_frames
        left_padding = padding_needed // 2
        right_padding = padding_needed - left_padding

        most_left_img = f"00001.jpg"
        most_right_img = f"{current_frames:05d}.jpg"

        for i in range(left_padding):
            img_path = os.path.join(video_dir_path, most_left_img)
            image_paths.append(img_path)

        for i in range(1, current_frames + 1):
            img_name = f"{i:05d}.jpg"
            img_path = os.path.join(video_dir_path, img_name)
            image_paths.append(img_path)

        for i in range(right_padding):
            img_path = os.path.join(video_dir_path, most_right_img)
            image_paths.append(img_path)

    if len(image_paths) != seq_len:
        raise ValueError(f"Missing images in dir: {video_dir_path}")

    return image_paths


@tf.function
def preprocess_frames(frames, label):
    brightness_delta = random.uniform(-0.2, 0.2)
    contrast_factor = random.uniform(0.8, 1.2)
    saturation_factor = random.uniform(0.8, 1.2)
    hue_delta = random.uniform(-0.02, 0.02)
    noise_stddev = 0.05

    augmentations = [
        lambda img: tf.image.adjust_brightness(img, brightness_delta),
        lambda img: tf.image.adjust_contrast(img, contrast_factor),
        lambda img: tf.image.adjust_saturation(img, saturation_factor),
        lambda img: tf.image.adjust_hue(img, hue_delta),
        lambda img: img + tf.random.normal(tf.shape(img), mean=0.0, stddev=noise_stddev)
    ]

    chosen_augmentations = random.sample(augmentations, 2)

    def process_image(img):
        for aug in chosen_augmentations:
            img = aug(img)
        return img

    preprocessed_images = tf.map_fn(process_image, frames, fn_output_signature=tf.float32)

    return preprocessed_images, label


@tf.function
def load_sequence_from_dir(image_paths: tf.Tensor, label: int, inp_shape: tuple[int, int, int]):
    def process_image(img_path):
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image, channels=inp_shape[2])
        image = tf.image.resize(image, inp_shape[:2])
        # image = tf.cast(image, tf.float32) / 255.0
        return image

    preprocessed_images = tf.map_fn(process_image, image_paths, fn_output_signature=tf.float32)

    return preprocessed_images, label


@tf.function
def one_hot_encode(path: tf.Tensor, label: tf.Tensor, classes_num: int):
    return path, tf.one_hot(label, classes_num, dtype=tf.int32)


@tf.function
def remove_one_dimensions(images: tf.Tensor, label: int):
    return tf.squeeze(images), label


@tf.function
def normalize_frames(frames, label):
    return (tf.cast(frames, tf.float32) / 255.0), label


@tf.function
def add_dimension(frames, label):
    frames = tf.expand_dims(frames, axis=0) 
    return frames, label


def create_pipeline(df: pd.DataFrame, *, num_classes: int, image_input_shape: InputShape, seq_len: int, batch_size: int, class_mapping: dict[int, int], is_training: bool = False, cache: bool = False) -> tf.data.Dataset:
    ds = create_dataset_from_dir(df, seq_len=seq_len, class_mapping=class_mapping) # (list with paths strs)
    ds = ds.map(lambda images, label: one_hot_encode(images, label, num_classes), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda path, label: load_sequence_from_dir(path, label, image_input_shape.as_tuple()), num_parallel_calls=tf.data.AUTOTUNE)
    if is_training:
        pass
        ds = ds.map(lambda frames, label: preprocess_frames(frames, label), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size=batch_size, drop_remainder=True)
    if cache:
        ds = ds.cache("./cache").prefetch(tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


class PipelineConfig:
    IMAGE_INPUT_SHAPE = InputShape(164, 164, 3)
    NUM_CLASSES = len(CLASSES)
    BATCH_SIZE = 16
    SEQ_LEN = 34


train_ds = create_pipeline(
    train_subset_df,
    num_classes=PipelineConfig.NUM_CLASSES,
    image_input_shape=PipelineConfig.IMAGE_INPUT_SHAPE,
    seq_len=PipelineConfig.SEQ_LEN,
    batch_size=PipelineConfig.BATCH_SIZE,
    class_mapping=CLASS_MAPPING,
    is_training=True
)

val_ds = create_pipeline(
    val_subset_df,
    num_classes=PipelineConfig.NUM_CLASSES,
    image_input_shape=PipelineConfig.IMAGE_INPUT_SHAPE,
    seq_len=PipelineConfig.SEQ_LEN,
    batch_size=PipelineConfig.BATCH_SIZE,
    class_mapping=CLASS_MAPPING
)

test_ds = create_pipeline(
    test_subset_df,
    num_classes=PipelineConfig.NUM_CLASSES,
    image_input_shape=PipelineConfig.IMAGE_INPUT_SHAPE,
    seq_len=PipelineConfig.SEQ_LEN,
    batch_size=PipelineConfig.BATCH_SIZE,
    class_mapping=CLASS_MAPPING,
    is_training=False
)


def build_model_from_articles_conv3D(input_shape, num_classes: int):
    model = keras.models.Sequential()
    model.add(keras.layers.Input(input_shape))
    model.add(keras.layers.Conv3D(32, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling3D(pool_size=(1, 2, 2)))
    model.add(keras.layers.Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling3D(pool_size=(1, 2, 2)))
    model.add(keras.layers.Conv3D(128, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling3D(pool_size=(1, 2, 2)))
    model.add(keras.layers.Conv3D(256, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv3D(256, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv3D(256, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.GlobalMaxPooling3D())
    model.add(keras.layers.RepeatVector(1))
    model.add(keras.layers.LSTM(256, return_sequences=True))
    model.add(keras.layers.LSTM(256))
    model.add(keras.layers.Dense(units=num_classes, activation='softmax', name="DENSE_OUTPUT"))

    optimizer = keras.optimizers.Adam(0.001)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[
            'accuracy',
            'precision',
            'recall',
            keras.metrics.F1Score(average='weighted')
        ]
    )

    return model

lstm_model = build_model_from_articles_conv3D(input_shape=(PipelineConfig.SEQ_LEN, *PipelineConfig.IMAGE_INPUT_SHAPE.as_tuple()), num_classes=len(CLASSES))

lstm_model.summary()

input("PASS? ")

print("Start Learning Best Model")

early_stopping_callback = keras.callbacks.EarlyStopping(
    patience=20,
    monitor='val_loss',
    mode="min"
)

checkpoint_callback = keras.callbacks.ModelCheckpoint(
    'model_weights.weights.h5',
    save_best_only=True,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    verbose=1
)

history = lstm_model.fit(
    train_ds,
    epochs=100,
    validation_data=val_ds,
    callbacks=[early_stopping_callback, checkpoint_callback],
    class_weight=class_weight_dict
)



plt.figure(figsize=(15, 7))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Dokładność na zbiorze treningowym')
plt.plot(history.history['val_accuracy'], label='Dokładność na zbiorze walidacyjnym')
plt.xlabel('Epoka')
plt.ylabel('Dokładność')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.grid()
plt.title('Krzywa Uczenia - Dokładność')

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Strata na zbiorze treningowym')
plt.plot(history.history['val_loss'], label='Strata na zbiorze walidacyjnym')
plt.xlabel('Epoka')
plt.ylabel('Strata')
plt.ylim([0, max(history.history['loss'])])
plt.legend(loc='upper right')
plt.grid()
plt.title('Krzywa Uczenia - Strata')


plt.savefig('training_plots.png')

print("Plots saved as 'training_plots.png'")
plt.figure(figsize=(15, 25))

plt.subplot(3, 2, 1)
plt.plot(history.history['accuracy'], label='Dokładność na zbiorze treningowym')
plt.plot(history.history['val_accuracy'], label='Dokładność na zbiorze walidacyjnym')
plt.xlabel('Epoka')
plt.ylabel('Dokładność')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.grid()
plt.title('Krzywa Uczenia - Dokładność')

plt.subplot(3, 2, 2)
plt.plot(history.history['loss'], label='Strata na zbiorze treningowym')
plt.plot(history.history['val_loss'], label='Strata na zbiorze walidacyjnym')
plt.xlabel('Epoka')
plt.ylabel('Strata')
plt.ylim([0, max(history.history['loss'])])
plt.legend(loc='upper right')
plt.grid()
plt.title('Krzywa Uczenia - Strata')

plt.subplot(3, 2, 3)
plt.plot(history.history['precision'], label='Precyzja na zbiorze treningowym')
plt.plot(history.history['val_precision'], label='Precyzja na zbiorze walidacyjnym')
plt.xlabel('Epoka')
plt.ylabel('Precyzja')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.grid()
plt.title('Krzywa Uczenia - Precyzja')

plt.subplot(3, 2, 4)
plt.plot(history.history['recall'], label='Czułość na zbiorze treningowym')
plt.plot(history.history['val_recall'], label='Czułość na zbiorze walidacyjnym')
plt.xlabel('Epoka')
plt.ylabel('Czułość')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.grid()
plt.title('Krzywa Uczenia - Czułość')

plt.subplot(3, 2, 5)
plt.plot(history.history['f1_score'], label='F1 Score na zbiorze treningowym')
plt.plot(history.history['val_f1_score'], label='F1 Score na zbiorze walidacyjnym')
plt.xlabel('Epoka')
plt.ylabel('F1 Score')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.grid()
plt.title('Krzywa Uczenia - F1 Score')

plt.tight_layout()
plt.savefig('training_plots_articles_based_grid.png')

print("Plots saved as 'training_plots_articles_based.png'")

plt.figure(figsize=(15, 15))
plt.plot(history.history['accuracy'], label='Dokładność na zbiorze treningowym')
plt.plot(history.history['val_accuracy'], label='Dokładność na zbiorze walidacyjnym')
plt.xlabel('Epoka')
plt.ylabel('Dokładność')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.title('Krzywa Uczenia - Dokładność')
plt.grid()
plt.savefig('training_plots_articles_based_acc_grid.png')

print("Plots saved as 'training_plots_articles_based.png'")
plt.figure(figsize=(15, 15))
plt.plot(history.history['loss'], label='Strata na zbiorze treningowym')
plt.plot(history.history['val_loss'], label='Strata na zbiorze walidacyjnym')
plt.xlabel('Epoka')
plt.ylabel('Strata')
plt.ylim([0, max(history.history['loss'])])
plt.legend(loc='upper right')
plt.title('Krzywa Uczenia - Strata')
plt.grid()
plt.savefig('training_plots_articles_based_loss.png')

print("Plots saved as 'training_plots_articles_based.png'")
plt.figure(figsize=(15, 15))
plt.plot(history.history['precision'], label='Precyzja na zbiorze treningowym')
plt.plot(history.history['val_precision'], label='Precyzja na zbiorze walidacyjnym')
plt.xlabel('Epoka')
plt.ylabel('Precyzja')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.title('Krzywa Uczenia - Precyzja')
plt.grid()
plt.savefig('training_plots_articles_based_prec.png')

print("Plots saved as 'training_plots_articles_based.png'")
plt.figure(figsize=(15, 15))
plt.plot(history.history['recall'], label='Czułość na zbiorze treningowym')
plt.plot(history.history['val_recall'], label='Czułość na zbiorze walidacyjnym')
plt.xlabel('Epoka')
plt.ylabel('Czułość')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.title('Krzywa Uczenia - Czułość')
plt.grid()
plt.savefig('training_plots_articles_based_recall.png')

print("Plots saved as 'training_plots_articles_based.png'")
plt.figure(figsize=(15, 15))
plt.plot(history.history['f1_score'], label='F1 Score na zbiorze treningowym')
plt.plot(history.history['val_f1_score'], label='F1 Score na zbiorze walidacyjnym')
plt.xlabel('Epoka')
plt.ylabel('F1 Score')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.title('Krzywa Uczenia - F1 Score')
plt.grid()
plt.savefig('training_plots_articles_based_f1.png')

print("Plots saved as 'training_plots_articles_based.png'")

y_pred = []
y_true = []

for images, labels in test_ds:
    predictions = lstm_model.predict(images)
    predicted_classes = np.argmax(predictions, axis=1)

    true_classes = np.argmax(labels, axis=1) if labels.shape[-1] > 1 else labels.numpy()

    y_pred.extend(predicted_classes)
    y_true.extend(true_classes)


cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))

jester_labels = {}
with open(os.path.join(JESTER_DATASET_DIR_PATH, "info", 'jester-v1-labels.csv')) as f:
    for idx, line in enumerate(f):
        jester_labels[idx] = line.strip()

display_info = [jester_labels.get(cls) for cls in CLASSES]


disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_info)


disp.plot(
    cmap=plt.cm.Blues,
    xticks_rotation=45,  
    values_format='.2f'
    # values_format='.0f'
)

fig = disp.figure_
fig.set_figwidth(30)
fig.set_figheight(30)

plt.xlabel('Przewidziana Etykieta', fontsize=20)
plt.ylabel('Prawdziwa Etykieta', fontsize=20)

plt.xticks(fontsize=15, rotation=45, ha='right')
plt.yticks(fontsize=15, rotation=0, va="center")

for text in disp.text_.ravel():
    text.set_fontsize(14)

plt.savefig('conf_mat_no_norm_test.png')
print("Plots saved as 'conf_mat_no_norm_test.png'")

cm = confusion_matrix(y_true, y_pred, normalize='true', labels=range(len(CLASSES)))

jester_labels = {}
with open(os.path.join(JESTER_DATASET_DIR_PATH, "info", 'jester-v1-labels.csv')) as f:
    for idx, line in enumerate(f):
        jester_labels[idx] = line.strip()

display_info = [jester_labels.get(cls) for cls in CLASSES]

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_info)


disp.plot(
    cmap=plt.cm.Blues,
    xticks_rotation=45,  
    values_format='.2f'
    # values_format='.0f'
)

fig = disp.figure_
fig.set_figwidth(30)
fig.set_figheight(30)

plt.xlabel('Przewidziana Etykieta', fontsize=20)
plt.ylabel('Prawdziwa Etykieta', fontsize=20)

plt.xticks(fontsize=15, rotation=45, ha='right')
plt.yticks(fontsize=15, rotation=0, va="center")

for text in disp.text_.ravel():
    text.set_fontsize(14)

plt.savefig('conf_mat_norm_test.png')
print("Plots saved as 'conf_mat_norm_test.png'")




lstm_model = build_model_from_articles_conv3D(
    input_shape=(PipelineConfig.SEQ_LEN, *PipelineConfig.IMAGE_INPUT_SHAPE.as_tuple()),
    num_classes=PipelineConfig.NUM_CLASSES
)


lstm_model.load_weights('model_weights.weights.h5')

results = lstm_model.evaluate(test_ds)

print("Test Loss:", results[0])
print("Test Accuracy:", results[1])
print("Test Precision:", results[2])
print("Test Recall:", results[3])
print("Test F1 Score:", results[4])




y_pred = []
y_true = []

for images, labels in val_ds:
    predictions = lstm_model.predict(images)
    predicted_classes = np.argmax(predictions, axis=1)

    true_classes = np.argmax(labels, axis=1) if labels.shape[-1] > 1 else labels.numpy()

    y_pred.extend(predicted_classes)
    y_true.extend(true_classes)


cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))

jester_labels = {}
with open(os.path.join(JESTER_DATASET_DIR_PATH, "info", 'jester-v1-labels.csv')) as f:
    for idx, line in enumerate(f):
        jester_labels[idx] = line.strip()

display_info = [jester_labels.get(cls) for cls in CLASSES]

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_info)

disp.plot(
    cmap=plt.cm.Blues,
    xticks_rotation=45,  
    values_format='.2f'
    # values_format='.0f'
)

fig = disp.figure_
fig.set_figwidth(30)
fig.set_figheight(30)

plt.xlabel('Przewidziana Etykieta', fontsize=20)
plt.ylabel('Prawdziwa Etykieta', fontsize=20)

plt.xticks(fontsize=15, rotation=45, ha='right')
plt.yticks(fontsize=15, rotation=0, va="center")

for text in disp.text_.ravel():
    text.set_fontsize(14)

plt.savefig('conf_mat_BEST_no_norm_val.png')
print("Plots saved as 'conf_BEST_val_no_norm.png'")


cm = confusion_matrix(y_true, y_pred, normalize='true', labels=range(len(CLASSES)))

jester_labels = {}
with open(os.path.join(JESTER_DATASET_DIR_PATH, "info", 'jester-v1-labels.csv')) as f:
    for idx, line in enumerate(f):
        jester_labels[idx] = line.strip()

display_info = [jester_labels.get(cls) for cls in CLASSES]

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_info)

disp.plot(
    cmap=plt.cm.Blues,
    xticks_rotation=45,  
    values_format='.2f'
    # values_format='.0f'
)

fig = disp.figure_
fig.set_figwidth(30)
fig.set_figheight(30)

plt.xlabel('Przewidziana Etykieta', fontsize=20)
plt.ylabel('Prawdziwa Etykieta', fontsize=20)

plt.xticks(fontsize=15, rotation=45, ha='right')
plt.yticks(fontsize=15, rotation=0, va="center")

for text in disp.text_.ravel():
    text.set_fontsize(14)

plt.savefig('conf_mat_BEST_norm_val.png')
print("Plots saved as 'conf_BEST_val_norm.png'")




y_pred = []
y_true = []

for images, labels in test_ds:
    predictions = lstm_model.predict(images)
    predicted_classes = np.argmax(predictions, axis=1)

    true_classes = np.argmax(labels, axis=1) if labels.shape[-1] > 1 else labels.numpy()

    y_pred.extend(predicted_classes)
    y_true.extend(true_classes)


cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))

jester_labels = {}
with open(os.path.join(JESTER_DATASET_DIR_PATH, "info", 'jester-v1-labels.csv')) as f:
    for idx, line in enumerate(f):
        jester_labels[idx] = line.strip()

display_info = [jester_labels.get(cls) for cls in CLASSES]

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_info)

disp.plot(
    cmap=plt.cm.Blues,
    xticks_rotation=45,  
    values_format='.2f'
    # values_format='.0f'
)

fig = disp.figure_
fig.set_figwidth(30)
fig.set_figheight(30)

plt.xlabel('Przewidziana Etykieta', fontsize=20)
plt.ylabel('Prawdziwa Etykieta', fontsize=20)

plt.xticks(fontsize=15, rotation=45, ha='right')
plt.yticks(fontsize=15, rotation=0, va="center")

for text in disp.text_.ravel():
    text.set_fontsize(14)

plt.savefig('conf_mat_BEST_no_norm_test.png')
print("Plots saved as 'conf_BEST_test_no_norm.png'")

cm = confusion_matrix(y_true, y_pred, normalize='true', labels=range(len(CLASSES)))

jester_labels = {}
with open(os.path.join(JESTER_DATASET_DIR_PATH, "info", 'jester-v1-labels.csv')) as f:
    for idx, line in enumerate(f):
        jester_labels[idx] = line.strip()

display_info = [jester_labels.get(cls) for cls in CLASSES]

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_info)

disp.plot(
    cmap=plt.cm.Blues,
    xticks_rotation=45,  
    values_format='.2f'
    # values_format='.0f'
)

fig = disp.figure_
fig.set_figwidth(30)
fig.set_figheight(30)

plt.xlabel('Przewidziana Etykieta', fontsize=20)
plt.ylabel('Prawdziwa Etykieta', fontsize=20)

plt.xticks(fontsize=15, rotation=45, ha='right')
plt.yticks(fontsize=15, rotation=0, va="center")

for text in disp.text_.ravel():
    text.set_fontsize(14)

plt.savefig('conf_mat_BEST_norm_test.png')
print("Plots saved as 'conf_BEST_test_norm.png'")
