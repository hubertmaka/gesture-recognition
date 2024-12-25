import cv2
import numpy as np
import logging
from collections import deque, Counter
import tensorflow as tf
import keras
from multiprocessing import Process, Queue


logger = logging.getLogger(__name__)

MODEL_WEIGHTS_PATH = '../MobileNetLSTM/model_weights.weights.h5'
LABELS_CSV_PATH = '../jester-v1-labels.csv'


def build_model_from_articles(input_shape: tuple, num_classes: int):
    mobilenet_model = keras.applications.MobileNetV3Large(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape[1:],
        include_preprocessing=True,
        pooling="avg",
    )
    model = keras.models.Sequential()
    model.add(keras.layers.Input(input_shape))
    model.add(keras.layers.TimeDistributed(mobilenet_model))
    for layer in model.layers:
        layer.trainable = False

    # LSTM Layers
    model.add(keras.layers.LSTM(units=256, return_sequences=True, kernel_regularizer=keras.regularizers.l2(0.0001),
                                name="LSTM_1"))
    model.add(keras.layers.Dropout(0.2, name="LSTM_DROPOUT_1"))
    model.add(keras.layers.LayerNormalization(name="LSTM_LNORM_1"))
    model.add(keras.layers.LSTM(units=256, return_sequences=True, kernel_regularizer=keras.regularizers.l2(0.0001),
                                name="LSTM_2"))
    model.add(keras.layers.Dropout(0.2, name="LSTM_DROPOUT_2"))
    model.add(keras.layers.LayerNormalization(name="LSTM_LNORM_2"))
    model.add(keras.layers.LSTM(units=256, return_sequences=False, kernel_regularizer=keras.regularizers.l2(0.0001),
                                name="LSTM_3"))
    model.add(keras.layers.Dropout(0.2, name="LSTM_DROPOUT_3"))
    model.add(keras.layers.LayerNormalization(name="LSTM_LNORM_3"))
    model.add(keras.layers.Dense(units=256, activation="tanh", kernel_regularizer=keras.regularizers.l2(0.001),
                                 name="DENSE_1"))
    model.add(keras.layers.Dropout(0.1, name="DENSE_DROPOUT_1"))
    model.add(keras.layers.LayerNormalization(name="DENSE_LNORM_1"))
    model.add(keras.layers.Dense(units=num_classes, name="DENSE_OUTPUT"))

    optimizer = keras.optimizers.Adam(0.001)

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[
            'accuracy',
            'precision',
            'recall',
        ]
    )
    return model


def load_gesture_labels(labels_csv_path):
    labels = []
    with open(labels_csv_path, 'r') as f:
        for line in f:
            label = line.strip()
            if label:
                labels.append(label)
    return labels


def preprocess_frames(frames):
    processed = []
    for frame in frames:
        frame_resized = cv2.resize(frame, (224, 224))
        frame_normalized = frame_resized
        processed.append(frame_normalized)
    processed = np.array(processed)
    processed = np.expand_dims(processed, axis=0)
    return processed


def prediction_process(input_queue: Queue, output_queue: Queue, model_weights_path: str, labels_csv_path: str):
    num_classes = 27
    input_shape = (34, 224, 224, 3)  # (SEQ_LEN, HEIGHT, WIDTH, CHANNELS)

    model = build_model_from_articles(input_shape=input_shape, num_classes=num_classes)
    model.load_weights(model_weights_path)
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[
            'accuracy',
            'precision',
            'recall',
        ]
    )

    gesture_labels = load_gesture_labels(labels_csv_path)
    if len(gesture_labels) != num_classes:
        raise ValueError(f"Number of labels ({len(gesture_labels)}) is not equal clas number ({num_classes}).")

    EMA_ALPHA = 0.7
    ema_probabilities = np.zeros(num_classes)

    PREDICTION_HISTORY_SIZE = 5
    prediction_history = deque(maxlen=PREDICTION_HISTORY_SIZE)

    while True:
        frames = input_queue.get()
        if frames is None:
            break

        input_data = preprocess_frames(frames)

        predictions = model.predict(input_data, verbose=0)
        probabilities = tf.nn.softmax(predictions, axis=1).numpy()[0]

        ema_probabilities = EMA_ALPHA * probabilities + (1 - EMA_ALPHA) * ema_probabilities

        predicted_index = np.argmax(ema_probabilities)
        predicted_gesture = gesture_labels[predicted_index]
        probability = ema_probabilities[predicted_index]

        prediction_history.append(predicted_gesture)

        most_common_gesture, count = Counter(prediction_history).most_common(1)[0]
        if count >= 3:
            display_gesture = most_common_gesture
            display_probability = probability
        else:
            display_gesture = predicted_gesture
            display_probability = probability

        output_queue.put((display_gesture, display_probability))

    model = None
    keras.backend.clear_session()


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        logger.error("Cannot use wideo camera.")
        exit()

    BUFFER_SIZE = 34
    frame_buffer = deque(maxlen=BUFFER_SIZE)
    PREDICTION_INTERVAL = 2
    frame_count = 0
    TEXT_COLOR = (255, 255, 255)
    BACKGROUND_COLOR = (50, 50, 50)
    PROGRESS_BAR_COLOR = (0, 255, 0)
    BACKGROUND_RECT_COLOR = (0, 0, 0)
    BAR_POSITION = (10, 60)
    BAR_WIDTH = 300
    BAR_HEIGHT = 25

    input_queue = Queue(maxsize=1)
    output_queue = Queue(maxsize=1)

    p = Process(target=prediction_process, args=(input_queue, output_queue, MODEL_WEIGHTS_PATH, LABELS_CSV_PATH))
    p.start()
    display_gesture = ""
    display_probability = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Cannot get a frame from wideo.")
                break

            frame_buffer.append(frame.copy())

            frame_count += 1

            if len(frame_buffer) == BUFFER_SIZE and frame_count % PREDICTION_INTERVAL == 0:
                if not input_queue.full():
                    input_queue.put(list(frame_buffer))

            while not output_queue.empty():
                display_gesture, display_probability = output_queue.get()

            cv2.rectangle(frame, (5, 5), (350, 100), BACKGROUND_RECT_COLOR, cv2.FILLED)
            cv2.rectangle(frame, (5, 5), (350, 100), BACKGROUND_COLOR, 2)

            gesture_text = f'Gesture: {display_gesture}'
            cv2.putText(frame, gesture_text, (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, TEXT_COLOR, 2, cv2.LINE_AA)

            cv2.putText(frame, 'Probability:',
                        (BAR_POSITION[0], BAR_POSITION[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, TEXT_COLOR, 2, cv2.LINE_AA)

            cv2.rectangle(frame, BAR_POSITION,
                          (BAR_POSITION[0] + BAR_WIDTH, BAR_POSITION[1] + BAR_HEIGHT),
                          BACKGROUND_COLOR, 2)

            filled_width = int(BAR_WIDTH * display_probability)

            if display_probability < 0.5:
                bar_color = (0, 0, 255)
            elif display_probability < 0.8:
                bar_color = (0, 255, 255)
            else:
                bar_color = PROGRESS_BAR_COLOR

            cv2.rectangle(frame, BAR_POSITION,
                          (BAR_POSITION[0] + filled_width, BAR_POSITION[1] + BAR_HEIGHT),
                          bar_color, cv2.FILLED)

            prob_text = f'{display_probability * 100:.2f}%'
            cv2.putText(frame, prob_text, (BAR_POSITION[0] + BAR_WIDTH + 10, BAR_POSITION[1] + BAR_HEIGHT + 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, TEXT_COLOR, 2, cv2.LINE_AA)

            cv2.imshow('Webcam - Gesture Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        logger.info("Exit by the user.")
    finally:
        input_queue.put(None)
        p.join()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
