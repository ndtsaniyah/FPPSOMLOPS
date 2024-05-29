import tensorflow as tf
from app.labeling import load_sentiment140_dataset, preprocess_text

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=5000, output_dim=16, input_length=100),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    train_dataset, test_dataset, _ = load_sentiment140_dataset()

    train_texts = []
    train_labels = []
    for text, label in train_dataset:
        train_texts.append(preprocess_text(text).numpy().decode("utf-8"))
        train_labels.append(label.numpy())

    test_texts = []
    test_labels = []
    for text, label in test_dataset:
        test_texts.append(preprocess_text(text).numpy().decode("utf-8"))
        test_labels.append(label.numpy())

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(train_texts)

    train_sequences = tokenizer.texts_to_sequences(train_texts)
    test_sequences = tokenizer.texts_to_sequences(test_texts)

    train_padded = tf.keras.preprocessing.sequence.pad_sequences(train_sequences, maxlen=100)
    test_padded = tf.keras.preprocessing.sequence.pad_sequences(test_sequences, maxlen=100)

    model = create_model()
    model.fit(train_padded, train_labels, epochs=10, validation_data=(test_padded, test_labels))
    model.save('sentiment_model.h5')

if __name__ == '__main__':
    main()
