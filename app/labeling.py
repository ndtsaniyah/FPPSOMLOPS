import tensorflow as tf
import tensorflow_datasets as tfds

def load_sentiment140_dataset():
    dataset, info = tfds.load('sentiment140', with_info=True, as_supervised=True)
    train_dataset, test_dataset = dataset['train'], dataset['test']
    return train_dataset, test_dataset, info

def preprocess_text(text):
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, '[^\w\s]', '')
    return text

def label_function(dataset):
    labeled_data = []

    def process_text(text, label):
        text = preprocess_text(text)
        return text, label

    for text, label in dataset:
        labeled_data.append(process_text(text, label))

    return labeled_data

def main():
    train_dataset, test_dataset, _ = load_sentiment140_dataset()
    train_labeled = label_function(train_dataset)
    test_labeled = label_function(test_dataset)
    for text, label in train_labeled[:5]:
        print(f'Text: {text.numpy().decode("utf-8")}, Label: {label.numpy()}')

if __name__ == '__main__':
    main()
