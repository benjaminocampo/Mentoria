from sklearn.preprocessing import LabelEncoder
from keras.preprocessing import text, sequence


def tokenize_features(x_train, x_test):
    # Define word tokenizer instance
    word_tokenizer = text.Tokenizer()

    # Fit tokenizer using training instances
    word_tokenizer.fit_on_texts(x_train)

    # Use trained tokenizer to get @x_train, and @x_test sequences
    x_train = word_tokenizer.texts_to_sequences(x_train)
    x_test = word_tokenizer.texts_to_sequences(x_test)

    # Add padding at the end of the sequences, so they all have the same length
    x_train = sequence.pad_sequences(x_train, padding='post')
    x_test = sequence.pad_sequences(x_test, padding='post')

    maxim = max(x_train.shape[1], x_test.shape[1])
    if x_train.shape[1] < maxim:
        x_train = np.hstack((x_train, np.zeros((x_train.shape[0], maxim- x_train.shape[1])))
    if x_test.shape[1] < maxim:
        x_test = np.hstack((x_test, np.zeros((x_test.shape[0], maxim - x_test.shape[1])))
    
    # Get learned vocabulary from tokenization
    vocab = word_tokenizer.word_index.keys()

    return x_train, x_test, vocab


def encode_labels(y_train, y_test):
    # Define label encoder instance
    le = LabelEncoder()

    # Fit encoder using training instances
    le.fit(y_train)

    # Encode labels
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)
    return y_train, y_test
