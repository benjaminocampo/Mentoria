from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


def create_vectorize_layer(length_long_sentence, output_mode):
    return TextVectorization(output_mode=output_mode,
                             output_sequence_length=length_long_sentence)


def encode_labels(y_train, y_test):
    # Define label encoder instance
    le = LabelEncoder()

    # Fit encoder using training instances
    le.fit(y_train)

    # Encode labels
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)
    return y_train, y_test
