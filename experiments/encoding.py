from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


def create_vectorize_layer(length_long_sentence, output_mode):
    return TextVectorization(output_mode=output_mode,
                             output_sequence_length=length_long_sentence)

def encode_labels(labels):
    le = LabelEncoder()
    return le.fit_transform(labels)
