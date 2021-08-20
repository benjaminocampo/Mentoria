import pandas as pd
import pdb
import tensorflow as tf
from train import load_embedding
from tensorflow.keras.preprocessing import text, sequence
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from keras.models import Sequential
from keras.layers import Embedding, Dense, Flatten
from preprocess import clean_text
from tqdm import tqdm

URL = "https://www.famaf.unc.edu.ar/~nocampo043/ml_challenge2019_dataset.csv"
SEED = 0


def main():
    df = pd.read_csv(URL)

    # Sample data so it's faster to debug
    df = df.sample(5000)
    # TODO only portuguese is considered. We should merge the sp and pt
    # embeddings.
    df = df[df["language"] == "portuguese"]

    # Activate progress bar in pandas
    tqdm.pandas()

    # Remove numbers, symbols, special chars, contractions, etc.
    df = df.assign(cleaned_title=df[["title", "language"]].progress_apply(
        lambda x: clean_text(*x), axis=1))

    # Split dataset into training and testing instances
    X, y = df["cleaned_title"], df["category"]

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=SEED)
    # Define tokenizers and label encoders, so they are trained with @X_train
    # instances.
    word_tokenizer = text.Tokenizer()
    word_tokenizer.fit_on_texts(X_train)
    le = LabelEncoder()
    le.fit(y_train)

    # Encode training instances
    X_train = word_tokenizer.texts_to_sequences(X_train)
    X_train = sequence.pad_sequences(X_train, padding='post')
    y_train = le.transform(y_train)

    # Encode testing instances
    X_test = word_tokenizer.texts_to_sequences(X_test)
    X_test = sequence.pad_sequences(X_test, padding='post')
    y_test = le.transform(y_test)

    # Get embedding matrix
    filename = "wiki.pt.vec"
    vocab = word_tokenizer.word_index.keys()
    embedding_dim = 300
    embedding_matrix, nof_hits, nof_misses = load_embedding(
        filename, vocab, embedding_dim)

    # Initialize embedding layer with weights
    embedding_layer = Embedding(
        len(vocab) + 1,
        embedding_dim,
        embeddings_initializer=tf.keras.initializers.Constant(
            embedding_matrix),
        trainable=False,
    )

    # Define model
    # TODO: Model should be defined inside the loop in order to tune
    # hyperparameters
    nof_classes = len(df["category"].unique())
    model = Sequential()
    model.add(embedding_layer)
    model.add(Dense(256, input_shape=(embedding_dim, ), activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(nof_classes, activation='softmax'))
    model.add(Flatten())

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Split on training to get cross-validation sets
    skf = StratifiedKFold(n_splits=5, shuffle=False)
    skf.get_n_splits(X_train, y_train)
    for train_indices, val_indices in skf.split(X_train, y_train):
        pdb.set_trace()
        history = model.fit(X_train[train_indices],
                            y_train[train_indices],
                            batch_size=128,
                            epochs=5,
                            verbose=1)
        loss, accuracy = model.evaluate(X_train[val_indices],
                                        y_train[val_indices],
                                        batch_size=128,
                                        verbose=0)
        print(f"loss: {loss}\taccuracy: {accuracy}")


if __name__ == "__main__":
    main()