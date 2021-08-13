import pandas as pd
import pdb
from train import load_embedding
from tensorflow.keras.preprocessing import text, sequence
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


URL = "https://www.famaf.unc.edu.ar/~nocampo043/ml_challenge2019_dataset.csv"
SEED = 0


def main():
    df = pd.read_csv(URL)

    # Preprocess
    df = df.assign(cleaned_title=df[["title", "language"]].apply(
        lambda x: clean_text(*x), axis=1))

    # Splitting
    X, y = df["title"], df["category"]

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=SEED)
    # Encoders
    word_tokenizer = text.Tokenizer()
    word_tokenizer.fit_on_texts(X_train)
    le = LabelEncoder()

    # Training encoding
    X_train = word_tokenizer.texts_to_sequences(X_train)
    X_train = sequence.pad_sequences(X_train, padding='post')
    y_train = le.fit_transform(y_train)

    # Testing encoding
    X_test = word_tokenizer.texts_to_sequences(X_test)
    X_test = sequence.pad_sequences(X_test, padding='post')
    y_test = le.fit_transform(y_test)

    # Cross validation
    skf = StratifiedKFold(n_splits=5, shuffle=False)
    skf.get_n_splits(X, y)

    splits = [{
        "X_train": X_train[train_index],
        "X_validation": X_train[validation_index],
        "y_train": y_train[train_index],
        "y_validation": y_train[validation_index]
    } for train_index, validation_index in skf.split(X_train, y_train)]

    # Embedding layer pretrained
    filename = "wiki.pt.vec"
    vocab = word_tokenizer.word_index.keys()
    embedding_dim = 300
    embedding_layer, nof_hits, nof_misses = load_embedding(filename, vocab, embedding_dim)
    pdb.set_trace()
    
    #embedding_layer.embeddings_initializer.value

    # Define model
    model = Sequential()
    model.add(embedding_layer)
    model.compile(loss='sparse_categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])


    # model.add(Dense(units = #Cantidad de clases))
    # model.add(Softmax)
    #model.add(Dropout(0.2))
    #model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    #model.fit(X_train, y_train, epochs = 100, batch_size = 32)

main()
