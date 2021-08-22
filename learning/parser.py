import argparse


def get_parser():
    parser = argparse.ArgumentParser(description="Process model arguments")
    parser.add_argument("--experiment_name",
                        help="name used by mlflow to record your experiments",
                        required=True)
    parser.add_argument("--kfolds",
                        help="number of folds used in cross-validation",
                        type=int,
                        default=5)
    parser.add_argument("--seed",
                        help="random seed state during train-test splitting",
                        type=int,
                        default=0)
    parser.add_argument("--batch_size",
                        help="batch size used in mini-batches",
                        type=int,
                        default=128)
    parser.add_argument("--embedding_dim",
                        help="dimension used in word embeddings",
                        type=int,
                        default=300)
    parser.add_argument("--embedding_file",
                        help="filename of pre-trained word embeddings",
                        type=str,
                        default="wiki.pt.vec")
    parser.add_argument("--nof_samples",
                        help="number of samples used of the dataset",
                        type=int,
                        default=5000)
    parser.add_argument(
        "--test_size",
        help="test size percentage used in stratified sampling splitting",
        type=float,
        default=.2)
    parser.add_argument("--epochs",
                        help="Number of epochs used when validating",
                        type=float,
                        default=5)
    parser.add_argument(
        "--dataset_path",
        help="url of where the database is located",
        default=
        "https://www.famaf.unc.edu.ar/~nocampo043/ml_challenge2019_dataset.csv"
    )

    return parser
