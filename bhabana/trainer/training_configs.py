THE_BOOK_OF_EXPERIMENTS = {
    "brahmaputra": [
        {
            "experiment_name": "SA_EMBED_NGRAM_CNN_RNN",
            "experiment_description": "Train with preloaded spacy vectors. "
                                      "Here we see the impact of freezing the"
                                      " embedding layer. This is a regression "
                                      "experiment without rescaling the ground "
                                      "truth to 0-1. Very little weight decay",
            "dataset": {
                "name": "IMDB",
                "n_workers": 10,
                "load_spacy_vectors": True,
                "max_seq_length": 400,
                "cuda": True
            },
            "setup": {
                "epochs": 30,
                "batch_size": 32,
                "evaluate_every": 450,
                "save_every": 450,
                "early_stopping_delta": 0.2,
                "patience": 12,
                "train_on_gpu": True,
                "save_embeddings": False
            },
            "pipeline": {
                "embedding_layer": {
                    "embedding_dims": 300,
                    "embedding_dropout": 0.5,
                    "preload_word_vectors": True,
                    "train_embeddings": False
                },
                "ngram_cnn": {
                    "cnn_kernel_dims": 500,
                    "cnn_kernel_sizes": [3, 5, 9, 13],
                    "cnn_layers": 1,
                    "cnn_dropout": 0.5
                },
                "rnn": {
                    "rnn_hidden_size": 600,
                    "rnn_layers": 2,
                    "bidirectional": True,
                    "rnn_dropout": 0.5,
                    "cell_type": "gru"
                },
                "regression": {
                    "activation": "relu"
                }
            },
            "optimizer" : {
                "learning_rate": 0.001,
                "weight_decay": 0.00001,
                "lr_scheduling_milestones": [2, 7, 15, 19]
            }
        },
        {
            "experiment_name": "SA_EMBED_NGRAM_CNN_RNN",
            "experiment_description": "Train with preloaded spacy vectors. "
                                      "Here we see the impact of rescaling "
                                      "the ground truth to 0-1 and adding a "
                                      "sigmoid activation function at the "
                                      "last layer (regression layer). No "
                                      "weight decay",
            "dataset": {
                "name": "IMDB",
                "n_workers": 9,
                "load_spacy_vectors": True,
                "max_seq_length": 400,
                "rescale": (0, 1),
                "cuda": True
            },
            "setup": {
                "epochs": 30,
                "batch_size": 32,
                "evaluate_every": 450,
                "save_every": 450,
                "early_stopping_delta": 0,
                "patience": 12,
                "train_on_gpu": True,
                "save_embeddings": False
            },
            "pipeline": {
                "embedding_layer": {
                    "embedding_dims": 300,
                    "embedding_dropout": 0.5,
                    "preload_word_vectors": True,
                    "train_embeddings": False
                },
                "ngram_cnn": {
                    "cnn_kernel_dims": 500,
                    "cnn_kernel_sizes": [3, 5, 9, 13],
                    "cnn_layers": 1,
                    "cnn_dropout": 0.5
                },
                "rnn": {
                    "rnn_hidden_size": 600,
                    "rnn_layers": 2,
                    "bidirectional": True,
                    "rnn_dropout": 0.5,
                    "cell_type": "gru"
                },
                "regression": {
                    "activation": "sigmoid"
                }
            },
            "optimizer" : {
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "lr_scheduling_milestones": [2, 7, 15, 19]
            }
        },
        {
            "experiment_name": "SA_EMBED_NGRAM_CNN_RNN",
            "experiment_description": "Train with preloaded spacy vectors. "
                                      "Here we see the impact of having a "
                                      "very small network. Rescaled GT with "
                                      "sigmoid",
            "dataset": {
                "name": "IMDB",
                "n_workers": 9,
                "load_spacy_vectors": True,
                "max_seq_length": 400,
                "rescale": (0, 1),
                "cuda": True
            },
            "setup": {
                "epochs": 30,
                "batch_size": 32,
                "evaluate_every": 450,
                "save_every": 450,
                "early_stopping_delta": 0,
                "patience": 12,
                "train_on_gpu": True,
                "save_embeddings": False
            },
            "pipeline": {
                "embedding_layer": {
                    "embedding_dims": 300,
                    "embedding_dropout": 0.5,
                    "preload_word_vectors": True,
                    "train_embeddings": False
                },
                "ngram_cnn": {
                    "cnn_kernel_dims": 50,
                    "cnn_kernel_sizes": [3, 5, 9, 13],
                    "cnn_layers": 1,
                    "cnn_dropout": 0.5
                },
                "rnn": {
                    "rnn_hidden_size": 100,
                    "rnn_layers": 1,
                    "bidirectional": True,
                    "rnn_dropout": 0.5,
                    "cell_type": "gru"
                },
                "regression": {
                    "activation": "sigmoid"
                }
            },
            "optimizer" : {
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "lr_scheduling_milestones": [2, 7, 15, 19]
            }
        },
        {
            "experiment_name": "SA_EMBED_NGRAM_CNN_RNN",
            "experiment_description": "Train with preloaded spacy vectors and "
                                      "use spacy's vocab. Here we also train"
                                      " the embeddings along with the "
                                      "network. Medium sized network. Weight "
                                      "decay",
            "dataset": {
                "name": "IMDB",
                "n_workers": 9,
                "use_spacy_vocab": True,
                "load_spacy_vectors": True,
                "max_seq_length": 300,
                "rescale": (0, 1),
                "cuda": True
            },
            "setup": {
                "epochs": 30,
                "batch_size": 32,
                "evaluate_every": 450,
                "save_every": 450,
                "early_stopping_delta": 0.2,
                "patience": 12,
                "train_on_gpu": True,
                "save_embeddings": True
            },
            "pipeline": {
                "embedding_layer": {
                    "embedding_dims": 300,
                    "embedding_dropout": 0.5,
                    "preload_word_vectors": True,
                    "train_embeddings": True
                },
                "ngram_cnn": {
                    "cnn_kernel_dims": 600,
                    "cnn_kernel_sizes": [3, 5, 9, 13, 20],
                    "cnn_layers": 1,
                    "cnn_dropout": 0.5
                },
                "rnn": {
                    "rnn_hidden_size": 1200,
                    "rnn_layers": 3,
                    "bidirectional": True,
                    "rnn_dropout": 0.5,
                    "cell_type": "gru"
                },
                "regression": {
                    "activation": "sigmoid"
                }
            },
            "optimizer" : {
                "learning_rate": 0.001,
                "weight_decay": 0.00001,
                "lr_scheduling_milestones": [2, 7, 15, 19]
            }
        },
        {
            "experiment_name": "SA_EMBED_NGRAM_CNN_RNN",
            "experiment_description": "Train with preloaded spacy vectors. "
                                      "Here we see the impact of freezing the"
                                      " embedding layer. This is a regression "
                                      "experiment without rescaling the ground "
                                      "truth to 0-1. Very little weight "
                                      "decay. Here we use an LSTM cell",
            "dataset": {
                "name": "IMDB",
                "n_workers": 9,
                "load_spacy_vectors": True,
                "max_seq_length": 400,
                "cuda": True
            },
            "setup": {
                "epochs": 30,
                "batch_size": 32,
                "evaluate_every": 450,
                "save_every": 450,
                "early_stopping_delta": 0.2,
                "patience": 12,
                "train_on_gpu": True,
                "save_embeddings": False
            },
            "pipeline": {
                "embedding_layer": {
                    "embedding_dims": 300,
                    "embedding_dropout": 0.5,
                    "preload_word_vectors": True,
                    "train_embeddings": False
                },
                "ngram_cnn": {
                    "cnn_kernel_dims": 500,
                    "cnn_kernel_sizes": [3, 5, 9, 13],
                    "cnn_layers": 1,
                    "cnn_dropout": 0.5
                },
                "rnn": {
                    "rnn_hidden_size": 600,
                    "rnn_layers": 2,
                    "bidirectional": True,
                    "rnn_dropout": 0.5,
                    "cell_type": "lstm"
                },
                "regression": {
                    "activation": "relu"
                }
            },
            "optimizer" : {
                "learning_rate": 0.001,
                "weight_decay": 0.00001,
                "lr_scheduling_milestones": [2, 7, 15, 19]
            }
        }
    ],
    "yamuna": [
        {
            "experiment_name": "SA_EMBED_NGRAM_CNN_RNN_CLASSIFICATION",
            "experiment_description": "Train with preloaded spacy vectors. "
                                      "Here we see the impact of freezing the"
                                      " embedding layer.",
            "dataset": {
                "name": "IMDB",
                "n_workers": 10,
                "load_spacy_vectors": False,
                "max_seq_length": 400,
                "cuda": True
            },
            "setup": {
                "epochs": 30,
                "batch_size": 32,
                "evaluate_every": 450,
                "save_every": 450,
                "early_stopping_delta": 0.0,
                "patience": 10,
                "train_on_gpu": True,
                "save_embeddings": False
            },
            "pipeline": {
                "embedding_layer": {
                    "embedding_dims": 300,
                    "embedding_dropout": 0.5,
                    "preload_word_vectors": True,
                    "train_embeddings": False
                },
                "ngram_cnn": {
                    "cnn_kernel_dims": 500,
                    "cnn_kernel_sizes": [3, 5, 9, 13],
                    "cnn_layers": 1,
                    "cnn_dropout": 0.5
                },
                "rnn": {
                    "rnn_hidden_size": 600,
                    "rnn_layers": 2,
                    "bidirectional": True,
                    "rnn_dropout": 0.5,
                    "cell_type": "gru"
                }
            },
            "optimizer" : {
                "learning_rate": 0.001,
                "weight_decay": 0.00001,
                "lr_scheduling_milestones": [2, 7, 15, 19]
            }
        },
        {
            "experiment_name": "SA_EMBED_NGRAM_CNN_RNN_CLASSIFICATION",
            "experiment_description": "Train with preloaded spacy vectors. "
                                      "Here we see the impact of having a "
                                      "very small network",
            "dataset": {
                "name": "IMDB",
                "n_workers": 9,
                "load_spacy_vectors": True,
                "max_seq_length": 400,
                "cuda": True
            },
            "setup": {
                "epochs": 30,
                "batch_size": 32,
                "evaluate_every": 450,
                "save_every": 450,
                "early_stopping_delta": 0,
                "patience": 12,
                "train_on_gpu": True,
                "save_embeddings": False
            },
            "pipeline": {
                "embedding_layer": {
                    "embedding_dims": 300,
                    "embedding_dropout": 0.5,
                    "preload_word_vectors": True,
                    "train_embeddings": False
                },
                "ngram_cnn": {
                    "cnn_kernel_dims": 50,
                    "cnn_kernel_sizes": [3, 5, 9, 13],
                    "cnn_layers": 1,
                    "cnn_dropout": 0.5
                },
                "rnn": {
                    "rnn_hidden_size": 100,
                    "rnn_layers": 1,
                    "bidirectional": True,
                    "rnn_dropout": 0.5,
                    "cell_type": "gru"
                }
            },
            "optimizer" : {
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "lr_scheduling_milestones": [2, 7, 15, 19]
            }
        },
        """
        {
            "experiment_name": "SA_EMBED_NGRAM_CNN_RNN_CLASSIFICATION",
            "experiment_description": "Train with preloaded spacy vectors and "
                                      "use spacy's vocab. Here we also train"
                                      " the embeddings along with the "
                                      "network. Medium sized network. Weight "
                                      "decay",
            "dataset": {
                "name": "IMDB",
                "n_workers": 9,
                "use_spacy_vocab": True,
                "load_spacy_vectors": True,
                "max_seq_length": 300,
                "cuda": True
            },
            "setup": {
                "epochs": 30,
                "batch_size": 32,
                "evaluate_every": 450,
                "save_every": 450,
                "early_stopping_delta": 0.0,
                "patience": 12,
                "train_on_gpu": True,
                "save_embeddings": True
            },
            "pipeline": {
                "embedding_layer": {
                    "embedding_dims": 300,
                    "embedding_dropout": 0.5,
                    "preload_word_vectors": True,
                    "train_embeddings": True
                },
                "ngram_cnn": {
                    "cnn_kernel_dims": 600,
                    "cnn_kernel_sizes": [3, 5, 9, 13, 20],
                    "cnn_layers": 1,
                    "cnn_dropout": 0.5
                },
                "rnn": {
                    "rnn_hidden_size": 1200,
                    "rnn_layers": 3,
                    "bidirectional": True,
                    "rnn_dropout": 0.5,
                    "cell_type": "gru"
                }
            },
            "optimizer" : {
                "learning_rate": 0.001,
                "weight_decay": 0.00001,
                "lr_scheduling_milestones": [2, 7, 15, 19]
            }
        }
        """
        ,
        {
            "experiment_name": "SA_EMBED_NGRAM_CNN_RNN_CLASSIFICATION",
            "experiment_description": "Train with preloaded spacy vectors. "
                                      "Here we see the impact of freezing the"
                                      " embedding layer. Very little weight "
                                      "decay. Here we use an LSTM cell",
            "dataset": {
                "name": "IMDB",
                "n_workers": 9,
                "load_spacy_vectors": True,
                "max_seq_length": 400,
                "cuda": True
            },
            "setup": {
                "epochs": 30,
                "batch_size": 32,
                "evaluate_every": 450,
                "save_every": 450,
                "early_stopping_delta": 0.0,
                "patience": 10,
                "train_on_gpu": True,
                "save_embeddings": False
            },
            "pipeline": {
                "embedding_layer": {
                    "embedding_dims": 300,
                    "embedding_dropout": 0.5,
                    "preload_word_vectors": True,
                    "train_embeddings": False
                },
                "ngram_cnn": {
                    "cnn_kernel_dims": 500,
                    "cnn_kernel_sizes": [3, 5, 9, 13],
                    "cnn_layers": 1,
                    "cnn_dropout": 0.5
                },
                "rnn": {
                    "rnn_hidden_size": 600,
                    "rnn_layers": 2,
                    "bidirectional": True,
                    "rnn_dropout": 0.5,
                    "cell_type": "lstm"
                }
            },
            "optimizer" : {
                "learning_rate": 0.001,
                "weight_decay": 0.00001,
                "lr_scheduling_milestones": [2, 7, 15, 19]
            }
        },
        {
            "experiment_name": "SA_EMBED_NGRAM_CNN_RNN_CLASSIFICATION",
            "experiment_description": "Train with preloaded spacy vectors. "
                                      "Here we see the impact of of having a "
                                      "very small network with very wide CNN "
                                      "filters",
            "dataset": {
                "name": "IMDB",
                "n_workers": 9,
                "load_spacy_vectors": True,
                "max_seq_length": 400,
                "cuda": True
            },
            "setup": {
                "epochs": 30,
                "batch_size": 32,
                "evaluate_every": 450,
                "save_every": 450,
                "early_stopping_delta": 0.0,
                "patience": 10,
                "train_on_gpu": True,
                "save_embeddings": False
            },
            "pipeline": {
                "embedding_layer": {
                    "embedding_dims": 300,
                    "embedding_dropout": 0.5,
                    "preload_word_vectors": True,
                    "train_embeddings": False
                },
                "ngram_cnn": {
                    "cnn_kernel_dims": 200,
                    "cnn_kernel_sizes": [3, 5, 9, 13, 20, 30, 40],
                    "cnn_layers": 1,
                    "cnn_dropout": 0.5
                },
                "rnn": {
                    "rnn_hidden_size": 400,
                    "rnn_layers": 2,
                    "bidirectional": True,
                    "rnn_dropout": 0.5,
                    "cell_type": "gru"
                }
            },
            "optimizer" : {
                "learning_rate": 0.001,
                "weight_decay": 0.00001,
                "lr_scheduling_milestones": [2, 7, 15, 19]
            }
        }
    ]
}