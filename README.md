<h1 align="center">Playing with AI (Machine Learning) module</h1>

<h3>Overview</h3>
=================
<!-- <img src="img/.png" width="300"> -->
<p align="justify"> 
This is an application with several machine learning algorithms, and can be used to create AI (Artificial Intelligence) models. A report with the metrics is also generated.
</p>

<h3>How to run?</h3>
=================

<p align="justify">
To Execute, run the main.py file and pass a json with the following parameters with mandatory fields:
</p>

<p align="justify">
<!--ts-->
    {
        "input_file": None,                            ---> Mandatory. Accepts values as True or False (None).
        "learning": None,                              ---> Mandatory. Accepts values as supervised, semisupervised, unsupervised or network.
        "reading":{                                    ---> Mandatory. 
            "reading_mode":None,                       ---> Mandatory. Accepts values as CSV, JSON, PARQUET or DATABASE. 
            "host": None,                              ---> Hostname or IP.
            "user": None,                              ---> Username.
            "password": None,                          ---> Username's password.
            "caminho": None,                           ---> Mandatory. Directory where the file is located.
            "nome_arquivo": None,                      ---> Mandatory. Filename.
            "database": None,                          ---> What is the database schema?
            "type_database": None,                     ---> What is the database? Mysql, Postgres, Redshift...
            "query": None                              ---> Query that will get data from the bank.
        },
        "pipeline": {                                  ---> Can be None.
                "numeric_features": ['col1','col2'],   ---> Columns you want to perform numerical normalization on.
                "categorical_features": None,          ---> Columns you want to perform categorical normalization on.
                "normalize_method_num":None,           ---> Accepts values as True or False (None).
                "normalize_method_cat": None           ---> Accepts values as True or False (None).
        },
        "validations":{                                ---> Can be None.
            "cross_validation": None,                  ---> If you want to do cross validation.
            "random": None,                            ---> If you want to perform rando type hyperparameter search.
            "grid": None,                              ---> If you want to perform grid type hyperparameter search.
            "param":{                                  ---> Mandatory for random and grid. Parameters you want to search for and what are the values.
                'model__optimizer': ['opt1', 'opt2'],
                'model__batch_size': [32, 64, 128]
            },                             
            "folders": None,                           ---> Mandatory for anything validation. Number of folders.
            "scoring": None                            ---> Mandatory for anything validation. Evaluation score.
        },
        "mode":{                                       ---> Mandatory. 
            "name_mode": None,                         ---> Mandatory. Model's name.
            "algorithm": None,                         ---> Mandatory. Algorithm's name.
            "target": None,                            ---> Mandatory for supervised and semisupervised. What is the name of the target column, or what is the json/parquet target field.
            "learning_network": None,                  ---> Mandatory for networks. Accepts values as supervised or unsupervised.
            "params": {                                ---> Mandatory. Algorithm configuration parameters. Each one has its own.
                HERE YOU PASS THE CONFIGURATIONS OF EACH ALGORITHM. IT IS EXPLAINED IN ALGORITHM SETTINGS.
            }
        }
    }
<!--te-->
</p>

<p>
The program will check if it is a valid JSON. 
If it is, the parameters will be set and the process will start. 
First, it gets the data and performs normalizations, if necessary. 
Then the model is configured and training begins. 
After training, validations are performed if requested, and the metrics for the chosen model are calculated. 
At the end, it saves the model and a file with the extension ".md", with the model evaluations.
</p>

<h3>Algorithm Settings</h3>
=================

<h4>Settings for each type of Algorithm</h4>

<h6><b>Supervised</b><h6>

<p align="justify">In this type of learning, the algorithm is trained using labeled data, that is, input data that already has the correct answer. The goal is for the algorithm to learn to map the inputs to the correct outputs so that it can make accurate predictions on new data.</p>

<p align="justify">Logistic Regression - Classification</p>
<p align="justify">
<!--ts-->
    penalty - Regularization type: 'l1', 'l2', 'elasticnet', 'none'
    C - Inverse of regularization strength
    solver - Algorithm for optimization: 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'
    max_iter - Maximum number of iterations
    random_state - Seed for reproducibility
<!--te-->
</p>

<p align="justify">Decision Tree - Classification</p>
<p align="justify">
<!--ts-->
    criterion - Function to measure the quality of a split: 'gini', 'entropy'
    max_depth - Maximum depth of the tree
    min_samples_split - Minimum number of samples needed to split a node
    min_samples_leaf - Minimum number of samples needed in a leaf
    random_state - Seed for reproducibility
<!--te-->
</p>

<p align="justify">Random Forests - Classification</p>
<p align="justify">
<!--ts-->
    n_estimators - Number of trees in the forest
    criterion - Function to measure the quality of a split: 'gini', 'entropy'
    max_depth - Maximum depth of each tree
    min_samples_split - Minimum number of samples needed to split a node
    min_samples_leaf - Minimum number of samples needed in a leaf
    random_state - Seed for reproducibility
<!--te-->
</p>

<p align="justify">Gradient Boosting - Classification</p>
<p align="justify">
<!--ts-->
    n_estimators - Number of boosting stages
    learning_rate - Learning rate
    max_depth - Maximum depth of each tree
    min_samples_split - Minimum number of samples needed to split a node
    min_samples_leaf - Minimum number of samples needed in a leaf
    random_state - Seed for reproducibility
<!--te-->
</p>

<p align="justify">Ada Boost - Classification</p>
<p align="justify">
<!--ts-->
     n_estimators - Number of estimators
     learning_rate - Learning rate
     algorithm - Algorithm: 'SAMME', 'SAMME.R'
     random_state - Seed for reproducibility
<!--te-->
</p>

<p align="justify">Naive Bayes - Classification</p>
<p align="justify">
<!--ts-->
    priors - Prior probabilities of the classes (if None, calculated from the data)
    var_smoothing - Smoothing parameter to avoid zero variance
<!--te-->
</p>

<p align="justify">SVM (Support Vector Machine) - Classification</p>
<p align="justify">
<!--ts-->
    C - Regularization parameter
    kernel - Kernel type: 'linear', 'poly', 'rbf', 'sigmoid'
    degree - Degree of the polynomial kernel (only for 'poly' kernels)
    gamma - Kernel coefficient (only for 'rbf', 'poly', 'sigmoid' kernels)
    probability - Enables probability calculation
    random_state - Seed for reproducibility
<!--te-->
</p>

<p align="justify">KNN (K-Nearest Neighbors) - Classification</p>
<p align="justify">
<!--ts-->
    n_neighbors - Number of neighbors
    weights - Neighbor weights: 'uniform', 'distance'
    algorithm - Algorithm for calculating neighbors: 'auto', 'ball_tree', 'kd_tree', 'brute'
    p - Minkowski distance parameter (1 for Manhattan, 2 for Euclidean)
<!--te-->
</p>

<p align="justify">Decision Tree - Regression</p>
<p align="justify">
<!--ts-->
    criterion - Function to measure the quality of a split: 'squared_error', 'friedman_mse', 'absolute_error'
    splitter - Strategy to choose the split: 'best', 'random'
    max_depth - Maximum depth of the tree
    min_samples_split - Minimum number of samples to split a node
    min_samples_leaf - Minimum number of samples in a leaf
    max_features - Maximum number of features considered for splitting
    random_state - Seed for reproducibility
<!--te-->
</p>

<p align="justify">Random Forests - Regression</p>
<p align="justify">
<!--ts-->
    n_estimators - Number of trees in the forest
    criterion - Function to measure the quality of a split
    max_depth - Maximum depth of each tree
    min_samples_split - Minimum number of samples to split a node
    min_samples_leaf - Minimum number of samples in a leaf
    max_features - Maximum number of features considered for splitting
    bootstrap - If True, uses sampling with replacement
    random_state - Seed for reproducibility
<!--te-->
</p>

<p align="justify">Gradient Boosting - Regression</p>
<p align="justify">
<!--ts-->
    n_estimators - Number of boosting stages
    learning_rate - Learning rate
    loss - Loss function: 'squared_error', 'absolute_error', 'huber', 'quantile'
    max_depth - Maximum depth of each tree
    min_samples_split - Minimum number of samples to split a node
    min_samples_leaf - Minimum number of samples in a leaf
    random_state - Seed for reproducibility
<!--te-->
</p>

<p align="justify">Ada Boost - Regression</p>
<p align="justify">
<!--ts-->
    base_estimator - Base estimator (if None, use DecisionTreeRegressor with max_depth=3)
    n_estimators - Number of estimators
    learning_rate - Learning rate
    loss - Loss function: 'linear', 'square', 'exponential'
    random_state - Seed for reproducibility
<!--te-->
</p>

<p align="justify">SVM (Support Vector Machine) - Regression</p>
<p align="justify">
<!--ts-->
    kernel - Kernel type: 'linear', 'poly', 'rbf', 'sigmoid'
    degree - Degree of the polynomial kernel (only for 'poly' kernels)
    gamma - Kernel coefficient (only for 'rbf', 'poly', 'sigmoid' kernels)
    C - Regularization parameter (the higher, the less regularization)
    epsilon - Margin of error for regression
    shrinking - If True, uses the shrinking heuristic to speed up training
    tol - Tolerance for stopping criterion
    max_iter - Maximum number of iterations (-1 = unlimited)
<!--te-->
</p>

<p align="justify">Ridge - Regression</p>
<p align="justify">
<!--ts-->
    alpha - Regularization parameter (the higher, the stronger the regularization)
    fit_intercept - If True, calculates the intercept (bias)
    copy_X - If True, copies the input data
    max_iter - Maximum number of iterations for the solver
    tol - Tolerance for stopping criterion
    solver - Optimization algorithm: 'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'
    random_state - Seed for reproducibility (used in 'sag' and 'saga')
<!--te-->
</p>

<p align="justify">Elasticnet - Regression</p>
<p align="justify">
<!--ts-->
    alpha - Regularization parameter (combines L1 and L2)
    l1_ratio - Ratio between L1 and L2 (0 = Ridge, 1 = Lasso)
    fit_intercept - If True, compute the intercept (bias)
    precompute - If True, precompute the Gram matrix
    max_iter - Maximum number of iterations
    tol - Tolerance for stopping criterion
    warm_start - If True, reuse the previous solution as initialization
    positive - If True, force the coefficients to be positive
    random_state - Seed for reproducibility
<!--te-->
</p>

<p align="justify">Lasso - Regression</p>
<p align="justify">
<!--ts-->
    alpha - Regularization parameter (the larger, the stronger the regularization)
    fit_intercept - If True, compute the intercept (bias)
    precompute - If True, precompute the Gram matrix to speed up training
    copy_X - If True, copy the input data
    max_iter - Maximum number of iterations
    tol - Tolerance for stopping criterion
    warm_start - If True, reuse the previous solution as initialization
    positive - If True, force the coefficients to be positive
    random_state - Seed for reproducibility
<!--te-->
</p>

<p align="justify">Linear - Regression</p>
<p align="justify">
<!--ts-->
    fit_intercept - If True, calculates the intercept (bias)
    copy_X - If True, copies the input data (avoids modifying the original data)
    n_jobs - Number of jobs to parallelize (None = 1, -1 = all cores)
<!--te-->
</p>

<h6><b>SemiSupervised</b><h6>

<p align="justify">This approach combines elements of supervised and unsupervised learning. The algorithm is trained using a combination of labeled and unlabeled data. This can be useful when data labeling is expensive or time-consuming.</p>

<p align="justify">Label Propagation - Classification </p>
<p align="justify">
<!--ts-->
    kernel - Type of kernel used to calculate the similarity between samples
    n_neighbors - Number of neighbors
    max_iter - Maximum number of iterations
    tol - Tolerance for convergence
    n_jobs - Number of parallel jobs to execute the algorithm
<!--te-->
</p>

<p align="justify">Label Spreading - Classification </p>
<p align="justify">
<!--ts-->
    kernel - Type of kernel used to calculate the similarity between samples
    n_neighbors - Number of neighbors
    alpha - Smoothing factor
    max_iter - Maximum number of iterations
    tol - Tolerance for convergence
    n_jobs - Number of parallel jobs to execute the algorithm
<!--te-->
</p>

<h6><b>Unsupervised</b><h6>

<p align="justify">In this case, the algorithm is trained using unlabeled data. The goal is for the algorithm to find hidden patterns and structures in the data, without the need for predefined <answers class=""></answers></p>

<p align="justify">K-means - Clustering </p>
<p align="justify">
<!--ts-->
    n_clusters = Number of clusters
    init - Centroid initialization method
    n_init - Number of times the algorithm will be executed with different seeds
    max_iter - Maximum number of iterations
    random_state - Seed for reproducibility
<!--te-->
</p>

<p align="justify">Hierarchical clustering - Clustering </p>
<p align="justify">
<!--ts-->
    n_clusters - Number of clusters
    affinity - Distance metric (e.g. 'euclidean', 'manhattan', 'cosine')
    linkage - Linkage criterion (e.g. 'ward', 'complete', 'average', 'single')
<!--te-->
</p>

<p align="justify"> DBSCAN (Density-Based Spatial Clustering of Applications with Noise) - Clustering </p>
<p align="justify">
<!--ts-->
    eps - Maximum distance between two points to be considered neighbors
    min_samples - Minimum number of points to form a cluster
    metric - Distance metric (e.g. 'euclidean', 'manhattan', 'cosine')
<!--te-->
</p>

<p align="justify">GMM (Gaussian Mixture Model) - Clustering </p>
<p align="justify">
<!--ts-->
    n_components - Number of components (clusters)
    covariance_type - Type of covariance matrix (e.g. 'full', 'tied', 'diag', 'spherical')
    random_state - Seed for reproducibility
<!--te-->
</p>

<p align="justify">Spectral clustering - Clustering </p>
<p align="justify">
<!--ts-->
    n_clusters - Number of clusters
    affinity - Affinity matrix construction method (e.g. 'rbf', 'nearest_neighbors')
    random_state - Seed for reproducibility
<!--te-->
</p>

<p align="justify">PCA (Principal Component Analysis) - Dimensionality Reduction </p>
<p align="justify">
<!--ts-->
    n_components - Number of principal components to retain
    whiten - If True, normalizes the principal components
    random_state - Seed for reproducibility
<!--te-->
</p>

<p align="justify">LDA (Linear Discriminant Analysis) - Dimensionality Reduction </p>
<p align="justify">
<!--ts-->
    n_components - Número de componentes a serem retidos
<!--te-->
</p>

<p align="justify">TSNE (t-distributed Stochastic Neighbor Embedding) - Dimensionality Reduction </p>
<p align="justify">
<!--ts-->
    n_components - Number of dimensions in the reduced space
    perplexity - Number of neighbors considered
    learning_rate - Learning rate
    random_state - Seed for reproducibility
<!--te-->
</p>

<p align="justify">ISOMAP (Isometric Mapping) - Dimensionality Reduction </p>
<p align="justify">
<!--ts-->
    n_components - Number of dimensions in the reduced space
    n_neighbors - Number of neighbors to build the graph
<!--te-->
</p>

<p align="justify">Factor Analysis - Dimensionality Reduction </p>
<p align="justify">
<!--ts-->
    n_components - Number of latent factors
    random_state - Seed for reproducibility
<!--te-->
</p>

<p align="justify">NMF (Non-negative Matrix Factorization) - Dimensionality Reduction </p>
<p align="justify">
<!--ts-->
    n_components - Number of components
    init - Initialization method
    random_state - Seed for reproducibility
<!--te-->
</p>

<h6><b>Networks</b><h6>

<p align="justify">Neural networks are computational models inspired by the workings of the human brain. They consist of layers of interconnected "neurons"1 that process information. Neural networks can be used in both supervised and unsupervised learning, and are particularly effective in complex tasks such as image recognition and natural language processing.</p>

<p align="justify">Perceptron - Classification</p>
<p align="justify">
<!--ts-->
    penalty - Regularization
    alpha - Regularization constant
    max_iter - Maximum number of iterations
    tol - Stopping tolerance
    eta0 - Initial learning rate
    shuffle - Shuffle the data after each epoch
    random_state - Seed for reproducibility
<!--te-->
</p>

<p align="justify">Som (Self Organization Maps) - Custering</p>
<p align="justify">
<!--ts-->
    map_size - Map size (10x10 neurons) (e.g. (10, 10))
    input_len - Number of features (data dimensions) (e.g. X.shape[1])
    sigma - Initial neighbor radius
    learning_rate - Initial learning rate
    random_seed = Seed for reproducibility
    epoch - Number of training epochs
    power - False for this type of network
<!--te-->
</p>

<p align="justify">MLP (Multilayer Perceptron) - Classification </p>
<p align="justify">
<!--ts-->
    hidden_layer_sizes - A hidden layer with neurons
    activation - Activation function
    solver - Optimization algorithm
    alpha - Regularization constant
    learning_rate - Learning rate
    max_iter - Maximum number of iterations
    tol - Stopping tolerance
    random_state - Seed for reproducibility
    power - False for this type of network
<!--te-->
</p>

<p align="justify">MLP (Multilayer Perceptron) - Regression </p>
<p align="justify">
<!--ts-->
    hidden_layer_sizes - A hidden layer with neurons
    activation - Activation function
    solver - Optimization algorithm
    alpha - Regularization constant
    learning_rate - Learning rate
    max_iter - Maximum number of iterations
    tol - Stopping tolerance
    random_state - Seed for reproducibility
    power - False for this type of network
<!--te-->
</p>

<p align="justify">DNN (Deep Neural Network) - Classification </p>
<p align="justify">
<!--ts-->
    n_camadas - number of layers
    activation_enter - Input layer activation function (string: 'tanh', 'relu', 'sigmoid', etc.)
    n_neuronio_enter - Number of input neurons, usually the number of data that is passed
    activation_intermediate - Intermediate layer activation function (string: 'tanh', 'relu', 'sigmoid', etc.)
    n_neuronio_intermediate - Number of neurons in the intermediate layer
    activation_end - Output layer activation function (string: 'tanh', 'relu', 'sigmoid', etc.)
    n_neuronio_end - Number of output neurons, usually the number of classes
    input_shape - Input data shape (tuple of integers)
    optimizer - Optimizer for training (string: 'adam', 'sgd', 'rmsprop', etc.)
    loss - Loss function (string: 'categorical_crossentropy', 'mse', 'binary_crossentropy', etc.)
    metrics - Metrics for evaluation (list of strings - example: accuracy)
    epoch - Number of training epochs
    batch_size - number of samples that will be processed in each training iteration
    power - True for this type of network
<!--te-->
</p>

<p align="justify">DNN (Deep Neural Network) - Regression </p>
<p align="justify">
<!--ts-->
    n_camadas - number of layers
    activation_enter - Input layer activation function
    n_neuronio_enter - Number of input neurons, usually the number of data passed
    activation_intermediate - Intermediate layer activation function
    n_neuronio_intermediate - Number of neurons in the intermediate layer
    activation_end - Output layer activation function
    n_neuronio_end - Number of output neurons, usually the number of classes
    input_shape - Input data shape (tuple of integers)
    optimizer - Optimizer for training (string: 'adam', 'sgd', 'rmsprop', etc.)
    loss - Loss function (string: 'categorical_crossentropy', 'mse', 'binary_crossentropy', etc.)
    metrics - Metrics for evaluation (list of strings - example: accuracy)
    epoch - Number of training epochs
    batch_size - number of samples that will be processed in each training iteration
    power - True for this type of network
<!--te-->
</p>

<p align="justify">CNN (Convolutional Neural Network) - Classification </p>
<p align="justify">
<!--ts-->
    n_camadas - number of layers
    activation_enter - Input layer activation function (string: 'tanh', 'relu', 'sigmoid', etc.)
    n_neuronio_enter - Number of input neurons, usually the number of data that is passed
    activation_intermediate - Intermediate layer activation function (string: 'tanh', 'relu', 'sigmoid', etc.)
    n_neuronio_intermediate - Number of neurons in the intermediate layer
    activation_end - Output layer activation function (string: 'tanh', 'relu', 'sigmoid', etc.)
    n_neuronio_end - Number of output neurons, usually the number of classes
    input_shape - Input data shape (tuple of integers: height, width, channels)
    kernel_enter - Kernel size in the input layer (tuple of integers)
    kernel_intermediate - Kernel size in the intermediate layers (tuple of integers)
    pooling_enter - Pooling size in the input layer (tuple of integers)
    pooling_intermediate - Pooling size in the intermediate layers (tuple of integers)
    optimizer - Optimizer for training (string: 'adam', 'sgd', 'rmsprop', etc.)
    loss - Loss function (string: 'categorical_crossentropy', 'mse', 'binary_crossentropy', etc.)
    metrics - Metrics for evaluation (list of strings - example: accuracy)
    epoch - Number of training epochs
    batch_size - number of samples that will be processed in each training iteration
    power - True for this type of network
<!--te-->
</p>

<p align="justify">CNN (Convolutional Neural Network) - Regression </p>
<p align="justify">
<!--ts-->
    n_camadas - number of layers
    activation_enter - Input layer activation function (string: 'tanh', 'relu', 'sigmoid', etc.)
    n_neuronio_enter - Number of input neurons, usually the number of data that is passed
    activation_intermediate - Intermediate layer activation function (string: 'tanh', 'relu', 'sigmoid', etc.)
    n_neuronio_intermediate - Number of neurons in the intermediate layer
    activation_end - Output layer activation function (string: 'tanh', 'relu', 'sigmoid', etc.)
    n_neuronio_end - Number of output neurons, usually the number of classes
    input_shape - Input data shape (tuple of integers: height, width, channels)
    kernel_enter - Kernel size in the input layer (tuple of integers)
    kernel_intermediate - Kernel size in the intermediate layers (tuple of integers)
    pooling_enter - Pooling size in the input layer (tuple of integers)
    pooling_intermediate - Pooling size in the intermediate layers (tuple of integers)
    optimizer - Optimizer for training (string: 'adam', 'sgd', 'rmsprop', etc.)
    loss - Loss function (string: 'categorical_crossentropy', 'mse', 'binary_crossentropy', etc.)
    metrics - Metrics for evaluation (list of strings - example: accuracy)
    epoch - Number of training epochs
    batch_size - number of samples that will be processed in each training iteration
    power - True for this type of network
<!--te-->
</p>

<p align="justify">RNN (Recurrent Neural Network) - Classification </p>
<p align="justify">
<!--ts-->
    n_camadas - number of layers
    return_sequences_enter - Return sequences in the input layer (boolean: True or False)
    return_sequences_intermediate - Return sequences in the intermediate layers (boolean: True or False)
    activation_enter - Input layer activation function (string: 'tanh', 'relu', 'sigmoid', etc.)
    n_neuronio_enter - Number of input neurons, usually the number of data that is passed
    activation_intermediate - Intermediate layer activation function (string: 'tanh', 'relu', 'sigmoid', etc.)
    n_neuronio_intermediate - Number of neurons in the intermediate layer
    activation_end - Output layer activation function (string: 'tanh', 'relu', 'sigmoid', etc.)
    n_neuronio_end - Number of output neurons, usually the number of classes
    input_shape - Input data shape (tuple of integers: time steps, features)
    optimizer - Optimizer for training (string: 'adam', 'sgd', 'rmsprop', etc.)
    loss - Loss function (string: 'categorical_crossentropy', 'mse', 'binary_crossentropy', etc.)
    metrics - Metrics for evaluation (list of strings - example: accuracy)
    epoch - Number of training epochs
    batch_size - number of samples that will be processed in each training iteration
    power - True for this type of network
<!--te-->
</p>

<p align="justify">RNN (Recurrent Neural Network) - Regression </p>
<p align="justify">
<!--ts-->
    n_camadas - number of layers
    return_sequences_enter - Return sequences in the input layer (boolean: True or False)
    return_sequences_intermediate - Return sequences in the intermediate layers (boolean: True or False)
    activation_enter - Input layer activation function (string: 'tanh', 'relu', 'sigmoid', etc.)
    n_neuronio_enter - Number of input neurons, usually the number of data that is passed
    activation_intermediate - Intermediate layer activation function (string: 'tanh', 'relu', 'sigmoid', etc.)
    n_neuronio_intermediate - Number of neurons in the intermediate layer
    activation_end - Output layer activation function (string: 'tanh', 'relu', 'sigmoid', etc.)
    n_neuronio_end - Number of output neurons, usually the number of classes
    input_shape - Input data shape (tuple of integers: time steps, features)
    optimizer - Optimizer for training (string: 'adam', 'sgd', 'rmsprop', etc.)
    loss - Loss function (string: 'categorical_crossentropy', 'mse', 'binary_crossentropy', etc.)
    metrics - Metrics for evaluation (list of strings - example: accuracy)
    epoch - Number of training epochs
    batch_size - number of samples that will be processed in each training iteration
    power - True for this type of network
<!--te-->
</p>

<p align="justify"><b><i>NOTE: REMEMBER YOU MUST PASS THESE ALGORITHM PARAMETERS IN THE 'PARAMS' SUB-FIELD OF THE 'MODE' FIELD AND IN THE 'PARAM' SUB-FIELD OF THE 'VALIDATIONS' FIELD (IF THIS IS NECESSARY).</i></b></p>

<h3>Example Json</h3>
=================

<p align="justify">
<!--ts-->
      {
            "input_file": True,
            "learning": "network",
            "reading":{
                "reading_mode":'csv',
                "host": None,
                "user": None,
                "password": None,
                "caminho": 'C:\\Documents',
                "nome_arquivo": 'iris.csv',
                "database": None,
                "type_database": None,
                "query": None
            },
            "pipeline": None,
            "validations":{
                "cross_validation": True,
                "random": None,
                "grid": None,
                "param":None,
                "folders":5,
                "scoring":'accuracy'
            },
            "mode":{
                "name_mode": "regressao",
                "algorithm": "dnn",
                "target": 'target',
                "learning_network": "supervised",
                "params": {
                    'n_camadas': 5, 
                    'activation_enter': 'relu',
                    'n_neuronio_enter': 256,
                    'activation_intermediate': 'relu',
                    'n_neuronio_intermediate': 128,
                    'activation_end': 'sigmoid',
                    'n_neuronio_end': 3,
                    'input_shape': (4,), 
                    'optimizer': 'adam',
                    'loss': 'sparse_categorical_crossentropy',
                    'metrics': ['accuracy'],
                    'epoch': 10,
                    'batch_size': 32,
                    'power': True
                }
            }
      }
<!--te-->
</p>

<p align="justify">For more configuration examples, go to the 'test' directory, and see the settings.</p>

<h3>Final considerations</h3>
=================

<p>For more information, access the documentation:</p>
<p><a href="https://scikit-learn.org/stable/">Scikit-Learn</a></p>
<p><a href="https://www.tensorflow.org/?hl=pt-br">TensowFlow</a></p>