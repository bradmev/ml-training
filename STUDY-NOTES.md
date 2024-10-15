# Certified Machine Learning Specialty 

> **Warning**
> These study notes have no coherent flow! I took notes across various sources and tried to group them logically. Then just used them for revision.
> &nbsp;



# Study Notes

[Data Collection](#data-collection)

[Data Streaming](#data-streaming)

[Data Preparation](#data-preparation)

[Data  Analysis and Visualization](#data-analysis-and-visualization)

[Modeling](#modeling)

[Algorithms](#algorithms)

[Overfitting vs Underfitting](#overfitting-vs-underfitting)

[Regression](#regression)

[Clustering](#clustering)

[Classification](#classification)

[Anomaly Detection](#anomaly-detection)

[Text Analysis](#text-analysis)

[Reinforcement Learning](#reinforcement-learning)

[Forecasting (supervised)](#forecasting-\(supervised\))

[Ensemble Learning](#ensemble-learning)

[Monitoring and Analyzing](#monitoring-and-analyzing)

[AI Developer Services](#ai-developer-services)

[Implementation and Deployments](#implementation-and-deployments)

[Security](#security)

[VPC Endpoints (private links service)](#vpc-endpoints-\(private-links-service\))

[Monitoring](#monitoring)

[Cloudwatch](#cloudwatch)

[Cloudtrail](#cloudtrail)

[Sagemaker: Security](#sagemaker:-security)

[Hyperband](#hyperband)

[Random search](#random-search)

[Grid search](#grid-search)

[Bayesian optimization](#bayesian-optimization)

[Remove stop words](#remove-stop-words)

[SMOTE](#smote)

[Amazon SageMaker Random Cut Forest](#amazon-sagemaker-random-cut-forest)

[Multiple data imputations](#multiple-data-imputations)

[Macro F1 Score](#macro-f1-score)

[Amazon Polly](#amazon-polly)

[SSML](#ssml)

[Lexicons](#lexicons)

[Amazon Glue](#amazon-glue)

[Algorithms](#algorithms-1)

[Blazing Text \- Supervised / Unsupervised](#blazing-text---supervised-/-unsupervised)

[Latent Dirichlet Allocation (LDA) Algorithm \- unsupervised](#latent-dirichlet-allocation-\(lda\)-algorithm---unsupervised)

[Neural Topic modeling (NTM)](#neural-topic-modeling-\(ntm\))

[Object2Vec](#object2vec)

[Sequence to Sequence  \- supervised](#sequence-to-sequence---supervised)

[Text Classification \- TensorFlow \- supervised](#text-classification---tensorflow---supervised)

[DeepAR Forecasting \- supervised](#deepar-forecasting---supervised)

[IP Insights \- unsupervised](#ip-insights---unsupervised)

[K-Means \- unsupervised](#k-means---unsupervised)

[Principal Component Analysis (PCA) \- unsupervised](#principal-component-analysis-\(pca\)---unsupervised)

[Random Cut Forest \- unsupervised](#random-cut-forest---unsupervised)

[Vision](#vision)

[Image Classification \- MXNet \- supervised](#image-classification---mxnet---supervised)

[SemanticSegmentation](#semanticsegmentation)

[AutoGluon-Tabular](#autogluon-tabular)

[CatBoost](#catboost)

[FactorizationMachines (supervised learning)](#factorizationmachines-\(supervised-learning\))

[K-NearestNeighbours (index-based algorithm) supervised](#k-nearestneighbours-\(index-based-algorithm\)-supervised)

[LinearLearner (supervised)](#linearlearner-\(supervised\))

[TabTransformer (supervised)](#tabtransformer-\(supervised\))

[XGBoost (supervised)](#xgboost-\(supervised\))

[Additional Notes](#additional-notes)

[Learning Rate \- Step Size](#learning-rate---step-size)

[Root Mean Squared Error (RMSE)](#root-mean-squared-error-\(rmse\))

[Area under the curve (AUC)](#area-under-the-curve-\(auc\))

[F1 Score](#f1-score)

[AWS Sagemaker Canvas](#aws-sagemaker-canvas)

[Changing the Sagemaker Endpoint](#changing-the-sagemaker-endpoint)

[Neural Network Optimization \- AdaGrad, RMSProp, and ADAM.](#neural-network-optimization---adagrad,-rmsprop,-and-adam.)

[Accuracy, Precision, Recall and F1](#accuracy,-precision,-recall-and-f1)

[Calculate Confusion Matrix](#calculate-confusion-matrix)

[t-SNE](#t-sne)

[RFE \- Recursive Feature Elimination](#rfe---recursive-feature-elimination)

[Regularization \- Address Overfitting](#regularization---address-overfitting)

[L1 Regularization](#l1-regularization)

[L2 Regularization](#l2-regularization)

[Sagemaker Notebook instances](#sagemaker-notebook-instances)

[SVM](#svm)

[Collaborative Filtering](#collaborative-filtering)

[Content Based Filtering](#content-based-filtering)

[Kinesis Data Streams](#kinesis-data-streams)

[Kinesis Data Firehose](#kinesis-data-firehose)

[Kinesis Data Analytics](#kinesis-data-analytics)

[SageMakerJobs](#sagemakerjobs)

[Amazon Athena Performance](#amazon-athena-performance)

[Reducing Overfitting](#reducing-overfitting)

[Random Notes](#random-notes)

[AWS Kendra](#aws-kendra)

[Transfer Learning](#transfer-learning)

[AWS DeepLens](#aws-deeplens)

[AWS CodeGuru](#aws-codeguru)

[AWS Rekognition](#aws-rekognition)

[AWS SageMaker and Parallel Training for deep learning workloads](#aws-sagemaker-and-parallel-training-for-deep-learning-workloads)

[AWS SageMaker Local Mode](#aws-sagemaker-local-mode)

[Normalization vs Standardization](#normalization-vs-standardization)

[Standardization Z-Score Calculation](#standardization-z-score-calculation)

[S3 Encrypted Access from SageMaker Notebook](#s3-encrypted-access-from-sagemaker-notebook)

[tf-idf Calculation](#tf-idf-calculation)

[References](#references)

## 

## 

# 

# Data Collection {#data-collection}

- Good data  
  - Large data sets  
  - Precise attribute types/feature rich  
  - Complete fields  
  - No missing values  
  - Values are consistent  
  - Solid distribution of outcomes  
  - Fair sampling  
- You should have at least 10x as many data pots as total number of features  
- Datasets  
  - Unstructured  
  - Semi-structured  
  - Structured  
- Databases  
  - Traditional Relation  
  - Transactional  
  - Strict schema  
- Data Warehouses  
  - Processing done on import  
  - Data is classified/stored with se in mind  
  - Ready to use with BI tools  
- Data Lakes  
  - Processing done on export  
  - Many different sources and formats  
  - Raw data may no be ready for use  
- Labeled vs Unlabeled Data  
- Categorical vs Continuous  
  - Categorical  
    - Values that are associated with a group  
    - Qualitative  
    - Discrete  
    - **Qualitative**  
  - Continuous  
    - Values that are expressed as number  
    - Quantitative  
    - Infinite  
    - **Quantitative**  
- Text Data  
  - Corpus  
- Ground Truth Data  
  - Has been successfully labeled  
- AZ SageMaker GroundTruth  
  - Easily create ground truth data  
  - Active learning and human labeling   
- Data stores  
  - S3  
  - RDS  
  - Dynamo DB  
  - Redshift  
  - Timestream  
  - Document DB  
- Redshift Spectrum  
  - Requires Redshift Cluster  
  - Leverages the cluster for queries  
  - Does not have to load s3 data into tables.  
- Helper Tools  
  - EMR  
  - Spark   
    - etl and machine learning library  
    - Large scale compute processing distributed  
  - HAdoop  
    - Large scale distributed storage  
  - Hive ETL   
    - Service SQL like HQL for querying Hadoop  
  - Tensor Flow  
    - Build ML models  
    -   
  - Glue  
  - Glue sources  
    - S3  
    - DynamoDB  
    - RDS  
    - Redshift  
    - Ec2  
  - Glue dest  
    - Athena  
    - EMR  
    - S3  
      - PutObject  
      - API  
      - Console  
    - Redshift  
  - 

# Data Streaming {#data-streaming}

- Amazon Kinesis Data Firehose is the easiest way to reliably load streaming data into data lakes, data stores, and analytics services. It can capture, transform, and deliver streaming data to Amazon S3, Amazon Redshift, Amazon Elasticsearch Service, generic HTTP endpoints, and service providers like Datadog, New Relic, MongoDB, and Splunk.  
- Correct Answer  
- Amazon Kinesis Data Streams (KDS) is a massively scalable and durable real-time data streaming service. KDS can continuously capture gigabytes of data per second from hundreds of thousands of sources such as website clickstreams, database event streams, financial transactions, social media feeds, IT logs, and location-tracking events. The data collected is available in milliseconds to enable real-time analytics use cases such as real-time dashboards, real-time anomaly detection, dynamic pricing, and more.  
- A single shard can ingest   
  - up to 1 MB of data per second  
  - Up to 1000 transactions per second  
- Kinesis Data Streams  
- Stored for 24hrs in shares (up to 7 days)  
- Capacity is dependant on number of shards and ECC consumers  
- Producers/Consumers then Storage  
- Made up of Shards with Fixed units of capacity shards=capacity  
- Order is maintained

- Kinesis Video Streams  
  - Used for video streams

Scaling kinesis Consumers using Kinesis Client Libraries

- It’s fine if number of shards exceed number of instances  
- CPU utilization should be measure for scaling  
- KCL would add another record processor when it detects another shard

Kinesis Data Streams

- Sources  
  - Kinesis Agent  
  - KPL  
  - SDK  
- Destinations  
  - Amazon Data Firehose  
  - Managed Flink  
  - KCL

Kinesis Firehose

- Automated do not have to worry about shards  
- Data is sent to s3 or possibly lambda  
- No retention data is sent straight away  
- No consumers  
- Sources  
  - Data Streams  
  - MSK  
  - Direct Put  
- Destination  
  - S3, OpenSearch, Redshift  
  - Multiple Third Parties

Kinesis Analytics

- Use SQL type query against the data.  
- Send to redshift  
- Query Data as it comes in (realtime)  
- Source  
  - Data Streams  
  - Firehose  
- Destination  
  - Lambda  
  - Firehose DeleiveryStream  
  - Data Stream

Amazon Kinesis Data Streams can ingest and store data streams for Lambda processing, which can transform and load the data into Amazon S3.

Amazon Kinesis Data Firehose can ingest streaming data from Amazon Kinesis Data Streams, which can leverage Lambda to transform the data and load into Amazon S3.

Amazon Kinesis Data Analytics can query, analyze and transform streaming data from Amazon Kinesis Data Streams and use Amazon Kinesis Data Firehose as a destination for loading data into Amazon S3.

### Full Bayesian vs Naive Bayesian

the main difference lies in the modeling assumptions and complexity: Naive Bayes is simple and assumes feature independence, making it suitable for basic classification tasks with modest-sized datasets. Full Bayesian Networks are more complex, allowing for flexible modeling of dependencies among variables.

# 

# Data Preparation {#data-preparation}

- Categorical Encoding  
  - Manipulating categorical variables when ML a logs expect numerical inputs  
- When to encode  
  - Predicting price of a home  
    - Linear Regression (encoding required)  
  - Determine the subject of a  given text corpus  
    - Naive Bayes (encoding not necessary)  
  - Detecting malignancy in radiology images  
    - Convoluted Neural Network (encoding necessary)  
- Nominal vs Ordinal  
  - Encoding nominal values to integers is a bad idea  
  - Ordinal  
    - Order does matter  
  - Nominal  
    - Order does not matter  
- Discrete vs Continuous  
- One-hot-encoding  
  - Transform nominal categorical features and creates new binary columns for each observation  
  - Not always a good choice when there are many categories  
  - Grouping by similarity could create fewer overall categories  
  - Mapping rare values  
- Text feature engineering  
  - Transforming text within our data so ML can better analyze  
  - Bag of words  
    - Tokenizes raw text and creates a statistical representation of the text  
  - N-Gram  
    - Produce groups of words  
    - Brakes up test by whitespace  
    - Size \= x sliding window  
  - Orthogonal Sparse Bigram (OSB)  
    - Creates groups of words of size n and outputs paid of words that includes the first word  
    - Define window size and delimiter  
    - he\_is he\_\_a he\_\_\_jedi  
    - Size \= 3 ^  
    - First and second items are one  
  - Term-frequency inverse document frequency (tf-idf)  
    - Represents how important a word or words are to a given set of text by providing appropriate weights to terms that are common and less common in the text  
    - Makes common words less important.  
    - Document Count , Unique n-grams vector  
  - Example usage  
    - Matching phrases \- n-gram  
    - Determining subject \- tf-idf/orthogonal sparse bigram  
  - Removing punctuation or   
  - Cartesian Product \- combining features to create a new feature  
- Feature Engineering: Dates  
  - Extract more information is\_weekend, day\_of\_week etc  
- Numeric Feature Engineering  
  - Scaling (normalization)   
    - change values so they fit on the same scale  
    - Scale between 0 and 1  
    - Outliers can through off scaling remove those first  
    - Required by many algorithms  
  - Binning  
    - grouping similar values  
    - Can result in irregular bins  
    - Use quantile binning for equal bins  
    - Combine quantile binning and one-hot encoding   
  - Standardization  
    - Mean is set to zero  
    - Z-score score away from the mean   
    - Rescales values by making the values of each feature in the data have zero mean (much less affected by outliers)  
  - When you’re done you can always translate back to get data to the original representation.  
- Image Feature Engineering  
- Audio Feature Engineering  
- Dataset Formats  
  - CSV, JSON, PArquet, PNG/JPG  
  - PIPE record-io-protobuf  
- Handling missing data  
  - Missing at Random (MAR)  
    - Related to some of the other observed data.  
  - Missing Completely at Random(MCAR)  
    - Nothing to do with its value or other values  
  - Missing not at random (MNAR)  
    - Dependant on some other factor or value  
  - How to replace (aka imputation)  
    - Supervised Learning  
      - Most difficult but can yield best results.  
    - Mean/Median/Mode  
      - Quick and easy can skew results  
    - Dropping Rows (not imputation)  
      - easiest

    Removes missing values

- Feature Selection   
  - An intuitive step made by the human  
  - PCA (Principal Component Analysis)   
    - An unsupervised learning algorithm that reduces the number of features while still retaining as much info as possible  
- Data prep tools  
  - AWS Glue  
    - Setup a crawler  which build out a schema   
    - Create jobs  to query that data and dump to EMR S3 or Redshift  
    - Glue Spark Jobs  
      - Python or Scala  
      - Glue Jobs Transform Reference  
        - Avro, csv, parquery, orc, xml, grokLog, ion  
    - Glue Notebooks  
      - Zeppelin or Jupyter  
  - SageMaker Jupyter Notebooks  
    - Conda and Pip access to install libraries  
  - EMR  
    - Includes hive, pig, mxnet tensorflow, hadoop, presto, hadoop, athena  
  - Data pipeline

# Data  Analysis and Visualization {#data-analysis-and-visualization}

- What are we looking to show  
  - Relationships  
    - Scatter plots \- 2 values  
    - Bubble plots \- three values (bubble size is third)  
    -   
  - Distributions  
    - Histograms  
    - Box Plots  
      - Shows max, min median, and 25th and 75th percentile  
    - Scatter Plots  
  - Comparisons  
    - Bar chart (qty)  
    - Line chart (time series)  
  - Compositions  
    - Pie  
    - Stacked Area  
    - Stacked Bar  
  - Heatmaps, values as color  
- Quicksight BI Data Vis tool  
- 

# 

# Modeling {#modeling}

- Take a problem, lots of data, an ML algo, try to figure out a mathematical formula that can generalize  
- Questions to ask  
  - What type of generalization are we seeking?  
  - Do we really need machine learning?  
  - How will it be consumed, real time or batch\>  
  - What data do we have to work with?  
  - How can I tell if it is working?  
- Supervised Learning  
  - Classification (Discrete)  
  - Regression (Continuous)  
- Unsupervised Learning  
  - Clustering (Discrete)  
  - Reduction of dimensionality (Continuous)  
- Reinforcement Learning  
  - Simulation Based (Discrete)  
  - Continuous (Autonomous Devices)  
- Reinforcement Learning  
- Examples  
  - Fraud detection \= Binary classification (only two possible outcomes)  
  - Predict rate of deceleration of car \- Heuristic (Well known formulas for speed)  
  - Determine the most efficient path to travel for robot \= Reinforcement Learning  
  - Breed of Dog \= Multi-Class Classification   
- Cascading algorithms  
  - Remove outliers  (Random-Cut forest)  
  - Identify relevant attributes (PCA)  
  - Cluster into groups (K-Means)  
  - Predict basket size (Linear Learner)  
  - 


|  |  | Outcome |  |
| :---: | :---: | :---: | :---: |
| Prediction |  | TRUE | FALSE |
|  | TRUE | Correct | (False Positive) |
|  | FALSE | Wrong(False Negative) | Correct |

- Data prep for model   
  - Randomize  
  - Split  
  - Train  
  - Test  
- K-Fold  
  - Helps reduce overfitting  
  - Data set is split into k ‘folds’ each folds acts as the validation set only once  
- Mechanical Turk  
  - Real human validation  
  - Crowdsourcing tasks  
- Sage Maker  
  - Groundtruth  
    - Setup and manage labeling jobs for training datasets using active learning and human labeling  
  - Notebook  
    - Access a managed Jupyter Notebook environment  
  - Training  
    - Train and tune models  
  - Inference  
    - Packageand deploy your machine learning models  
  - SageMaker Console  
    - Use Jupyter for building models  
  - SageMaker SDK   
    - Use Spark for building models  
  - CSV is accepted by most algorithms  
    - Be sure COntent/Type is set as text/csv in S3  
    - Target value should be in first column with no header  
    - For unsupervised algos we specify the absence of labels by setting content type to text/csv;label\_size=0  
  - CreateTrainingJob API  
    - Specify training algo  
    - Supply algo specific hyper params  
    - Specify input and output configuration  
  - Hyperparameter  
    - Values set before the learning process  
  - Parameter  
    - Values derived via the learning process  
- Training  
  - ASIC \- burned in least flexible most performant  
  - FPGA  
  - GPU  
  - CPU  
- ECR contains images for SageMaker  
  - Inference Images  
    - Uses CreateModel API Call  
  - Training Images  
    - Uses CreateTrainingJob API Call  
  - Tags  
    - Use :1 for stable and prod  
    - Use :latest for backward compatible  
  - Training Image and Reference Images have a path, training input mode, file type and instance class to choose from  
- ECS will run the job for you and pull put data from input and output paths  
- Information is logged to Cloudwatch

# Algorithms {#algorithms}

- Algo  
  - A set for steps to follow to solve a problem with repeatable outcome  
- Heuristic  
  - Rule of thumb shortcut  
- How to train  
  - Supervised  
    - Training and test Data  
  - Unsupervised  
    - No Training  
  - Reinforcement  
    - Maximize reward

### Overfitting vs Underfitting {#overfitting-vs-underfitting}

In machine learning, \*\*overfitting\*\* occurs when a model learns the training data too well, including noise and details that don’t generalize to new, unseen data. This results in high accuracy on training data but poor performance on test data. It's like memorizing answers to specific questions without understanding the underlying concepts.

#### Class Probability

Decreasing the class probability threshold makes the model more sensitive and, therefore, marks more cases as the positive class, which is fraud in this case. This will increase the likelihood of fraud detection. However, it comes at the price of lowering precision.

\*\*Underfitting\*\*, on the other hand, happens when a model is too simple to capture the patterns in the data. It performs poorly on both training and test data because it fails to learn enough from the data, similar to giving general, vague answers that miss key details.

In short, overfitting is when a model is too tailored to the training set, and underfitting is when it's too basic to capture important patterns. The goal is to find a balance that allows the model to generalize well to new data.

Underfitting the model most likely has not enough data to identify features. Especially the case if accuracy on training and validation data sets.

## Regression {#regression}

- Linear models  
  - You give the models labels (x,y) with x being a high-dimensional vector  
  - Y is a numeric label  
  - You need a number or list of numbers that yields your answer  
  - Stochastic Gradient Descent  
    - Adjusts so all the distances from the line are as small as possible  
    - Local Min vs Global min  
    - Stochastic Gradient Descent (SGD) is a cost function that seeks to find the minimal error.  
  - Linear Learner Algo (supervised)  
    - Very flexible  
    - Can explorer difference training objectives  
    - Built in tuning \- internal tuning of hyper-p’s  
    - Good first choice  
  - Factorization Machines (supervised)  
    - Good for sparse data  
    - Good for binary classification and regression.  
    - Sagemakers instance will only analyze rel;ationships of two paris of features at a time  
    - CSV is not supported  
    - File and Pipe mode supported with Float32 tensors  
    - Doesn’t work for multiclass  
    - Needs lots of data  
    - Other algos are better aqt full sets of data.

## Clustering {#clustering}

- Unsupervised  
  - K-Means  (unsupervised)  
    - Attempts to find discrete groups within data  
    - Will take a list of things with attributes  
    - Similarity is calculated based on the distance between the identifying attributes  
    - Expects tabular data   
    - Rows are the items you wish to cluster  
    - Define identifying attributes  
    - CPU instances recommended  
    - Training is still a thing  
  - MNIST

## Classification {#classification}

- K-Nearest Neighbor (supervised)  
  - For classification or regression  
    - You choose the number of neighbors  
    - Does not generalize   
    - KNN doesn't learn but rather uses the data points held in memory to decide on similar samples  
    - Examples   
      - Recommendation engine  
      - Grouping people based on credit risk  
  - Image Classification  
  - Object Detection  
    - Classifieds objects within   
  - Semantic Segmentation  
    - Identify shapes within an image  
    - Low level analysis of individual pixels  
    - Accepts PNG file input  
    - Only supports GPU instances for training  
    - But can deploy on CPU or GPU instances

## Anomaly Detection {#anomaly-detection}

- Random Cut Forest (unsupervised)  
  - Detect anomalies data points within a set   
    - Work with n-dimensional input  
    - Find outliers (usually more than 3 std dev)  
    - Low scores indicates normal high score indicates outlier  
    - Scales well and does not benefit from gpu  
  - IP Insights (unsupervised)  
    - Identify behavior from insights  
    - GPUs recommended for training  
    - CPUs recommended for inference

## Text Analysis {#text-analysis}

- LDA \- Latent Dirichlet Allocation  
  - Used to figure out how similar documents are, based on the freq of similar words.  
    - Article recommendation  
  - Neural Topic Model (unsupervised)  
    - Similar to LDA may yield different results.  
    - Semantic similarity  
  - Sequence to Sequence (supervised)  
    - Language translation engine  
    - We must supply training data and vocab.  
    - Embedding,m Encoding, Decoding using Neural Net  
    - Only GPU instances are supported.  
    - Speech to text  
    - Attention mechanism is designed to handle long sequences  
    -   
  - Blazing Text (supervised \- text classification / unsupervised \- word2vec)  
    - Optimized versions of word2vec and text classification  
    - Sentiment analysis, entity recognition   
    - Optimized way to determine contextual semantic relationships  
    - Modes  
      - Word2Vec (unsupervised)  
        - Single cpu  
        -   
      - Text Classification (supervised)  
      - Continuous bag of words  
    - Expects single pre-processed text file  
      - Each line should contain single sentence  
    - 20x faster than FastText  
  - Object2Vec (supervised)  
    - A way to map out thinks in a d-dimensional space to figure out how similar they might be to one another  
    - Expects pairs of things  
    - Can be used for downstream supervised tasks like classification or regression  
    - Requires labeled data fro training (can be provided with natural clustering)  
    - Examples  
      - Movie rating prediction  
      - Document classification

## Reinforcement Learning {#reinforcement-learning}

- Learn a strategy (policy)  
  - That optimizes for an agent acting in an environment  
  - Markov Decision Process (MDP)  
    - Agent  
    - Environment  
    - Reward  
    - State   
    - Action  
    - Observation  
    - Episodes  
    - Policy  
  - Examples   
    - Autonomous vehicles  
    - Intelligent hvac control  
    - 

## Forecasting (supervised) {#forecasting-(supervised)}

- DeepAR  
  - For scalar time series using recurrent neural networks (RNN)  
    - Can predict both point in time values and estimated values over a timeframe  
    - More time series is better  
    - Must supply at least 300 observations  
    - Some hyperparams are required  
      - Context Length  
      - Epochs  
      - Prediction Length  
      - Time Frequency  
    - Automatic eval of the model   
    - Examples  
      - Forecasting product performance  
      - Predict Labor Needs

## Ensemble Learning {#ensemble-learning}

- XGBoost  
  - OpenSource implementation of the gradient boosted trees algorithm  
    - Multi use for regression, classification and ranking   
    - 2 \- 35 hyperparams  
    - Accepts csv and libsvm for training and inference  
    - Use an instance with enough memory to hold entire training set  
    - You can call XGBoost direct from within the Spark environment.  
    - Examples  
      - Ranking   
      - Fraud Detection

# Monitoring and Analyzing {#monitoring-and-analyzing}

- Choose appropriate metrics for monitoring  
- Underfitting  
  - Not predicting well   
  - To prevent add more data or train longer  
- Overfitting  
  - Our model is too dependent on the data we used to train  
  - We have trained the model to memorize rather than generalize  
  - To prevent overfitting  
    - More data  
    - Early stopping   
    - Data could be too clean \- add some noise  
    - Ditch some features \- too many irrelevant features  
    - Regulate the data \- smooth irregularities.  
- Just Right  
  - Model does well on training and new data  
  - It can deal with noise in the data  
- Residual  
  - Difference between predicted vs actual  
  - RMSE is a measure of residual  
  - Lower value is better  
- False Positive   
  - TYPE I error  
- False Negative  
  - TYPE II error  
- F1 Score  
  - Recall vs Precision  
  - A larger value indicates better predictive accuracy  
  - Multiclass f1 score per class with a macro average f1 score  
- Model Tuning  
  - Automatic hyper param tuning  
  - Bayesian Optimization  
- A/B Testing  
  - 99% / 1%   
  - Variant 1 Current Model  
  - Variant 2 New Model

# AI Developer Services {#ai-developer-services}

- Amazon Comprehend  
  - Natural Language Processing  
  - Amazon Comprehend is a natural language processing (NLP) service that uses machine learning to find meaning and insights in text.  
  - Used to to identify the language of the text, extract key phrases, places, people, brands, or events, understand sentiment about products or services, and identify the main topics from a library of documents.  
  - Can provide topic modeling.   
  - Part of speech tagging and key phrase extraction.  
      
- SageMaker Canvas  
  -  a service that you can use to create ML models without having to write code. You can use SageMaker Canvas to build a custom model trained with your data. SageMaker Canvas can perform no-code data preparation, feature engineering, algorithm selection, training and tuning, inference, continuous model monitoring, and other tasks.  
- Amazon Forecast  
  - Amazon Forecast is a fully managed machine learning service that automates time-series forecasting. It supports a wide range of input data types (e.g., item metadata, related time series) to improve prediction accuracy. Users can customize models for specific domains such as retail demand, financial metrics, or resource allocation. Forecast handles preprocessing, model training, hyperparameter tuning, and deployment, delivering forecasts via a REST API. It integrates with AWS services like S3 for data storage and Lambda for automation, providing scalable and highly accurate forecasts without requiring ML expertise.  
- Amazon LEX  
  - Conversational interfaces  
  - Understand intent  
- Amazon Personalize  
  - Recommendation Engine  
- Amazon Poly  
  - Text to speech service  
- Amazon Rekognition  
  - Rekognition provides features such as object detection, facial recognition, scene detection, text recognition, and even the ability to detect inappropriate or unsafe content for images and video.  
- Amazon Texttract  
  - Extract text and context from scanned documents  
- Amazon Transcribe  
  - Speech to text as a service  
- Amazon Translate  
  - Translate between languages  
- Amazon Macie  
  - A data security service that discovers sensitive data using machine learning and pattern matching, provides visibility into data security risks, and enables automated protection against those risks.  
- Amazon Fraud Detector   
  - Amazon Fraud Detector is designed for online fraud use cases requiring real-time ML modeling and rules-based evaluation. For example:  
    - New account fraud, within an account sign-up process  
    - Online identity fraud  
    - Payment fraud for online orders  
    - Guest checkout fraud  
    - Loyalty account protection  
    - Account takeover detection  
    - Seller fraud in online marketplaces

# Implementation and Deployments {#implementation-and-deployments}

- Offline usage or online usage  
- Offline  
  - SageMaker Batch Transform  
- Online  
  - Sagemaker Hosting Services  
    - Create a Model  
      - CreateModel API call is used to launch an inference container.  
    - Create a Endpoint Configuration  
      - Model  
      - Instance  
      - Variant name and weight (production variant)  
    - Create an Endpoint  
      - Publish the model to SagerMaker  InvokeEndpoint()  
- Inference Pipelines  
  - Can be used for both batch or real time  
  - 2 \- 5 containers  
  - Built-in or custom algorithms  
- SageMaker Neo  
  - Optimize ML models for a variety of architectures  
    - ARM  
    - Intel  
    - nVidia  
- Elastic Inference   
  - Speeds up throughput and decreases latency of real-time inferences  
  - Uses only CPU based instances   
  - More cost-effective than a GPU based instance  
  - Elastic Inference accelerates deep learning models during the inference phase, which typically requires less GPU power than training.  
  - By attaching GPU acceleration selectively, you save costs compared to running inference on a full GPU instance.  
  - Elastic Inference works with models built in popular frameworks like TensorFlow, Apache MXNet, PyTorch nd ONNX.  
  - Instead of running an entire SageMaker endpoint on a GPU instance for inference, you can use a cheaper CPU instance with Elastic Inference attached, reducing costs while maintaining high performance.  
  -   
- Auto Scaling  
  - Scale on InvocationsPerInstance  
- InitialInstanceCount  
  - Set for variant within endpoint configuration  
  - 

# Security {#security}

### VPC Endpoints (private links service) {#vpc-endpoints-(private-links-service)}

- Interface  
  - An ENI interface with a private address in your vpc  
  - Multiple AWS services are supported including SageMaker, S3 ad Dynamo  
- Gateway  
  - A target for your route in a route table  
  - Only for s3 and DynamoDB

### VPC Endpoint Policy

An endpoint policy is a resource-based policy that you attach to a VPC endpoint to control which AWS principals can use the endpoint to access an AWS service.

# Monitoring {#monitoring}

## Cloudwatch {#cloudwatch}

- Endpoint invocation, instance, Training, Transform and Ground Truth available in addition to algo specific metrics  
- Metrics available in 1min resolution  
- Stored for 15 months (available in api)  
- Only 2 weeks available in the console.  
- Any stdout or stderr messages from notebook instances algo containers or model containers is sent to amazon cloudwatch logs

## Cloudtrail {#cloudtrail}

- Log API Access \- Captures any API calls made.  
- Last 90 days of events are visible in CloudTrail event history  
- Setup a CloudTrail to store logs on S3 to reference for as long as you’d like

### Sagemaker: Security {#sagemaker:-security}

In SageMaker, the iam:PassRole action is needed for the Amazon SageMaker action sagemaker:CreateModel. This allows the user to pass authorization to SageMaker to actually create models

### Hyperband {#hyperband}

Hyperband is a dynamic tuning strategy in SageMaker that reallocates resources based on the performance of training jobs. It combines elements of other methods, using both intermediate and final results to prioritize promising hyperparameter configurations. This approach aims to achieve a balance between computational efficiency and the accuracy of the tuned model.

#### Random search {#random-search}

Random search in SageMaker samples hyperparameter values randomly from specified ranges. This method is less computationally demanding than grid search, offering a quicker but potentially less optimal exploration of the hyperparameter space. It's a balance between computational efficiency and the chance of finding the best values.

#### Grid search {#grid-search}

Grid search in Amazon SageMaker methodically explores combinations of hyperparameter values from specified categorical ranges. It's thorough and evaluates every possible combination within the given categorical parameters. This exhaustive approach can be computationally intensive, especially when dealing with large hyperparameter spaces.

#### Bayesian optimization {#bayesian-optimization}

Bayesian optimization in SageMaker treats hyperparameter tuning as a regression problem, predicting which combinations might yield the best results. Using a statistical model, it intelligently selects hyperparameter values based on previous evaluations, aiming for efficiency and accuracy. It's a sophisticated method that balances exploration and exploitation of the hyperparameter space.

### Remove stop words {#remove-stop-words}

Before we use corpus data in some Machine Learning processes like language translation, sentiment analysis, or spam filtering it is important we properly apply text processing to the data. Some of the important text processing that needs to be done is tokenization. This includes  removing stop words — frequent words such as ”the”, ”is”, etc. that do not have specific meaning.

### SMOTE {#smote}

Reaching out to the company and gaining more data is a great option in terms of minimizing information loss. Also using techniques like SMOTE (Synthetic Minority Over-Sample Technique) can create more samples of the abnormal observations to even out the number of samples. If we did use the undersample approach (removing normal behaviors until normal and abnormal match) then we are getting rid of a very large portion of our data 

### Amazon SageMaker Random Cut Forest {#amazon-sagemaker-random-cut-forest}

Amazon SageMaker Random Cut Forest (RCF) is an unsupervised algorithm for detecting anomalous data points within a data set. When using RCF the optional test channel is used to compute accuracy, precision, recall, and F1-score metrics on labeled data. Train and test data content types can be either application/x-recordio-protobuf or text/csv and AWS recommends using ml.m4, ml.c4, and ml.c5 instance families for training. 

### Multiple data imputations {#multiple-data-imputations}

Multiple imputation for missing data makes it possible for the researcher to obtain approximately unbiased estimates of all the parameters from the random error. The researcher cannot achieve this result from deterministic imputation, which the multiple imputation for missing data can do. [Multiple Imputation for Missing Data: Concepts and New Development](https://stats.idre.ucla.edu/wp-content/uploads/2016/02/multipleimputation.pdf)

Both in training and inference, need to be vectors of integers representing word counts. This is so-called bag-of-words (BOW) representation. To convert plain text to BOW, we need to first “tokenize” our documents, that is, identify words and assign an integer ID to each of them. Then, we count the occurrence of each of the tokens in each document and form BOW vectors. [Introduction to the Amazon SageMaker Neural Topic Model | AWS Machine Learning Blog](https://aws.amazon.com/blogs/machine-learning/introduction-to-the-amazon-sagemaker-neural-topic-model/)

Resample the dataset using oversampling/undersampling and use the F beta (F1) score as the objective metric. Finally, apply the XGBoost algorithm.

When the data is highly imbalanced, using accuracy as the objective metric is not a good idea. Instead using precision, recall, or F1 score is a better option. Since this is a binary classification problem, the XGBoost is a great algorithm for this. Random Cut Forest (RCF) is used for anomaly detection. [XGBoost Algorithm \- Amazon SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html) [Create a model for predicting orthopedic pathology using Amazon SageMaker | AWS Machine Learning Blog](https://aws.amazon.com/blogs/machine-learning/create-a-model-for-predicting-orthopedic-pathology-using-amazon-sagemaker/)

### Macro F1 Score {#macro-f1-score}

The Macro F1 Score is an unweighted average of the F1 score across all classes and is typically used to evaluate the accuracy of multi-class models. A number closer to 1 indicates higher accuracy. [Multiclass Model Insights \- Amazon Machine Learning](https://docs.aws.amazon.com/machine-learning/latest/dg/multiclass-model-insights.html)

### Amazon Polly {#amazon-polly}

#### SSML {#ssml}

Using SSML-enhanced input text gives you additional control over how Amazon Polly generates speech from the text you provide. Using these tags allows you to substitute a different word (or pronunciation) for selected text such as an acronym or abbreviation. You can also create a dictionary lexicon to apply to any future tasks instead of apply SSML to each individual document.

#### Lexicons {#lexicons}

Pronunciation lexicons enable you to customize the pronunciation of words. Amazon Polly provides API operations that you can use to store lexicons in an AWS region. Those lexicons are then specific to that particular region. You can use one or more of the lexicons from that region when synthesizing the text by using the `SynthesizeSpeech` operation. This applies the specified lexicon to the input text before the synthesis begins.

### Amazon Glue {#amazon-glue}

When you create jobs you can encrypt metadata and data written to s3 by jobs crawlers and dev endpoints.

AWS Glue can connect to a wide range of data sources for both inputs (to extract data) and outputs (to store transformed data). Here are the key data sources:

\#\#\# \*\*Input Data Sources\*\* (for data extraction):  
1\. \*\*Amazon S3\*\*: Glue can read structured, semi-structured, and unstructured data (e.g., CSV, JSON, Parquet, ORC, Avro, etc.) stored in Amazon S3.  
2\. \*\*Amazon RDS\*\*: Supports relational databases such as MySQL, PostgreSQL, MariaDB, Oracle, and SQL Server on RDS.  
3\. \*\*Amazon Redshift\*\*: Allows reading data from Redshift clusters.  
4\. \*\*Amazon DynamoDB\*\*: Supports DynamoDB tables as a source.  
5\. \*\*JDBC Connections\*\*: Enables connections to various on-premise or cloud databases via JDBC (e.g., MySQL, PostgreSQL, Oracle, SQL Server).  
6\. \*\*AWS Data Exchange\*\*: Can access third-party datasets from AWS Data Exchange.  
7\. \*\*Amazon Kinesis Data Streams\*\*: Supports reading real-time data streams.  
8\. \*\*AWS Lake Formation\*\*: Can read from a data lake created with AWS Lake Formation.  
9\. \*\*Glue Data Catalog\*\*: Serves as a central repository of metadata for Glue jobs, allowing them to reference datasets in various sources.

\#\#\# \*\*Output Data Sources\*\* (for data storage):  
1\. \*\*Amazon S3\*\*: Glue can write transformed data to S3 in various formats (CSV, JSON, Parquet, ORC, Avro).  
2\. \*\*Amazon Redshift\*\*: Allows writing transformed data back to Redshift clusters.  
3\. \*\*Amazon RDS\*\*: Transformed data can be written to RDS databases (MySQL, PostgreSQL, etc.).  
4\. \*\*Amazon DynamoDB\*\*: Supports writing to DynamoDB tables.  
5\. \*\*JDBC Connections\*\*: Enables writing to on-premise or cloud databases via JDBC.  
6\. \*\*Amazon Kinesis Data Streams\*\*: Transformed data can be output to Kinesis for real-time analytics.  
7\. \*\*Amazon OpenSearch Service (formerly Elasticsearch)\*\*: Glue can write output to OpenSearch for search and analysis.

AWS Glue provides flexibility to move and transform data between these sources, supporting ETL (Extract, Transform, Load) processes across diverse data environments.

AWS Glue FindsMatches transform (ml tool) even when they done match exactly.  
Can detect fuzzy matches.

Accuracy  
The number of correct predictions to the number of total predictions  
Less effective with lots of true negatives

Precision  
The proportion of positive predictions that are actually correct.  
Good metric when cost of false positives is high.

Recall  
The proportion of correct sets that are identified as positive  
Good metric when cost of false negatives are high.

**Area Under the Curve**  
Commonly used to eval classification models.

**F1 Score**  
Particularly in classification tasks

F1=2 × (Precision+Recal / Precision×Recall​)

* The F1 score is useful when you need to balance precision and recall and avoid situations where one metric is significantly higher than the other.  
* It's particularly helpful in cases with **imbalanced datasets**, where the positive class is much smaller than the negative class. In such cases, accuracy might be misleading, and the F1 score provides a clearer picture of model performance.  
* A high F1 score (close to 1\) means both precision and recall are high, indicating a good model.  
* A low F1 score indicates poor performance in either or both precision and recall.

## Algorithms {#algorithms-1}

### Blazing Text \- Supervised / Unsupervised {#blazing-text---supervised-/-unsupervised}

- WordtoVec and Text Classification  
- Word to vec is used for sentiment analysis, named entity recognition, machine translation  
- Accelerated training of the fastText text classifier on multi-core CPUs or GPUs  
- Inputs  
  - Expects a single preprocessed text file with space separated tokens. Each line should contain a single sentence  
  - Wor2Vec \- upload a file under the train channel  
  - Classification \- supervised train with file mode, prefix single lines with labels  
    - \_\_ **label\_\_**4  
    - UPload train and optional validation file   
    - Augmented Manifest Text fFormat \- upload jsonl lines with sour and label properties. This can be trained in pipe mode without creating RecordIO files.  
- Outputs  
  - Supervised outputs creates model.bin file that can be run by BlazingText hosting. For inference a JSON file is accepted containing a list of sentences.  
- Params  
  - Word2Vec  
    - Mode, batch\_size, buckets, epochs (complete passes through the training data)  
  - Text Class  
- Tuning  
  - Word2vec reports ona  single metric  train:mean\_rho  
  - Text class reports on a single metric validation: accuracy

### Latent Dirichlet Allocation (LDA) Algorithm \- unsupervised {#latent-dirichlet-allocation-(lda)-algorithm---unsupervised}

- Used to discover a number of topics within a text corpus  
- Supports single instance CPU training  
- Better on perplexity \- geometric mean per word  
- Per word log-likelihood \- larger value is better  
- Inputs  
  - Data to be provided in train channel and optional test channel  
  - Supports RecordIO-wrapped-protobuf, CSV   
  - Trained in file or pipe mode (file for csv)  
  - For inference text/csv, application/json and x-recordio-protobuf are supported  
- Outputs  
  -  LDA inference returns application/json or application/x-recordio-protobuf *predictions*, which include the topic\_mixture vector for each observation.  
- Params  
  - Alpha0 and num\_topics can effect the LDA objective metric test\_pwll  
- Tuning  
  - Reports a single metric test:pwll

### Neural Topic modeling (NTM) {#neural-topic-modeling-(ntm)}

- Used to organize a text corpus into topics based on statistical distribution  
- Four data channels: train, text, validation and auxiliary (train is only one required)  
- Inputs  
  - Supports RecordIO-wrapped-protobuf, CSV   
  - You can use either file or pipe mode   
- Outputs  
  - NTM inference returns application/json or application/x-recordio-protobuf *predictions*, which include the topic\_weights v  
- Params  
  - Feature\_dim  
  - Num\_topics \- number required topics  
- Tuning  
  - *Training loss* measures how well the model fits the training data. *Validation loss* measures how well the model can generalize to data that it is not trained on. Low training loss indicates that a model is a good fit to the training data. Low validation loss indicates that a model has not overfit the training data and so should be able to model documents successfully on which is has not been trained.  
  -  NTM algorithm reports a single metric that is computed during training: validation:total\_loss.

## Object2Vec {#object2vec}

- general purpose neural embedding algo  
- Preserves the relationship between objects  
- Inputs  
  - A discrete token, which is represented as a list of a single integer-id. For example, \[10\].  
  - A sequences of discrete tokens, which is represented as a list of integer-ids. For example, \[0,12,10,13\].  
  - Input label can be   
    - Categorical label  
    - Score \= similarity  
    - 

### Sequence to Sequence  \- supervised {#sequence-to-sequence---supervised}

- Input is sequence of tokens and output is sequence of tokens  
- Machine translation, text summarization, language to language  
- Uses RNN or CNN  
- Inputs  
  - Expects recordIO-protobuf as integers  
  - Generate 32bit tensors and vocab files  
  - 3 channels  
    - Train  
    - Validation  
    - Vocab  
- Inference  
  - Json or recordio-protobuf  
  - Batch transform expects jsonlines  
- Instances  
  - Only supports GPU types P2, P3 G4dn G5  
- Tuning  
  - Choose one of the following as objectives  
    - validation:accuracy  
    - validation: bleu  
    - validation:perplexity

### Text Classification \- TensorFlow \- supervised {#text-classification---tensorflow---supervised}

- Transfer learning with many pre-trained models  
- Input  
  - **Training data input format:** A directory containing a data.csv file. Each row of the first column should have integer class labels between 0 and the number of classes. Each row of the second column should have the corresponding text data  
  - Any raw text formats for inference must be content type application/x-text.  
  -   
- Tuning  
  - Use validation:accuracy

### DeepAR Forecasting \- supervised {#deepar-forecasting---supervised}

- Forecasting scalar on-dimensional time series using recurrent neural networks RNN  
- Input  
  - Json, josnz, parquet  
  - Start \- date  
  - Target \- target value  
  - Dynamic\_feat \- features related that may impact value  
  - Cat \- assist with grouping things

### IP Insights \- unsupervised {#ip-insights---unsupervised}

- Learns the usage patterns for ipv4 addresses  
- Can learn vector representations of ip addresses   
- Use them in measuring similarities between IP addresses  
- Input  
  - Training and validation data channels  
  - Data format needs to be CSV  
  - The first column of the CSV data is an opaque string that provides a unique identifier for the entity. The second column is an IPv4 address in decimal-dot notation. IP Insights currently supports only File mode  
  - For inference Csv, json josnlines  
- Output  
  - Each record in the output data contains the corresponding dot\_product (or compatibility score) for each input data point.  
  -   
- Produces an AUC metric  
- Runs on CPU or GPU instances  
- Tuning   
  - validation:discriminator\_auc

### K-Means \- unsupervised {#k-means---unsupervised}

- Attempts to find discrete groupings within data.  
- You define the attributes you want the algo to use to deinfe similarity  
- Input  
  - Expects train channel and optional test channel  
  - Recordio-protobuff and csv formats supported  
  - File or pipe mode supported  
  - For inference text.csv and application/json and x-recordio-protobuf  
- Output  
  - k-means returns a closest\_cluster label and the distance\_to\_cluster for each observation.  
- Instances  
  - CPU recommended  
- Tuning  
  - Metrics  
    - Test:msd \- mean squared distances between record  
    - Test:ssd \- sum of squared distances

### Principal Component Analysis (PCA) \- unsupervised {#principal-component-analysis-(pca)---unsupervised}

- Finds a new set of features called components.  
- Operates in regular or randomized  
- Reduces dimensionality  
- Input  
  - recordIO-wrapped-protobuf and CSV  
  - Inference csv/json/x-recordio-protobuf  
- Output  
  - Results are returned in either application/json or application/x-recordio-protobuf format with a vector of "projections."  
- 

### Random Cut Forest \- unsupervised  {#random-cut-forest---unsupervised}

- Detects anomalous values within a dataset  
- Low scores normal  
- High scores anomaly  
- Inputs  
  - Train and test data channels  
  - Test channel is used to compute accuracy, precision and real and f1 score on labeled data.  
  - You can use either File mode or Pipe mode to train RCF models on data that is formatted as recordIO-wrapped-protobuf or as CSV  
- Output  
  - RCF inference returns application/x-recordio-protobuf or application/json formatted output  
- Although the algorithm could technically run on GPU instance types it does not take advantage of GPU hardware.  
- Tuning  
  - Metric  
    - test:f1 F1-score on the test dataset based on the difference between calculated labels and actual labels.  
  - Params  
    - num\_samples\_per\_tree  
    - num\_trees

## Vision {#vision}

Image Classification \- TensorFlow

### Image Classification \- MXNet \- supervised {#image-classification---mxnet---supervised}

- Input  
  - application/x-recordio)   
  - image (image/png, image/jpeg, and application/x-image)  
- Output   
  - Json  
    - {"prediction": \[prob\_0, prob\_1, prob\_2, prob\_3, ...\]}

Object Detection \- MXNet \- supervised  
Object Detection \- TensorFlow

### SemanticSegmentation {#semanticsegmentation}

- Pixel level approach tagging every pixel  
- Input  
  - S3  
  - Two channels train and validation using four directories two for images and two for annotations train, train\_annotation, validation, and validation\_annotation  
  - The dataset also expected to have one label\_map.json file per channel for train\_annotation and validation\_annotation respectively.  
- Output  
  - The segmentation output is represented as a grayscale image, called a *segmentation mask*. A segmentation mask is a grayscale image with the same shape as the input image.  
- Instances  
  - Only supports GPU  
- Tuning  
  - Metrics  
    - validation:mIOU  
    - validation:pixel\_accuracy  
  - Params  
    - learning\_rate  
    - mini\_batch\_size  
    - momentum  
    - optimizer  
    - Weight\_decay  
    - 

Instances  
Ml.p2, p3 g4, g5

### AutoGluon-Tabular {#autogluon-tabular}

- AutoML uses ensembling multiple models and stacking them in multiple layers  
- Input  
  - S3  csv for training and inference  
  - Use training and validation channels  
- Output  
  - The segmentation output is represented as a grayscale image, called a *segmentation mask*. A segmentation mask is a grayscale image with the same shape as the input image.  
- Instances  
  - Only supports GPU  
- Tuning

### CatBoost {#catboost}

- High-performance open source implementation of gradient boosting decision tree algorithm. GBDT is a supervised learning algorithm. Includes ordered boosting  
- Input  
  - Use training and validation channels  
- Output  
  - The segmentation output is represented as a grayscale image, called a *segmentation mask*. A segmentation mask is a grayscale image with the same shape as the input image.  
- Instances  
  - Only supports GPU  
- Tuning  
  - learning\_rate	  
  - depth  
  - l2\_leaf\_reg  
  - Random\_strength

### FactorizationMachines (supervised learning) {#factorizationmachines-(supervised-learning)}

- General purpose for both binary classification and regression. Extension of a linear learner model. Designed to capture interactions between features within high dimensional sparse data sets.  
- Input  
  - Test channel \+ train channel data sets  
  - Only supports recordIO-protobuf format with Float32 tensors  
  - File and pipe mode supported  
  - Inference \- application/json and x-recordio-protobuf formats  
- Output  
  - Score and label returned  
- Instances  
  - CPU recommended for sparse and dense data  
- Tuning  
  - Metrics  
    - Test:rmse  
  - Params  
    - test:binary\_classification\_accuracy	  
    - test:binary\_classification\_cross\_entropy	  
    - test:binary\_f\_beta

### K-NearestNeighbours (index-based algorithm) supervised {#k-nearestneighbours-(index-based-algorithm)-supervised}

- Classification and regression  
- Three phases sampling, dimension reduction, index building   
- Input  
  - Test channel \+ train channel data sets  
  - text/csv and application/x-recordio-protobuf  
  - Use file or pipe mode  
  -   
- Output  
  - application/json and application/x-recordio-protobuf  
- Instances  
  - P and g instances CPU or GPU  
- Tuning  
  - Metrics  
    - Test:rmse  
  - Params  
    - test:binary\_classification\_accuracy	  
    - test:binary\_classification\_cross\_entropy	  
    - Test:binary\_f\_beta

### LinearLearner (supervised) {#linearlearner-(supervised)}

- Classification or Regression Problems  
- Three phases sampling, dimension reduction, index building   
- Input  
  - Use train, validation and test channels  
  - recordIO-wrapped protobuf  (only float 32 tensors are supported)  
  - CSV  
- Output  
  - **For regression** (predictor\_type='regressor'), the score is the prediction produced by the model.   
  - **For classification** (predictor\_type='binary\_classifier' or predictor\_type='multiclass\_classifier'), the model returns a score and also a predicted\_lab  
- Instances  
  - P2, P3, G4dn, and G5  
- Tuning  
  - Metrics  
    -  To avoid overfitting, we recommend tuning the model against a validation metric instead of a training metric.  
  - Params  
    - wd  
    - l1  
    - learning\_rate  
    - mini\_batch\_size  
    - use\_bias  
    - Positive\_example\_weight\_mult

### TabTransformer (supervised) {#tabtransformer-(supervised)}

The Transformer layers transform the embeddings of categorical features into robust contextual embeddings to achieve higher prediction accuracy.

- Classification or Regression Problems  
-   
- Input  
  -   
- Output  
  -   
- Instances  
  - P2, P3, G4dn, and G5  
- Tuning  
  - Metrics  
    - r2	  
    - F1\_score binary cross entropy  
    - Accuracy\_score  
  - Params  
    - learning\_rate	  
    - input\_dim	  
    - N\_blocks  
    - attn\_dropout	  
    - mlp\_dropout	  
    - Frac\_shared\_embed

### XGBoost (supervised) {#xgboost-(supervised)}

a popular and efficient open-source implementation of the gradient boosted trees algorithm. Gradient boosting is a supervised learning algorithm that tries to accurately predict a target variable by combining multiple estimates from a set of simpler mode

- Multi and binary) Classification or Regression Problems and Rankin  
- You can use the new release of the XGBoost algorithm as either:  
  - A Amazon SageMaker built-in algorithm.  
  - A framework to run training scripts in your local environments.  
- You can use scale\_pos\_weight param to adjust the balance of positive and negative weights. It will give more importance to minority.  
    
- Input  
  - *text/libsvm* (default)  
  - *text/csv*  
  - *application/x-parquet*  
  - *application/x-recordio-protobuf*  
  - Can assign weights to labeled data  
- Output  
  -   
- Instances  
  - Supports CPU and GPU  
  - Supports distributed  
- Tuning  
  - Metrics  
    - validation:accuracy  
    - Validation:auc  
    - validation:error  
    - validation:f1  
    - validation:logloss  
    - validation:mae   
    - validation:map  
    - validation:merror  
    - validation:mlogloss  
    - Validation:mse  
    - validation:ndcg      
    - validation:rmse  
  - Params  
    - alpha  
    - colsample\_bylevel   
    - colsample\_bynode   
    - colsample\_bytree     
    - eta      
    - gamma     
    - lambda      
    - max\_delta\_step     
    - max\_depth      
    - min\_child\_weight      
    - num\_round    
    - subsample  
    - 

## Additional Notes {#additional-notes}

### Learning Rate \- Step Size {#learning-rate---step-size}

It controls the size of the steps the model takes while adjusting its parameters (weights and biases) to minimize the error or loss function.

**Small Learning Rate**:

* **Pro**: The updates to the model parameters are small, which may lead to a more stable convergence towards the optimal solution.  
* **Con**: Training becomes slower, and the model might get stuck in local minima or take a very long time to converge.

**Large Learning Rate**:

* **Pro**: Training is faster because the model takes larger steps toward the optimal solution.  
* **Con**: The updates might be too large, causing the model to overshoot the optimal solution or even diverge (fail to converge).

**Very Large Learning Rate**:

* **Con**: The model may oscillate wildly or fail to converge entirely as it jumps back and forth without ever settling at a minimum.

### Root Mean Squared Error (RMSE) {#root-mean-squared-error-(rmse)}

For regression tasks, the industry standard Root Mean Square Error (RMSE) metric. It is a distance measure between the predicted numeric target and the actual numeric answer (ground truth). The smaller the value of the RMSE, the better is the predictive accuracy of the model. A model with perfectly correct predictions would have an RMSE of 0\. [Regression Model Insights \- Amazon Machine Learning](https://docs.aws.amazon.com/machine-learning/latest/dg/regression-model-insights.html)

Example question \- What is a good target metric to use generally when comparing different regression models?

### Area under the curve (AUC) {#area-under-the-curve-(auc)}

AUC-ROC (Area Under the Receiver Operating Characteristic curve) is a performance measurement for classification problems at various thresholds settings. It tells how much model is capable of distinguishing between classes, taking into account both the true positive rate and the false positive rate, and is not sensitive to imbalanced classes. It is suitable when equal importance is given to all misclassification types and you need to evaluate the model performance across all possible classification thresholds.

Example question- What is a good target metric to use when comparing different binary classification models?

### F1 Score {#f1-score}

The F1 score is a weighted average of precision and recall, hence it is more useful than accuracy, especially if you have an uneven class distribution. However, it does not consider all possible classification thresholds  
When calculating trigrams calculate unique vectors for 1,2 and 3 word grams

### AWS Sagemaker Canvas {#aws-sagemaker-canvas}

Correct. SageMaker Canvas is a service that you can use to create ML models without having to write code. You can use SageMaker Canvas to build a custom model trained with your data. SageMaker Canvas can perform no-code data preparation, feature engineering, algorithm selection, training and tuning, inference, continuous model monitoring, and other tasks. SageMaker Canvas can generate models that are accurate, even when datasets are highly imbalanced. Because this scenario requires time series forecasting to generate predictions, SageMaker Canvas is a suitable option to build the custom model.

### Changing the Sagemaker Endpoint {#changing-the-sagemaker-endpoint}

De-register the endpoint as a scalable target. Update the endpoint config. Register the endpoint 

### Neural Network Optimization \- AdaGrad, RMSProp, and ADAM. {#neural-network-optimization---adagrad,-rmsprop,-and-adam.}

These optimization algorithms are stochastic gradient descent with momentum, AdaGrad, RMSProp, and ADAM.

* Local optima and saddle points of the loss function pose problems where the simple gradient method reaches its limits when training neural networks  
* With AdaGrad, RMSProp and ADAM there are technical possibilities to make the gradient descent more efficient when finding the optimal weights.  
* ADAM is usually the better optimization algorithm for training the neural networks.

What contributed to wide adoption of neural networks

- Cheaper GPUS  
- Increased data collection  
- Efficient algorithms

 API does the Amazon SageMaker SDK use to create and interact with the Amazon SageMaker hyperparameter tuning jobs

- HyperparameterTuner()

### Amazon Forecast

PerfirmAutoML to true   
If you want Amazon Forecast to evaluate each algorithm and choose the one that minimizes the `objective function`

`Forecast input`  
`Csv and json data from s3.`

ForecastFrequency
The frequency of predictions in a forecast.

[ForecastTypes](https://docs.aws.amazon.com/forecast/latest/dg/API_CreatePredictor.html#API_CreatePredictor_RequestSyntax)
Specifies the forecast types used to train a predictor. You can specify up to five forecast types. Forecast types can be quantiles from 0.01 to 0.99, by increments of 0.01 or higher. You can also specify the mean forecast with mean.

[HPOConfig](https://docs.aws.amazon.com/forecast/latest/dg/API_CreatePredictor.html#API_CreatePredictor_RequestSyntax)
Provides hyperparameter override values for the algorithm. If you don't provide this parameter, Amazon Forecast uses default values. The individual algorithms specify which hyperparameters support hyperparameter optimization (HPO). For more information, see Amazon Forecast Algorithms.

[PerformHPO](https://docs.aws.amazon.com/forecast/latest/dg/API_CreatePredictor.html#API_CreatePredictor_RequestSyntax)

Whether to perform hyperparameter optimization (HPO). HPO finds optimal hyperparameter values for your training data. The process of performing HPO is known as running a hyperparameter tuning job.

[ForecastHorizon](https://docs.aws.amazon.com/forecast/latest/dg/API_CreatePredictor.html#API_CreatePredictor_RequestSyntax)

Specifies the number of time-steps that the model is trained to predict. The forecast horizon is also called the prediction length.

### Accuracy, Precision, Recall and F1 {#accuracy,-precision,-recall-and-f1}

**Precision** (also called [positive predictive value](https://en.wikipedia.org/wiki/Positive_predictive_value)) is the fraction of relevant instances among the retrieved instances. Written as a formula:

**Recall** (also known as [sensitivity](https://en.wikipedia.org/wiki/Sensitivity_and_specificity)) is the fraction of relevant instances that were retrieved. Written as a formula:

### Calculate Confusion Matrix {#calculate-confusion-matrix}

![][image1]

Accuracy \= TP \+ TN / (TP \+ TN \+ FP \+ FN)  
Correct Predictions / All Predictions

Precision \= TP ÷ (TP \+ FP)  
24 / (24+2) \= 0.92  
Good for reducing False Positives

Recall \= TP ÷ (TP \+ FN)  
24 / (24+6) \= 0.8  
Good for reducing False Negatives

F1  
F1 \= 2 × (Precision × Recall) ÷ (Precision \+ Recall)  
2 x (.92 x .8  / .92 \+ .8) \= 0.855  
Good for balancing both recall and precision

### t-SNE {#t-sne}

t-SNE is something called **nonlinear dimensionality reduction**. What that means is this algorithm allows us to separate data that cannot be separated by any straight line, let me show you an example: Similar to PCA which is linear dimensionality reduction.

### RFE \- Recursive Feature Elimination {#rfe---recursive-feature-elimination}

RFE is popular because it is easy to configure and use and because it is effective at selecting those features (columns) in a training dataset that are more or most relevant in predicting the target variable.  
There are two important configuration options when using RFE: the choice in the number of features to select and the choice of the algorithm used to help choose features. Both of these hyperparameters can be explored, although the performance of the method is not strongly dependent on these hyperparameters being configured well.

### Regularization \- Address Overfitting {#regularization---address-overfitting}

Altering the loss function to reduce overfitting  
In order to create less complex (parsimonious) model when you have a large number of features in your dataset, some of the Regularization techniques used to address over-fitting and feature selection are:

#### L1 Regularization {#l1-regularization}

L1 regularization technique is called **Lasso Regression** (Least Absolute Shrinkage and Selection Operator) adds “*absolute value of magnitude*” of coefficient as penalty term to the loss function.

#### L2 Regularization {#l2-regularization}

L2 is called ***Ridge Regression***.  **Ridge regression** adds “*squared magnitude*” of coefficient as penalty term to the loss function. Here the *highlighted* part represents L2 regularization element.

### Sagemaker Notebook instances {#sagemaker-notebook-instances}

Based on EC2 instances run on service accounts. Isolated from the customer account. EBS volumes are also managed in Service Account.

### Linear Support Vector Machine

Suitable for linear classification

### SVM  {#svm}

Suitable for  Non linear classification. Think scatter plot overlay boundary

### LTSM 

Long short-term memory good for linear

### Single perceptron

linear

### Collaborative Filtering {#collaborative-filtering}

To address some of the limitations of content-based filtering, collaborative filtering uses *similarities between users and items simultaneously* to provide recommendations. This allows for serendipitous recommendations; that is, collaborative filtering models can recommend an item to user A based on the interests of a similar user B. Furthermore, the embeddings can be learned automatically, without relying on hand-engineering of features.

### Content Based Filtering {#content-based-filtering}

Content-based filtering uses item features to recommend other items similar to what the user likes, based on their previous actions or explicit feedback.

ROC Curve   
For understand how different classification techniques will impact model performance when tuning the ideal classification threshold   
The **ROC (Receiver Operating Characteristic) curve** is a graphical representation used to evaluate the performance of a binary classification model in machine learning. It plots the **True Positive Rate (TPR)**, also known as sensitivity or recall, against the **False Positive Rate (FPR)** at various classification thresholds.

### AWS Transcribe Vocab Filtering 

A custom vocabulary filter is a text file that contains a custom list of individual words that you want to modify in your transcription output.

### Sagemaker LifeCycle Configuration Script

A *lifecycle configuration* (LCC) provides shell scripts that run only when you create the notebook instance or whenever you start one. When you create a notebook instance, you can create a new LCC or attach an LCC that you already have.  They run as root.

## Kinesis Data Streams {#kinesis-data-streams}

Kinesis Data Streams **stores data** for later processing by applications (key difference with Firehose which delivers data directly to AWS services).

* Producers  
  * A producer creates the data that makes up the stream. Producers can be used through the following:  
  * Kinesis Streams API.  
  * Kinesis Producer Library (KPL).  
  * Kinesis Agent.  
* By default, records of a stream are accessible for up to 24 hours from the time they are added to the stream (can be raised to 7 days by enabling extended data retention).  
* A data blob is the data of interest your data producer adds to a data stream.  
* The maximum size of a data blob (the data payload before Base64-encoding) within one record is 1 megabyte (MB).  
* A shard is the base throughput unit of an Amazon Kinesis data stream.  
* One shard provides a capacity of 1MB/sec data input and 2MB/sec data output.  
* Each shard can support up to 1000 PUT records per second.  
* A stream is composed of one or more shards.  
* The total capacity of the stream is the sum of the capacities of its shards.  
* EMR can read directly form Kinesis Data Streams

### Kinesis Data Firehose {#kinesis-data-firehose}

* Kinesis Data Firehose is the easiest way to load streaming data into data stores and analytics tools.  
* Captures, transforms, and loads streaming data.  
* Enables near real-time analytics with existing business intelligence tools and dashboards.  
* Kinesis Data Streams can be used as the source(s) to Kinesis Data Firehose.  
* You can configure Kinesis Data Firehose to transform your data before delivering it.  
* With Kinesis Data Firehose you don’t need to write an application or manage resources.  
* Firehose can batch, compress, and encrypt data before loading it.  
* Firehose synchronously replicates data across three AZs as it is transported to destinations.  
* Each delivery stream stores data records for up to 24 hours.  
* A source is where your streaming data is continuously generated and captured.  
* A delivery stream is the underlying entity of Amazon Kinesis Data Firehose.  
* A record is the data of interest your data producer sends to a delivery stream.  
* The maximum size of a record (before Base64-encoding) is 1000 KB.  
* A destination is the data store where your data will be delivered.  
* Firehose Destinations include:  
  * Amazon S3.  
  * Amazon Redshift.  
  * Amazon Elasticsearch Service.  
  * Splunk.  
* Producers provide data streams.  
* No shards, totally automated.  
* Can encrypt data with an existing AWS Key Management Service (KMS) key.  
* Server-side-encryption can be used if Kinesis Streams is used as the data source.  
* Firehose can invoke an AWS Lambda function to transform incoming data before delivering it to a destination.

### Kinesis Data Analytics {#kinesis-data-analytics}

* Amazon Kinesis Data Analytics is the easiest way to process and analyze real-time, streaming data.  
* Can use standard SQL queries to process Kinesis data streams.  
* Provides real-time analysis.  
* Use cases:  
* Generate time-series analytics.  
* Feed real-time dashboards.  
* Create real-time alerts and notifications.  
* Quickly author and run powerful SQL code against streaming sources.  
* Can ingest data from Kinesis Streams and Kinesis Firehose.  
* Output to S3, RedShift, Elasticsearch and Kinesis Data Streams.  
* Sits over Kinesis Data Streams and Kinesis Data Firehose.  
* A Kinesis Data Analytics application consists of three components:  
* Input – the streaming source for your application.  
* Application code – a series of SQL statements that process input and produce output.  
* Output – one or more in-application streams to hold intermediate results.  
* Kinesis Data Analytics supports two types of inputs: streaming data sources and reference data sources:  
* A streaming data source is continuously generated data that is read into your application for processing.  
* A reference data source is static data that your application uses to enrich data coming in from streaming sources.  
* Can configure destinations to persist the results.  
* Supports Kinesis Streams and Kinesis Firehose (S3, RedShift, Elasticsearch) as destinations.  
* IAM can be used to provide Kinesis Analytics with permissions to read records from sources and write to destinations.  
* C-bow continuous bag of words  
* When you need to identify the most relevant keywords within a document relative to a collection, TF-IDF provides a clear picture. Preference over BlazingText CBOW and NLTK stemming and stop world removal

KMeans Elbow  
![][image2]

Determine if regression Model is Overestimating or Underestimating  
Residual plots

Decision Trees   
Can create non-linear separation between classification classes. (circle scatter plot)  
Provides higher performance/recall.

### SageMakerJobs  {#sagemakerjobs}

Required MUST specify params

- The training channel  
- IAM Role  
- Output Path

Ec2 instance type (sagemaker has defaults but it can be best practice).

### Amazon Athena Performance {#amazon-athena-performance}

Storing files in s3 csv and json are row based and good for readability. Parquet is column based and has improved performance for reading and writing.  
RecordIO is good for sequential patterns for ml processing.  
Parquet is good for athena query performance.

### Reducing Overfitting {#reducing-overfitting}

Increase Regularization \- add a penalty to loss function L1 or L2  
Increase Dropout \- Randomly drop neurons during training to make a less fit more general model.  
Reduce Feature Combinations \- PCA / t-SNE

## Random Notes {#random-notes}

* You can use a glue schema from firehose to convert from json to parquet.  
* Amazon Athena has a JDBC Connector allowing you to connect BI tools to query S3 Data.  
* K-means is NOT used from anomaly detection or is not optimal. RCF is better.  
* KDA can provide real time anomaly detection. It enables you to continuously process streams of data in real-time and discover patterns, correlations, trends, outliers, and other insights. KDA provides built-in functions for filtering, mapping, aggregating, scaling, and enriching your data streams as they arrive, allowing you to perform complex calculations on raw data directly within the service itself.  
* CNN convolutional neural networks are often used for image related ML problems.  
* Glue has native support for pyspark.  
* Glue can combine large data sets in S3.  
* Sage MAker Endpoints \- A single SageMaker endpoint cannot serve two different models  
* Counting unique worlds when calculating uni gram  
* Imputation Techniques \-  Last observation Carried Forward  
  * Can be used for time series data.  
* Logistic Regression \- Binary Classification  
* Linear Regression \- Predicting Continuous value,   
* Multicollinearity  
  * Instances where identical features in a training set can result in a singular matrix during optimization which fails to define a unique solution  
* Early Stopping \- Stops training after validation error has reached minimum.  
* Area under precision recall curve (focus on precision for positive classiufication focus)  
* You can enable network issolcation for training jobs.  
  * You can enable network isolation when you create your training job or model by setting the value of the `EnableNetworkIsolation` parameter to `True` when you call [`CreateTrainingJob`](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTrainingJob.html), [`CreateHyperParameterTuningJob`](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateHyperParameterTuningJob.html), or [`CreateModel`](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateModel.html).   
* A **private workforce** is a group of workers that *you* choose. These can be employees of your company or a group of subject matter experts from your industry. For example, if the task is to label medical images, you could create a private workforce of people knowledgeable about the images in question.   
* For recommendation training test split “identify most recent x% transactions fro each use. Split these off for test set.  
* You can disable root access on the sagemaker notebook instances.  
* You can restrict notebook presigned ursl to specific ip addresses used by a company.  
* Amazon Comprehend custom entity recognition  
* Stopoping and starting restarting the notebook instance applies latest updates and patches.  
* DepLens for local processing,   
* Recognition can run locally on deeplens  
* You can deploy sagemaker model to a DeepLens device.  
* IP insights is unsupervised and does not require training.  
* sagemaker::CreatePOresignedNotebookInstanceUrl and sagemaker:DescribeNotebookInstance actions from only the VPC endpoints. Apply the policy to all users and grous and roles to use the notebook instrances.  
* AWS Cloud HSM \- hardware security module  
* Sagemaker built-in transient keys for only processing.  
* Glue can discover schema from DynamiDB also.  
* Lex be aware slot types vs synonyms, dont watn to introduce more categories/slots.  
  * Add synonyms in the custom slot type  
* IAM Best Practice  
  * Define resources and actions allowed to role that is applied to notebook instance.  
* Sagemaker Debugger supports creating  SHAP Values  
  * SHapley Additive exPlandations  
* When using SageMaker Endpoint Model Monitor you must update the model monitor baseline after model changes to reflect the new model baseline.  
* For forecasting granularity matters.  
* Read confusion matrix closely look at performance of individual classes. Look at proportions.  
* Univariate selection  
  * Rank features based on target (feature selection)  
* MARL \- Multi Agent Reinforcement learning \- muti agent traffic lights.  
* Min-Max feature scaling operates ont he individual features themselves.

### AWS DataSync

Purpose: Designed for large-scale data transfer and replication between various storage locations, including on-premises, cloud, and different AWS services like S3, EFS, FSx, etc.  
Transfer modes: Offers one-time transfers, scheduled replications, and continuous data synchronization.

Key features: Automated data discovery, performance optimization, security encryption, integration with other AWS services.

### AWS Transfer

Purpose: Facilitates secure file transfers between your applications and storage solutions like S3 and EFS.  
Data types: Focuses on file-based data (various formats).  
Transfer protocols: Supports FTP, FTPS, SFTP, and AS2.

Choose DataSync for large-scale data movement and replication across diverse storage locations, including relational databases.  
Choose Transfer Family for simple, secure file transfers between your applications and S3/EFS, especially when using traditional file transfer protocols like FTP/FTPS/SFTP.

### Amazon Augmented AI

Amazon Augmented AI (Amazon A2I) enables you to build the workflows required for human review of ML predictions. Amazon A2I brings human review to all developers, removing the undifferentiated heavy lifting associated with building human review systems or managing large numbers of human reviewers.

### AWS Kendra {#aws-kendra}

Amazon Kendra is a highly accurate and easy-to-use enterprise search service that’s powered by machine learning (ML).

### Transfer Learning {#transfer-learning}

Take advantage of pre-trained models on a large dataset. Use pre-trained weights.

### AWS DeepLens  {#aws-deeplens}

Used for smaller installations not suited for 10k plus installations.

### AWS CodeGuru {#aws-codeguru}

Amazon CodeGuru is a developer tool that provides intelligent recommendations to improve code quality and identify an application’s most expensive lines of code.

- Security \- analyze code and look for vulns (Java, Python, Javascript)  
- Profiler \- hello operators and optimize operating  (Java, Kotlin, Python preview)

### AWS Rekognition  {#aws-rekognition}

Rekognition Video for capturing streams  
Rekognition Image to detect faces.

### AWS SageMaker and Parallel Training for deep learning workloads {#aws-sagemaker-and-parallel-training-for-deep-learning-workloads}

What is Horovod? It is a framework allowing a user to distribute a deep learning workload among multiple compute nodes and take advantage of inherent parallelism of deep learning training process. It is available for both CPU and GPU AWS compute instances. Horovod follows the Message Passing Interface (MPI) model in All-Reduce fashion. This is a popular standard for passing messages and managing communication between nodes in a high-performance distributed computing environment.

When you train a model with a large amount of data, you should distribute the training across multiple GPUs on either a single instance or multiple instances. Deep learning frameworks provide their own methods to support multi-GPU training or distributed training. However, there is another way to accomplish this using distributed deep learning framework such as [Horovod](https://github.com/horovod/horovod).

### AWS SageMaker Local Mode {#aws-sagemaker-local-mode}

The SageMaker Python SDK supports local mode, which allows you to create estimators, processors, and pipelines, and deploy them to your local environment. This is a great way to test your deep learning scripts before running them in SageMaker’s managed training or hosting environments. Local Mode is supported for frameworks images (TensorFlow, MXNet, Chainer, PyTorch, and Scikit-Learn) and images you supply yourself.

### Normalization vs Standardization {#normalization-vs-standardization}

| Normalization | Standardization |
| ----- | ----- |
| Rescales values to a range between 0 and 1 | Centers data around the mean and scales to a standard deviation of 1 |
| Useful when the distribution of the data is unknown or not Gaussian | Useful when the distribution of the data is Gaussian or unknown |
| Sensitive to outliers | Less sensitive to outliers |
| Retains the shape of the original distribution | Changes the shape of the original distribution |
| May not preserve the relationships between the data points | Preserves the relationships between the data points |
| Equation: (x – min)/(max – min) | Equation: (x – mean)/standard deviation |

### Standardization Z-Score Calculation {#standardization-z-score-calculation}

Let's take the value 5\. To calculate the standardization value we use the following formula z \= (x \- u) / s where 'z' is the standardized value, where 'x' is our observed value, where 'u' is the mean value of the feature, and 's' is the standard deviation.

Z \= x \- mean / stddev

### S3 Encrypted Access from SageMaker Notebook {#s3-encrypted-access-from-sagemaker-notebook}

Assign Role to SageMaker notebook with S3 Read Access also Grant permission in the KMS Key policy to that role.

### tf-idf Calculation {#tf-idf-calculation}

1. Ignore duplicate words  
2. Count all unique words from all sentences  
3. Count bigrams within each sentence 

### High-Degree Polynomial Transformation

A **high-degree polynomial transformation** is a math trick that helps us fit more complex curves to this data.

Think of a polynomial as a mathematical equation with powers (like squares or cubes) of your variables. For example, instead of just xxx, you might have x2x^2x2 (which makes a curve), x3x^3x3 (an even twistier curve), and so on.

So, when we apply a high-degree polynomial transformation, we’re adding these powers of variables to capture the twists and turns in the data that a straight line can’t explain. This helps the model learn patterns in the data that might not be obvious with simpler math.

However, if you use too high of a degree, the model can overfit—meaning it tries too hard to fit every tiny detail, even if it’s just noise. So, it’s a balance between fitting the data well and not going overboard.

High-degree polynomial transformations are used in machine learning to capture complex, non-linear relationships between features and the target variable. By adding polynomial features (e.g., squares, cubes) of the original features, models like linear regression can fit more complex patterns.

### Logarithmic Transformation

Logarithmic transformation is used in machine learning to reduce skewness and handle non-linear relationships by transforming variables with a wide range of values. It compresses the range of large values while expanding smaller ones, making the data more normally distributed.  (handles left skewed data)

### Exponential Transformation

Handle Right Skewed data\!\!\!

### Polynomial Transformation

Handle noliniaer issues in data. Does not necessarily handle skewness in data

### Sinusoidal Transformation

Used to model periodic data. Does not handle skewness.

### EMR Clusters

Master Node \- must be up  
Cor Nodes handel data storage and job execution cant afford to loose. Must be alive 100%  
Task Nodes \- ok for spot instances

### Object Detection SingleShot Multibox detector 

**Single Shot:** this means that the tasks of object localization and classification are done in a *single* *forward pass* of the network  
**MultiBox:** this is the name of a technique for bounding box regression developed by Szegedy et al. (we will briefly cover it shortly)  
**Detector:** The network is an object detector that also classifies those detected objects

### 

### Tensor Flow Script Mode

TernsorFlow can read TFRecord data format from S3 in script mode.

### 

### 

### CNN vs RNN

CNN  is typically used for image and spatial analysis  
CNNs are ideal for grid-like data such as images, where they apply filters to capture spatial hierarchies and local patterns.

RNN is used for time series data can capture temporal dependencies nd patterns over time.  
RNNs are designed for sequential data, like time series or natural language, where information flows through the network over time, capturing temporal dependencies.

InScope  
Analytics: • Amazon Athena • Amazon Data Firehose • Amazon EMR • AWS Glue • Amazon Kinesis • Amazon Kinesis Data Streams • AWS Lake Formation • Amazon Managed Service for Apache Flink • Amazon OpenSearch Service • Amazon QuickSight

Management and Governance: • AWS CloudTrail • Amazon CloudWatch Networking and Content Delivery: • Amazon VPC Security, Identity, and Compliance: • AWS Identity and Access Management (IAM) Storage: • Amazon Elastic Block Store (Amazon EBS) • Amazon Elastic File System (Amazon EFS) • Amazon FSx • Amazon S3

Database: • Amazon Redshift Internet of Things: • AWS IoT Greengrass Machine Learning: • Amazon Bedrock • Amazon Comprehend • AWS Deep Learning AMIs (DLAMI) • Amazon Forecast • Amazon Fraud Detector • Amazon Lex • Amazon Kendra • Amazon Mechanical Turk • Amazon Polly • Amazon Q • Amazon Rekognition • Amazon SageMaker • Amazon Textract • Amazon Transcribe • Amazon Translate

Compute: • AWS Batch • Amazon EC2 • AWS Lambda Containers: • Amazon Elastic Container Registry (Amazon ECR) • Amazon Elastic Container Service (Amazon ECS) • Amazon Elastic Kubernetes Service (Amazon EKS) • AWS Fargate

### AWS Deep Learning AMIs (DLAMI) 

provides ML practitioners and researchers with a curated and secure set of frameworks, dependencies, and tools to accelerate deep learning on Amazon EC2. Built for Amazon Linux and Ubuntu, Amazon Machine Images (AMIs) come preconfigured with TensorFlow, PyTorch, NVIDIA CUDA drivers and libraries, Intel MKL, Elastic Fabric Adapter (EFA), and AWS OFI NCCL plugin, allowing you to quickly deploy and run these frameworks and tools at scale

## References {#references}

- [https://docs.aws.amazon.com/sagemaker/latest/dg/security\_iam\_id-based-policy-examples.html\#api-access-policy](https://docs.aws.amazon.com/sagemaker/latest/dg/security_iam_id-based-policy-examples.html#api-access-policy)  
- [https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-considerations.html](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-considerations.html)  
- [https://docs.aws.amazon.com/sagemaker/latest/dg/security\_iam\_id-based-policy-examples.html\#api-access-policy](https://docs.aws.amazon.com/sagemaker/latest/dg/security_iam_id-based-policy-examples.html#api-access-policy)  
- [https://en.wikipedia.org/wiki/Precision\_and\_recall](https://en.wikipedia.org/wiki/Precision_and_recall)  
- [https://docs.aws.amazon.com/sagemaker/latest/dg/algorithms-choose.html](https://docs.aws.amazon.com/sagemaker/latest/dg/algorithms-choose.html)  
- [https://artemoppermann.com/optimization-in-deep-learning-adagrad-rmsprop-adam/](https://artemoppermann.com/optimization-in-deep-learning-adagrad-rmsprop-adam/)  
- [https://towardsdatascience.com/t-sne-clearly-explained-d84c537f53a](https://towardsdatascience.com/t-sne-clearly-explained-d84c537f53a)  
- [https://digitalcloud.training/amazon-kinesis/](https://digitalcloud.training/amazon-kinesis/)  
- [https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c](https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c)  
- [https://www.youtube.com/watch?v=eGCpUhE1wNc\&list=PL7GozF-qZ4Kdjv39v0-OoFyegxCrFNp1G\&index=5](https://www.youtube.com/watch?v=eGCpUhE1wNc&list=PL7GozF-qZ4Kdjv39v0-OoFyegxCrFNp1G&index=5)  
- [https://docs.aws.amazon.com/sagemaker/latest/APIReference/API\_CreateTrainingJob.html](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTrainingJob.html)  
- [https://sagemaker.readthedocs.io/en/stable/overview.html\#local-mode](https://sagemaker.readthedocs.io/en/stable/overview.html#local-mode)  
- [https://www.tecracer.com/blog/2024/06/an-unsung-hero-of-amazon-sagemaker-local-mode.html](https://www.tecracer.com/blog/2024/06/an-unsung-hero-of-amazon-sagemaker-local-mode.html)  
- Ml university   
  - [https://www.youtube.com/playlist?list=PL8P\_Z6C4GcuWfAq8Pt6PBYlck4OprHXsw](https://www.youtube.com/playlist?list=PL8P_Z6C4GcuWfAq8Pt6PBYlck4OprHXsw)  
- 

[image1]: <data:image/png;base64,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>

[image2]: <data:image/png;base64,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>