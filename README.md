# NetworkSecurityProject for Phishing Data

##Data: 
We downloaded Phisphing Data from Intenernet and is present in Networ_Data folder

## ETL Pipeline: Extract, Transform, Load

![alt text](<Screenshot 2024-10-28 at 4.23.01 PM.png>)
Step1: Extraction: We read data from multiple data sources and store it locally
Step2: Transformation
* It invovles basic preprocessing
* It involved cleaning raw data
* It invovles combining data into a single dataframe as a .json for loading purpsoses to mongodb data base
Step3: Loading
We load it into a database in our case Mongodb Atlas Cloud


![alt text](image.png)

### Data Ingestion:

The Data Ingestion component is responsible for initiating and executing the data retrieval process. It reads raw data from MongoDB, processes it for ML tasks, and splits it into training and testing datasets. The ingestion component creates a **DataIngestionArtifact**, which contains the paths to the processed train and test datasets.

#### Configuration
* **DataIngestionConfig**: Configures the Data Ingestion component by defining the data sources, output paths, and any parameters required to perform the data ingestion.
* **DataIngestionArtifact**: Serves as the output from the ingestion process, housing the **train_file_path** and **test_file_path** that are passed to subsequent components.
  
#### Data Ingestion Process
The steps involved in the Data Ingestion process are:

1. **export_collection_as_dataframe()**:
   * Reads data directly from MongoDB and returns it as a Pandas DataFrame, enabling further processing.
2. **export_data_into_feature_store**:
   * Stores the retrieved data into a feature store, making it accessible for transformation and modeling stages.
3. **split_data_as_train_test**:
   * Splits the data from the feature store into training and testing datasets, preparing it for machine learning tasks.

#### Running Data Ingestion:

    To run the data ingestion component, execute the main.py file which configures and runs the Data Ingestion process, initializing it through DataIngestionConfig and capturing its output in the DataIngestionArtifact.










### Data Validation:

### Note:

* Each component, including Data Ingestion, outputs an artifact, encapsulated within artifact_entity.py.
* Input configurations for each component are managed via respective config_entity classes, ensuring modularity and reusability.

* final_model folder houses the best model found and the data preprocessor object associated with the data for it  pickle files

Steps in project:
1. Write 4 Components:
   1. Data Ingestion Process
   2. Data Validation Process
   3. Data Transformation Process
   4. Model Training Process
2. Create a Training Pipeline
3. Creating API Prediction using Fast API:-
Input is :
- best model (model.pkl)
- preprocessor object (preprocessor.pkl)
  Steps:
   1. Training 
   2. Batch Prediction by data uploaded in Fast API:
      1. Takes batch of test data (without dependent variable) in valid_data folder
      2. Gived Batch prediction output n prediction_output folder with last column as the dependent variable we're trying to predict
4. Deployment of the NetworkPhishing Data Detection application as a docker 

![AWS CI/CD Pipeline with Self-Hosted App Runner on EC2 Linux Using Dockerized Application ](image-2.png)



# Deployment Steps of Network Security Application in AWS:
![alt text](image-1.png)