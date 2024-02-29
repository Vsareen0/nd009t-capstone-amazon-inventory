# Inventory Monitoring for Distribution Centre

Distribution centres often use robots to move objects as a part of their operations. Objects are carried in bins which can contain multiple objects. Sometimes items get misplaced and to prevent mismatch while maintaining inventory records an efficient system can be put in place.

In this project, we will have to build a model that can count the number of objects in each bin. Creating a system like this can be used to track inventory and make sure that delivery consignments have the correct number of items.

# Environment
We used an AWS SageMaker instance *ml.t3.medium* type with the following configurations:

- two virtual CPUs
- four GiB memory

And the main software pre-requisites for the project are:

- Python 3.10
- Pytorch: 2.0

## Project Set Up and Installation

To run this project you can open **sagemaker.ipynb** and run all the cells in order.

## Dataset

The dataset has been provided from Amazon https://registry.opendata.aws/amazon-bin-imagery/ 

### Overview
The data has been provided from Amazon and I had already been given a portion of whole dataset in **file_list.json** which I used to download data from s3 s3://aft-vbi-pds/ onto sagemaker studio and then to our s3.

The data has 10,441 images, where 1-5 represent subdirectory/labels for images.   

### Access
I used boto3 to connect to s3 and used below snippet to download the code from source.

`s3_client.download_file('aft-vbi-pds', os.path.join('bin-images', file_name),
                             os.path.join(directory, file_name))`

To upload the data I used this command:
```!aws s3 cp train s3://sagemaker-us-east-1-372402537355/capstone_project_amazon/train --recursive```

## Model Training
**TODO**: What kind of model did you choose for this experiment and why? Give an overview of the types of hyperparameters that you specified and why you chose them. Also remember to evaluate the performance of your model.
I choose ResNet50 with transfer learning because it performs well on image classification tasks. I used ```"batch-size": 128, "learning-rate": 0.006``` which I got from hyperparameter tuning. 

I used three metric for evaluating performance of my model (for each class):
```
INFO:__main__:Testing Accuracy: 0.3336510962821735
INFO:__main__:Recall Computed: {0: 0.43548387096774194, 1: 0.45021645021645024, 2: 0.3694029850746269, 3: 0.3907563025210084, 4: 0.0}
INFO:__main__:F1 Computed: {0: 0.46956521739130436, 1: 0.3747747747747748, 2: 0.3350253807106599, 3: 0.34831460674157305, 4: 0}
```

## Machine Learning Pipeline

The steps were taken in order to create pipeline

1. Downloaded the s3 data.
2. Split into train, test and validation and uploaded to s3.
3. Perfomed hyperparameter tuning to get best starting values using **hpo.py**.
4. Setup model profiler and debugger
5. Perform Training on model using **train.py** script.
6. Get Reports and also check metrics for evaluation.
7. Deploy Model.
8. Perform inference.
9. If done, delete endpoint.

## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.

1. Hyperparameter Tuning
2. Model Deployment
3. Inference
4. Profiling And Report
