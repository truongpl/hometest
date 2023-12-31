# Initial
This take home test requires two weeks to tackle the problem. The dataset is the the [Microsoft Azure Predictive Maintenance](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance) dataset. I have dissmissed the company name.

Below is the test requirements:
![Page_1](https://github.com/truongpl/hometest/blob/main/assets/HT1_P1.png)
![Page_2](https://github.com/truongpl/hometest/blob/main/assets/HT1_P2.png)

# Solution
## Dataset description
- We have 100 machine, the sensors of machine are recorded every hour in a year so we have totally: 100 x 365 x 24 records
- After doing some EDA, the dataset is imbalance on the label of "failure", there are only 719 failure cases.
- The failure time is also imbalance, most of the machines 's failure occur at 6:00.

Upon initial consideration and made some quick research, I intend to generating sentences like: 
"Given voltage as 0.12345, pressure as 0.6789...."  (Similar to the main idea of the paper https://arxiv.org/pdf/2210.10723.pdf)
for each sample and feeding these sentences into a Language Model (LM) to create an embedding vector. The plan was to develop a traditional LSTM-based time series prediction model. However, this approach may not align with the specific requirements of the task because, in this case, the LSTM model would serve as the pattern recognizer not the LM.

In the pursuit of solving this problem, a few challenges have arise:

- The primary goal is to predict the time of failure in the future
- Standard example notebooks available Kaggle may not be directly applicable, because most of them use the input of a failure day to predict the output. 
- With above information, I decide to calculate the time_to_fail and use it as a base feature.

In short, the problem that I am trying to solve:
*Determine if a machine will fail within the next 24 hours and predict the time of its failure.*

it's necessary to approach the problem in a two-fold manner:

## Binary Classification

One aspect of the problem involves classification, aimed at determining whether a machine is likely to fail within the next 24 hours. Here's a breakdown of the classification task:

- **Input Data**: The input consists of a time series dataset, including parameters such as voltage (volt), rotation (rotate), pressure (pressure), vibration (vibration), and age (age). We only use the input that has been collected before the failure date

- **Negative Samples**: For negative samples, consider selecting time series data from the beginning up to a point 24 hours before the time of failure (time_to_fail - 24).

- **Positive Samples**: For positive samples, select time series data that includes the time range from zero-ttf to 24 hours before the failure occurs (time_to_fail - 24).

## Multi Class Classification

The other dimension of the problem involves multi classification. When a machine failure is predicted, it's important to estimate the specific time frame within which this failure is likely to occur. Here are the details:

- **Target**: The multi classification task aims to predict the time frame of failure, expressed as a time range between 00:00 and 23:00.

By approaching the problem from both classification and multi classification problems, we can try to finetune a LLM with the output is (is_failure,failure_time)


# Model choosing
I considered various Language Model (LLM) options for fine-tuning to address the task. After researching available resources, there are many of them, e.g:
- LLaMA2 
- Vicuna 
- Fastchat 
- Falcon 180B
- ChatGPT 3.5

I decide to finetune with the ChatGPT because of the production readiness of it. 

# First trial
In the initial trial, I created a system prompt instructing the LLM to respond exclusively in JSON format. The prompt specified that I would provide time series data, with the first line serving as the header. The LLM's task was to predict the following:

Does the machine require maintenance within the next 24 hours? (Respond with "Yes" or "No")
If "Yes," specify the predicted time.


Input data will be fed to user prompt as below:
volt,rotate,pressure,vibration,age
182.467109259603,501.918972726944,85.7626146951866,51.0214861151087,18
162.879223,402.747490,95.460525,43.413973,18
.....

I don't want to normalize the input data by MinMaxScaler because I believe that normalizing it to the [0,1] range might reduce the distinguish-ability of the LLM.

Response for the LLM is the JSON:
{
    "is_maintenance": (0,1),
    "predict_ttf": (0-23)
}

This approach fail on both the cost and efficiency, cost me 20.95$ to finetune but the accuracy on test set is very low.
One of the issue is the choosing of time series to create user prompts. 
Another drawback is that the cost of predictions: the total token for each request may be larger than 10k token which made the request cost (10 * 0.008 = 0.08$) which makes this approach unfeasible for the test.

# Second trial
Recognizing the challenges of the first trial, an alternative approach was adopted. Feeding raw time series data to the Large Language Model (LLM) was deemed impractical due to cost and inference time considerations. Instead, I calculate the mean of values by day, resulting in a more manageable dataset:

The dataset consists of data for 100 machines over 365 days, with data recorded every hour, so I will have 100 x 365 average data instead of 100 x 365 x 24 (volt,rotate,pressure,vibration) data.

To tackle the imbalance problem, I only pick 2 "normal" value for finetuning:

- A value from the day preceding a failure is considered a positive class (which give the answer: the **next day will have failure**)
- Two normal days, one is the next day after the failure are considered a negative class.

To optimize cost, I simplified the prompt to:
"Giving input volt,rotate,pressure,vibration,age answer with number,number, first number is 0/1, second is regression in 0-23 range'"

With these adjustments, the finetuning cost is 3.20$ and the model performed as intended. However, it's worth noting that this model has a drawback: the accuracy of the binary classification affects the multi-class classification. In other words, if the binary classification result is incorrect, the multi-class classification is also fail too.

# Wrap up the project
## Source code structure

- dataset: we should symlink that dataset to this folder
- deployment: has subfolders that contain ingress,platform, website and secret deployment.
- docker: has Dockerfile of platform and website
- experiment: I create the training data for ChatGPT
- src: I separate to api and service: api is Restful implementation, service is use to call LLM endpoint.
- main.py: use to test local before Docker's building
- wsgi.py: use for gunicorn
            
## Backend development:
- Flask and flask-restx were utilized to develop the Restful API, which serves at `/pdm`.

## Web application:
- I explored creating a web application but faced limitations due to my familiarity with HTML. Instead, I attempted to generate an `index.html` file using the Large Language Model (LLM) and adapted it to suit the problem. The resulting page features 24 rows for user input for sensor data. Users are not required to fill in all 24 rows, but the system considers it sufficient if more than 5 rows are provided.


## Hosting preparation
-  Two Virtual Private Clouds (VPCs) were created on DigitalOcean to take advantage of their free credits: one for the backend and one for the website.
-   Two subdomains were created: [] for the website and [] for the backend.
-   SSL certificates were generated for both the website and backend in https://zerossl.com/.

## Deployment
The deployment steps by steps as below:
-   I create the cluster by Microk8s
-   Docker images were created for both the backend and frontend, with details specified in `docker/Dockerfile.platform` and `docker/Dockerfile.website`.
-   Images were pushed to the DigitalOcean Container Registry.
-   Ingress and Service resources were deployed.
-   Create three secrets: regcred for CTR, api-tls, fe-tls for SSL of backend and website.
-   MicroK8s Metallb was enabled, and the LoadBalancer was directed to the IP address of the backend node.

## Testing
- In the *data_processing_v2.ipynb* I have use (8,2,2) ratio, first 8 months for finetuning, next 2 months for validation. We can use the remaining 2 months to test the accuracy of the model.
- Testing result:
- - Binary classification: 0.9157509157509157
- - Multi classification: 0.9157509157509157 - this is pretty useless because the failure time is mostly at 6:00 (702 cases / 719 failure). So the trivia solution (always tell the failure time is 6:00) will have accuracy = 0,97635605).

## Page site
- At [], I have 24 rows for input and a submit button. After pressing the "Submit", a message will be display right below the submit button to show the message prediction.
