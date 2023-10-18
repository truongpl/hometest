# Initial
This take home test requires 3-5 days to tackle the problem. The dataset is the the [SEED](https://www.kaggle.com/datasets/phhasian0710/seed-iv?resource=download) dataset. I have dissmissed the company name.

Below is the test requirements:
![Page_1](https://github.com/truongpl/hometest/blob/main/assets/HT0_P1.jpg)
![Page_2](https://github.com/truongpl/hometest/blob/main/assets/HT0_P2.jpg)
![Page_3](https://github.com/truongpl/hometest/blob/main/assets/HT0_P3.jpg)

# Solution
## Dataset description
This solution investigates the application of machine learning techniques for the classification of human emotions using electroencephalography (EEG) data. Emotion classification plays a crucial role in understanding and enhancing human-computer interaction, mental health assessment, and cognitive research. The study explores various preprocessing methods for EEG data, including noise reduction and feature extraction. Different classification algorithms, such as Support Vector Machines, Neural Networks, and Random Forests, are employed to discern emotional states from EEG signals. Performance evaluation is conducted using metrics such as accuracy, precision, recall, and F1-score. The SEED-IV is used for this work.

The SEED-IV has 15 subjects participate into the research, each subject is requested to watch 24 different film/clips on 3 different days. The clips have four labels: Neutral, Happy, Fear, Sad. During the experiment, the researchers attach 62 electrocudes on the subject's head to collect EEG signal and an eye movement tracking device. The dataset contains raw EEG signal from 62 electrocudes and data that was collected from the eye. Description of the dataset as below:

````bash
├── eeg_feature_smooth : EEG feature
│ ├── 1
│ ├── 2
│ └── 3
├── eeg_raw_data : Eye movement raw data
│ ├── 1
│ ├── 2
│ └── 3
├── eye_feature_smooth : Eye feature
│ ├── 1
│ ├── 2
│ └── 3
└── eye_raw_data : 62 electrocudes raw data signal
````

## Problem 1
### Calculate the power band
This is a raw signal processing process. I hope code in the function calculate_band_power on the
notebook speaks for itself. Steps are below:
- Calculate the Welch periodogram, it is power density of signal.
- Select the correct bin of signal. E.g: with Delta band we pick the values from [1.5-4] range.
- Calculate the accumulation power density for that range

### Data insight
At first, I prepare the data for analysis, the output of the preparation is:
- Data sample for each band and emotion. Each sample will have size = (62,15*18) = (62,270). 15
is the number of subjects, 18 is the total label of a emotion class over 3 session. This data is used for
hypothesis testing.
- Correlation coefficients: because the electrodes place densely on the skull so maybe a pair of
neighbor electrodes will both change during emotion change.

### Hypothesis testing:
Adapt from a basic statistical problem, I want to know that each electrode will react differently
between emotion. So I built a set of null hypothesis testing for a power band. For example:

- Null hypothesis:There is no different between gamma means of each electrode when emotion is
sad.
- Null hypothesis: There is no different between gamma means of each electrode when emotion is
neutral.
- Null hypothesis for gamma: There is no different between gamma means of each electrode when
emotion is happy.
- Null hypothesis for gamma: There is no different between gamma means of each electrode when
emotion is fear.


### Testing result

Sample of testing result on Gamma band

| Power Band   | Emotion | F statistic      | P value                | Reject |
|--------------|---------|------------------|------------------------|--------|
| Gamma        | Neutral | 1.1850261017569  | 0.1533769400864        | No     |
| Gamma        | Sad     | 2.4241121191908  | 4.02708119871393e-09   | Yes    |
| Gamma        | Fear    | 1.9432534839470  | 1.5305477157827447e-05 | Yes    |
| Gamma        | Happy   | 2.3539977530464  | 1.4487386032285e-08    | Yes    |

#### Interpret the result for Gamma
When we reject a null hypothesis, that mean there is a different node that cause the emotion. It means that we can use the electrodes to classify the emotion


#### Correlation coefficient:
In addition from the hypothesis testing, I calculate the correlation coefficients between each pairs of electrode to know which pair will be significantly change for each emotion. E.g: top 5 pairs of electrodes for gamma during sad motion is:
(P2 P1) (CPZ CZ) (FCZ FC2) (CP4 C4) (CP3 C3)
That means that these pair both increase or decrease when we are sad or not.

## Problem 2
### Feature define
Exploring the “eeg_feature_smooth” data, the authors define that it has below information:
- Shape: 62*W*5, the maximum of W = 64
- There are four features, named:['de_movingAve', 'de_LDS', 'psd_movingAve', 'psd_LDS']
Exploring the “eye_feature_smooth” data:
- Shape: 31 * W, the maximum of W = 64
- It is a flatten feature

### Network design
Because the eye feature is a kind of motion feature, it doesn't on a same kind of EEG feature so I
decide to use a multimodal network: one has input is the EEG feature, the other has input is Eye
feature.
https://github.com/truongpl/hometest/blob/main/assets/HT0_ND0.png
![Network illustrate](https://github.com/truongpl/hometest/blob/main/assets/HT0_ND0.png)
![Network Parameters](https://github.com/truongpl/hometest/blob/main/assets/HT0_ND1.png)

### Experimental
I create a train and test set with ratio = 0.2, implement a data generator to spawn the data for
training. For training set, I train with Kfold = 4, achieve average accuracy across fold = 81%. After
CV is finished ,I retrain the network with full train set, using below setup:
- Learning rate = 0.001
- Early stopping = 5 epochs if val_loss doesn't decrease
- Save best model into “best_model.h5”
- Final accuracy on test set is: F1-score: 0.8485 - Precision: 0.8559 - Recall: 0.8516
In addition, I also try a SVM with GridSearchCV but the result is not good, it can only achieve the
score = 26%.
### Improvements:
- I think about doing some PCA/LDA method before feeding the whole feature set into the network
because there are some high correlations features on EEG. This may reduce the training time.
- Because this is a classification problem, I think we can apply a technique like triplet embedding of
face recognition to boost the accuracy.


## System specifications and setup
### System specifications:
I am familiar with my PC so I run the test on my local machine with below specifications:
- OS: Ubuntu 20.04.05 LTS
- CPU: I5-8400
- RAM: 32GB DDR4 2400 Mhz
- GPU: Tesla K80 12GB
- NVIDIA Drivers: Driver 460.160 Server – Cuda Driver 11.2 – CuDNN 8.1
### Setup:
#### Working tree of submission:

````bash
├── data
├── Problem1
└── Problem2
├── modules
└── output
```` 


- Install package in the submission folder: pip install -r requirements.txt
- Symbolic link the dataset into submission folder: `ln -s [path_to_dataset] data`
- The solution of Problem 1 is written on Jupyter notebook. The solution of Problem 2 is written by Python code.
#### Running:
- Run solution 1: jupyter notebook and execute each code blocks like Google Colab
- Run solution 2: cd Problem2; python train.py

#### Troubleshoot:
If you have multiple GPU on machine, you need to run “export CUDA_VISIBLE_DEVICES=0” before training the model for Problem 2.
