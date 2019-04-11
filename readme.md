# Predicting Length of Survival of ALS Subjects

Amyotrophic Lateral Sclerosis - aka ALS, Lou Gehrig’s Disease, Motor Neuron Disease   

## Project Overview  

Using machine learning, several predictive models were built to determine the length of time from onset of disease until death in ALS subjects. The final model was a random forest regression alogorithm that can predict the day of death with an abolute error of 151 days, an R-squared of 0.73, and median absolute percentage error of 12%. The model takes in information about the onset of disease, demographic data, and repeated functional assessment scores taken since onset. An interactive html tool allows a user to enter a subject's ALS data and retrieve a prediction.

### ALS

Amyotrophic lateral sclerosis (ALS), also known as motor neurone disease (MND) or Lou Gehrig's disease, is a fatal disease affecting the motor neurons of the brain and spinal cord.  Motor neurons are responsible for controling body movements including such movements as walking, talking, eating, and breathing. Progressive degeneration of the motor neurons in ALS eventually leads to neural necrosis. When the motor neurons die, the ability of the brain to initiate and control muscle movement is permanently lost.  

ALS is a terminal, progressive disease. Each year approximately 5,600 American's are diagnosed with ALS. Currently, there is **NO CURE FOR ALS**.
        
This project sought to predict the length of survival of ALS subjects. Death from onset of disease is rapid but also heterogeneous, with 50% dying within three years of symptom onset, 75% within the first five years, and 90% within 10 years. Most ALS subjects die as a result of resperatory failure. 

Disease progression is measured using validated functional assessments including the ALSFRS, Forced Vital Capacity, and Slow Vital Capacity, and clinical laboratory results.  
 

### Data


Data for this project included information from over 10,000 clinical trials subjects. Despite this number, few subjects had data across all features, and thus the overall number of subjects available for analysis was significantly more limited.

All data was provided by the PRO-ACT ALS organization. The dataset contained data on over 10,000 clinical trial participants from over 20 clinical trials. All trails failed to show a significant difference between treatment and placebo groups.

See this document for more information on the data collection and curation. [Data Dictionary](https://nctu.partners.org/ProACT/Document/DisplayLatest/2)


### Model Features

The dataset contained multiple features, including: laboratory data, FVC, SVC, ALSFRS scores (functional assessments), demographic data (age, sex, weight, etc), symptoms at onset, family history, and others. These values were either subject level, one value per subject (symptoms, demographic), or repeated measures (e.g. ALSFRS, FVC).

As expected in a study spaning multiple studies, not all assessments were performed in every study and studies with similar assessments did not always perform testing on the same schedule. 

The model utilizes the following features:

1. ALSFRS Functional Assessments (10 questions)
2. Age at Onset
3. Subject used Riluzole
4. Whether race was Caucasian
5. First symptom was weakness
6. Body location where onset occured



## Feature Details
#### The Amyotrophic Lateral Sclerosis Functional Rating Scale (ALSFRS)

Overview:
The Amyotrophic Lateral Sclerosis Functional Rating Scale (ALSFRS) is an instrument for evaluating the functional status of patients with Amyotrophic Lateral Sclerosis. It can be used to monitor functional change in a patient over time.

Measures:
- speech
- salivation
- swallowing
- handwriting
- cutting food and handling utensils (with or without gastrostomy)
- dressing and hygiene
- turning in bed and adjusting bed clothes
- walking
- climbing stairs
- respiratory

These data where collected over several time points. To use them in the model I determined an anchor-onset linear model for each subject. I used the slope of this line in the model. I also fit the data with a 2nd-degree polynomial, as there appeared to be curvature in the data. Using the first and second derivatives as features in the model did not improve the model perfomance. 

### Diease Onset Location

ALS can affect different neuron groups at the onset of disease. Generally, the location of neuronal involvement is mapped to neuron clusters in either the bulbar or spinal regions. Bulbar neurons are located in the head region and control such movements as speech and swallowing. The spinal region nerons control the distal limbs. There was evidence in the literature that bulbar involvelment at onset was associated with worsening disease. Indeed there was some signal suggesting this but its impact in predicting survival in the model presented here was negligible.

The location data in the original dataset was messy. Most text strings contained paraphrased, abbreviated, or misspelled words. After cleaning the values were once-hot encoded into several categories.

### Death

Death, the target variable, was present for only those subjects whom died during a clinical trial.


## Feature Engineering
### 

## The model
Several models were evaluated. 

## Surpises

Approximatly 90% of ALS subjects ultimatly die from respiratory failure. Interestingly, respiratory functional test data (FVC and SVC) were not predictors of length of survival. Literature from research studies appear to support this finding. Hopefully, a better respiratory test can be developed to test respiration forces in ALS subjects which does not involve the confounding confounding factor of subject volition.

