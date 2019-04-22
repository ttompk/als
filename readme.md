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

All data was provided with permission by the PRO-ACT ALS organization. 

Data for this project included information from over 10,000 clinical trials subjects who participated in over 20 clincial trials. The dataset included both treated and placebo subjects, however no information on the drug, e.g. name, dose, route of administration, was provided. Importantly, all trials failed to show a significant difference between treatment and placebo groups.

Despite the large number of large subjects, few had data across all features, and thus the overall number of subjects available for modeling was significantly more limited.

See this document for more information on the data collection and curation. [Data Dictionary](https://nctu.partners.org/ProACT/Document/DisplayLatest/2)


### Features

The dataset contained multiple features, including: laboratory data, forced vital capcity (FVC), slow vital capcity (SVC), ALSFRS scores (functional assessments), demographic data (age, sex, weight, etc), symptoms at onset, family history, and others. These values were either subject level, one value per subject (e.g. symptoms, demographic), or repeated measures (e.g. ALSFRS, FVC, labs). 

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
- Q1: speech
- Q2: salivation
- Q3: swallowing
- Q4: handwriting
- Q5: cutting food and handling utensils (with or without gastrostomy)
- Q6: dressing and hygiene
- Q7: turning in bed and adjusting bed clothes
- Q8: walking
- Q9: climbing stairs
- Q10: respiratory (replaced with revised scale R1: , R2: , R3: )

Scores from individual questions are summed to create a 'Total' score ranging from 40, maximal ordinary function, to 0, no function at all. 

Some subjects had a revised functional assessment score. The revised scale replaced question 10, respiratory, with three additional questions. To maximize the number of subjects in the analysis, the first of the three revised scores, R1, was used to represent Q10 from the unrevised scale, thereby maintaining a total score of 40.

These data where collected over several time points. To use the ALSFRS scores in a model, the repeated measures had to be distilled to one value (or 2 in case of a ploynomial fit) for each subject. This value would then be used in the model to predict survival. As suggested by (Karanevich et al)[https://www.ncbi.nlm.nih.gov/pubmed/29409450] an additional maximal score of 4 for each question (40 total) was added for each subject at the time of disease onset. An assumption was made that each subject was completely functional prior to disease symptoms.

![alt text](https://github.com/ttompk/als/blob/master/images/alsfrs_start.png)

To detect rate of decreasing function, a linear regression line was fitted to each subject's indivual ALSFRS responses and the combined score. The onset-anchored point was critical to stabilizing the slope of the linear model for each subject. Additional fits using multi-degree polynomials were also fit using the first and second derivatives as features in the model. These polynomial fits generally provided worse fitting of the model and were rejected for incorporation in the final model.

### Diease Onset Location

ALS can affect different neuron groups at the onset of disease. Generally, the location of neuronal involvement at onset is mapped to neuron clusters in either the bulbar or spinal regions. Bulbar neurons are located in the head region and control such movements as speech and swallowing. The spinal region nerons control the distal limbs. There was evidence in the literature that bulbar involvelment at onset was associated with worsening disease. [reference] Indeed there was some signal suggesting this but its impact in predicting survival in the model presented here was negligible.

The location data in the original dataset was messy. Most text strings contained paraphrased, abbreviated, or misspelled words. After cleaning, the values were one-hot encoded into several categories and evaluated in the model.

### Death

Death, the target variable, was present for only those subjects whom died during a clinical trial. The fact that a majority of subjects were lost to followup limited the amount of data available for the analysis.


### 





## The model
Several models were evaluated. 

## Surpises

Approximatly 90% of ALS subjects ultimatly die from respiratory failure. Interestingly, respiratory functional test data (FVC and SVC) were not predictors of length of survival. Literature from research studies appear to support this finding. Hopefully, a better respiratory test can be developed to test respiration forces in ALS subjects which does not involve the confounding confounding factor of subject volition.

