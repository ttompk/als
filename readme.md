# Predict length of survival of ALS subjects

Amyotrophic Lateral Sclerosis - aka ALS, Lou Gehrig’s Disease, Motor Neuron Disease   

## Project Overview  

Using machine learning, a predictive model was built to determine the length of time from onset of disease until death. The model can predict the day of death with an abolute error of 149 days and an R-squared of 0.75. The model requires information from the onset of disease and functional assessments taken since onset. As a result, the model cannot predict the length of survival at the time of diagnosis unless a functional assessment has been made.

### ALS

Amyotrophic lateral sclerosis (ALS), also known as motor neurone disease (MND) or Lou Gehrig's disease, is a fatal disease affecting the motor neurons of the brain and spinal cord.  Motor neurons are responsible for controling body movements including such movements as walking, talking, eating, and breathing. Progressive degeneration of the motor neurons in ALS eventually leads to neural necrosis. When the motor neurons die, the ability of the brain to initiate and control muscle movement is permanently lost.  

Each year approximately 5,600 American's are diagnosed with ALS. Currently, there is **NO CURE FOR ALS**.
                
This project sought to predict the length of survival of ALS subjects. Death from onset of disease is rapid but also heterogeneous, with 50% dying within three years of symptom onset, 75% within the first five years, and 90% within 10 years. 

Data was provided by the PRO-ACT ALS organization. The dataset contained data on over 10,000 clinical trial participants, and included ... However, few values were available across multiple domains for each subject.  
                
ALS is a terminal, progressive disease. Most ALS subjects die as a result of resperatory failure. Disease progression is measured using validated functional assessments including the ALSFRS, Forced Vital Capacity, and Slow Vital Capacity.  

This project used data collected from over 25 ALS clinical trials - all trials failed to show improvement in the treatment group!
 

### Data

Data for this project included information from over 10,000 clinical trials subjects. Despite this number, few subjects had data across all features, and thus the overall number of subjects available for analysis was significantly more limited.

See this document for more information on the data collection and curation. [Data Dictionary](https://nctu.partners.org/ProACT/Document/DisplayLatest/2)


### Model Features
The model utilizes the following features:

1. ALSFRS Functional Assessment Questions:  The ALSFRS is a clinically-validated tool to assess neuron function over the course of ALS. The test measures the ability of subjects to perform specific tasks. To utilize this important feature, the slope of an onset anchored linear model of the assessment scores over time was utilized. 
- Q1_Speech, Q2_Salivation, Q3_Swallowing', Q4_Handwriting', slope_Q6_Dressing_and_Hygiene', slope_Q7_Turning_in_Bed', slope_Q8_Walking', Q9_Climbing_Stairs', Q10_Respiratory', updated_ALSFRS_Total',  

features = 'Subject_used_Riluzole', 'Race_Caucasian','age_at_onset',
            'symptom_weakness','loc_spinal','loc_speech_or_mouth'

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

### Diease Onset Location

ALS can affect different neuron groups at the onset of disease. Generally, the location of neuronal involvment includes the include either the which includes the distal limbs 

### Death
