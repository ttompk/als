# Predicting Length of Survival of ALS Subjects

Amyotrophic Lateral Sclerosis - aka ALS, Lou Gehrig’s Disease, Motor Neuron Disease   

## Project Overview  

Using machine learning, a predictive model was built to determine the length of time from onset of disease until death in ALS subjects. The prediction tool was designed to aid in subject selection for ALS clinical trials. Model features include symptoms and neuron set affected at onset of disease, subject demographic data, and ALS functional assessment scores. The final model utilized a random forest regression alogorithm to predict the day of death with an abolute error of 151 days, an R-squared of 0.73, and median absolute percentage error of 12%.  An interactive html tool was built to allow clinical teams to enter a subject's ALS data and retrieve a prediction.

![website_demo](https://github.com/ttompk/als/blob/master/images/als_website_demo.gif)


### What is ALS

![als_man](https://github.com/ttompk/als/blob/master/images/als_man.png)  

Amyotrophic lateral sclerosis (ALS), also known as motor neurone disease (MND) or Lou Gehrig's disease, is a fatal disease affecting the motor neurons of the brain and spinal cord.  Motor neurons are responsible for controling body movements including walking, talking, eating, and breathing. Progressive degeneration of the motor neurons in ALS eventually leads to neural necrosis. When the motor neurons die, the ability of the brain to initiate and control muscle movement is permanently lost.  

ALS is a terminal, progressive disease. Each year approximately 5,600 Americans are diagnosed with ALS. Currently, there is **NO CURE FOR ALS**.
        
This project sought to predict the length of survival of ALS subjects. Death from onset of disease is rapid but also heterogeneous, with 50% dying within three years of symptom onset, 75% within the first five years, and 90% within 10 years. Most ALS subjects die as a result of respiratory failure. 

Disease progression is measured using validated functional assessments including the ALSFRS, Forced Vital Capacity, and Slow Vital Capacity, and clinical laboratory results.  
 

### Data

All data was provided with permission by the PRO-ACT ALS organization. 

![pro-act-logo](https://github.com/ttompk/als/blob/master/images/Pro-Act-logo.gif)  

Data for this project included information from over 10,000 clinical trials subjects who participated in over 20 clincial trials. The dataset included both treated and placebo subjects, however no information on the drug, e.g. name, dose, route of administration, was provided. Importantly, all trials in the dataset failed to show a significant difference between treatment and placebo groups.

Despite the large number of subjects few had data across all features and thus the overall number of subjects available for modeling was significantly more limited.

See this document for more information on the data collection and curation. [Data Dictionary](https://nctu.partners.org/ProACT/Document/DisplayLatest/2)


### Features

The dataset contained multiple features, including: laboratory data, forced vital capcity (FVC), slow vital capcity (SVC), ALSFRS scores (functional assessments), demographic data (age, sex, weight, etc), symptoms at onset, family history, and others. These values were either subject level, one value per subject (e.g. symptoms, demographic), or repeated measures (e.g. ALSFRS, FVC, labs). 

As expected in a study spaning multiple studies, not all assessments were performed in every study and studies with similar assessments did not always perform testing on the same schedule. Therefore several strategies were evaluated to enable data from repeated measures to be incorporated into the model.

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
- Q10: respiratory (replaced with revised scale R1: dyspnea, R2: orthopnea, R3: respiratory insufficiency)

Scores from individual questions are summed to create a 'Total' score ranging from 40, maximal ordinary function, to 0, no function at all. 

Some subjects had a revised functional assessment score. The revised scale replaced question 10, respiratory, with three additional questions. To maximize the number of subjects in the analysis, the first of the three revised scores, R1, was used to represent Q10 from the unrevised scale, thereby maintaining a total score of 40.

These data where collected over several time points. To use the ALSFRS scores in the model, the repeated measures had to be distilled to one value (or 2 in case of a polynomial fit) for each subject. This value would then be used in the model to predict survival. 

To detect rate of decreasing function, a linear regression line was fitted to each subject's indivual ALSFRS responses and the total combined score. As suggested by [Karanevich *et al*](https://www.ncbi.nlm.nih.gov/pubmed/29409450), an additional maximal score of 4 for each question (40 points total) was added for each subject at the time of disease onset. This onset-anchored point was critical to stabilizing the slope of the linear model for each subject. Usage of this point required the assumption that each subject was completely functional in all areas prior to disease symptoms.

![pre_mod_alsfrs](https://github.com/ttompk/als/blob/master/images/alsfrs_start.png)

![post_mod_alsfrs](https://github.com/ttompk/als/blob/master/images/alsfrs_onset_slope.png)

Additional fits using multi-degree polynomials were also fit using the first and second derivatives as features in the model. These polynomial fits generally provided worse fitting and were excluded from the final set of features.

### Forced Vital Capacity (FVC) and Slow Vital Capatcity (SVC)

A similar approach to ALSFRS scores, with or without onset-anchoring, was performed for FVC values. A linear regression line was fitted to each subject's respiratory function values and the corresponding slope was evaluated during model selection.

### Diease Onset Location

ALS can affect different neuron groups at the onset of disease. Generally, the location of neuronal involvement at onset is mapped to neuron clusters in either the bulbar or spinal regions. Bulbar neurons are located in the head region and control such movements as speech and swallowing. The spinal region nerons control the distal limbs. There was evidence in the literature, [Knibb *et al*](https://jnnp.bmj.com/content/jnnp/87/12/1361.full.pdf), that bulbar involvement at onset was associated with worsening disease. Indeed there was some signal suggesting this but its impact in predicting survival in the model presented here was negligible compared to the importance of the ALSFRS scores.

The location data in the original dataset included many misspelled, abbreviated, paraphrased, and truncated word forms. After cleaning and mapping, the values were one-hot encoded into several categories and evaluated in the model.

### Death

Death, the target variable, was present for only those subjects whom died during a clinical trial. The fact that the day of death of a majority of subjects were lost to follow up limited the amount of data available for the analysis.

### Demographic Features

The dataset contained subject demographic data including age, sex, weight, and race. Also present was the subject's family history of ALS. All of these features were one-hot encoded (if needed) and evaluated during modeling.


## Workflow
Numerous permutations of features and modeling algorithms were evaluated using cross-validation and hyperparameter tuning. 

![workflow](https://github.com/ttompk/als/blob/master/images/workflow.png)  

Models evaluated: linear regression, random forest regression, gradient boosting regression

The random forest regression model performed the best after cross-validation and/or hyperparameter tuning. 

### Model Performance

Model performance was evaluated using residual plots, R<sup>2</sup>, mean absolute error, and median absolute percentage error.

Plot of predicted length of survival (days) vs actual length of survival (days)
![predict_plot](https://github.com/ttompk/als/blob/master/images/pred_plot.png)

The final model had the following test prediction error:
- Mean Absolute Error: **151 days**
- Median Absolute Percentage Error (MAPE):  **12%**  (10th Quart: 2.5%, 90th Quart: 34%).

#### Comparison to Other Models
As a comparison, if one predicted that all subjects survived exactly as long as the mean value, 956 days, the mean absolute error would be 308 days with a MAPE of 26.7%. [Knibb *et al*](https://jnnp.bmj.com/content/jnnp/87/12/1361.full.pdf) suggest that bulbar symptoms at disease onset is linked to reduced survival times. Even though the neuron group affected at onset is a feature in the model, the data was modeled using only subjects with bulbar neural group involvement at onset. This model performed better than the 'mean' only model but not as well as the final model. The bulbar-group model had a mean absolute error of 233 days and MAPE of 24.4%. 

### Residuals
Plot of prediction residuals:![resid_plot](https://github.com/ttompk/als/blob/master/images/resids.png)

As can be seen from the plot of residuals above, the prediction response did not have a continous varaiance across time, i.e. the response was heteroscedastic. This implies that the prediction error is greater the further away from onset a subject is predicted to live. This error is understandable given that the most important features are the slopes of the ALSFRS scores. This implies that the model is more accurate in detecting how long subjects will live among subjects with more negative slopes (those declining in function more rapidly) and less accurate at predicting those that live longer. As expected, small variations in slope among the longer lived subjects had a greater impact on the model's prediction error. 

Distribution of residuals:![resid_dist](https://github.com/ttompk/als/blob/master/images/resid_dist.png)
 

## Conclusion

The model provided here can predict length of survival in ALS with an error of approximately 6 months. The model out performs or is comparable to [pubished reports](https://jnnp.bmj.com/content/jnnp/87/12/1361.full.pdf). Importantly, the model presented here relies primarily on ongoing functional assessments for its predictive strength, whereas other survival models attempt to infer survival at the time of onset. Not exactly an apples to apples comparison. 

The model more accurately predicts shorter lifespans than longer, ie. fast progressors vs. slower progressors. Despite this shortcoming, the tool remains a practical choice for evaluating subjects for clinical trial participation. The lack of prediction strength years after onset is not as critical to subject selection when the objective is to determine candidates who meet a defined exclusion criteria, e.g. exclude subjects whose decline in function is too severe or alternatively whose decline is minimal. Both scenarios can be problematic for statistical analyses.

### Surpises

Approximately 90% of ALS subjects ultimately die from respiratory failure. Interestingly, respiratory functional test data (FVC and SVC) were not predictors of length of survival and where not included in the final model. Literature from research studies appear to contrast with these findings, [Mousavi *et al.*](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4240929/), [Schmidt *et al.*](https://onlinelibrary.wiley.com/doi/pdf/10.1002/mus.20450), [Wolf *et al.*](https://bmcneurol.biomedcentral.com/articles/10.1186/s12883-014-0197-9). These papers report grouping subjects by setting a threshold for pulmonary function, e.g. below 75% of percent normalized function. This is a good idea for further work. Additionaly, FVC measurements were taken across multiple studies which potentially introduced a source of variation. Presumably FVC measurements and neuromuscular function do not always correlate because the test requires subject volition to measure respiration forces. A less subjective measurement may enhance the predictive power of respiration forces in predicting length of survival.

