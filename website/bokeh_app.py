# ALS survival prediction 
'''
To run app, use this statement in the command line:

**
bokeh serve --show bokeh_app.py
**

HTML-based app written with bokeh and python. The file is hosted on a bokeh server.

Takes a pickled model and user inputed
values and performs a prediction for how long someone is likely to survive with 
ALS. 
 
Default values for two repeats of ALSFRS scores were included to show program's ability. 
These default values can be overwritten by actual values.
'''


# load dependency libraries
import pandas as pd
import numpy as np
from os.path import dirname, join
from datetime import date
from datetime import timedelta as td
from datetime import datetime as dt
from collections import defaultdict
from scipy.stats import linregress
import pickle

# load bokeh libraries and methods
from bokeh.io import output_file, show, curdoc
from bokeh.layouts import widgetbox, row, gridplot, layout
from bokeh.layouts import column as bokehcolumn
from bokeh.plotting import figure
from bokeh.models.widgets import TextInput, Paragraph, Button, MultiSelect, RadioGroup
from bokeh.models.widgets import DatePicker, Panel, Tabs, Div, RadioButtonGroup, Button
from bokeh.models.widgets import Select, DataTable, TableColumn
from bokeh.models import ColumnDataSource, DateFormatter, Span
from bokeh.palettes import Viridis3


### Set Initial State load files  ###
# set the intial state so that prediction will not run on start up.
pred_run=False

# load pickled model
filename = 'finalized_model.sav'
global model
model = pickle.load(open(filename, 'rb'))

# set current data variable
crnt_date=dt.now()

''' ---   function: linear regression of alsfrs  --- '''
# this function fits a linear regression line to each repeated question.

def linreg_scalers(df):
    '''
    creates linear regression slopes, intercepts for every alsfrs value in a df.
    input:
        df = df
        subject_col = name of subject id column
        y_col = name of first y data column
        x_col = name of column with x values
        y_label = name of y feature (for table output)
    output:   None
    returns:  slopes and intercepts as dataframe.
    '''
    
    slopes = defaultdict()
    # format df for analysis
    df.dropna(inplace=True)
    x=np.array(df['day_from_onset'].astype(float))
    
    df.drop(['date','test', 'day_from_onset', 'R2', 'R3'], axis=1, inplace=True)
    df.astype(float, inplace=True)
    
    # for each question
    for q in list(df.columns):
        y = np.array(df[q].astype(float))
        
        # linregress is part of scipy.stats. 
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        slopes[q] = (q, slope, intercept)

    # format as dataframe
    full_result = pd.DataFrame(slopes).transpose().reset_index(drop=True)
    full_result.rename(index=str, columns={"index": "subject_id", 
                                           0: 'question' ,
                                           1: 'slope', 
                                           2: 'intercept'}, inplace=True)
    full_result['slope'].astype(float)
    full_result['intercept'].astype(float)
    
    # return slopes and intercepts
    return full_result
    #end of function

# bokeh file
output_file("prediction_ui.html")

# test for paragraph at the top of the page. stored in HTML.
description = Div(text=open(join(dirname(__file__), "description.html")).read(), width=800)


''' --- Left side of page: Onset and demographic data --- '''
## Onset Panel
# date onset
def dt_date_onset_handler(attr, old, new):
    print(type(old))
    print(type(new))
    print('old was {} and new is {}'.format(old,new))
dt_date_onset=DatePicker(title="Date of ALS Onset", 
                         min_date=date(1990,1,1),
                         max_date=date.today(),
                         value=date(2015,3,1))
dt_date_onset.on_change("value", dt_date_onset_handler)

# subject id
def subject_id_handler(attr, old, new):
    print("Previous label: " + tpye(old))
    print("Updated label: " + new)
subject_id = TextInput(title="Subject ID:", value="ref number")
#subject_id.on_change("value", subject_id_handler)

# age at onset
def age_onset_handler(attr, old, new):
    print("Previous label: " + type(old))
    print("Updated label: " + new)
age_onset = TextInput(title="Age at ALS Onset (years)", value="")
#age_onset.on_change("value", age_onset_handler)

# onset location
def onset_loc_detail_handler(attr, old, new):
    print ('Radio button option ' + str(new) + ' selected.')
onset_loc_detail = MultiSelect(title="Location Detail (can pick multiple)", value=["unk"],
                           options=[("unk", "unknown"),("hands", "Hands"), 
                                    ("arms", "Arms"), ("feet", "Feet"), ("legs", "Legs")])
#onset_loc_detail.on_change("value", onset_loc_detail_handler)

# symptom selection
def symptom_handler(new):
    print ('Symptom selection ' + str(new) + ' selected.')
symptom = Select(title="Symptom Type:", value="Unknown", 
                 options=["Unknown", "Sensory changes", "Speech", 
                          "Stiffness", "Swallowing",
                         'Weakness', 'Atrophy', 'Cramps',
                         'Fasciculations', "Gait changes"])

# onset radio button
def onset_loc_handler(new):
    print ('Radio button option' + str(new) + ' selected.')
pp_onset = Paragraph(text="""Onset Neuron Group""", width=250, height=15)
onset_loc = RadioButtonGroup(labels=["Bulbar", "Spinal", "Both", "*Unk"], active=3)
#onset_loc.on_click(onset_loc_handler)

# riluzole radio button
def riluzole_handler(new):
    print ('Radio button option ' + str(new) + ' selected.')
pp_riluzole = Paragraph(text="""Subject used Riluzole""", width=250, height=15)
riluzole = RadioButtonGroup(labels=["No", "Yes", "*Unk"], active=2)
riluzole.on_click(onset_loc_handler)

# caucasian radio button
def caucasian_handler(new):
    print ('Radio button option ' + str(new) + ' selected.')
pp_caucasian = Paragraph(text="""Race Caucasian""", width=250, height=15)
caucasian = RadioButtonGroup(labels=["No", "Yes", "*Unk"], active=2) 
caucasian.on_click(caucasian_handler)

# sex radio button
def sex_handler(new):
    print ('Radio button option ' + str(new) + ' selected.')
pp_sex = Paragraph(text="""Sex""", width=250, height=15)
sex = RadioButtonGroup(labels=["Female", "Male", "*Unk"], active=2)
sex.on_click(sex_handler)


''' --- right side of page: ALSFRS scores --- '''
#### ALSFRS panels
# dates of ALS scores
dt_alsfrs_1=DatePicker(title='Date of Test 1: ', 
                       min_date=date(1990,1,1),
                       max_date=date.today(), 
                       value=date(2015,9,1),
                       width=100)

dt_alsfrs_2=DatePicker(title='Date of Test 2: ', 
                       min_date=date(1990,1,1),
                       max_date=date.today(), 
                       value=date(2016,3,1),
                       width=100)

dt_alsfrs_3=DatePicker(title='Date of Test 3: ', 
                        min_date=date(1990,1,1),
                        max_date=date.today(), 
                       value=date.today(),
                       width=100)

dt_alsfrs_4=DatePicker(title='Date of Test 4: ', 
                        min_date=date(1990,1,1),
                        max_date=date.today(), 
                       value=date.today(),
                       width=100)

dt_alsfrs_5=DatePicker(title='Date of Test 5: ', 
                        min_date=date(1990,1,1),
                        max_date=date.today(), 
                       value=date.today(),
                       width=100)

dt_alsfrs_6=DatePicker(title='Date of Test 6: ', 
                        min_date=date(1990,1,1),
                        max_date=date.today(), 
                       value=date.today(),
                       width=100)

# test boxes for alsfrs questions
wdbox=140
Q1_1 = TextInput(value="3", title="Q1 Speech:", width=wdbox)
Q1_2 = TextInput(value="3", title="Q1 Speech:", width=wdbox)
Q1_3 = TextInput(value="2", title="Q1 Speech:", width=wdbox)
Q1_4 = TextInput(value="", title="Q1 Speech:", width=wdbox)
Q1_5 = TextInput(value="", title="Q1 Speech:", width=wdbox)
Q1_6 = TextInput(value="", title="Q1 Speech:", width=wdbox)

Q2_1 = TextInput(value="3", title="Q2 Salivation:", width=wdbox)
Q2_2 = TextInput(value="3", title="Q2 Salivation:", width=wdbox)
Q2_3 = TextInput(value="2", title="Q2 Salivation:", width=wdbox)
Q2_4 = TextInput(value="", title="Q2 Salivation:", width=wdbox)
Q2_5 = TextInput(value="", title="Q2 Salivation:", width=wdbox)
Q2_6 = TextInput(value="", title="Q2 Salivation:", width=wdbox)

Q3_1 = TextInput(value="3", title="Q3 Swallowing:", width=wdbox)
Q3_2 = TextInput(value="2", title="Q3 Swallowing:", width=wdbox)
Q3_3 = TextInput(value="2", title="Q3 Swallowing:", width=wdbox)
Q3_4 = TextInput(value="", title="Q3 Swallowing:", width=wdbox)
Q3_5 = TextInput(value="", title="Q3 Swallowing:", width=wdbox)
Q3_6 = TextInput(value="", title="Q3 Swallowing:", width=wdbox)

Q4_1 = TextInput(value="3", title="Q4 Handwriting:", width=wdbox)
Q4_2 = TextInput(value="3", title="Q4 Handwriting:", width=wdbox)
Q4_3 = TextInput(value="1", title="Q4 Handwriting:", width=wdbox)
Q4_4 = TextInput(value="", title="Q4 Handwriting:", width=wdbox)
Q4_5 = TextInput(value="", title="Q4 Handwriting:", width=wdbox)
Q4_6 = TextInput(value="", title="Q4 Handwriting:", width=wdbox)

Q5_1 = TextInput(value="3", title="Q5 Cutting Food:", width=wdbox)
Q5_2 = TextInput(value="2", title="Q5 Cutting Food:", width=wdbox)
Q5_3 = TextInput(value="2", title="Q5 Cutting Food:", width=wdbox)
Q5_4 = TextInput(value="", title="Q5 Cutting Food:", width=wdbox)
Q5_5 = TextInput(value="", title="Q5 Cutting Food:", width=wdbox)
Q5_6 = TextInput(value="", title="Q5 Cutting Food:", width=wdbox)

Q6_1 = TextInput(value="3", title="Q6 Dressing:", width=wdbox)
Q6_2 = TextInput(value="3", title="Q6 Dressing:", width=wdbox)
Q6_3 = TextInput(value="3", title="Q6 Dressing:", width=wdbox)
Q6_4 = TextInput(value="", title="Q6 Dressing:", width=wdbox)
Q6_5 = TextInput(value="", title="Q6 Dressing:", width=wdbox)
Q6_6 = TextInput(value="", title="Q6 Dressing:", width=wdbox)

Q7_1 = TextInput(value="3", title="Q7 Turning in Bed:", width=wdbox)
Q7_2 = TextInput(value="2", title="Q7 Turning in Bed:", width=wdbox)
Q7_3 = TextInput(value="3", title="Q7 Turning in Bed:", width=wdbox)
Q7_4 = TextInput(value="", title="Q7 Turning in Bed:", width=wdbox)
Q7_5 = TextInput(value="", title="Q7 Turning in Bed:", width=wdbox)
Q7_6 = TextInput(value="", title="Q7 Turning in Bed:", width=wdbox)

Q8_1 = TextInput(value="3", title="Q8 Walking:", width=wdbox)
Q8_2 = TextInput(value="3", title="Q8 Walking:", width=wdbox)
Q8_3 = TextInput(value="2", title="Q8 Walking:", width=wdbox)
Q8_4 = TextInput(value="", title="Q8 Walking:", width=wdbox)
Q8_5 = TextInput(value="", title="Q8 Walking:", width=wdbox)
Q8_6 = TextInput(value="", title="Q8 Walking:", width=wdbox)

Q9_1 = TextInput(value="3", title="Q9 Climbing Stairs:", width=wdbox)
Q9_2 = TextInput(value="2", title="Q9 Climbing Stairs:", width=wdbox)
Q9_3 = TextInput(value="1", title="Q9 Climbing Stairs:", width=wdbox)
Q9_4 = TextInput(value="", title="Q9 Climbing Stairs:", width=wdbox)
Q9_5 = TextInput(value="", title="Q9 Climbing Stairs:", width=wdbox)
Q9_6 = TextInput(value="", title="Q9 Climbing Stairs:", width=wdbox)

R1_1 = TextInput(value="3", title="Q10 Respiratory/R1:", width=wdbox)
R1_2 = TextInput(value="3", title="Q10 Respiratory/R1:", width=wdbox)
R1_3 = TextInput(value="2", title="Q10 Respiratory/R1:", width=wdbox)
R1_4 = TextInput(value="", title="Q10 Respiratory/R1:", width=wdbox)
R1_5 = TextInput(value="", title="Q10 Respiratory/R1:", width=wdbox)
R1_6 = TextInput(value="", title="Q10 Respiratory/R1:", width=wdbox)

R2_1 = TextInput(value="3", title="R2", width=wdbox)
R2_2 = TextInput(value="2", title="R2:", width=wdbox)
R2_3 = TextInput(value="", title="R2:", width=wdbox)
R2_4 = TextInput(value="", title="R2:", width=wdbox)
R2_5 = TextInput(value="", title="R2:", width=wdbox)
R2_6 = TextInput(value="", title="R2:", width=wdbox)

R3_1 = TextInput(value="3", title="R3", width=wdbox)
R3_2 = TextInput(value="3", title="R3:", width=wdbox)
R3_3 = TextInput(value="", title="R3:", width=wdbox)
R3_4 = TextInput(value="", title="R3:", width=wdbox)
R3_5 = TextInput(value="", title="R3:", width=wdbox)
R3_6 = TextInput(value="", title="R3:", width=wdbox)

# create dummys for spacing in tab layout
dumbdiv1a = Div()
dumbdiv2a = Div()
dumbdiv3a = Div()
dumbdiv4a = Div()
dumbdiv5a = Div()
dumbdiv6a = Div()


#### arrange ALSFRS boxes in a 'tab'

wd = 200
l1a = row(widgetbox(Q1_1, Q2_1, Q3_1, Q4_1, width=wd), 
          widgetbox(Q5_1, Q6_1, Q7_1, Q8_1, width=wd), 
          widgetbox(Q9_1, R1_1, R2_1, R3_1, width=wd))
l1 = layout([dt_alsfrs_1], l1a, sizing_mode='scale_width')

l2a = row(widgetbox(Q1_2, Q2_2, Q3_2, Q4_2, width=wd), 
          widgetbox(Q5_2, Q6_2, Q7_2, Q8_2, width=wd), 
          widgetbox(Q9_2, R1_2, R2_2, R3_2, width=wd))
l2 = layout([dt_alsfrs_2], [dumbdiv2a], l2a, sizing_mode='scale_width')

l3a = row(widgetbox(Q1_3, Q2_3, Q3_3, Q4_3, width=wd), 
          widgetbox(Q5_3, Q6_3, Q7_3, Q8_3, width=wd), 
          widgetbox(Q9_3, R1_3, R2_3, R3_3, width=wd))
l3 = layout([dt_alsfrs_3], [dumbdiv3a], l3a, sizing_mode='scale_width')

l4a = row(widgetbox(Q1_4, Q2_4, Q3_4, Q4_4, width=wd), 
          widgetbox(Q5_4, Q6_4, Q7_4, Q8_4, width=wd), 
          widgetbox(Q9_4, R1_4, R2_4, R3_4, width=wd))
l4 = layout([dt_alsfrs_4], [dumbdiv4a], l4a, sizing_mode='scale_width')

l5a = row(widgetbox(Q1_5, Q2_5, Q3_5, Q4_5, width=wd), 
          widgetbox(Q5_5, Q6_5, Q7_5, Q8_5, width=wd), 
          widgetbox(Q9_5, R1_5, R2_5, R3_5, width=wd))
l5 = layout([dt_alsfrs_5], [dumbdiv5a], l5a, sizing_mode='scale_width')

l6a = row(widgetbox(Q1_6, Q2_6, Q3_6, Q4_6, width=wd), 
          widgetbox(Q5_6, Q6_6, Q7_6, Q8_6, width=wd), 
          widgetbox(Q9_6, R1_6, R2_6, R3_6, width=wd))
l6 = layout([dt_alsfrs_6], [dumbdiv6a], l6a, sizing_mode='scale_width')

tab1 = Panel(child= l1, title="ALSFRS-1")
tab2 = Panel(child= l2, title="ALSFRS-2")
tab3 = Panel(child= l3, title="ALSFRS-3")
tab4 = Panel(child= l4, title="ALSFRS-4")
tab5 = Panel(child= l5, title="ALSFRS-5")
tab6 = Panel(child= l6, title="ALSFRS-6")

tabs = Tabs(tabs=[ tab1, tab2, tab3, tab4, tab5, tab6], width = 800)


''' --- DataFrames for updated scores  --- '''
### alsfrs tables

# make df of input values
def create_alsfrs_data():
    # collapse text box inputs into one list
    alsfrs_entries = {'test': ['onset', 'test1', 'test2', 'test3', 'test4', 'test5', 'test6'], 
                      'date': [dt_date_onset.value, dt_alsfrs_1.value, dt_alsfrs_2.value,
                              dt_alsfrs_3.value, dt_alsfrs_4.value, dt_alsfrs_5.value, dt_alsfrs_6.value],
                      'day_from_onset':[0, "","","","","",""], 
                      'Q1':[4, Q1_1.value, Q1_2.value, Q1_3.value, Q1_4.value, Q1_5.value, Q1_6.value], 
                      'Q2':[4, Q2_1.value, Q2_2.value, Q2_3.value, Q2_4.value, Q2_5.value, Q2_6.value],
                      'Q3':[4, Q3_1.value, Q3_2.value, Q3_3.value, Q3_4.value, Q3_5.value, Q3_6.value],
                      'Q4':[4, Q4_1.value, Q4_2.value, Q4_3.value, Q4_4.value, Q4_5.value, Q4_6.value],
                      'Q5':[4, Q5_1.value, Q5_2.value, Q5_3.value, Q5_4.value, Q5_5.value, Q5_6.value],
                      'Q6':[4, Q6_1.value, Q6_2.value, Q6_3.value, Q6_4.value, Q6_5.value, Q6_6.value],
                      'Q7':[4, Q7_1.value, Q7_2.value, Q7_3.value, Q7_4.value, Q7_5.value, Q7_6.value],
                      'Q8':[4, Q8_1.value, Q8_2.value, Q8_3.value, Q8_4.value, Q8_5.value, Q8_6.value],
                      'Q9':[4, Q9_1.value, Q9_2.value, Q9_3.value, Q9_4.value, Q9_5.value, Q9_6.value],
                      'Q10_R1':[4, R1_1.value, R1_2.value, R1_3.value, R1_4.value, R1_5.value, R1_6.value],
                      'R2':[4, R2_1.value, R2_2.value, R2_3.value, R2_4.value, R2_5.value, R2_6.value],
                      'R3':[4, R3_1.value, R3_2.value, R3_3.value, R3_4.value, R3_5.value, R3_6.value]}
    
    alsfrs_df = pd.DataFrame(alsfrs_entries)
    for c in list(alsfrs_df.columns):
        alsfrs_df[c] = alsfrs_df[c].where(alsfrs_df[c]!="", np.NaN)
        alsfrs_df[c] = alsfrs_df[c].where(alsfrs_df[c]!="(depreciated)", np.NaN)
    
    # calculate date differences
    def days_from_onset_func(val):
        if val==None:
            return np.NaN
        else:
            return (val - dt_date_onset.value).days
    
    # apply date fucntion to alsfrs data
    alsfrs_df['day_from_onset'] = alsfrs_df['date'].apply(days_from_onset_func)
    alsfrs_df.dropna(inplace=True)
    
    # add a total row (combined score)
    col_list= list(alsfrs_df.columns)
    
    # clean up dataframe
    col_list.remove('day_from_onset')
    col_list.remove('date')
    col_list.remove('test')
    alsfrs_df[col_list] = alsfrs_df[col_list].apply(pd.to_numeric, errors='coerce')
    col_list.remove('R3')
    col_list.remove('R2')
    alsfrs_df['ALSFRS_Total'] = alsfrs_df[col_list].sum(axis=1).astype(float)
    
    return alsfrs_df
    # end function

# instaniate alsfrs table
alsfrs_df = create_alsfrs_data()

# make a columndata source of the data table output for image updating
alsfrs_source = ColumnDataSource(alsfrs_df)


'''   --- Data for models ---   '''
### create model data

### load in baseline data on all modeling features or generic prediction
def create_base_data():
    '''
    The base model data contains the average or most common values from the features
    which were included in the model. This function oads the base model data file and 
    creates a ColumnDataSource for use by bokeh methods.
    This table will be updated with new data where present - therefore the model will
    still provide a result without all data being present.
    '''
    base = pd.read_csv("baseline_df.csv")
    global baseline_source
    baseline_source = ColumnDataSource(base)
    return base

base = create_base_data()


# update the baseline model with new data
def model_data_update(slope):
    '''
    Base data updated with form entries
    '''
    global base
    base = create_base_data()
    
    if age_onset.value != "":
        base['age_at_onset'] = age_onset.value
    else:
        base['age_at_onset'] = 57.514399
    
    if riluzole.active == 2:
        base['Subject_used_Riluzole'] = 1
    else:
        base['Subject_used_Riluzole'] = riluzole.active
    
    if  caucasian.active == 2:
        base['Race_Caucasian'] = 1
    else:
        base['Race_Caucasian'] = caucasian.active
    
    if symptom.value == "Weakness":
        base['symptom_weakness'] = 1
    else:
        base['symptom_weakness'] = 0
    
    if onset_loc.active == 3:
        base['loc_spinal'] = 1
        base['loc_speech_or_mouth'] = 0
    elif onset_loc.active == 2:
        base['loc_spinal'] = 1
        base['loc_speech_or_mouth'] = 1
    elif onset_loc.active == 1:
        base['loc_spinal'] = 1
        base['loc_speech_or_mouth'] = 0
    else:
        base['loc_spinal'] = 0
        base['loc_speech_or_mouth'] = 1
    
    # update with question slope values
    base['slope_Q1_Speech'] = float(slope['slope'].loc[slope['question']=='Q1'])
    base['slope_Q2_Salivation'] = float(slope['slope'].loc[slope['question']=='Q2'])
    base['slope_Q3_Swallowing'] = float(slope['slope'].loc[slope['question']=='Q3'])
    base['slope_Q4_Handwriting'] = float(slope['slope'].loc[slope['question']=='Q4'])
    base['slope_Q6_Dressing_and_Hygiene'] = float(slope['slope'].loc[slope['question']=='Q6'])
    base['slope_Q7_Turning_in_Bed'] = float(slope['slope'].loc[slope['question']=='Q7'])
    base['slope_Q8_Walking'] = float(slope['slope'].loc[slope['question']=='Q8'])
    base['slope_Q9_Climbing_Stairs'] = float(slope['slope'].loc[slope['question']=='Q9'])
    base['slope_Q10_Updated'] = float(slope['slope'].loc[slope['question']=='Q10_R1'])
    base['slope_updated_ALSFRS_Total'] = float(slope['slope'].loc[slope['question']=='ALSFRS_Total'])
    #end function

    
    
'''---  Prediction Button  ---'''

## button to run prediction 
def run_predict_button():
    '''
    When button is clicked:
        1. runs the alsfrs slope function on new data
        2. updates the model data with form data
        3. runs prediction on the updated model data
        4. updates the distribution graph
    This button does not return anything
    '''
    global prediction
    global pred_run
    global data_table
    pred_run=True
    alsfrs_df = create_alsfrs_data()
    update_data = None
    data_table.source.data.update(alsfrs_df)
    slope = linreg_scalers( alsfrs_df )
    model_data_update(slope)
    plotting_slope_df = plotting_slope()
    
    # run prediction on pickle model
    prediction = model.predict(base)[0]
    predict_output_handler()
    print(prediction)
    print(type(prediction))
    vline.location = prediction
    pred_run=True
    #end function

predict_button = Button(label="Run Prediction", button_type="success")
predict_button.on_click(run_predict_button)

# update the text with predict button click
predict_output = Div(text='Current Survival Prediction (Base Model): ' + '</br>' + '1205.19 days' + '</br>', 
                     width=800, style={'font-size': '200%', 'color': 'black'}) #, height=height)
def predict_output_handler():
    predict_output.text = 'Updated Survival Prediction: ' + '</br>' + str(prediction) + '</br>'



'''---  Output  ---'''

### DataTable
#return alsfrs_source
#def alsfrs_source_cols():
alsfrs_source_columns = [
        TableColumn(field="test", title="Test Number"),
        TableColumn(field="date", title="Date", formatter=DateFormatter(format="%m/%d/%Y")),
        TableColumn(field="day_from_onset", title="Days from Onset"),
        TableColumn(field="Q1", title="Q1"),
        TableColumn(field="Q2", title="Q2"),
        TableColumn(field="Q3", title="Q3"),
        TableColumn(field="Q4", title="Q4"),
        TableColumn(field="Q5", title="Q5"),
        TableColumn(field="Q6", title="Q6"),
        TableColumn(field="Q7", title="Q7"),
        TableColumn(field="Q8", title="Q8"),
        TableColumn(field="Q9", title="Q9"),
        TableColumn(field="Q10_R1", title="Q10_R1"),
        TableColumn(field="R2", title="R2"),
        TableColumn(field="R3", title="R3") ]

data_table = DataTable(source=alsfrs_source, columns=alsfrs_source_columns, 
                           width=1000, height=280)

pp_data_table = Paragraph(text="""ALSFRS Data Table""", width=250, height=15)


### FIGURES

def plotting_slope():
    '''
    this function not in operation. 
    '''
    xxx = create_alsfrs_data()['day_from_onset'].values
    yyy = create_alsfrs_data()['ALSFRS_Total'].values

    # determine best fit line
    par = np.polyfit(xxx, yyy, 1, full=True)
    slope=par[0][0]
    intercept=par[0][1]
    y_predicted = [slope*i + intercept  for i in xxx]
    plotting_slope_df = pd.DataFrame({'xxx': xxx, 'y_predicted': y_predicted}, columns=['xxx', 'y_predicted'])
    print(plotting_slope_df)
    return plotting_slope_df

# run plotting slope for first time.
plotting_slope_df = plotting_slope()
slope1_source = ColumnDataSource(plotting_slope_df)


def ALSFRS_plot():
    '''
    output: image of linear regression line for alsfrs total score over 
    repeated measures from alsfrs form
    '''
    ALSFRS_plot = figure( plot_width=400, plot_height=400, title="ALSFRS Total Score")
    #gg= alsfrs_source
    plotting_slope()
    # plot it bokeh style
    #ALSFRS_plot.circle(xxx,yyy)
    ALSFRS_plot.line('day_from_onset', 'ALSFRS_Total', source=alsfrs_source, line_width=2, line_color="gray")
    ALSFRS_plot.line('xxx', 'y_predicted', line_color='red', line_width=3, legend="slope", source=slope1_source)
    ALSFRS_plot.xaxis.axis_label = "Days From Onset"
    ALSFRS_plot.yaxis.axis_label = "ALSFRS Total Score"
    ALSFRS_plot.xgrid.visible = False
    ALSFRS_plot.ygrid.visible = False
    return ALSFRS_plot

## test figure
def this_awesome_plot():
    plot = figure(plot_width=400, plot_height=400)
    plot.multi_line([[1, 3, 2], [3, 4, 6, 6]], [[2, 1, 4], [4, 7, 8, 5]],
             color=["firebrick", "navy"], alpha=[0.8, 0.3], line_width=4)
    return plot

# test figure
def testplot1():
    '''
    This function not in operation. Example of a plot - for further exploration.
    '''
    x = list(range(11))
    y0 = x
    y1 = [10 - i for i in x]
    y2 = [abs(i - 5) for i in x]

    # create three plots
    p1 = figure(plot_width=250, plot_height=250, title=None)
    p1.circle(x, y0, size=10, color=Viridis3[0])
    return p1

def make_hist_plot(title, hist, edges, x, pdf):
    '''
    plots the distribution of death days from onset from teh training data 
    with a red line indicating predicted day of death from the user inputed data.
    '''
    p = figure(title=title, tools='', plot_width=400, plot_height=400, x_range=(-50, 3500))
    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
           fill_color="#1C2833", line_color="white", alpha=0.6)
    if pred_run == True:
        death_day=prediction
    else: 
        death_day=1205
    global vline
    vline = Span(location=death_day, dimension='height', line_color='red', line_width=4)
    p.renderers.extend([vline])
    p.line(x, pdf, line_color="gray", line_width=4, alpha=0.7, legend="PDF")
    p.y_range.start = 0
    p.legend.location = "center_right"
    p.legend.background_fill_color = "#F8F9F9"
    p.xaxis.axis_label = 'days'
    p.yaxis.axis_label = 'subjects'
    p.grid.grid_line_color="white"
    return p


# Death Distribution
def death_dist():
    ''' 
    loads distribution data for plot of day of death
    sets distribution values
    '''
    died_days = pd.read_csv("death_days.csv")
    mu = 958
    sigma = 442
    measured = np.random.normal(mu, sigma, 1000)
    hist, edges = np.histogram(died_days['death_day_since_onset'], density=True, bins=50)
    x = np.linspace(0, 1800, 1000)
    pdf = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2 / (2*sigma**2))
    return make_hist_plot("Day of Death Distribution", hist, edges, x, pdf)


''' ---  Output Display in bokeh server  --- '''

grid=gridplot([[ALSFRS_plot(), death_dist()], [None, None ]])
    
# put all show/hide elements in a group
show_output = layout( [predict_output], [pp_data_table], [grid], [data_table])


### Display
curdoc().title = "ALS_Predict"
curdoc().add_root( layout([description], row(
    widgetbox(subject_id, dt_date_onset, age_onset, pp_onset, 
                onset_loc, onset_loc_detail, symptom, pp_riluzole, riluzole, 
                pp_caucasian, caucasian, pp_sex, sex,
                width=300),
                     tabs), 
                          [predict_button], 
                          show_output))