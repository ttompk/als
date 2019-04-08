#bokeh_app
import pandas as pd
import numpy as np
from os.path import dirname, join
from datetime import date
from datetime import timedelta as td
from datetime import datetime as dt
import pickle

from bokeh.io import output_file, show, curdoc
#from bokeh.plotting import curdoc
from bokeh.layouts import widgetbox, row, gridplot, layout
from bokeh.layouts import column as bokehcolumn
from bokeh.plotting import figure
from bokeh.models.widgets import TextInput, Paragraph, Button, MultiSelect, RadioGroup
from bokeh.models.widgets import DatePicker, Panel, Tabs, Div, RadioButtonGroup, Button
from bokeh.models.widgets import Select, DataTable, TableColumn

from bokeh.models import ColumnDataSource

from bokeh.palettes import Viridis3


### load in baseline data on all modeling features or generic prediction
baseline_source = ColumnDataSource(pd.read_csv("baseline_df.csv"))  
# create the alsfrs 
alsfrs_entries = {'day':[0], 'Q1':[4], 'Q2':[4], 'Q3':[4], 'Q4':[4], 'Q5':[4], 'Q6':[4], 'Q7':[4], 
               'Q8':[4], 'Q9':[4], 'Q10':[4], 'R1':[4], 'R2':[4], 'R3':[4]}
alsfrs_data = pd.DataFrame(alsfrs_entries)
alsfrs_source = ColumnDataSource(alsfrs_data)

# load pickled model
filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))


output_file("prediction_ui.html")

# the top of the page
description = Div(text=open(join(dirname(__file__), "description.html")).read(), width=800)
# End of button example  



## Onset Panel
# date onset
def dt_date_onset_handler(attr, old, new):
    print("Previous label: " + old)
    print("Updated label: " + new)
dt_date_onset=DatePicker(title="Date of ALS Onset", 
                                       min_date=date(1990,1,1),
                                       max_date=date.today())
dt_date_onset.on_change("value", dt_date_onset_handler)

# subject id
def subject_id_handler(attr, old, new):
    print("Previous label: " + old)
    print("Updated label: " + new)
subject_id = TextInput(title="Subject ID:", value="ref number")
subject_id.on_change("value", subject_id_handler)

# age at onset
def age_onset_handler(attr, old, new):
    print("Previous label: " + old)
    print("Updated label: " + new)
age_onset = TextInput(title="Age at ALS Onset (years)", value="")
age_onset.on_change("value", age_onset_handler)

# onset location
def onset_loc_detail_handler(attr, old, new):
    print ('Radio button option ' + str(new) + ' selected.')
onset_loc_detail = MultiSelect(title="Location Detail (can pick multiple)", value=["unk"],
                           options=[("unk", "unknown"),("hands", "Hands"), 
                                    ("arms", "Arms"), ("feet", "Feet"), ("legs", "Legs")])
onset_loc_detail.on_change("value", onset_loc_detail_handler)


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
onset_loc.on_click(onset_loc_handler)

# riluzole radio button
def riluzole_handler(new):
    print ('Radio button option ' + str(new) + ' selected.')
    
pp_riluzole = Paragraph(text="""Subject used Riluzole""", width=250, height=15)
riluzole = RadioButtonGroup(labels=["Yes", "No", "*Unk"], active=2)
riluzole.on_click(onset_loc_handler)

# caucasian radio button
def caucasian_handler(new):
    print ('Radio button option ' + str(new) + ' selected.')
    
pp_caucasian = Paragraph(text="""Race Caucasian""", width=250, height=15)
caucasian = RadioButtonGroup(labels=["Yes", "No", "*Unk"], active=2) 
caucasian.on_click(caucasian_handler)

# sex radio button
def sex_handler(new):
    print ('Radio button option ' + str(new) + ' selected.')
    
pp_sex = Paragraph(text="""Sex""", width=250, height=15)
sex = RadioButtonGroup(labels=["Male", "Female", "*Unk"], active=2)
sex.on_click(sex_handler)


dumbdiv = Div()


## ALSFRS panels
dt_alsfrs_1=DatePicker(title='Date of Test 1: ', 
                       min_date=date(1990,1,1),
                       max_date=date.today(), width=100)
dt_alsfrs_2=DatePicker(title='Date of Test 2: ', 
                       min_date=date(1990,1,1),
                        max_date=date.today(), width=100)
dt_alsfrs_3=DatePicker(title='Date of Test 3: ', 
                        min_date=date(1990,1,1),
                        max_date=date.today(), width=100)
dt_alsfrs_4=DatePicker(title='Date of Test 4: ', 
                        min_date=date(1990,1,1),
                        max_date=date.today(), width=100)
dt_alsfrs_5=DatePicker(title='Date of Test 5: ', 
                        min_date=date(1990,1,1),
                        max_date=date.today(), width=100)
dt_alsfrs_6=DatePicker(title='Date of Test 6: ', 
                        min_date=date(1990,1,1),
                        max_date=date.today(), width=100)
wdbox=140
Q1_1 = TextInput(value="3", title="Q1 Speech:", width=wdbox)
Q1_2 = TextInput(value="", title="Q1 Speech:", width=wdbox)
Q1_3 = TextInput(value="", title="Q1 Speech:", width=wdbox)
Q1_4 = TextInput(value="", title="Q1 Speech:", width=wdbox)
Q1_5 = TextInput(value="", title="Q1 Speech:", width=wdbox)
Q1_6 = TextInput(value="", title="Q1 Speech:", width=wdbox)

Q2_1 = TextInput(value="3", title="Q2 Salivation:", width=wdbox)
Q2_2 = TextInput(value="", title="Q2 Salivation:", width=wdbox)
Q2_3 = TextInput(value="", title="Q2 Salivation:", width=wdbox)
Q2_4 = TextInput(value="", title="Q2 Salivation:", width=wdbox)
Q2_5 = TextInput(value="", title="Q2 Salivation:", width=wdbox)
Q2_6 = TextInput(value="", title="Q2 Salivation:", width=wdbox)

Q3_1 = TextInput(value="3", title="Q3 Swallowing:", width=wdbox)
Q3_2 = TextInput(value="", title="Q3:", width=wdbox)
Q3_3 = TextInput(value="", title="Q3:", width=wdbox)
Q3_4 = TextInput(value="", title="Q3:", width=wdbox)
Q3_5 = TextInput(value="", title="Q3:", width=wdbox)
Q3_6 = TextInput(value="", title="Q3:", width=wdbox)

Q4_1 = TextInput(value="3", title="Q4 Handwriting:", width=wdbox)
Q4_2 = TextInput(value="", title="Q4:", width=wdbox)
Q4_3 = TextInput(value="", title="Q4:", width=wdbox)
Q4_4 = TextInput(value="", title="Q4:", width=wdbox)
Q4_5 = TextInput(value="", title="Q4:", width=wdbox)
Q4_6 = TextInput(value="", title="Q4:", width=wdbox)

Q5_1 = TextInput(value="3", title="Q5 Cutting Food:", width=wdbox)
Q5_2 = TextInput(value="", title="Q5:", width=wdbox)
Q5_3 = TextInput(value="", title="Q5:", width=wdbox)
Q5_4 = TextInput(value="", title="Q5:", width=wdbox)
Q5_5 = TextInput(value="", title="Q5:", width=wdbox)
Q5_6 = TextInput(value="", title="Q5:", width=wdbox)

Q6_1 = TextInput(value="3", title="Q6 Dressing:", width=wdbox)
Q6_2 = TextInput(value="", title="Q6:", width=wdbox)
Q6_3 = TextInput(value="", title="Q6:", width=wdbox)
Q6_4 = TextInput(value="", title="Q6:", width=wdbox)
Q6_5 = TextInput(value="", title="Q6:", width=wdbox)
Q6_6 = TextInput(value="", title="Q6:", width=wdbox)

Q7_1 = TextInput(value="3", title="Q7 Turning in Bed:", width=wdbox)
Q7_2 = TextInput(value="", title="Q7:", width=wdbox)
Q7_3 = TextInput(value="", title="Q7:", width=wdbox)
Q7_4 = TextInput(value="", title="Q7:", width=wdbox)
Q7_5 = TextInput(value="", title="Q7:", width=wdbox)
Q7_6 = TextInput(value="", title="Q7:", width=wdbox)

Q8_1 = TextInput(value="3", title="Q8 Walking:", width=wdbox)
Q8_2 = TextInput(value="", title="Q8:", width=wdbox)
Q8_3 = TextInput(value="", title="Q8:", width=wdbox)
Q8_4 = TextInput(value="", title="Q8:", width=wdbox)
Q8_5 = TextInput(value="", title="Q8:", width=wdbox)
Q8_6 = TextInput(value="", title="Q8:", width=wdbox)

Q9_1 = TextInput(value="3", title="Q9 Climbing Stairs:", width=wdbox)
Q9_2 = TextInput(value="", title="Q9:", width=wdbox)
Q9_3 = TextInput(value="", title="Q9:", width=wdbox)
Q9_4 = TextInput(value="", title="Q9:", width=wdbox)
Q9_5 = TextInput(value="", title="Q9:", width=wdbox)
Q9_6 = TextInput(value="", title="Q9:", width=wdbox)

#Q10_1 = TextInput(value="(depreciated)", title="Q10 Respiratory:", width=wdbox)
#Q10_2 = TextInput(value="(depreciated)", title="Q10:")
#Q10_3 = TextInput(value="(depreciated)", title="Q10:")
#Q10_4 = TextInput(value="(depreciated)", title="Q10:")
#Q10_5 = TextInput(value="(depreciated)", title="Q10:")
#Q10_6 = TextInput(value="(depreciated)", title="Q10:")

R1_1 = TextInput(value="3", title="Q10 Respiratory/R1:", width=wdbox)
R1_2 = TextInput(value="", title="Q10 Respiratory/R1:", width=wdbox)
R1_3 = TextInput(value="", title="Q10 Respiratory/R1:", width=wdbox)
R1_4 = TextInput(value="", title="Q10 Respiratory/R1:", width=wdbox)
R1_5 = TextInput(value="", title="Q10 Respiratory/R1:", width=wdbox)
R1_6 = TextInput(value="", title="Q10 Respiratory/R1:", width=wdbox)

R2_1 = TextInput(value="3", title="R2", width=wdbox)
R2_2 = TextInput(value="", title="Label:", width=wdbox)
R2_3 = TextInput(value="", title="Label:", width=wdbox)
R2_4 = TextInput(value="", title="Label:", width=wdbox)
R2_5 = TextInput(value="", title="Label:", width=wdbox)
R2_6 = TextInput(value="", title="Label:", width=wdbox)

R3_1 = TextInput(value="3", title="R3", width=wdbox)
R3_2 = TextInput(value="", title="Label:", width=wdbox)
R3_3 = TextInput(value="", title="Label:", width=wdbox)
R3_4 = TextInput(value="", title="Label:", width=wdbox)
R3_5 = TextInput(value="", title="Label:", width=wdbox)
R3_6 = TextInput(value="", title="Label:", width=wdbox)


## ALSFRS boxes
wd = 200
l1a = row(widgetbox(Q1_1, Q2_1, Q3_1, Q4_1, width=wd), 
          widgetbox(Q5_1, Q6_1, Q7_1, Q8_1, width=wd), 
          widgetbox(Q9_1, R1_1, R2_1, R3_1, width=wd))
l1 = layout([dt_alsfrs_1], l1a, sizing_mode='scale_width')

l2a = row(widgetbox(Q1_2, Q2_2, Q3_2, Q4_2, width=wd), 
          widgetbox(Q5_2, Q6_2, Q7_2, Q8_2, width=wd), 
          widgetbox(Q9_2, R1_2, R2_2, R3_2, width=wd))
l2 = layout([dt_alsfrs_2], l2a, sizing_mode='scale_width')

l3a = row(widgetbox(Q1_3, Q2_3, Q3_3, Q4_3, width=wd), 
          widgetbox(Q5_3, Q6_3, Q7_3, Q8_3, width=wd), 
          widgetbox(Q9_3, R1_3, R2_3, R3_3, width=wd))
l3 = layout([dt_alsfrs_3], l3a, sizing_mode='scale_width')

l4a = row(widgetbox(Q1_4, Q2_4, Q3_4, Q4_4, width=wd), 
          widgetbox(Q5_4, Q6_4, Q7_4, Q8_4, width=wd), 
          widgetbox(Q9_4, R1_4, R2_4, R3_4, width=wd))
l4 = layout([dt_alsfrs_4], l4a, sizing_mode='scale_width')

l5a = row(widgetbox(Q1_5, Q2_5, Q3_5, Q4_5, width=wd), 
          widgetbox(Q5_5, Q6_5, Q7_5, Q8_5, width=wd), 
          widgetbox(Q9_5, R1_5, R2_5, R3_5, width=wd))
l5 = layout([dt_alsfrs_5], l5a, sizing_mode='scale_width')

l6a = row(widgetbox(Q1_6, Q2_6, Q3_6, Q4_6, width=wd), 
          widgetbox(Q5_6, Q6_6, Q7_6, Q8_6, width=wd), 
          widgetbox(Q9_6, R1_6, R2_6, R3_6, width=wd))
l6 = layout([dt_alsfrs_6], l6a, sizing_mode='scale_width')

tab1 = Panel(child= l1, title="ALSFRS-1")
tab2 = Panel(child= l2, title="ALSFRS-2")
tab3 = Panel(child= l3, title="ALSFRS-3")
tab4 = Panel(child= l4, title="ALSFRS-4")
tab5 = Panel(child= l5, title="ALSFRS-5")
tab6 = Panel(child= l6, title="ALSFRS-6")

tabs = Tabs(tabs=[ tab1, tab2, tab3, tab4, tab5, tab6], width = 800)


## date things
crnt_date=dt.now()

def callback(attr,old,new):
    print(type(old))
    print('old was {} and new is {}'.format(old,new))

### alsfrs tables
## this data table code should be a function called by the prediction button
# collapse text box inputs into one list
alsfrs_entries = {'test': ['onset', 'test1', 'test2', 'test3', 'test4', 'test5', 'test6'], 
                  'day':[0, 1,2,3,4,5,6], 
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

alsfrs_source = ColumnDataSource(alsfrs_df)
alsfrs_source_columns = [
    TableColumn(field="test", title="Test Number"),
    #TableColumn(field="day", title="Date", formatter=DateFormatter()),
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
    TableColumn(field="R3", title="R3")
    
]

data_table = DataTable(source=alsfrs_source, columns=alsfrs_source_columns, 
                       width=800, height=280)
pp_data_table = Paragraph(text="""ALSFRS Data Table""", width=250, height=15)


###  make prediction button   
def run_predict_button():
   # 
    data = alsfrs_source.data   
    newdata = expand_data(data)  # get the new values from the form and add to data
    source2.data = newdata2

## button to run prediction
predict_button = Button(label="Run Prediction", button_type="success")

# part of predict family button
def expand_data(data):
    # add the als form data to the df
    
    # get the current textbox values
    
    
    # update the alsfrs table
    data
    #
    return data



### FIGURES

#lm example
#x=np.array([0,1,2,3,4,5,6,7,8])
alsfrs_total = alsfrs_df.copy()
alsfrs_total.dropna()
alsfrs_total.drop(['R2','R3'], axis=1, inplace=True)
alsfrs_total['Total'] = alsfrs_total.sum(axis=1)


y=np.array([1,2,3,5,4,6,8,7,9])
# determine best fit line
par = np.polyfit(x, y, 1, full=True)
slope=par[0][0]
intercept=par[0][1]
y_predicted = [slope*i + intercept  for i in x]
# plot it bokeh style
plot_slope_alsfrs=figure(plot_width=250, plot_height=250)
plot_slope_alsfrs.circle(x,y)
plot_slope_alsfrs.line(x,y_predicted,color='red',
                       legend='y='+str(round(slope,2))+'x+'+str(round(intercept,2)))

## test figures
x = list(range(11))
y0 = x
y1 = [10 - i for i in x]
y2 = [abs(i - 5) for i in x]

# create three plots
p1 = figure(plot_width=250, plot_height=250, title=None)
p1.circle(x, y0, size=10, color=Viridis3[0])
p2 = figure(plot_width=250, plot_height=250, title=None)
p2.triangle(x, y1, size=10, color=Viridis3[1])
p3 = figure(plot_width=250, plot_height=250, title=None)
p3.square(x, y2, size=10, color=Viridis3[2])

# make a grid
grid = gridplot([[p1, p2], [None, plot_slope_alsfrs]])  # can also fill in with None


### Display
curdoc().title = "ALS_Predict"
'''
curdoc().add_root(
            row(widgetbox(
                subject_id, dt_date_onset, age_onset,
                pp_onset, 
                onset_loc, onset_loc_detail, symptom,
                pp_riluzole, riluzole, 
                pp_caucasian, caucasian,
                pp_sex, sex,
                width=250),
                tabs),
            predict_button, grid )
'''
# these removed for troubleshooting
#

curdoc().add_root( layout([description], row(
    widgetbox(subject_id, dt_date_onset, age_onset, pp_onset, 
                onset_loc, onset_loc_detail, symptom, pp_riluzole, riluzole, 
                pp_caucasian, caucasian, pp_sex, sex,
                width=300),
                     tabs), 
                          [predict_button], 
                          [pp_data_table],
                          [data_table], 
                          [grid]))