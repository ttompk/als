#bokeh_app
from os.path import dirname, join
from datetime import date
from datetime import timedelta as td
from datetime import datetime as dt

from bokeh.io import output_file, show
from bokeh.plotting import curdoc
from bokeh.layouts import widgetbox, row, gridplot, layout
from bokeh.layouts import column as bokehcolumn
from bokeh.plotting import figure
from bokeh.models.widgets import TextInput, Paragraph, Button, MultiSelect, RadioGroup
from bokeh.models.widgets import DatePicker, Panel, Tabs, Div, RadioButtonGroup, Button

from bokeh.palettes import Viridis3

output_file("text_input.html")

desc = Div(text=open(join(dirname(__file__), "description.html")).read(), width=800)


## Onset Panel
dt_date_onset=DatePicker(title="Date of ALS Onset", 
                                       min_date=date(1990,1,1),
                                       max_date=date.today())
subject_id = TextInput(title="Subject ID:", value="ref number")
age_onset = TextInput(title="Age at ALS Onset", value="years")
onset_loc_detail = MultiSelect(title="Location Detail (can pick multiple)", value=["unk"],
                           options=[("unk", "unknown"),("hands", "Hands"), 
                                    ("arms", "Arms"), ("feet", "Feet"), ("legs", "Legs")])

# onset radio button
def onset_loc_handler(new):
    print ('Radio button option ' + str(new) + ' selected.')
    
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
dt_alsfrs_1=DatePicker(title='Date of Test: ', 
                       min_date=date(1990,1,1),
                       max_date=date.today(), width=100)
dt_alsfrs_2=DatePicker(title='Date of 2nd ALSFRS: ', 
                                       min_date=date(1990,1,1),
                                       max_date=date.today())
dt_alsfrs_3=DatePicker(title='Date of 3rd ALSFRS: ', 
                                       min_date=date(1990,1,1),
                                       max_date=date.today())
dt_alsfrs_4=DatePicker(title='Date of 4th ALSFRS: ', 
                                       min_date=date(1990,1,1),
                                       max_date=date.today())
dt_alsfrs_5=DatePicker(title='Date of 5th ALSFRS: ', 
                                       min_date=date(1990,1,1),
                                       max_date=date.today())
dt_alsfrs_6=DatePicker(title='Date of 6th ALSFRS: ', 
                                       min_date=date(1990,1,1),
                                       max_date=date.today())
wdbox=40
Q1_1 = TextInput(value="", title="Q1:", width=wdbox)
Q1_2 = TextInput(value="", title="Q1:")
Q1_3 = TextInput(value="", title="Q1:")
Q1_4 = TextInput(value="", title="Q1:")
Q1_5 = TextInput(value="", title="Q1:")
Q1_6 = TextInput(value="", title="Q1:")

Q2_1 = TextInput(value="", title="Q2:", width=wdbox)
Q2_2 = TextInput(value="", title="Q2:")
Q2_3 = TextInput(value="", title="Q2:")
Q2_4 = TextInput(value="", title="Q2:")
Q2_5 = TextInput(value="", title="Q2:")
Q2_6 = TextInput(value="", title="Q2:")

Q3_1 = TextInput(value="", title="Q3:", width=wdbox)
Q3_2 = TextInput(value="", title="Q3:")
Q3_3 = TextInput(value="", title="Q3:")
Q3_4 = TextInput(value="", title="Q3:")
Q3_5 = TextInput(value="", title="Q3:")
Q3_6 = TextInput(value="", title="Q3:")

Q4_1 = TextInput(value="", title="Q4:", width=wdbox)
Q4_2 = TextInput(value="", title="Q4:")
Q4_3 = TextInput(value="", title="Q4:")
Q4_4 = TextInput(value="", title="Q4:")
Q4_5 = TextInput(value="", title="Q4:")
Q4_6 = TextInput(value="", title="Q4:")

Q5_1 = TextInput(value="", title="Q5:", width=wdbox)
Q5_2 = TextInput(value="", title="Q5:")
Q5_3 = TextInput(value="", title="Q5:")
Q5_4 = TextInput(value="", title="Q5:")
Q5_5 = TextInput(value="", title="Q5:")
Q5_6 = TextInput(value="", title="Q5:")

Q6_1 = TextInput(value="", title="Q6:", width=wdbox)
Q6_2 = TextInput(value="", title="Q6:")
Q6_3 = TextInput(value="", title="Q6:")
Q6_4 = TextInput(value="", title="Q6:")
Q6_5 = TextInput(value="", title="Q6:")
Q6_6 = TextInput(value="", title="Q6:")

Q7_1 = TextInput(value="", title="Q7:", width=wdbox)
Q7_2 = TextInput(value="", title="Q7:")
Q7_3 = TextInput(value="", title="Q7:")
Q7_4 = TextInput(value="", title="Q7:")
Q7_5 = TextInput(value="", title="Q7:")
Q7_6 = TextInput(value="", title="Q7:")

Q8_1 = TextInput(value="", title="Q8:", width=wdbox)
Q8_2 = TextInput(value="", title="Q8:")
Q8_3 = TextInput(value="", title="Q8:")
Q8_4 = TextInput(value="", title="Q8:")
Q8_5 = TextInput(value="", title="Q8:")
Q8_6 = TextInput(value="", title="Q8:")

Q9_1 = TextInput(value="", title="Q9:", width=wdbox)
Q9_2 = TextInput(value="", title="Q9:")
Q9_3 = TextInput(value="", title="Q9:")
Q9_4 = TextInput(value="", title="Q9:")
Q9_5 = TextInput(value="", title="Q9:")
Q9_6 = TextInput(value="", title="Q9:")

Q10_1 = TextInput(value="(depreciated)", title="Q10:", width=wdbox)
Q10_2 = TextInput(value="(depreciated)", title="Q10:")
Q10_3 = TextInput(value="(depreciated)", title="Q10:")
Q10_4 = TextInput(value="(depreciated)", title="Q10:")
Q10_5 = TextInput(value="(depreciated)", title="Q10:")
Q10_6 = TextInput(value="(depreciated)", title="Q10:")

R1_1 = TextInput(value="", title="R1:", width=wdbox)
R1_2 = TextInput(value="", title="Label:")
R1_3 = TextInput(value="", title="Label:")
R1_4 = TextInput(value="", title="Label:")
R1_5 = TextInput(value="", title="Label:")
R1_6 = TextInput(value="", title="Label:")

R2_1 = TextInput(value="", title="R2", width=wdbox)
R2_2 = TextInput(value="", title="Label:")
R2_3 = TextInput(value="", title="Label:")
R2_4 = TextInput(value="", title="Label:")
R2_5 = TextInput(value="", title="Label:")
R2_6 = TextInput(value="", title="Label:")

R3_1 = TextInput(value="", title="R3", width=wdbox)
R3_2 = TextInput(value="", title="Label:")
R3_3 = TextInput(value="", title="Label:")
R3_4 = TextInput(value="", title="Label:")
R3_5 = TextInput(value="", title="Label:")
R3_6 = TextInput(value="", title="Label:")


## ALSFRS boxes
wd = 200
l1a = row(widgetbox(Q1_1, Q2_1, Q3_1, Q4_1, width=wd), 
          widgetbox(Q5_1, Q6_1, Q7_1, Q8_1,width=wd), 
          widgetbox(Q9_1, Q10_1, width=wd), 
          widgetbox(R1_1, R2_1, R3_1, width=wd))

l1 = layout([dt_alsfrs_1], l1a, sizing_mode='scale_width')
                 #widgetbox(Q9_1, Q10_1, R1_1)
                 #widgetbox(R2_1, R3_1))),, sizing_mode='fixed')
l2 = layout([dt_alsfrs_2, Q1_2, Q2_2, Q3_2, Q4_2, Q5_2, Q6_2, Q7_2,
            Q8_2, Q9_2, Q10_2, R1_2, R2_2, R3_2], sizing_mode='scale_width')

l3 = layout([dt_alsfrs_3, Q1_3, Q2_3, Q3_3, Q4_3, Q5_3, Q6_3, Q7_3,
            Q8_3, Q9_3, Q10_3, R1_3, R2_3, R3_3], sizing_mode='fixed')

l4 = layout([dt_alsfrs_4, Q1_4, Q2_4, Q3_4, Q4_4, Q5_4, Q6_4, Q7_4,
            Q8_4, Q9_4, Q10_4, R1_4, R2_4, R3_4], sizing_mode='fixed')

l5 = layout([dt_alsfrs_5, Q1_5, Q2_5, Q3_5, Q4_5, Q5_5, Q6_5, Q7_5,
            Q8_5, Q9_5, Q10_5, R1_5, R2_5, R3_5], sizing_mode='fixed')

l6 = layout([dt_alsfrs_6, Q1_6, Q2_6, Q3_6, Q4_6, Q5_6, Q6_6, Q7_6,
            Q8_6, Q9_6, Q10_6, R1_6, R2_6, R3_6], sizing_mode='fixed')


tab1 = Panel(child= l1, title="ALSFRS-1")
tab2 = Panel(child= l2, title="ALSFRS-2")
tab3 = Panel(child= l3, title="ALSFRS-3")
tab4 = Panel(child= l4, title="ALSFRS-4")
tab5 = Panel(child= l5, title="ALSFRS-5")
tab6 = Panel(child= l6, title="ALSFRS-6")

tabs = Tabs(tabs=[ tab1, tab2, tab3, tab4, tab5, tab6], width = 800)

text_input = TextInput(value="default", title="this")


## button to run prediction
predict_button = Button(label="Run Prediction", button_type="success")

## date things
crnt_date=dt.now()

def callback(attr,old,new):
    print(type(old))
    print('old was {} and new is {}'.format(old,new))

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
grid = gridplot([[p1, p2], [None, p3]])


### Display
dt_alsfrs_1.on_change('value',callback)
#curdoc().add_root(bokehcol(dt_pckr_strt))

show(layout([desc],
            row(widgetbox(
                subject_id, dt_date_onset, age_onset,
                pp_onset, 
                onset_loc, onset_loc_detail, 
                pp_riluzole, riluzole, 
                pp_caucasian, caucasian,
                pp_sex, sex,
                width=250),
                tabs),
            predict_button, grid ))

