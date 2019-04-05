#bokeh_app

from bokeh.io import output_file, show
from bokeh.plotting import curdoc
from bokeh.layouts import widgetbox, row, gridplot, layout
from bokeh.layouts import column as bokehcolumn
from bokeh.plotting import figure
from bokeh.models.widgets import TextInput, Paragraph, Button, MultiSelect, RadioGroup
from bokeh.models.widgets import DatePicker, Panel, Tabs, Div
from datetime import date
from datetime import timedelta as td
from datetime import datetime as dt

from bokeh.palettes import Viridis3


output_file("text_input.html")

p = Paragraph(text="""Your text is initialized with the 'text' argument.  The
remaining Paragraph arguments are 'width' and 'height'. For this example, those values
are 200 and 100 respectively.""",
width=200, height=100)

#show(widgetbox(p))

subject_id = TextInput(value="default", title="Subject ID:")
date_onset = TextInput(value="default", title="Date of ALS Onset")
onset_loc = RadioGroup(labels=["Bulbar", "Spinal", "Both"], active=0)
onset_loc_detail = MultiSelect(title="Location Detail (can pick multiple)", value=["unk"],
                           options=[("unk", "unknown"),("hands", "Hands"), 
                                    ("arms", "Arms"), ("feet", "Feet"), ("legs", "Legs")])

dumbdiv = Div()

## ALSFRS panels
dt_alsfrs_1=DatePicker(title='Date of 1st ALSFRS: ', 
                                       min_date=date(1990,1,1),
                                       max_date=date.today())
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

Q1_1 = TextInput(value="default", title="Q1:")
Q1_2 = TextInput(value="default", title="Label:")
Q1_3 = TextInput(value="default", title="Label:")
Q1_4 = TextInput(value="default", title="Label:")
Q1_5 = TextInput(value="default", title="Label:")
Q1_6 = TextInput(value="default", title="Label:")

Q2_1 = TextInput(value="default", title="Q2:")
Q2_2 = TextInput(value="default", title="Label:")
Q2_3 = TextInput(value="default", title="Label:")
Q2_4 = TextInput(value="default", title="Label:")
Q2_5 = TextInput(value="default", title="Label:")
Q2_6 = TextInput(value="default", title="Label:")

Q3_1 = TextInput(value="default", title="Q3:")
Q3_2 = TextInput(value="default", title="Label:")
Q3_3 = TextInput(value="default", title="Label:")
Q3_4 = TextInput(value="default", title="Label:")
Q3_5 = TextInput(value="default", title="Label:")
Q3_6 = TextInput(value="default", title="Label:")

Q4_1 = TextInput(value="default", title="Q4:")
Q4_2 = TextInput(value="default", title="Label:")
Q4_3 = TextInput(value="default", title="Label:")
Q4_4 = TextInput(value="default", title="Label:")
Q4_5 = TextInput(value="default", title="Label:")
Q4_6 = TextInput(value="default", title="Label:")

Q5_1 = TextInput(value="default", title="Q5:")
Q5_2 = TextInput(value="default", title="Label:")
Q5_3 = TextInput(value="default", title="Label:")
Q5_4 = TextInput(value="default", title="Label:")
Q5_5 = TextInput(value="default", title="Label:")
Q5_6 = TextInput(value="default", title="Label:")

Q6_1 = TextInput(value="default", title="Q6:")
Q6_2 = TextInput(value="default", title="Label:")
Q6_3 = TextInput(value="default", title="Label:")
Q6_4 = TextInput(value="default", title="Label:")
Q6_5 = TextInput(value="default", title="Label:")
Q6_6 = TextInput(value="default", title="Label:")

Q7_1 = TextInput(value="default", title="Label:")
Q7_2 = TextInput(value="default", title="Label:")
Q7_3 = TextInput(value="default", title="Label:")
Q7_4 = TextInput(value="default", title="Label:")
Q7_5 = TextInput(value="default", title="Label:")
Q7_6 = TextInput(value="default", title="Label:")

Q8_1 = TextInput(value="default", title="Label:")
Q8_2 = TextInput(value="default", title="Label:")
Q8_3 = TextInput(value="default", title="Label:")
Q8_4 = TextInput(value="default", title="Label:")
Q8_5 = TextInput(value="default", title="Label:")
Q8_6 = TextInput(value="default", title="Label:")

Q9_1 = TextInput(value="default", title="Label:")
Q9_2 = TextInput(value="default", title="Label:")
Q9_3 = TextInput(value="default", title="Label:")
Q9_4 = TextInput(value="default", title="Label:")
Q9_5 = TextInput(value="default", title="Label:")
Q9_6 = TextInput(value="default", title="Label:")

Q10_1 = TextInput(value="default", title="Label:")
Q10_2 = TextInput(value="default", title="Label:")
Q10_3 = TextInput(value="default", title="Label:")
Q10_4 = TextInput(value="default", title="Label:")
Q10_5 = TextInput(value="default", title="Label:")
Q10_6 = TextInput(value="default", title="Label:")

R1_1 = TextInput(value="default", title="Label:")
R1_2 = TextInput(value="default", title="Label:")
R1_3 = TextInput(value="default", title="Label:")
R1_4 = TextInput(value="default", title="Label:")
R1_5 = TextInput(value="default", title="Label:")
R1_6 = TextInput(value="default", title="Label:")

R2_1 = TextInput(value="default", title="Label:")
R2_2 = TextInput(value="default", title="Label:")
R2_3 = TextInput(value="default", title="Label:")
R2_4 = TextInput(value="default", title="Label:")
R2_5 = TextInput(value="default", title="Label:")
R2_6 = TextInput(value="default", title="Label:")

R3_1 = TextInput(value="default", title="Label:")
R3_2 = TextInput(value="default", title="Label:")
R3_3 = TextInput(value="default", title="Label:")
R3_4 = TextInput(value="default", title="Label:")
R3_5 = TextInput(value="default", title="Label:")
R3_6 = TextInput(value="default", title="Label:")


l2 = layout([Q1_1, Q2_1 ], sizing_mode='fixed')
l3 = layout([Q1_1, Q2_1 ], sizing_mode='fixed')
l4 = layout([Q1_1, Q2_1 ], sizing_mode='fixed')
l5 = layout([Q1_1, Q2_1 ], sizing_mode='fixed')
l6 = layout([Q1_1, Q2_1 ], sizing_mode='fixed')


l1 = layout(row(widgetbox(dt_alsfrs_1, width=100),
                widgetbox(Q1_1, Q2_1, Q3_1, Q4_1, width=100),
             widgetbox(Q5_1, Q6_1, Q7_1, Q8_1, width=100), 
             widgetbox(Q9_1, Q10_1, R1_1, width=100),
             widgetbox(R2_1, R3_1, width=100)), sizing_mode='fixed')

l2 = layout([[widgetbox(dt_alsfrs_2, width=100)], Q1_2, Q2_2, Q3_2, Q4_2, Q5_2, Q6_2, Q7_2,
            Q8_2, Q9_2, Q10_2, R1_2, R2_2, R3_2], sizing_mode='fixed')

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

button = Button(label="Foo", button_type="success")

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

# show the results
dt_alsfrs_1.on_change('value',callback)
#curdoc().add_root(bokehcol(dt_pckr_strt))
show(layout(row(widgetbox(p,subject_id,date_onset, onset_loc, onset_loc_detail, width=200),grid), tabs))

