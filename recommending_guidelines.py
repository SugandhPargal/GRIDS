
"""Plotting OUTPUT"""

import plotly.express as px
import pandas as pd
stages = ["Speed Profile", "Maneuvers", "Environmental Obstacles", "Front Vehicle Profile"]
df_mtl = pd.DataFrame(dict(number=[39, 27.4, 20.6, 11], stage=stages))
df_mtl['office'] = 'Montreal'
df_toronto = pd.DataFrame(dict(number=[52, 36, 18,11], stage=stages))
df_toronto['office'] = 'Toronto'
df = pd.concat([df_mtl, df_toronto], axis=0)
fig = px.funnel(df, x='number', y='stage', color='office')
fig.show()

!pip install -U kaleido

import kaleido
engine= "kaleido"

"""D1 vs NewYork

"""



import plotly
plotly.__version__

pip install -c plotly plotly-orca

pip install plotly==5.3.1

pip install -U kaleido

from plotly import graph_objects as go

fig = go.Figure()

fig.add_trace(go.Funnel(
    name = 'High',
    y = ["<b>SP</b>","<b>DM</b>","<b>TO</b>","<b>PV</b>"],
    x = [30, 30, 30, 60],
    text = ["<b>Drive Slow</b>","<b>Take Wider <br> Turn</b>", "<b>Keep An Eye <br>On Peer Cars</b>","<b>Maintain Relative Speed <br> Between The Preceding Vehicle </b>"],
    
    textinfo = "text",
    marker = {"color": ["peachpuff", "peachpuff", "peachpuff", "peachpuff"]}))

fig.add_trace(go.Funnel(
    name = 'Moderate',
    orientation = "h",
    y = ["<b>SP</b>", "<b>DM</b>","<b>TO</b>","<b>PV</b>"],
    x = [30, 30, 30, 60],
    text = ["<b>Press Clutch <br> Fully Down</b>","<b>Maintain Your <br> Lane</b>", "<b>Slow Down At <br> Cross Walks</b>", "<b>Drive Defensively</b>"],
    
    textposition = "inside",
    textinfo = "text",
    marker = {"color": ["lightblue", "lightblue", "lightblue", "lightblue"]}))

fig.add_trace(go.Funnel(
    name = 'Least',
    orientation = "h",
    y = ["<b>SP</b>", "<b>DM</b>"],
    x = [30, 30],
    text = ["<b>Do not <br> Oversteer</b>","<b>Take Brakes<br> Smoothly</b>"],
    textposition = "inside",
    textinfo = "text",
    marker = {"color": ["mediumaquamarine", "mediumaquamarine"]}))
fig.update_traces(textposition='inside',textfont_size=20)    
fig.show()

"""Temporal recommendation along with events behind"""

from plotly import graph_objects as go

fig = go.Figure()

fig.add_trace(go.Funnel(
    name = 'High',
    y = ["<b>SP</b>", "<b>DM</b>","<b>PV</b>"],
    x = [30, 30,30],
    text = ["<b>Drive Slow</b>","<b>Take Wider Turns</b>","<b>Keep Distance From <br> Preceding Vehicle</b>"],
    
    textinfo = "text",
    marker = {"color": ["peachpuff", "peachpuff", "peachpuff", "peachpuff"]}))

fig.add_trace(go.Funnel(
    name = 'Moderate',
    orientation = "h",
    y = ["<b>SP</b>", "<b>DM</b>","<b>PV</b>"],
    x = [30, 30, 30],
    text = ["<b>Donot Oversteer</b>","<b>Donot Change <br> Lane Frequently</b>", "<b>Follow Preceding <br> Vehicles' Braking Action</b>"],
    
    textposition = "inside",
    textinfo = "text",
    marker = {"color": ["lightblue", "lightblue", "lightblue", "lightblue"]}))

"""fig.add_trace(go.Funnel(
    name = 'Least',
    orientation = "h",
    y = ["<b>SP</b>", "<b>DM</b>"],
    x = [30, 30],
    text = ["<b>Slight Poor <br> Road Condition</b>","<b>Few Abrupt Turns <br> In Visiting City</b>"],
    textposition = "inside",
    textinfo = "text",
    marker = {"color": ["mediumaquamarine", "mediumaquamarine"]}))"""
fig.update_traces(textposition='inside',textfont_size=20)    
fig.show()

from plotly import graph_objects as go

fig = go.Figure()

fig.add_trace(go.Funnel(
    name = 'High',
    y = ["<b>SP</b>", "<b>DM</b>","<b>PV</b>"],
    x = [10, 10,10],
    text = ["<b>Drive Slow</b>","<b>Maintain Your Lane</b>","<b>Follow Preceding <br> Vehicles' Braking Action</b>"],
    
    textinfo = "text",
    marker = {"color": ["peachpuff", "peachpuff", "peachpuff", "peachpuff"]}))

"""fig.add_trace(go.Funnel(
    name = 'Moderate',
    orientation = "h",
    y = ["<b>SP</b>", "<b>DM</b>","<b>PV</b>"],
    x = [30, 30, 30],
    text = ["<b>Donot Oversteer</b>","<b>Donot Change <br> Lane Frequently</b>", "<b>Follow Preceding <br> Vehicles' Braking Action</b>"],
    
    textposition = "inside",
    textinfo = "text",
    marker = {"color": ["lightblue", "lightblue", "lightblue", "lightblue"]}))"""

"""fig.add_trace(go.Funnel(
    name = 'Least',
    orientation = "h",
    y = ["<b>SP</b>", "<b>DM</b>"],
    x = [30, 30],
    text = ["<b>Slight Poor <br> Road Condition</b>","<b>Few Abrupt Turns <br> In Visiting City</b>"],
    textposition = "inside",
    textinfo = "text",
    marker = {"color": ["mediumaquamarine", "mediumaquamarine"]}))"""
fig.update_traces(textposition='inside',textfont_size=20)    
fig.show()

from plotly import graph_objects as go

fig = go.Figure()

fig.add_trace(go.Funnel(
    name = 'High',
    y = ["<b>DM</b>", "<b>SP</b>","<b>TO</b>","<b>PV</b>"],
    x = [10, 10,10,10],
    text = ["<b>Take Brake Smoothly</b>","<b>Donot Fluctuate Speed</b>","<b>Slowdown At Cross-Walks</b>","<b>Follow Preceding <br> Vehicles' Braking Action</b>"],
    
    textinfo = "text",
    marker = {"color": ["peachpuff", "peachpuff", "peachpuff", "peachpuff"]}))

"""fig.add_trace(go.Funnel(
    name = 'Moderate',
    orientation = "h",
    y = ["<b>SP</b>", "<b>DM</b>","<b>PV</b>"],
    x = [30, 30, 30],
    text = ["<b>Donot Oversteer</b>","<b>Donot Change <br> Lane Frequently</b>", "<b>Follow Preceding <br> Vehicles' Braking Action</b>"],
    
    textposition = "inside",
    textinfo = "text",
    marker = {"color": ["lightblue", "lightblue", "lightblue", "lightblue"]}))"""

"""fig.add_trace(go.Funnel(
    name = 'Least',
    orientation = "h",
    y = ["<b>SP</b>", "<b>DM</b>"],
    x = [30, 30],
    text = ["<b>Slight Poor <br> Road Condition</b>","<b>Few Abrupt Turns <br> In Visiting City</b>"],
    textposition = "inside",
    textinfo = "text",
    marker = {"color": ["mediumaquamarine", "mediumaquamarine"]}))"""
fig.update_traces(textposition='inside',textfont_size=20)    
fig.show()

"""D5 vs New York

"""

from plotly import graph_objects as go

fig = go.Figure()

fig.add_trace(go.Funnel(
    name = 'High',
    y = ["<b>SP</b>", "<b>DM</b>","<b>PV</b>"],
    x = [30, 30, 60],
    text = ["<b>Drive Slow</b>","<b>Take Wider <br> Turns</b>","<b>Maintain Relative Speed <br> Between The Preceding Vehicle </b>"],
    
    textinfo = "text",
    marker = {"color": ["peachpuff", "peachpuff", "peachpuff"]}))

fig.add_trace(go.Funnel(
    name = 'Moderate',
    orientation = "h",
    y = ["<b>SP</b>", "<b>DM</b>"],
    x = [30, 30],
    text = ["<b>---</b>","<b>Strictly Maintain <br> Your Lane</b>"],
    
    textposition = "inside",
    textinfo = "text",
    marker = {"color": ["lightblue", "lightblue", "lightblue"]}))

fig.add_trace(go.Funnel(
    name = 'Least',
    orientation = "h",
    y = ["<b>SP</b>", "<b>DM</b>"],
    x = [30, 30],
    text = ["<b>---</b>","<b>Take Brake <br> Smoothly</b>"],
    textposition = "inside",
    textinfo = "text",
    marker = {"color": ["mediumaquamarine", "mediumaquamarine"]}))
fig.update_traces(textposition='inside',textfont_size=20)
fig.show()

"""D1 vs San Fransisco"""

from plotly import graph_objects as go

fig = go.Figure()

fig.add_trace(go.Funnel(
    name = 'High',
    y = ["<b>SP</b>", "<b>DM</b>","<b>TO</b>","<b>PV</b>"],
    x = [30, 30, 30, 50],
    text = ["<b>Do not Fluctuate<br> Speed</b>","<b>Take Less <br>Wider Turns</b>","<b>Keep An Eye <br> On Peer Cars</b>", "<b>Donot Fluctuate Relative Speed <br> Between Preceding Vehicle</b>"],
    
    textinfo = "text",
    marker = {"color": ["peachpuff", "peachpuff", "peachpuff", "peachpuff"]}))

fig.add_trace(go.Funnel(
    name = 'Moderate',
    orientation = "h",
    y = ["<b>SP</b>", "<b>DM</b>"],
    x = [30, 30, 30, 30],
    text = ["<b>Press Clutch <br> Fully</b>","<b>Avoid Changing <br> Lanes Frequently</b>"],
    
    textposition = "inside",
    textinfo = "text",
    marker = {"color": ["Lightblue", "Lightblue", "Lightblue", "Lightblue"]}))


fig.update_traces(textposition='inside',textfont_size=20)
fig.show()

"""D5 vs San Francisco"""

from plotly import graph_objects as go

fig = go.Figure()

fig.add_trace(go.Funnel(
    name = 'High',
    y = ["<b>SP</b>", "<b>DM</b>","<b>PV</b>"],
    x = [30, 30, 60],
    text = ["<b>Do not Fluctuate <br> Speed</b>","<b>Take Less <br> Wider Turns</b>", "<b>Maintain Relative Distance <br> Between Preceding Vehicle</b>"], 
    textinfo = "text",
    marker = {"color": ["peachpuff", "peachpuff", "peachpuff", "peachpuff"]}))

fig.add_trace(go.Funnel(
    name = 'Moderate',
    orientation = "h",
    y = ["<b>SP</b>", "<b>DM</b>"],
    x = [30, 30],
    text = ["<b>---</b>","<b>Avoid Changing <br> Lane Frequently</b>"],
    
    textposition = "inside",
    textinfo = "text",
    marker = {"color": ["Lightblue", "Lightblue", "Lightblue", "Lightblue"]}))

fig.add_trace(go.Funnel(
    name = 'Least',
    orientation = "h",
    y = ["<b>SP</b>", "<b>DM</b>"],
    x = [30, 30],
    text = ["<b>---</b>","<b>Take Brake <br> Smoothly</b>"],
    textposition = "inside",
    textinfo = "text",
    marker = {"color": ["mediumaquamarine", "mediumaquamarine"]}))

fig.update_traces(textposition='inside',textfont_size=20)
fig.show()

from plotly import graph_objects as go

fig = go.Figure(go.Funnelarea(
    text = ["Median","Mean", "Jerkiness", "Snap", "Variation"],
    values = [5, 4, 3, 2, 1]
    ))
fig.show()

import matplotlib.pyplot as plt
 
# line 1 points
x1 = [1,2,3,4,5,6,7]
y1 = [12,3,2,10,11,2,3]
# plotting the line 1 points
plt.plot(x1, y1, label = "sensor1")
 
# line 2 points
x2 = [1,2,3,4,5,6,7]
y2 = [17,7,6,20,12,10,8]
# plotting the line 2 points
plt.plot(x2, y2, label = "sensor2")

# line 3 points
x3 = [1,2,3,4,5,6,7]
y3 = [12,17,9,15,10,8,15]
# plotting the line 2 points
plt.plot(x3, y3, label = "sensor3")
 
# naming the x axis
plt.xlabel('TIME',fontweight ='bold', fontsize='15')
# naming the y axis
plt.ylabel('SENSOR DATA',fontweight ='bold', fontsize='15')
# giving a title to my graph

 
# show a legend on the plot
plt.legend()
 
# function to show the plot
plt.show()

