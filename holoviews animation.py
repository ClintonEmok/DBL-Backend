import pandas as pd
import numpy as np
import holoviews as hv


from bokeh.io import curdoc
from bokeh.layouts import layout
from bokeh.models import Slider, Button
from holoviews import dim, opts


renderer = hv.renderer('bokeh')
hv.extension('bokeh')

metro_data = pd.read_csv('./metro_data.csv', sep = ';', encoding='latin-1')
mapdata = metro_data[metro_data['StimuliName'] == '01_Antwerpen_S1.jpg']

userCount = 1
countList = [1]

for i in range(1, len(mapdata)):
    if mapdata.iloc[i, 6] == mapdata.iloc[i-1, 6]:
        userCount = userCount + 1
        countList.append(userCount)
    else:
        userCount = 1
        countList.append(userCount)

mapdata['User Index'] = countList

kdims='MappedFixationPointX'
vdims='MappedFixationPointY'

start = 0
end = 43

plot = hv.Scatter(mapdata, kdims, vdims, ).opts(width=800, height= 600, invert_yaxis = True)
plot = renderer.get_plot(plot)

# Define custom widgets
def animate_update():
    index = slider.value + 1
    if index > end:
        index = start
    slider.value = index

# Update the holoviews plot by calling update with the user index.
def slider_update(attrname, old, new):
    plot.update(slider.value)

slider = Slider(start=start, end=end, value=0, step=1, title="User Index")
slider.on_change('value', slider_update)


def animate():
    if button.label == '► Play':
        button.label = '❚❚ Pause'
        curdoc().add_periodic_callback(animate_update, 200)
    else:
        button.label = '► Play'
        curdoc().remove_periodic_callback(animate_update)

button = Button(label='► Play', width=60)
button.on_click(animate)

layout = layout([
    [plot.state],
    [slider, button],
], sizing_mode='fixed')

curdoc().add_root(layout)


