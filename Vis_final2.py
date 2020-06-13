#!/usr/bin/env python
# coding: utf-8

# In[1]:


AUTHOR_NAME = 'Jane Deijnen'
AUTHOR_ID_NR = '1354396'
AUTHOR_DATE = '2020-05-05'

AUTHOR_NAME, AUTHOR_ID_NR, AUTHOR_DATE

# In[2]:


import numpy as np
import pandas as pd
import bokeh as bk
import sqlite3
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import seaborn as sns
import holoviews as hv
from bokeh.models import Slider, Button
from holoviews import dim, opts
from holoviews.util import Dynamic
from holoviews.streams import Stream, param
from sklearn.cluster import KMeans  # for clustering

renderer = hv.renderer('bokeh')
hv.extension('bokeh')
sns.set()  # set Seaborn defaults
plt.rcParams['figure.figsize'] = 10, 5  # default hor./vert. size of plots, in inches
plt.rcParams['lines.markeredgewidth'] = 1  # to fix issue with seaborn box plots; needed after import seaborn
from bokeh.io import output_notebook, show, reset_output, curdoc
from bokeh.models import Slider, CustomJS, Select, Panel, Tabs
from bokeh.plotting import figure
from bokeh.layouts import layout, column
from bokeh.layouts import gridplot
from bokeh.transform import factor_cmap, transform
from bokeh.models.annotations import Label, LabelSet
from bokeh.models import (
    ColumnDataSource, Div,
    HoverTool,
    LinearColorMapper,
    BasicTicker,
    PrintfTickFormatter,
    ColorBar,
    FactorRange,
    ImageURL,
    CategoricalColorMapper,
    LinearInterpolator
)
from bokeh.palettes import BuPu
from bokeh.palettes import Colorblind8, Viridis256, Turbo256, Plasma256, Spectral11, all_palettes

output_notebook()

# In[3]:


conn_countries = sqlite3.connect('db.sqlite3')
query_all = 'SELECT * FROM upload_metro;'

# In[4]:


data_database = pd.read_sql_query(query_all, conn_countries)
pd.read_sql_query(query_all, conn_countries)

# In[5]:


# import eye tracking data
Eyetracking_data = pd.read_csv('metro_data.csv', encoding='latin1', sep=";")
Eyetracking_data.head()

# In[6]:


# replacing weird coding fails in stimuli names
Eyetracking_data['StimuliName'] = Eyetracking_data['StimuliName'].replace(
    {'24_Z?rich_S1.jpg': '24_Zurich_S1.jpg', '24_Zrich_S1.jpg': '24_Zurich_S1.jpg',
     '24_Zrich_S2.jpg': '24_Zurich_S2.jpg', '24_Z?rich_S2.jpg': '24_Zurich_S2.jpg',
     '24b_Z?rich_S1.jpg': '24b_Zurich_S1.jpg', '24b_Zrich_S1.jpg': '24b_Zurich_S1.jpg',
     '24b_Z?rich_S2.jpg': '24b_Zurich_S2.jpg', '24b_Zrich_S2.jpg': '24b_Zurich_S2.jpg',

     '12_Br?ssel_S1.jpg': '12_Brussel_S1.jpg', '12_Brssel_S1.jpg': '12_Brussel_S1.jpg',
     '12_Br?ssel_S2.jpg': '12_Brussel_S2.jpg', '12_Brssel_S2.jpg': '12_Brussel_S2.jpg',
     '12b_Br?ssel_S1.jpg': '12b_Brussel_S1.jpg', '12b_Brssel_S1.jpg': '12b_Brussel_S1.jpg',
     '12b_Br?ssel_S2.jpg': '12b_Brussel_S2.jpg', '12b_Brssel_S2.jpg': '12b_Brussel_S2.jpg',

     '14_D?sseldorf_S1.jpg': '14_Dusseldorf_S1.jpg', '14_Dsseldorf_S1.jpg': '14_Dusseldorf_S1.jpg',
     '14_D?sseldorf_S1.jpg': '14_Dusseldorf_S2.jpg', '14_Dsseldorf_S2.jpg': '14_Dusseldorf_S2.jpg',
     '14b_D?sseldorf_S1.jpg': '14b_Dusseldorf_S1.jpg', '14b_Dsseldorf_S1.jpg': '14b_Dusseldorf_S1.jpg',
     '14b_D?sseldorf_S2.jpg': '14b_Dusseldorf_S2.jpg', '14b_Dsseldorf_S2.jpg': '14b_Dusseldorf_S2.jpg',

     '15_G?teborg_S1.jpg': '15_Goteborg_S1.jpg', '15_Gteborg_S1.jpg': '15_Goteborg_S1.jpg',
     '15_G?teborg_S2.jpg': '15_Goteborg_S2.jpg', '15_Gteborg_S2.jpg': '15_Goteborg_S2.jpg',
     '15b_G?teborg_S1.jpg': '15b_Goteborg_S1.jpg', '15b_Gteborg_S1.jpg': '15b_Goteborg_S1.jpg',
     '15b_G?teborg_S2.jpg': '15b_Goteborg_S2.jpg', '15b_Gteborg_S2.jpg': '15b_Goteborg_S2.jpg',

     '04_K?ln_S1.jpg': '04_Koln_S1.jpg', '04_Kln_S1.jpg': '04_Koln_S1.jpg',
     '04_K?ln_S2.jpg': '04_Koln_S2.jpg', '04_Kln_S1.jpg': '04_Koln_S2.jpg',
     '04b_K?ln_S1.jpg': '04b_Koln_S1.jpg', '04b_Kln_S1.jpg': '04b_Koln_S1.jpg',
     '04b_K?ln_S2.jpg': '04b_Koln_S2.jpg', '04b_Kln_S2.jpg': '04b_Koln_S2.jpg', })

# In[7]:


# color palette
colors = ['#0072B2', '#E69F00', '#F0E442', '#009E73', '#56B4E9', '#D55E00', '#CC79A7', '#000000',
          '#990F02', '#871F78', '#581845']
# list of map names
maps = Eyetracking_data['StimuliName'].unique().tolist()
maps.sort()
# list of users
users = Eyetracking_data['user'].unique().tolist()
users = sorted(users, key=lambda x: int("".join([i for i in x if i.isdigit()])))
users.insert(0, "All")

# In[8]:


# Slider and drop down menu
slider_cluster = Slider(title="Amount of Clusters/AOIs", start=1, end=12, value=2, step=1)
stimulimap = Select(title="Select stimulus", value=maps[0], options=maps)
select_user = Select(title="Select user:", value="All", options=users)
button = Button(label='► Play Animation', width=80)

# In[9]:


# create source for AOI plot
source = ColumnDataSource(data=dict(x=[], y=[], c=[], u=[], color=[], gradient=[], time=[], index=[]))

# draw empty AOI plot
TOOLS = "hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select"
TOOLTIPS = [("Participant", "@u"),
            ("AOI", "@c"),
            ("X-coordinate", "@x"),
            ("Y-coordinate", "@y")
            ]

p1 = figure(title="AOI plot",
            plot_width=525,
            plot_height=525,
            tools=TOOLS,
            tooltips=TOOLTIPS,
            x_axis_label='x-axis',
            y_axis_label='y-axis')
# p1.image_url(url=['01_Antwerpen_S1.jpg'], x=0, y=0, w=1650, h=1200)
p1.circle(x='x', y='y', color='color', legend_field='c', source=source, fill_alpha=0.2, size=10)
p1.xgrid.grid_line_color = None
p1.ygrid.grid_line_color = None
p1.legend.orientation = "vertical"
p1.legend.location = "bottom_right"
p1.legend.title = 'AOI'
p1.y_range.flipped = True

# In[10]:


# fixation heatmap
source_heat = ColumnDataSource(data=dict(x=[], y=[], color=[], user=[], gradient=[]))

mapper = LinearColorMapper(palette="Turbo256", low=33, high=1200, low_color="blue", high_color="red")
color_bar = ColorBar(color_mapper=mapper, width=8, location=(0, 0))

p2 = figure(title="Fixation heat map",
            plot_width=p1.plot_width,
            plot_height=p1.plot_width,
            tools=TOOLS,
            tooltips=TOOLTIPS,
            x_range=p1.x_range,
            y_range=p1.y_range,
            sizing_mode="scale_both",
            x_axis_label='x-axis',
            y_axis_label='y-axis')
p2.circle(x="x", y="y", source=source, size=10, fill_color=transform("gradient", mapper),
          line_color=None)
p2.y_range.flipped = True
p2.xgrid.grid_line_color = None
p2.ygrid.grid_line_color = None
p2.add_layout(color_bar, 'right')

# In[11]:


TOOLTIPS3 = [("Value", "@value"),
             ("AOI transition from", "@y"),
             ("to", "@x")]

# Create source for transition matrix
source_matrix = ColumnDataSource(data=dict(x=[], y=[], value=[]))

# Draw empty transition matrix figure

colormap = cm.get_cmap("BuPu")
bokehpalette = [mpl.colors.rgb2hex(m) for m in colormap(np.arange(colormap.N))]
mapper = LinearColorMapper(palette=bokehpalette, low=0.0, high=1.0)

p3 = figure(title="Transition Matrix",
            plot_width=p1.plot_width,
            plot_height=p1.plot_width,
            toolbar_sticky=False,
            tools=TOOLS,
            tooltips=TOOLTIPS3,
            x_axis_label='AOI',
            y_axis_label='AOI')
p3.rect(x='x', y='y', width=1, height=1, source=source_matrix,
        fill_color={'field': 'value', 'transform': mapper}, line_color=None)

color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="5pt",
                     ticker=BasicTicker(desired_num_ticks=8),
                     label_standoff=6, border_line_color=None, location=(0, 0))
p3.add_layout(color_bar, 'right')

# In[12]:


TOOLTIPS4 = [
    ("Timestamp", "@time"),
    ("Participant", "@u"),
    ("AOI", "@c"),
    ("X-coordinate", "@x"),
    ("Y-coordinate", "@y"),
]

# add colours to the scatter points
color_mapper = CategoricalColorMapper(factors=list(Eyetracking_data.user.unique()), palette=Viridis256)

# set size of the dots in the future scatter plot
size_mapper = LinearInterpolator(x=[Eyetracking_data.FixationDuration.min(), Eyetracking_data.FixationDuration.max()],
                                 y=[10, 50])

# scatter plot //  gaze plot

labels = LabelSet(x='x',
                  y='y',
                  text='index',
                  level='glyph',
                  x_offset=0,
                  y_offset=0,
                  source=source,
                  render_mode='canvas')

p4 = figure(title='Gaze plot',
            plot_width=p1.plot_width,
            plot_height=p1.plot_width,
            x_range=p1.x_range,
            y_range=p1.y_range,
            tools=TOOLS,
            tooltips=TOOLTIPS4,
            x_axis_label='x-axis',
            y_axis_label='y-axis')
p4.line(x='x', y='y', source=source, color='purple', line_width=2)
p4.circle(x='x', y='y', source=source,
          color={'field': 'u', 'transform': color_mapper},
          size={'field': 'gradient', 'transform': size_mapper},
          fill_alpha=0.5)

p4.y_range.flipped = True
p4.xgrid.grid_line_color = None
p4.ygrid.grid_line_color = None
p4.add_layout(labels)

# In[13]:


# Database plot
source_database = ColumnDataSource(data=dict(x=[], y=[], color=[], user=[], gradient=[]))
TOOLTIPS5 = [
    ("Participant", "@user"),
    ("X-coordinate", "@x"),
    ("Y-coordinate", "@y"),
]

mapper = LinearColorMapper(palette="Magma256", low=33, high=1500, low_color="blue", high_color="red")

p5 = figure(title="Database plot", tools=TOOLS, tooltips=TOOLTIPS5, sizing_mode="scale_both")
p5.square(x="x", y="y", source=source_database, size=10, fill_color=transform("gradient", mapper),
          line_color=None)
p5.y_range.flipped = True
p5.xgrid.grid_line_color = None
p5.ygrid.grid_line_color = None
p5.xaxis.axis_label = 'x-axis'
p5.yaxis.axis_label = 'y-axis'
p5.xgrid.grid_line_color = None
p5.ygrid.grid_line_color = None


# In[14]:


# function calculating new dataframe based on slider value & selecting stimulus
def calc_clusters():
    stimulimap_val = stimulimap.value
    selected = Eyetracking_data.copy()
    userselect_val = select_user.value.strip()

    if (stimulimap_val != ""):
        selected = selected[selected['StimuliName'].str.contains(stimulimap_val) == True]
        users = selected['user'].unique().tolist()
        users = sorted(users, key=lambda x: int("".join([i for i in x if i.isdigit()])))
        users.insert(0, "All")
        select_user.options = users
    if (userselect_val != "All"):
        selected = selected[selected['user'] == userselect_val]

    X_km = selected[['MappedFixationPointX', 'MappedFixationPointY']].copy()
    km = KMeans(slider_cluster.value)
    km.fit(X_km)
    centers = pd.DataFrame(km.cluster_centers_, columns=X_km.columns)
    X_km['cluster'] = km.labels_

    user = selected["user"]
    FixationDuration = selected["FixationDuration"]
    FixationIndex = selected["FixationIndex"]
    Timestamp = selected['Timestamp']

    X_km = X_km.join(user)
    X_km = X_km.join(FixationDuration)
    X_km = X_km.join(FixationIndex)
    X_km = X_km.join(Timestamp)
    X_km = X_km.reset_index()

    X_km_adj = X_km.copy()

    for i in range(X_km.index[-1] + 1):
        X_km_adj.loc[i, 'cluster'] = X_km_adj['cluster'][i] + 1

    return X_km_adj


# In[15]:


# function calculating matrix dataframe based on slider value & selecting stimulus
def calc_matrix():
    stimulimap_val = stimulimap.value
    n_clusters = slider_cluster.value

    selected = Eyetracking_data.copy()
    if (stimulimap_val != ""):
        selected = selected[selected['StimuliName'].str.contains(stimulimap_val) == True]
    X_km = selected[['MappedFixationPointX', 'MappedFixationPointY']].copy()
    km = KMeans(n_clusters)
    km.fit(X_km)
    centers = pd.DataFrame(km.cluster_centers_, columns=X_km.columns)
    X_km['cluster'] = km.labels_

    user = selected["user"]
    FixationDuration = selected["FixationDuration"]
    X_km = X_km.join(user)

    X_km = X_km.reset_index()

    matrix = []
    AOI = []
    count = 0
    for i in range(0, n_clusters):
        matrix.append([])
        AOI.append(count + 1)
        count += 1
    for i in range(0, n_clusters):
        for j in range(0, n_clusters):
            matrix[i].append(j)
            matrix[i][j] = 0

    cluster = X_km.loc[0, 'cluster']
    user = X_km.loc[0, 'user']
    for n in range(1, X_km.index[-1] + 1):
        cluster_compare = X_km.loc[n, 'cluster']
        user_compare = X_km.loc[n, 'user']
        if cluster != cluster_compare and user == user_compare:
            matrix[cluster][cluster_compare] = matrix[cluster][cluster_compare] + 1
        cluster = cluster_compare
        user = user_compare

    matrix = np.array(matrix)

    m = np.amax(matrix)
    norm_matrix = (1 / m) * matrix
    df_norm_matrix = pd.DataFrame(norm_matrix, index=AOI, columns=AOI)

    matrix_r = df_norm_matrix.reset_index()
    matrix_rows = pd.melt(matrix_r, id_vars=['index'], value_vars=AOI, var_name='target_AOI')

    return matrix_rows


# In[16]:


def database_callback():
    stimulimap_val = stimulimap.value
    selected = data_database.copy()

    if (stimulimap_val != ""):
        selected = selected[selected['StimuliName'].str.contains(stimulimap_val) == True]

    return selected


# In[17]:


# function updating column data source base on slider value & stimulus selection
def update():
    df = calc_clusters()
    df_db = database_callback()
    x2 = list(df_db['MappedFixationX'])
    y2 = list(df_db['MappedFixationY'])
    colorlist = []
    for i in range(df.index[-1] + 1):
        index = df['cluster'][i]
        colorlist.append(colors[index])

    color = colorlist
    source.data = dict(
        x=df['MappedFixationPointX'],
        y=df['MappedFixationPointY'],
        c=df['cluster'],
        u=list(df['user']),
        color=color,
        gradient=df["FixationDuration"],
        index=df["FixationIndex"],
        time=df["Timestamp"]
    )

    # update matrix data source
    matrix_rows = calc_matrix()

    source_matrix.data = dict(
        x=matrix_rows['target_AOI'],
        y=matrix_rows['index'],
        value=matrix_rows['value']
    )

    color_mapper.factors = list(df['user'])
    size_mapper.x = [df.FixationDuration.min(), df.FixationDuration.max()]

    # update database data source
    plot_database = database_callback()
    source_database.data = dict(
        x=x2,
        y=y2,
        color=plot_database["description"],
        user=plot_database["user"],
        gradient=plot_database["FixationDuration"]
    )


# In[18]:
def calc_animation(stimulimap_val):
    userselect_val = select_user.value.strip()

    mapdata = Eyetracking_data.copy()

    if stimulimap_val != "":
        mapdata = mapdata[mapdata['StimuliName'] == stimulimap_val]
        users = mapdata['user'].unique().tolist()
        users = sorted(users, key=lambda x: int("".join([i for i in x if i.isdigit()])))
        users.insert(0, "All")
        select_user.options = users

    if userselect_val != "All":
        mapdata = mapdata[mapdata['user'] == userselect_val]

    userCount = 1
    countList = [1]

    for i in range(1, len(mapdata)):
        if mapdata.iloc[i, 6] == mapdata.iloc[i - 1, 6]:
            userCount = userCount + 1
        else:
            userCount = 1
        countList.append(userCount)

    mapdata['user_index'] = countList
    return mapdata



def animationFunction(trackdata):
    hvData = hv.Dataset(trackdata)

    kdims = ['MappedFixationPointX', 'MappedFixationPointY']
    vdims = ['FixationDuration', 'user', 'StimuliName']

    hvtracking = hvData.to(hv.Points, kdims, vdims, 'user_index')
    
    # Define custom widgets
    def animate_update():
        user_index = animation_slider.value + 1
        if user_index > end:
            user_index = start
        animation_slider.value = user_index

    # Update the holoviews plot by calling update with the user index.
    def slider_update(attrname, old, new):
        hvplot.update((new,))

    callback_id = None

    def animate():
        global callback_id
        print('animate() ', trackData.head())  # This always shows the old stimuli
        if button.label == '► Play':
            button.label = '❚❚ Pause'
            callback_id = doc.add_periodic_callback(animate_update, 600)
        else:
            button.label = '► Play'
            callback_id = doc.remove_periodic_callback(callback_id)

    start, end = hvData.range('user_index')
    animation_slider = Slider(start=start, end=end, value=0, step=1, title="User Index")
    animation_slider.on_change('value', slider_update)

    button.on_click(animate)

    doc = curdoc()
    hvplot = renderer.get_plot(hvtracking, doc)
    hvplot.update((1,))

    plot = layout([[hvplot.state], [animation_slider]])


    return plot


# In[19]:

# layout execution
controls = [stimulimap, select_user, slider_cluster]
for control in controls:
    control.on_change('value', lambda attr, old, new: update())

controls.append(button)

inputs = column(*controls)
# inputs.sizing_mode = "fixed"

temp = calc_animation()
p6 = animationFunction(temp)
print('\nshit: ', temp.head())

#grid = gridplot([[inputs, p1, p3], [None, p2, p4, p5], [None, p6]], plot_width=450, plot_height=400)
grid = gridplot([[inputs, p6]], plot_width=450, plot_height=400)

update()  # initial load of the data

curdoc().add_root(grid)
curdoc().title = "Eyetracking Data Visualizations"

# In[20]:


show(grid)
