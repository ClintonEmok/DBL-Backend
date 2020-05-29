import numpy as np
import pandas as pd
import bokeh as bk
import random
import numpy as np

import networkx as nx

from graphviz import Digraph
import pydot
from networkx.drawing.nx_agraph import graphviz_layout

from networkx.drawing.nx_pydot import graphviz_layout

import holoviews as hv
from holoviews import opts, dim
from holoviews.element.graphs import layout_nodes

from pprint import pprint

import math
#%matplotlib inline
import matplotlib.pyplot as plt

import matplotlib.cm as cm
import matplotlib as mpl
import seaborn as sns
sns.set()  # set Seaborn defaults
plt.rcParams['figure.figsize'] = 10, 5  # default hor./vert. size of plots, in inches
plt.rcParams['lines.markeredgewidth'] = 1  # to fix issue with seaborn box plots; needed after import seaborn
from sklearn.cluster import KMeans  # for clustering

#from bkcharts import Chord
from bokeh.io import output_notebook, show, reset_output, curdoc
from bokeh.models import Slider, CustomJS, Select, Arrow, NormalHead
from bokeh.plotting import figure, from_networkx
from bokeh.layouts import layout, column
from bokeh.models import (
    ColumnDataSource, Div,
    HoverTool,
    LinearColorMapper,
    BasicTicker,
    PrintfTickFormatter,
    ColorBar,
    FactorRange,
    Oval,
    GraphRenderer,
    StaticLayoutProvider,
    ImageURL
)
from bokeh.palettes import BuPu, Colorblind8, Spectral8
from bokeh.palettes import Colorblind8

#import eye tracking data
Eyetracking_data = pd.read_csv('metro_data.csv', encoding = 'latin1', sep = ";")
Eyetracking_data.head()

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

# Grouping data per map
Antwerpen_S1 = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '01_Antwerpen_S1.jpg']
Antwerpen_S2 = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '01_Antwerpen_S2.jpg']

Antwerpen_S1b = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '01b_Antwerpen_S1.jpg']
Antwerpen_S2b = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '01b_Antwerpen_S2.jpg']

Berlin_S1 = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '02_Berlin_S1.jpg']
Berlin_S2 = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '02_Berlin_S2.jpg']

Berlin_S1b = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '02b_Berlin_S1.jpg']
Berlin_S2b = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '02b_Berlin_S2.jpg']

Bordeaux_S1 = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '03_Bordeaux_S1.jpg']
Bordeaux_S2 = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '03_Bordeaux_S2.jpg']

Bordeaux_S1b = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '03b_Bordeaux_S1.jpg']
Bordeaux_S2b = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '03b_Bordeaux_S2.jpg']

Köln_S1 = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '04_Koln_S1.jpg']
Köln_S1b = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '04b_Koln_S1.jpg']
Köln_S2b = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '04b_Koln_S2.jpg']

Frankfurt_S1 = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '05_Frankfurt_S1.jpg']
Frankfurt_S2 = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '05_Frankfurt_S2.jpg']

Frankfurt_S1b = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '05b_Frankfurt_S1.jpg']
Frankfurt_S2b = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '05b_Frankfurt_S2.jpg']

Hamburg_S1 = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '06_Hamburg_S1.jpg']
Hamburg_S2 = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '06_Hamburg_S2.jpg']

Hamburg_S1b = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '06b_Hamburg_S1.jpg']
Hamburg_S2b = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '06b_Hamburg_S1.jpg']

Moskau_S1 = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '07_Moskau_S1.jpg']
Moskau_S2 = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '07_Moskau_S2.jpg']

Moskau_S1b = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '07b_Moskau_S1.jpg']
Moskau_S2b = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '07b_Moskau_S2.jpg']

Riga_S1 = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '08_Riga_S1.jpg']
Riga_S2 = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '08_Riga_S1.jpg']

Riga_S1b = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '08b_Riga_S1.jpg']
Riga_S2b = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '08b_Riga_S1.jpg']
Tokyo_S1 = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '09_Tokyo_S1.jpg']
Tokyo_S2 = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '09_Tokyo_S2.jpg']

Tokyo_S1 = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '09b_Tokyo_S1.jpg']
Tokyo_S2 = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '09b_Tokyo_S2.jpg']

Barcelona_S1 = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '10_Barcelona_S1.jpg']
Barcelona_S2 = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '10_Barcelona_S2.jpg']

Barcelona_S1 = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '10b_Barcelona_S1.jpg']
Barcelona_S2 = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '10b_Barcelona_S2.jpg']

Bologna_S1 = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '11_Bologna_S1.jpg']
Bologna_S2 = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '11_Bologna_S2.jpg']

Bologna_S1b = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '11b_Bologna_S1.jpg']
Bologna_S2b = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '11b_Bologna_S2.jpg']

Brüssel_S1 = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '12_Brüssel_S1.jpg']
Brüssel_S2 = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '12_Brüssel_S2.jpg']

Brüssel_S1b = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '12b_Brüssel_S1.jpg']
Brüssel_S2b = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '12b_Brüssel_S2.jpg']

Budapest_S1 = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '13_Budapest_S1.jpg']
Budapest_S2 = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '13_Budapest_S2.jpg']

Budapest_S1b = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '13b_Budapest_S1.jpg']
Budapest_S2b = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '13b_Budapest_S2.jpg']

Düsseldorf_S1 = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '14_Dusseldorf_S1.jpg']
Düsseldorf_S2 = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '14_Dusseldorf_S2.jpg']

Düsseldorf_S1b = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '14b_Dusseldorf_S1.jpg']
Düsseldorf_S2b = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '14b_Dusseldorf_S2.jpg']

Göteborg_S1 = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '15_Goteborg_S1.jpg']
Göteborg_S2 = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '15_Goteborg_S2.jpg']

Göteborg_S1b = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '15b_Göteborg_S1.jpg']
Göteborg_S2b = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '15b_Göteborg_S2.jpg']

Hong_Kong_S1 = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '16_Hong_Kong_S1.jpg']
Hong_Kong_S2 = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '16_Hong_Kong_S2.jpg']

Hong_Kong_S1b = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '16b_Hong_Kong_S1.jpg']
Hong_Kong_S2b = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '16b_Hong_Kong_S2.jpg']

Krakau_S1 = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '17_Krakau_S1.jpg']
Krakau_S2 = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '17_Krakau_S2.jpg']

Krakau_S1b = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '17b_Krakau_S1.jpg']
Krakau_S2b = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '17b_Krakau_S2.jpg']

Ljubljana_S1 = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '18_Ljubljana_S1.jpg']
Ljubljana_S2 = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '18_Ljubljana_S2.jpg']

Ljubljana_S1b = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '18_Ljubljana_S1.jpg']
Ljubljana_S2b = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '18_Ljubljana_S2.jpg']

New_York_S1 = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '19_New_York_S1.jpg']
New_York_S2 = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '19_New_York_S2.jpg']

New_York_S1b = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '19b_New_York_S1.jpg']
New_York_S2b = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '19b_New_York_S2.jpg']

Paris_S1 = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '20_Paris_S1.jpg']
Paris_S2 = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '20_Paris_S2.jpg']

Paris_S1b = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '20b_Paris_S1.jpg']
Paris_S2b = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '20b_Paris_S2.jpg']

Pisa_S1 = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '21_Pisa_S1.jpg']
Pisa_S2 = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '21_Pisa_S2.jpg']

Pisa_S1b = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '21b_Pisa_S1.jpg']
Pisa_S2b = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '21b_Pisa_S2.jpg']

Venedig_S1 = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '22_Venedig_S1.jpg']
Venedig_S2 = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '22_Venedig_S2.jpg']

Venedig_S1b = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '22b_Venedig_S1.jpg']
Venedig_S2b = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '22b_Venedig_S2.jpg']

Warschau_S1 = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '23_Warschau_S1.jpg']
Warschau_S2 = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '23_Warschau_S2.jpg']

Warschau_S1 = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '23b_Warschau_S1.jpg']
Warschau_S2 = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '23b_Warschau_S2.jpg']

Zürich_S1 = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '24_Zürich_S1.jpg']
Zürich_S2 = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '24_Zürich_S2.jpg']

Zürich_S1b = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '24b_Zürich_S1.jpg']
Zürich_S2b = Eyetracking_data.loc[Eyetracking_data['StimuliName'] == '24b_Zürich_S2.jpg']

n_clusters = 6 #input
selected = Antwerpen_S1.copy()

#selected = Eyetracking_data.copy()
X_km = selected[['MappedFixationPointX', 'MappedFixationPointY']].copy()
km = KMeans(n_clusters)
km.fit(X_km)
centers = pd.DataFrame(km.cluster_centers_, columns=X_km.columns)
X_km['cluster'] = km.labels_

user = selected["user"]
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
    if cluster != cluster_compare and user == user_compare:  # I think this solves the problem with the increment between users
        matrix[cluster][cluster_compare] = matrix[cluster][cluster_compare] + 1
    cluster = cluster_compare
    user = user_compare

matrix = np.array(matrix)

m = np.amax(matrix)
norm_matrix = (1 / m) * matrix
df_norm_matrix = pd.DataFrame(norm_matrix, index=AOI, columns=AOI)

print(norm_matrix)
print(type(norm_matrix))

print(df_norm_matrix)

matrix_r = df_norm_matrix.reset_index()
matrix_rows = pd.melt(matrix_r, id_vars=['index'], value_vars=AOI, var_name='target_AOI')

#print(matrix_rows)

# Above was basically Jane's code without interactivity, here come's mine:

#making a dataframe without zeros, makes plot clearer
matrix_nozero = matrix_rows[matrix_rows['value'] > 0]
#print(matrix_nozero)

end_matrix = matrix_nozero.copy()
#end_matrix['max_value'] = end_matrix.groupby('index')['value'].transform('max')
end_matrix = end_matrix.loc[end_matrix['value'] == end_matrix.groupby('index')['value'].transform('max')]

#print(end_matrix)



#graph with networkx
states = ['1', '2', '3', '4', '5', '6'] #un hard code this

def _get_markov_edges(Q):
    edges = {}
    for col in Q.columns:
        for idx in Q.index:
            edges[(idx,col)] = Q.loc[idx,col]
    return edges

edges_wts = _get_markov_edges(df_norm_matrix)
pprint(edges_wts)


# create graph object
G = nx.MultiDiGraph()

# nodes correspond to states
G.add_nodes_from(states)
print('Nodes:\n{G.nodes()}\n')

# edges represent transition probabilities
for k, v in edges_wts.items():
    tmp_origin, tmp_destination = k[0], k[1]
    G.add_edge(tmp_origin, tmp_destination, weight=v, label=v)
#print('Edges:')
end_graph = G
pprint(G.edges(data=True))

pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='dot')
nx.draw_networkx(G, pos)



# create edge labels for jupyter plot but is not necessary
edge_labels = {(n1, n2): d['label'] for n1, n2, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
end_graph = nx.drawing.nx_pydot.write_dot(G, 'pet_dog_markov.dot')
#plt.show()

plot = figure(title="Networkx Integration Demonstration", x_range=(-5, 5), y_range=(-5, 5),
              tools="", toolbar_location=None)

graph = from_networkx((G, pos), nx.circular_layout, scale=2, center=(0,0))
plot.renderers.append(graph)

show(plot)