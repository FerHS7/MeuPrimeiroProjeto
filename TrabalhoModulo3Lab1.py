#step 1:importing libraries
import dash
from dash import html, dcc
from dash.dependencies import Output, Input

# Initialize the Dash app
app = dash.Dash(__name__)
import folium
import os
import pandas as pd
import wget
from folium.plugins import MarkerCluster,MousePosition
from folium.features import DivIcon
from math import sin, cos, sqrt, atan2, radians

#Step 2: loading the data

spacex_csv_file = wget.download(
    'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/spacex_launch_geo.csv'
)
spacex_df = pd.read_csv(spacex_csv_file)

# Select relevant columns
spacex_df = spacex_df[['Launch Site', 'Lat', 'Long', 'class']]

#step 3: Create a map with locals of the launches
launch_sites_df = spacex_df.groupby('Launch Site', as_index=False).first()[['Launch Site', 'Lat', 'Long']]
site_map = folium.Map(location=[29.559684888503615, -95.0830971930759], zoom_start=5)

# Add place markers
for _, row in launch_sites_df.iterrows():
    folium.Circle([row['Lat'], row['Long']], radius=1000, color='#d35400', fill=True).add_to(site_map)
    folium.map.Marker(
        [row['Lat'], row['Long']],
        icon=DivIcon(icon_size=(20,20), icon_anchor=(0,0),
        html=f'<div style="font-size: 12; color:#d35400;"><b>{row["Launch Site"]}</b></div>')
    ).add_to(site_map)

site_map

#step 4:adding markers the successful and failed
# Add column with color
def assign_marker_color(outcome):
    return 'green' if outcome == 1 else 'red'

spacex_df['marker_color'] = spacex_df['class'].apply(assign_marker_color)

# Create cluster de markers
marker_cluster = MarkerCluster()
site_map.add_child(marker_cluster)

for _, row in spacex_df.iterrows():
    marker = folium.Marker(
        location=[row['Lat'], row['Long']],
        icon=folium.Icon(color=row['marker_color']),
        popup=row['Launch Site']
    )
    marker_cluster.add_child(marker)

site_map

#step 5:Enable mouse position (to get coordinates)
mouse_position = MousePosition(
    position='topright',
    separator=' Long: ',
    prefix='Lat:',
    lat_formatter="function(num) {return L.Util.formatNum(num, 5);}",
    lng_formatter="function(num) {return L.Util.formatNum(num, 5);}"
)
site_map.add_child(mouse_position)

#✅Step 6: Calculate distance between two points
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6373.0  # raio da Terra em km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

#✅ Step 7: Mark point of interest and draw distance line
# Example: coordinates of the point of interest (such as coastline, city, road, etc.)
coast_lat, coast_lon = 28.56367, -80.57163
launch_lat, launch_lon = 28.5623, -80.5774

# Calcular distância
distance = calculate_distance(launch_lat, launch_lon, coast_lat, coast_lon)

# Criar marcador da distância
folium.Marker(
    [coast_lat, coast_lon],
    icon=DivIcon(
        icon_size=(20,20),
        icon_anchor=(0,0),
        html=f'<div style="font-size: 12; color:#d35400;"><b>{distance:.2f} KM</b></div>'
    )
).add_to(site_map)

# Linha entre os pontos
folium.PolyLine(locations=[[launch_lat, launch_lon], [coast_lat, coast_lon]], weight=2).add_to(site_map)

site_map

#Continuity Module 3 lab 2
