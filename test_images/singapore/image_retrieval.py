#@title imports

import pandas as pd
#import fiona
import geopandas as gpd
import contextily as cx

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from shapely import wkt
from shapely.geometry import Point, Polygon, LineString, GeometryCollection

import os
import numpy as np
import json
import seaborn as sns
import requests

#@title main

import urllib.parse

bipv_image_links = "/Users/durana/Documents/web-tool/test_images/singapore/links.rtf"

def nones(n) -> list:
    return [None for _ in range(n)]

class gsv_image_object(object):
    def __init__(self, url):
        self.url = url
        self.latitude, self.longitude, self.fov, self.heading, self.pitch, self.panoID, self.date = nones(7)

        # Scope 1: main data

        start = url.index('/@') + 2
        finish = url.index('/data')
        chunks = url[start:finish].split(',')

            # Latitude & Longitude
        self.latitude, self.longitude = float(chunks[0]), float(chunks[1])
        for chunk in chunks[3:]:
            char = chunk[-1]
            data = chunk[:-1]
            if char == 'a':
                continue

            # FOV
            if char == 'y':
                self.fov = round(float(data), 2)
                continue

            # Heading
            if char == 'h':
                self.heading = round(float(data), 2)
                continue

            # Pitch
            if char == 't':
                self.pitch = round(float(data) - 90, 2)
                continue

        # Scope 2: additional data

            # PanoID
        self.panoID = urllib.parse.unquote(url.split("!1s")[1].split("!2e")[0])

            # Date
            # Pedram: This part is to extract the date for old GSV photos.
            # The GSV Static API doesn't yet have an implementation for
            # old photos, but keeping this here in case they add it.
        if '!5s' in url and 'T000000' in url:
            start = url.index('!5s') + 3
            finish = url.index('T000000')
            chunk = url[start:finish]
            self.date = f"{chunk[:4]}-{chunk[4:6]}"
    def __str__(self) -> str:
        return f"location: ({self.latitude},{self.longitude}), fov: {self.fov}, heading: {self.heading}, pitch: {self.pitch}, panoID: {self.panoID}, date: {self.date}"

with open(bipv_image_links, 'r') as file:
    lines = file.readlines()

df_index = -1
project_index = None
gsv_link = None
loose_links = []

prev = False
df = pd.DataFrame(columns=['project_index', 'gsv_link', 'loose_links'])

for counter, line in enumerate(lines):
    # value type 1: index integer
    if line.startswith('>'):

        # consolidate loose links from the previous index if they exist
        # consolidate GSV link from the previous index if it exists
        if prev:
            df.at[df_index, 'project_index'] = project_index

            df.at[df_index, 'loose_links'] = loose_links
            df.at[df_index, 'loose_links_count'] = int(len(loose_links))

            df.at[df_index, 'gsv_link'] = gsv_link
            if gsv_link:
                gsv_image = gsv_image_object(gsv_link)
                df.at[df_index, 'gsv_latitude'] = gsv_image.latitude
                df.at[df_index, 'gsv_longitude'] = gsv_image.longitude
                df.at[df_index, 'gsv_fov'] = gsv_image.fov
                df.at[df_index, 'gsv_heading'] = gsv_image.heading
                df.at[df_index, 'gsv_pitch'] = gsv_image.pitch
                df.at[df_index, 'gsv_panoID'] = gsv_image.panoID
                df.at[df_index, 'gsv_date'] = gsv_image.date

            # clear values for next batch
            project_index, gsv_link = None, None
            loose_links = []

        if line == '>end':
            continue

        # collect new indices and update
        df_index += 1
        project_index = line[1:].strip()

        if prev == False:
            prev = True

        continue

    # value type 2: google street view link
    if 'https://www.google.com/maps' in line:
        gsv_link = line

        continue

    # value type 3: loose image link from the web
    loose_links.append(line)

    continue

df['loose_links_count'] = df['loose_links_count'].astype(int)

image_link_df = df.copy()
df = None

print(image_link_df)