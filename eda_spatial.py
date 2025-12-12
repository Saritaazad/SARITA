import os
os.environ['PROJ_LIB'] = r"C:\Users\User\anaconda3\envs\rapids\Lib\site-packages\pyproj\proj_dir\share\proj"


import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from shapely.geometry import Polygon

# Define longitude and latitude grids (adjust to match your data resolution)
longitude = np.linspace(72, 81, 86)  # Example: 86 points of longitude
latitude = np.linspace(28.5, 37, 86)  # Example: 86 points of latitude

# Example REOF pattern data (replace with your actual REOF data)
reof_patterns = np.random.uniform(0, 1, size=(86, 86))  # Replace with actual data

# Create a figure with a Cartopy projection for India
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})

# Set the extent to focus on India (adjust based on your coordinates)
ax.set_extent([72, 81, 28.5, 37], crs=ccrs.PlateCarree())  # [lon_min, lon_max, lat_min, lat_max]

# Plot the REOF mode 2 data (adjust the colormap and levels as needed)
contour = ax.contourf(longitude, latitude, reof_patterns, 
                      levels=np.linspace(-10, 10, 21),  # Adjust levels as per your data
                      cmap='coolwarm', extend='both')

# Add features like coastlines, borders, etc.
# ax.coastlines(resolution='10m', color='black')
# ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle=':', edgecolor='black')

# Load the .shp file using GeoPandas
shp_path = 'shap.shp'  # Add correct path to your shapefile
gdf = gpd.read_file(shp_path, engine='fiona')#.to_crs("EPSG:4326")

# Plot the shapefile data on the Cartopy plot
gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1)


# Add gridlines (optional)
ax.gridlines(draw_labels=True, linewidth=0.5, color='black', alpha=0.5, linestyle='--')

# Show the plot
plt.title('Shapefile')
plt.show()
