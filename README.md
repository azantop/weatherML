# weatherML -  Weather prediction based on historical data
Multi-step time series forecasting with physics informed neural networks (PINN) on multivariate weather data.  
We use the meteostat library to access wordwide hourly and daily historical weather data.  
For data processing we use a combination of Pandas and Numpy. Machine learning is written with Tensorflow.   

For this project, we load historical weather data in Europe on a rectengular 10x10 grid.  
Data processing interpolates missing values (sometimes a stations miss a measurement)
from nearby stations. 

## Input data visualization: Temperature / Wind speed
The historic weather data is taken for all points shown on the following map, along with a contour plot of the temperature.
We use hourly data for 10 years after 2012. Here, we show a day in June. 
<img src="https://github.com/azantop/weatherML/blob/main/images/heatmap.png?raw=true" alt="temperatures" width="600"/>

Wind speed is visualized as streamline plot, where the line thickness corresponds to the wind speed.  
<img src="https://github.com/azantop/weatherML/blob/main/images/windmap.png?raw=true" alt="temperatures" width="600"/>

## Machine Learning Model and Forecast length:

The Model receives the last 72 hours of the weather of all stations 
and makes a forecast of the future 72 hours on the whole map.

## Results: Temperature forecast for a single point:
We observe a good agreement between predicted an actual future weather data.  
<img src="https://github.com/azantop/weatherML/blob/main/images/prediction.png?raw=true" alt="temperatures" width="500"/>

### Results: Temperature forecast on the whole map:
<img src="https://github.com/azantop/weatherML/blob/main/images/prediction_map.png?raw=true" alt="temperatures" width="800"/>
