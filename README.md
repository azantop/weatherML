# weatherML -  Weather prediction based on historical data
Multi-step time series forecasting with physics informed neural networks (PINN) on multivariate weather data.  
We use the meteostat library to access wordwide hourly and daily historical weather data.  
For data processing we use a combination of Pandas and Numpy. Machine learning is written with Tensorflow.   

For this project, we load historical weather data in Europe on a rectengular 10x10 grid.  
Data processing interpolates missing values (sometimes a stations miss a measurement)
from nearby stations. 

## Temperature visualization:
<img src="https://github.com/azantop/weatherML/blob/main/images/heatmap.png?raw=tru" alt="temperatures" width="200"/>

## Wind speed visualization:
![alt text](https://github.com/azantop/weatherML/blob/main/images/windmap.png?raw=true)
