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
The model is defined as follows:
<code>
def build_model():
    channels = 6
    features = 3
    past = 72
    future = 72
    grid_past  = layers.Input((past,grid_size,grid_size,channels), name="grid_past") 
    grid_now   = layers.Input((grid_size,grid_size,channels), name="grid_now")
    dropout_rate = 0.0
    
    # encoder for the grid state:
    grid_encoder = layers.TimeDistributed( layers.LocallyConnected2D( channels, (3,3) ), name="local2D_1" )( grid_past )
    grid_encoder = layers.TimeDistributed( layers.MaxPooling2D(), name="max_pooling_1" )( grid_encoder )
    grid_encoder = layers.TimeDistributed( layers.LocallyConnected2D( channels, (3,3) ), name="local2D_2" )( grid_encoder )
    grid_encoder = layers.TimeDistributed( layers.MaxPooling2D(), name="max_pooling_2" )( grid_encoder )
    grid_encoder = layers.TimeDistributed( layers.Flatten() )( grid_encoder )
    
    # recurrent network for the temporal dynamics:
    time_encoder = layers.LSTM( 32, name="recurrent", return_sequences=True )( grid_encoder ) 
    time_encoder = layers.Flatten()( time_encoder )
    
    # merge with the grid state of current weather state:
    merge_layer_grid = layers.Concatenate(name="concat")( [ time_encoder, layers.Flatten()( grid_now ) ] )   
    
    # decoder: 
    output_grid = layers.Dense( future*(grid_size-2)*(grid_size-2)*features )( merge_layer_grid )
    output_grid = layers.Reshape( (future,(grid_size-2),(grid_size-2),features) )( output_grid )
    output_grid = layers.TimeDistributed( tf.keras.layers.Conv2DTranspose( features, 3, input_shape=(8,8,1) ), 
                                          name="upscaler"  )( output_grid )
     
    return keras.Model( [grid_past,grid_now], [output_grid] )  
</code>
This architecture results in 40,186,236 trainalble parameters.

## Results: Temperature forecast for a single point:
We observe a good agreement between predicted an actual future weather data.  
<img src="https://github.com/azantop/weatherML/blob/main/images/prediction.png?raw=true" alt="temperatures" width="500"/>

## Results: Temperature forecast on the whole map:
<img src="https://github.com/azantop/weatherML/blob/main/images/map_prediction.png?raw=true" alt="temperatures" width="800"/>
