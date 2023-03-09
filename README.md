# weatherML -  weather forecast based on historical data

<img src="https://github.com/azantop/weatherML/blob/main/images/forecast.gif?raw=true" alt="temperatures" width="600"/>
Demonstration of multi-step time series forecasting with physics informed neural networks (PINN) on multivariate weather data.  
We use the meteostat library to access wordwide hourly and daily historical weather data.  
For data processing we use a combination of Pandas and Numpy. Machine learning is written with Tensorflow.   

For this project, we load historical weather data in Europe on a rectengular 10x10 grid.  
Data processing interpolates missing values from nearby stations. 

## Input data visualization: Temperature / Wind speed
The historic weather data is taken for all points shown on the following map,  
along with a contour plot of the temperature.  
We use hourly data for 10 years after 2012. Here, we show a day in June. 
<img src="https://github.com/azantop/weatherML/blob/main/images/heatmap.png?raw=true" alt="temperatures" width="600"/>

Wind speed is visualized as streamline plot, where the line thickness corresponds to the wind speed.  
<img src="https://github.com/azantop/weatherML/blob/main/images/windmap.png?raw=true" alt="temperatures" width="600"/>

## Machine Learning Model and Forecast length:

The model receives the last 72 hours of the weather of all stations and makes a forecast  
of the future 72 hours on the whole map. The model is defined as follows: 
* a time distributed spatial encoder consisting of locally connected and max pooling layers applied to the grid
* a LSTM recurrent layer for the temporal dynamics 
* an decoder consisting of a dense layer and a de-convolution layer applied.
<pre>
<code>def build_model( grid_size, channels, features, past, future ):
    """Creates tensorflow model 
    Params
        ======
            grid_size: length of the 2D input grid
            channels: number of input data channels
            features: features represented in the output of the model
            past: hours of data
            future: hours of forecast
    """
    grid_past  = layers.Input((past,grid_size,grid_size,channels), name="grid_past") 
    grid_now   = layers.Input((grid_size,grid_size,channels), name="grid_now")
    
    # encoder for the grid state:
    grid_encoder = layers.TimeDistributed( layers.LocallyConnected2D( channels, (3,3) ), name="local2D_1" )( grid_past )
    grid_encoder = layers.TimeDistributed( layers.MaxPooling2D(), name="max_pooling_1" )( grid_encoder )
    grid_encoder = layers.TimeDistributed( layers.LocallyConnected2D( channels, (3,3) ), name="local2D_2" )( grid_encoder )
    grid_encoder = layers.TimeDistributed( layers.MaxPooling2D(), name="max_pooling_2" )( grid_encoder )
    grid_encoder = layers.TimeDistributed( layers.Flatten() )( grid_encoder )
    
    # recurrent network for the temporal dynamics:
    time_encoder = layers.LSTM( 64, name="recurrent", return_sequences=True )( grid_encoder ) 
    time_encoder = layers.Flatten()( time_encoder )
    
    # merge with the grid state of current weather state:
    merge_layer_grid = layers.Concatenate(name="concat")( [ time_encoder, layers.Flatten()( grid_now ) ] )   
    
    # decoder: 
    output_grid = layers.Dense( future*(grid_size-2)*(grid_size-2)*features )( merge_layer_grid )
    output_grid = layers.Reshape( (future,(grid_size-2),(grid_size-2),features) )( output_grid )
    output_grid = layers.TimeDistributed( tf.keras.layers.Conv2DTranspose( features, 3, input_shape=(8,8,1) ), 
                                          name="upscaler"  )( output_grid )
     
    return keras.Model( [grid_past,grid_now], [output_grid] )</code>
</pre>
This architecture results in 40,186,236 trainalble parameters.

## Results: Temperature forecast for a single point:
We observe a good agreement between predicted an actual future weather data.  
<img src="https://github.com/azantop/weatherML/blob/main/images/forecast.png?raw=true" alt="temperatures" width="500"/>

## Results: Temperature forecast on the whole map:
The complete ouptut of the network looks as follows:
<img src="https://github.com/azantop/weatherML/blob/main/images/map_forecast.png?raw=true" alt="temperatures" width="800"/>  
Drawn on the world map, we obtain the video shown above:  
<img src="https://github.com/azantop/weatherML/blob/main/images/forecast.gif?raw=true" alt="temperatures" width="600"/>

