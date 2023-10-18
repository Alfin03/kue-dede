import streamlit as st
import pandas as pd
import numpy
import openpyxl


y=0
x=0
time = 0
dfa = []
data = []

# Allow only .csv and .xlsx files to be uploaded
uploaded_file = st.file_uploader("Upload spreadsheet", type=["xlsx"])

# Check if file was uploaded
if uploaded_file:
    # Check MIME type of the uploaded file
    if uploaded_file.type == "text/csv":
        df1 = pd.read_csv(uploaded_file)
        y=1
    else:
        #df1 = pd.read_excel(uploaded_file)
        workbook = openpyxl.load_workbook(uploaded_file)
        for sheet in workbook:
            dfa.append(pd.read_excel(uploaded_file, sheet_name=sheet.title))
            data = pd.concat(dfa, axis=0)
        
        option = st.selectbox(
            'Masukkan Periode peramalan',
            ('7 hari','10 hari', '14 hari'))
        if st.button('Mulai'):
            if option=='7 hari':
                y=7
            if option=='10 hari':
                y=10
            if option=='14 hari':
                y=14    
            #st.title("Ini adalah dataset")    
            #data
            x=1
        
        
         



import numpy as np
#import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


#nilai = []
#xyz = nilai   
#workbook = openpyxl.load_workbook(xyz)

#dfs = []

#for sheet in workbook:
    #dfs.append(pd.read_excel(xyz, sheet_name=sheet.title))

#data = pd.concat(dfs, axis=0)
if x==1:
    data['small_holiday'] = (data['Holiday'] == 1).astype(int)
    data['big_holiday'] = (data['Holiday'] == 2).astype(int)
    data.drop('Holiday', axis=1, inplace=True)
    data.set_index('TANGGAL', inplace=True)
    variables = ['KUE SUS', 'BIKANGDOANG', 'DONUT COKLAT', 'PUDDING',
       'RESOLES', 'ROTI KEJU', 'ROTI AYAM', 'BLUDER', 'SIRAM COKLAT',
       'APANG PARANGGI','small_holiday', 'big_holiday']
    def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()

        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [(f'{data.columns[j]}(t-{i})') for j in range(n_vars)]

        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [(f'{data.columns[j]}(t)') for j in range(n_vars)]
            else:
                names += [(f'{data.columns[j]}(t+{i})') for j in range(n_vars)]

        agg = pd.concat(cols, axis=1)
        agg.columns = names

        if dropnan:
            agg.dropna(inplace=True)
        return agg


    time_steps=14

    data_supervised = series_to_supervised(data[variables], n_in=time_steps, n_out=1)

    train_size = int(len(data_supervised) * 0.8)
    train_data = data_supervised[:train_size]
    val_data = data_supervised[train_size:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_data_scaled = scaler.fit_transform(train_data)
    val_data_scaled = scaler.transform(val_data)

    train_X  = train_data_scaled[:, :-len(variables)]
    train_y = train_data_scaled[:, -len(variables):]
    val_X  = val_data_scaled[:, :-len(variables)]
    val_y = val_data_scaled[:, -len(variables):]
#input data to 3D arrays
    train_X = train_X.reshape((train_X.shape[0], time_steps, len(variables)))
    val_X = val_X.reshape((val_X.shape[0], time_steps, len(variables)))

    # Reshape the input data
    train_X_reshaped = train_X.reshape(train_X.shape[0], -1)
    val_X_reshaped = val_X.reshape(val_X.shape[0], -1)

    # Create the model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(168, activation='relu', input_shape=(time_steps * len(variables),)))

    model.add(tf.keras.layers.Dense(64, activation='relu')) 

    model.add(tf.keras.layers.Dense(len(variables)))
    model.compile(loss='mse', optimizer='adam')

    # Train the model
    history = model.fit(train_X_reshaped, train_y, epochs=80, batch_size=64, validation_data=(val_X_reshaped, val_y), verbose=1)


    look_back = model.input_shape[1]

    current_input = val_X_reshaped
    print(current_input.shape)

    forcasts= model.predict(current_input)

    dates = data.index[-len(val_y):]
    forecast_dates = pd.date_range(start=dates[-1], periods=len(forcasts)+1, freq='D')[1:]

    forecast_df = pd.DataFrame(forcasts, columns=[variables[i] for i in range(forcasts.shape[1])])

    arr = forecast_df.values
    arr = np.resize(arr, (time_steps, time_steps * 12 +12 ))
    values_df=pd.DataFrame(scaler.inverse_transform(arr))

    values_df[values_df<0] = 0
    values_df = np.rint(values_df)

    values_df=values_df.iloc[:,0:10+2]

    values_df.columns = forecast_df.columns[:12]
    dates = pd.to_datetime(forecast_dates[:14])
    
    values_df['Tanggal'] = dates
    st.title("Berikut ini adalah hasil peramalan")
    if y==10:
        values_df[["Tanggal","KUE SUS", "BIKANGDOANG", "DONUT COKLAT", "PUDDING", "RESOLES", "ROTI KEJU", "ROTI AYAM", "BLUDER"
          , "SIRAM COKLAT", "APANG PARANGGI"]].iloc[:10]
    if y==7:
        values_df[["Tanggal","KUE SUS", "BIKANGDOANG", "DONUT COKLAT", "PUDDING", "RESOLES", "ROTI KEJU", "ROTI AYAM", "BLUDER"
            , "SIRAM COKLAT", "APANG PARANGGI"]].iloc[:7]
    if y==14:
        values_df[["Tanggal","KUE SUS", "BIKANGDOANG", "DONUT COKLAT", "PUDDING", "RESOLES", "ROTI KEJU", "ROTI AYAM", "BLUDER"
            , "SIRAM COKLAT", "APANG PARANGGI"]].iloc[:14]

    st.write(values_df)

    #import matplotlib.pyplot as plt
    #fig_size = (9, 3)  
    #for a, variable in enumerate(variables):
        #plt.figure(figsize=fig_size)  
        #plt.plot(dates,val_y[:, a], label='Actual')
        #plt.plot(forecast_dates,np.array(forcasts)[:, a], label='Predicted')
        #plt.title(variable)
 
        #plt.ylim(0, 1)
        #plt.xlabel("Time steps")
        #plt.ylabel(variable+" values")
        #plt.legend()
        #plt.show()

    # streamlit_app.py

    
