# EC_estimator.py
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers.experimental.preprocessing import Normalization, IntegerLookup, Rescaling #CategoryEncoding
from tensorflow.keras.models import Model
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import os


global num_feature_dims

def set_num_feature_dims(d):
    global num_feature_dims
    num_feature_dims = d

# this is kludgy and needs to be populated by a call to calc_lags_feature
lags_feature = None

root_logdir = os.path.join(os.curdir, "tf_training_logs")


def feature_names():
    return list(num_feature_dims.keys())

def root_mean_squared_error(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true)))

def days_to_ops(ser):
    pass

def block_history(series,maxday=7,nblock=10,blocksize=11):
    df = series.to_frame()
    base_name = df.columns[0]
    tups = [(base_name,"0d")]
    for iday in range(1,(maxday+1)):
        lagname = f"{iday}d"
        df[f"{base_name}_{lagname}"] = series.shift(iday)
        tups.append((base_name,lagname))

    past = series.shift(maxday)
    past = past.rolling(blocksize).mean()
    for iday in range(1,(nblock+1)):
        lagname = f"{iday}ave"
        df[f"{base_name}_{lagname}"] = past.shift((iday-1)*blocksize+1)
        tups.append((base_name,lagname))
    indices = pd.MultiIndex.from_tuples(tups,names=["var","lag"])
    df.columns = indices
    return df

def load_data(file_name):
    df = pd.read_csv(file_name)
    cols = df.columns
    old = cols[0] 
    df = df.rename({old: 'date'},axis=1)
    return df

def split_data(df, train_rows, test_rows):
    df_train = df.tail(train_rows)
    df_test = df.head(test_rows)
    return df_train, df_test


def build_model_inputs(df):
    inputs = []
    for feature,fdim in num_feature_dims.items():
        feature_input = Input(shape=(fdim,), name=f"{feature}")
        inputs.append(feature_input)
    return inputs

def calc_lags_feature(df):
    global lags_feature
    lags_feature = {feature: df.loc[:, pd.IndexSlice[feature,:]].columns.get_level_values(level='lag')[0:num_feature_dims[feature]] 
                    for feature in feature_names()}

def df_by_variable(df):
    """ Convert a dataset with a single index with var_lag as column names and convert to MultiIndex with (var,ndx)
        This facilitates queries that select only lags or only variables. As a side effect this routine will store
        the name of the active lags for each feature, corresponding to the number of lags in the dictionary num_feature_dims)
        into the module variable lag_features.

        Parameters
        ----------
        df : pd.DataFrame 
            The DataFrame to be converted

        Returns
        -------
        df_var : A DataFrame with multiIndex based on var,lag  (e.g. 'sac','4d')
    """
    indextups = []
    for col in list(df.columns):
        var = col
        lag = ""
        for key in num_feature_dims.keys():
            if col.startswith(key):
                var = key
                lag = col.replace(key,"").replace("_","")
                if lag is None or lag == "": lag = "0d"
                continue
        if var == "EC": lag = "0d"
        indextups.append((var,lag))
 
    ndx = pd.MultiIndex.from_tuples(indextups, names=('var', 'lag'))
    df.columns = ndx

    # This is a side effect. Maybe improve to function
    calc_lags_feature(df)
    names = feature_names()
    if "EC" in df.columns.get_level_values(0):
        names = names + ["EC"]
    df2=df.reindex(names,axis="columns",level="var")
    
    df2.index=df.date

    return df2

def preprocessing_layers(df_var, inputs,thresh=None):
    global lags_feature
    layers = []
    for fndx,feature in enumerate(feature_names()):
        if lags_feature is None: raise ValueError("lags_feature not calculated yet")
        #print(f"feature: {feature}, lags_feature: {lags_feature[feature]}")
        station_df = df_var.loc[:, pd.IndexSlice[feature,lags_feature[feature]]]
        if feature in ["dcc","smscg"] and False:
            feature_layer = Normalization(axis=None) #Rescaling(1.0)
        elif feature == 'sac' and thresh is not None:
            feature_layer = Rescaling(1/thresh)  #Normalization(axis=None)
        else:
            feature_layer = Normalization(axis=None)
            feature_layer.adapt(station_df.values.reshape(-1, num_feature_dims[feature]))  
            #print("Creating feature")
            #print(feature_layer.mean)
            #print(np.sqrt(feature_layer.variance))
            #print(feature)        
        layers.append(feature_layer(inputs[fndx]))
    return layers





def build_model(layers, inputs):
    """ Builds the standard CalSIM ANN
        Parameters
        ----------
        layers : list  
        List of tf.Layers

        inputs: dataframe
    """        

    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=root_logdir)
    x = tf.keras.layers.concatenate(layers)
    
    # First hidden layer with 8 neurons and sigmoid activation function
    x = Dense(units=8, activation='sigmoid', input_dim=x.shape[1], kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Second hidden layer with 2 neurons and sigmoid activation function
    x = Dense(units=2, activation='sigmoid', kernel_initializer="he_normal",name="hidden")(x) 
    x = tf.keras.layers.BatchNormalization(name="batch_normalize")(x)
    
    # Output layer with 1 neuron
    output = Dense(units=1,name="ec",activation="relu")(x)
    ann = Model(inputs = inputs, outputs = output)

    ann.compile(
        optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001), 
        loss=root_mean_squared_error, 
        metrics=['mean_absolute_error'],
        run_eagerly=True
    )
    
    return ann, tensorboard_cb



def train_model(model, tensorboard_cb, X_train, y_train, X_test, y_test,nepoch=100):
    tf.config.run_functions_eagerly(True)
    tf.data.experimental.enable_debug_mode()
    history = model.fit(
        X_train, y_train, 
        validation_data=(X_test, y_test), 
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", 
            patience=min(nepoch,1000), 
            mode="min", 
            restore_best_weights=True), 
            tensorboard_cb
        ], 
        batch_size=128, 
        epochs=nepoch, 
        verbose=0
    )
    return history, model

def calculate_metrics(model_name, y_train, y_train_pred, y_test, y_test_pred):
    y_train_np = y_train.values.ravel()
    y_train_pred_np = y_train_pred.ravel()

    # Calculate metrics for training data
    r2_train = r2_score(y_train_np, y_train_pred_np)
    rmse_train = np.sqrt(mean_squared_error(y_train_np, y_train_pred_np))
    percentage_bias_train = np.mean((y_train_pred_np - y_train_np) / y_train_np) * 100

    y_test_np = y_test.values.ravel()
    y_test_pred_np = y_test_pred.ravel()

    # Calculate metrics for test data
    r2_test = r2_score(y_test_np, y_test_pred_np)
    rmse_test = np.sqrt(mean_squared_error(y_test_np, y_test_pred_np))
    percentage_bias_test = np.mean((y_test_pred_np - y_test_np) / y_test_np) * 100

    # Return results as a dictionary
    return {
        'Model': model_name,
        'Train_R2': round(r2_train, 2),
        'Train_RMSE': round(rmse_train, 2),
        'Train_Percentage_Bias': round(percentage_bias_train, 2),
        'Test_R2': round(r2_test, 2),
        'Test_RMSE': round(rmse_test, 2),
        'Test_Percentage_Bias': round(percentage_bias_test, 2),
    }

def plot_history(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    print(len(history.history['loss']))
    plt.title('Training and Validation Loss Over Epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    #plt.ylim(0,4)
    #plt.xlim(0,80)
    plt.show()

def save_model(model, model_save_path):
    model.save(model_save_path)
    print(f"Model saved at location: {model_save_path}")

from tensorflow.keras.models import load_model

def load_model(model_path, loss_function):
    model = load_model(model_path, custom_objects={loss_function.__name__: loss_function})
    return model

def make_predictions(model, data, num_features):
    X_new = [data[feature] for feature in num_features]
    predictions = model.predict(X_new)
    return predictions






