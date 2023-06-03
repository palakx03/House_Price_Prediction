import json
import pickle
import numpy as np

__locations = None
__data_columns = None
__model = None

def get_estimated_price(location,sqft,bhk,bath):
    try: 
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1
    
    x_values= np.zeros(len(__data_columns))
    x_values[0]= sqft
    x_values[1] = bath
    x_values[2] = bhk
    if loc_index>=0:
        x_values[loc_index] = 1

    return round(__model.predict([x_values])[0],2)


def get_location_names():
    return __locations

def load_saved_artifacts():
    print("Loading saved artifacts..start")
    global __data_columns
    global __locations

    with open("./artifacts/columns.json",'r') as f:
         __data_columns = json.load(f)['data_columns']
         __locations = __data_columns[3:]

    global __model
    if __model is None:
        with open("./artifacts/Bangluru_House_Data_Model.pickle",'rb') as f:
            __model = pickle.load(f)
    print("Loading saves artifacts..done") 

def get_location_names():
    return __locations

def get_data_columns():
    return __data_columns
if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('1st Phase JP Nagar', 1000, 3, 2))
    print(get_estimated_price('1st Phase JP Nagar', 1000, 2, 2))
    print(get_estimated_price('Kalhalli', 1000, 2, 2))   #other location
    print(get_estimated_price('Ejipura', 1000, 2, 2))