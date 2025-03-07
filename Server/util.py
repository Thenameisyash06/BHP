import json
import pickle
import numpy as np

__location = None
__data_columns = None
__model = None

def get_estimated_price(location,sqft,bhk,bath):
    try:
        loc_index = __data_columns.index(location.lower()) # type: ignore
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns)) # type: ignore
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index>=0:
        x[loc_index] = 1

    return round(__model.predict([x])[0],2) # type: ignore

def get_location_names():
    return __location

def get_data_columns():
    return __data_columns

def load_saved_artifacts():
    print("Loading Artifacts....Start")
    global __location
    global __data_columns
    global __model

    with open("./Server/artifacts/columns.json",'r') as f:
       __data_columns = json.load(f)['data_columns']
       __location = __data_columns[3:]

    with open("./Server/artifacts/banglore_house_price_prediction.pickle",'rb') as f:
        __model = pickle.load(f)
    print("Loading Artifacts....Done")

if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('1st Phase JP Nagar',1000, 3, 3))
    print(get_estimated_price('1st Phase JP Nagar', 1000, 2, 2))
    print(get_estimated_price('Kalhalli', 1000, 2, 2)) # other location
    print(get_estimated_price('Ejipura', 1000, 2, 2))  # other location