import pandas as pd
import geo_cal as geo
import h5py


arrays = {}
pairs = pd.read_csv('data/pairs_from_clean_1_300.csv')
base = h5py.File('data/baseID2xy.mat')['baseID2xy']

dis = []
for i in range(pairs.shape[0]):
    w_lng = base[1][list(base[0]).index(pairs.iloc[i]['work_station_code'])]
    w_lat = base[2][list(base[0]).index(pairs.iloc[i]['work_station_code'])]
    h_lng = base[1][list(base[0]).index(pairs.iloc[i]['home_station_code'])]
    h_lat = base[2][list(base[0]).index(pairs.iloc[i]['work_station_code'])]
    pairs['w_lng'] = w_lng
    pairs['w_lat'] = w_lat
    pairs['h_lng'] = h_lng
    pairs['h_lat'] = h_lat
    dis.append(geo.Manhattan(w_lng, w_lat, h_lng, h_lat))
pairs['dis'] = dis
pairs.to_csv('data/pairs_from_clean_1_300.csv', index=None)
