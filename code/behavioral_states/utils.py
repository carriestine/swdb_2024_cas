import os
import pandas as pd
from allensdk.brain_observatory.ecephys.behavior_ecephys_session import BehaviorEcephysSession
from scipy.signal.windows import exponential
from scipy.ndimage.filters import convolve1d

cache_path = '/root/capsule/data/visual-behavior-neuropixels/behavior_ecephys_sessions'
cache_metadata_path = '/root/capsule/data/visual-behavior-neuropixels/project_metadata'

def get_session(sessionid):
    
    sessionid = str(sessionid)
    session_file_path = os.path.join(cache_path, sessionid, 'ecephys_session_'+sessionid+'.nwb')
    
    return BehaviorEcephysSession.from_nwb_path(session_file_path)


def get_metadata_table(tablename):
    return pd.read_csv(os.path.join(cache_metadata_path, tablename + '.csv'))

def get_session_table():
    return get_metadata_table('ecephys_sessions')

def get_channels_table():
    return get_metadata_table('channels')

def get_probes_table():
    return get_metadata_table('probes')

def get_units_table():
    return get_metadata_table('units')


def exponential_convolve(response_vector, tau=1, symmetrical=False):
    
    center = 0 if not symmetrical else None
    exp_filter = exponential(10*tau, center=center, tau=tau, sym=symmetrical)
    exp_filter = exp_filter/exp_filter.sum()
    filtered = convolve1d(response_vector,exp_filter[::-1])
    
    return filtered


def baseline_subtract(response_vector, baseline_window=slice(0,20)):
    
    return response_vector - np.mean(response_vector[baseline_window])