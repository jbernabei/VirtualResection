#!/usr/bin/python
'''
Utility module run Echobase operations on ECoG network.
'''

import os
import sys
import json
import h5py
import re
import uuid
import datetime

from multiprocessing import Pool
import warnings
import nibabel as nib
import matplotlib.pyplot as plt

import numpy as np
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats
from scipy.ndimage import morphology
from scipy.signal import convolve

import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
# Delete below line
import sys
print sys.path
sys.path.append('/home1/j/johnbe/gdrive/util/Echobase')

#from Echobase import *
import Echobase
# Delete above line
from Echobase.Common import errors
from Echobase.Network.Metrics.globaltopo import synchronizability
from Echobase.Network.Metrics.nodetopo import node_control
from Echobase.Network.Transforms import lesion
from Echobase.Network.Rewire import geometry
from Echobase.Pipelines.ecog_network import *

np.random.seed(sum(map(ord, "aesthetics")))

with open('../data/DATA.json') as json_data_file:
    data = json.load(json_data_file)

warnings.filterwarnings('ignore')

## Utility Functions
def correspond_label_names(eeg_channel_labels, cartoon_map_labels):
    '''
    Function for corresponding channel labels from ECoG (usually set as a montage) and electrode labels on clinical cartoon maps.
    Parameters
    ----------
        eeg_channel_labels: list of str
            Channel labels as found on IEEG.org.

        cartoon_map_labels: list of str
            Electrode labels from cartoon map. Does not have to be complete set of labels.

    Returns
    -------
        dict
            Dictionary contains as key the cartoon map label and its corresponding EEG channel label along with index in eeg_channel_labels
    '''
    labels = {}
    for ii,channel_label in enumerate(eeg_channel_labels):
        for cartoon_label in cartoon_map_labels:
            if('EEG' in channel_label and 'Ref' in channel_label):
                for p in re.findall(r'EEG[ ]*([A-Za-z]+)[ _-]*([0-9]{0,2})-Ref',channel_label):
                    channel_label_prefix = p[0]
                    try:
                        channel_label_num = str(int(p[1]))
                    except ValueError:
                        channel_label_num = ''

                    cartoon_label_prefix = re.match(r'([A-Za-z]+)([0-9]*)',cartoon_label).group(1)
                    try:
                        cartoon_label_num = re.match(r'([A-Za-z]+)([0-9]*)',cartoon_label).group(2)
                    except AttributeError:
                        cartoon_label_num = ''

                    if channel_label_prefix == cartoon_label_prefix and channel_label_num == cartoon_label_num:
                        labels[cartoon_label] = (ii,channel_label)
            else:
                if(channel_label == cartoon_label): # For CHOP patients ???
                    labels[cartoon_label] = (ii,channel_label)
                else:
                    for p in re.findall(r'([A-Za-z]+)[ _-]*([0-9]{0,2})',channel_label):
                        channel_label_prefix = p[0]
                        try:
                            channel_label_num = str(int(p[1]))
                        except ValueError:
                            channel_label_num = ''

                        cartoon_label_prefix = re.match(r'([A-Za-z]+)([0-9]*)',cartoon_label).group(1)
                        try:
                            cartoon_label_num = re.match(r'([A-Za-z]+)([0-9]*)',cartoon_label).group(2)
                        except AttributeError:
                            cartoon_label_num = ''

                        if channel_label_prefix == cartoon_label_prefix and channel_label_num == cartoon_label_num:
                            labels[cartoon_label] = (ii,channel_label)
    return labels

def get_outcome(outcome):
    """
    Function for determing poor and favorable surgical outcome.
    Parameters
    ----------
        outcome: str
            Surgical outcome as either Engel or ILAE.

    Returns
    -------
        str
            Returns either good or bad.
    """
    switcher = {
        '1': 'Good',
        '1.1': 'Good',
        '1.2': 'Good',
        '1.3': 'Good',
        '1.4': 'Good',
        '1A': 'Good',
        '1B': 'Good',
        '1C': 'Good',
        '1D': 'Good',
        'IA': 'Good',
        'IB': 'Good',
        'IC': 'Good',
        'ID': 'Good',
        '2': 'Poor',
        '2.1': 'Poor',
        '2.2': 'Poor',
        '2.3': 'Poor',
        '2.4': 'Poor',
        '2a': 'Poor',
        '2b': 'Poor',
        '2c': 'Poor',
        '2d': 'Poor',
        '3': 'Poor',
        '4': 'Poor',
        'II': 'Poor',
        'III': 'Poor',
        'IV': 'Poor',
        'ILAE1': 'Good',
        'ILAE2': 'Good',
        'ILAE3': 'Poor',
        'ILAE4': 'Poor',
        'ILAE5': 'Poor'
    }

    return switcher.get(outcome, "Good")

def mean_confidence_interval(data, confidence=0.95):
    """
    Function for computing confidence interval.
    Parameters
    ----------
        data: 1D array
            Data array to compute the CI.

        confidence: float
            Percent confidence

    Returns
    -------
        tuple
            The mean, lower bound and higher bound for the confidence interval.
    """
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h

def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p)>=0

def patient_electrodes(patient_id,dilate_radius=0, data=data):
     # Generate list of cartoon map labels
    labels = map(lambda x: x.split(',')[4].replace('\n',''), open(os.path.expanduser(
        data['PATIENTS'][patient_id]['ELECTRODE_LABELS']
        ),'r').readlines())

    # Get path
    comp_dir = os.path.expanduser(data['COMP_DIR'])
    data_dir = os.path.expanduser(data['DATA_DIR'])

    # Load ignored node labels
    ignored_node_labels = data['PATIENTS'][patient_id]['IGNORE_ELECTRODES']
    for ignored_node_label in ignored_node_labels:
        if(ignored_node_label not in labels):
            labels.append(ignored_node_label)

    # Create output UUID codx
    unique_idx = []

    # Load ictal clips and get data as T x N for T = epoch_length (seconds) * fs
    for event_type, events in data['PATIENTS'][patient_id]['Events'].items():
        unique_id = str(uuid.uuid4())
        for event_id in events.keys():
            try:
                if(events[event_id]['STATUS'] == 'ALL_DROPOUT'):
                        continue # unusable clip
            except KeyError:
                pass

            fn = os.path.join(data_dir, patient_id, 'eeg', events[event_id]['FILE'])
            channels = []

            # Get channels, ECoG Data, Fsx
            with h5py.File(fn) as f:
                evData = f['evData'].value
                Fs = f['Fs'].value
                for column in f['channels']:
                    row_data = []
                    for row_number in range(len(column)):
                        row_data.append(''.join(map(unichr, f[column[row_number]][:])))
                    channels.append(row_data)
            Fs = int(Fs[0][0])
            channels = channels[0]
            # evData = scipy.stats.zscore(evData,axis=1)
            T = evData.shape[0]

            # Correspond label names
            labels_dict = correspond_label_names(channels, labels)

            # Load electrodes to ignore
            ignored_node_idx  = map(lambda x: labels_dict[x][0], ignored_node_labels)
            for ii,node_id in enumerate(ignored_node_idx):
                print 'Ignoring node label: %s because label %s is in IGNORE_ELECTRODES'%(channels[node_id],ignored_node_labels[ii])
            channels = list(np.delete(np.array(channels),ignored_node_idx))

            # Recorrespond label names
            labels_dict = correspond_label_names(channels, labels)

            # Generate list of resected electrodes and write to CSV file
            try:
                if(dilate_radius == 0):
                    resected_node_labels = data['PATIENTS'][patient_id]['RESECTED_ELECTRODES']
                elif(dilate_radius > 0 or dilate_radius < 0):
                    resected_node_labels = data['PATIENTS'][patient_id]['RESECTED_ELECTRODES']
                    for fringe_node_label in data['PATIENTS'][patient_id]['RESECTED_FRINGE_ELECTRODES']:
                        resected_node_labels.append(fringe_node_label)
                else:
                    return []
            except Exception:
                resected_electrodes_fn = write_resected_electrodes(patient_id, dilate_radius, data, labels_dict)

                # Load resected electrodes
                try:
                    resected_nodes = map(lambda x: int(x.split(',')[0]), open(resected_electrodes_fn,'r').readlines())
                    resected_node_labels = map(lambda x: x.split(',')[1].replace('\n',''), open(resected_electrodes_fn,'r').readlines())
                except IndexError:
                    print 'ERROR! Resected electrodes %s does not have any electrodes. Skipping'%(resected_electrodes_fn)
                    return []

            # Map the resected electrodes to channels
            clean_resected_node_labels = []
            for resected_node_label in resected_node_labels:
                if resected_node_label in ignored_node_labels:
                    continue
                else:
                    clean_resected_node_labels.append(resected_node_label)
            resected_node_idx = map(lambda x: labels_dict[x][0], clean_resected_node_labels)
            for ii,node_id in enumerate(resected_node_idx):
                print 'Virtually resecting node label: %s because label %s is in the resection zone'%(channels[node_id],resected_node_labels[ii])

            # Map the NON-resected electrodes to channels
            all_node_idx = map(lambda x: labels_dict[x][0], labels_dict.keys())
            non_resected_node_idx = []
            for idx in all_node_idx:
                if(idx in resected_node_idx):
                    continue
                else:
                    non_resected_node_idx.append(idx)
            # non_resected_node_idx = np.array(non_resected_node_idx)

    return resected_node_idx


