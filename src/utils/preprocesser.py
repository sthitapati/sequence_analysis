# this script includes all the classes and functions needed to process the raw data

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from typing import Dict, Tuple, List, Union, Optional
import pandas as pd
import warnings
import pickle
import json
import argparse
import datetime
import shutil
import yaml
import re

def parse_arguments():
    """
    Function to parse command-line arguments.

    Returns:
        dict: A dictionary containing the values of the 'metadata' and 'replace' arguments.
    """

    # Initialize the ArgumentParser object
    parser = argparse.ArgumentParser(description="This script processes input arguments.")

    # Adding argument for metadata file with additional help text
    parser.add_argument("-md", 
                        "--metadata", 
                        help = "Provide the path to the metadata file.")

    # Adding argument for replace option with additional help text
    parser.add_argument("-r", 
                        "--replace", 
                        help = "Replace existing data if this argument is passed.", 
                        action = "store_true")

    # Parsing the arguments
    args = parser.parse_args()

    # Create a dictionary to store argument values
    arguments = {
        'metadata': args.metadata,
        'replace': args.replace
    }

    # Return dictionary of arguments
    return arguments

class ConfigParser:
    """
    This class parses the metadata file in YAML format and extracts the relevant information.
    """

    def __init__(self, metadata_file):
        self.metadata_file = metadata_file
        # Initialize variables to store the parsed data
        self.experiment = None
        self.animal_ids = []  # List of animal IDs
        self.group = {}  # Mapping of animal_id to Group
        self.sessions_to_plot = {}  # Mapping of animal_id to Sessions_to_plot
        self.input_directory = ''
        self.output_directory = ''
        self.camera_directory = ''
        self.animals_to_exclude_from_plotting = []
        self.opto_sessions = {}
        # Call the parse function upon initialization
        self._parse_yaml()

    def parse_metadata(self):
        """
        Parses the metadata file based on its extension (expecting '.yml' or '.yaml').

        Raises:
        - ValueError: If the file format is not supported.
        """
        file_extension = os.path.splitext(self.metadata_file)[1].lower()
        
        if file_extension in ['.yml', '.yaml']:
            self._parse_yaml()
        else:
            raise ValueError(f"Unsupported metadata file format: {file_extension}")

    def _parse_yaml(self):
        """
        Parses the metadata from a YAML file.
        """
        with open(self.metadata_file, 'r') as file:
            metadata = yaml.safe_load(file)

        # Parse data based on the structure provided by the YAML content
        self.experiment = metadata.get("Experiment", "DefaultExperiment")
        self.input_directory = metadata.get("Path_To_Raw_Data", "")
        self.output_directory = metadata.get("Output_Folder", "")
        self.camera_directory = metadata.get("Camera_Folder", "")
        # Process each animal's data to populate animal_ids, group, and sessions_to_plot
        animals_data = metadata.get("Animals", {})
        self.animal_ids = list(animals_data.keys())
        self.group = {animal_id: animal_data.get('Group', 'Unknown') for animal_id, animal_data in animals_data.items()}
        self.sessions_to_plot = {animal_id: animal_data.get('Sessions_to_plot', []) for animal_id, animal_data in animals_data.items()}

        self.animals_to_exclude_from_plotting = metadata.get("Animals_to_exclude_from_plotting", [])
        self.opto_sessions = metadata.get("Opto_Settings", [])


##############################################################################################################
# Functions for loading and saving data
##############################################################################################################

def loadmat(filename: str) -> Dict:
    '''
    Load a .mat file and convert all mat-objects to nested dictionaries.

    Parameters:
    filename (str): The name of the .mat file to load.

    Returns:
    dict: A dictionary containing the contents of the .mat file.
    '''
    # Load the .mat file
    data = scipy.io.loadmat(filename, simplify_cells=True)
    return data

def import_bpod_data_files(input_path: str) -> Tuple[Dict[int, Dict], int, List[str], List[str]]:
    '''
    Load all '.mat' files in a given folder and convert them to Python format.

    Parameters:
    input_path (str): The path to the folder containing the '.mat' files.

    Returns:
    Tuple[Dict[int, Dict], int, list, list]: A tuple containing the converted data, the number of sessions,
    the list of file paths, and the list of file dates.
    '''
    # Get a list of all files in the input path
    behav_path = sorted(os.listdir(input_path))
    behav_data = {}  # Set up file dictionary
    session_dates = []
    sessions = 0  # For naming each data set within the main dictionary

    # Loop through each file in the input path
    for file in [f for f in behav_path if f.endswith('.mat') and os.stat(input_path + f).st_size > 200000]:
        # Check if the file is not the weird hidden file
        if file != '.DS_Store':
            # Load the '.mat' file and add it to the dictionary
            current_file = loadmat(input_path + file)
            behav_data[sessions] = current_file
            sessions += 1
            session_dates.append(file[-19:-4])

    return behav_data, sessions, behav_path, session_dates

##############################################################################################################
# Functions for various useful data wrangling 
##############################################################################################################

def extract_poke_times(behavior_data: Dict) -> Tuple[List, List, List]:
    """
    Extracts all port in/out times across the session for each port. 
    It aligns them to trial start timestamps so that the port in times 
    are across the whole session.

    Parameters:
    behavior_data (dict): The dictionary containing behavior data for the session.

    Returns:
    Tuple: Lists of all port in times, port out times, and corresponding port references.
    """
    # Initialize lists to store port in times, port out times and corresponding port references
    all_port_in_times = []
    all_port_out_times = []
    all_port_references = []

    # Iterate over each port
    for port in range(1, 9):

        # Initialize lists to store port in/out times for each port
        port_in_times = []
        port_out_times = []

        # Iterate over each trial
        for trial_index in range(behavior_data['SessionData']['nTrials']):

            # Extract port in times
            if f'Port{port}In' in behavior_data['SessionData']['RawEvents']['Trial'][trial_index]['Events']:
                trial_start_timestamp = behavior_data['SessionData']['TrialStartTimestamp'][trial_index]
                port_in_ts_offset = behavior_data['SessionData']['RawEvents']['Trial'][trial_index]['Events'][f'Port{port}In']
                port_in_ts = trial_start_timestamp + port_in_ts_offset

                # If port in timestamp is a single value, convert it to a list
                if isinstance(port_in_ts, np.float64):
                    port_in_ts = [port_in_ts]

                # Add port in times to the list
                port_in_times.extend(port_in_ts)

            # Extract port out times
            if f'Port{port}Out' in behavior_data['SessionData']['RawEvents']['Trial'][trial_index]['Events']:
                trial_start_timestamp = behavior_data['SessionData']['TrialStartTimestamp'][trial_index]
                port_out_ts_offset = behavior_data['SessionData']['RawEvents']['Trial'][trial_index]['Events'][f'Port{port}Out']
                port_out_ts = trial_start_timestamp + port_out_ts_offset

                # If port out timestamp is a single value, convert it to a list
                if isinstance(port_out_ts, np.float64):
                    port_out_ts = [port_out_ts]

                # Add port out times to the list
                port_out_times.extend(port_out_ts)

        # Check if the number of port in times and port out times are equal
        # If not, apply error check and fix
        if len(port_in_times) != len(port_out_times):
            port_in_times, port_out_times = error_check_and_fix(port_in_times, port_out_times)

        # Add port in times, port out times and port references to the overall lists
        all_port_references.extend([port] * len(port_in_times))
        all_port_in_times.extend(port_in_times)
        all_port_out_times.extend(port_out_times)

    return all_port_in_times, all_port_out_times, all_port_references

def error_check_and_fix(port_in_times: List, port_out_times: List) -> Tuple[List, List]:
    """
    Checks and corrects mismatches in the length of port in and port out times lists.
    If lengths are unequal, 'nan' is inserted at the appropriate position or appended to the shorter list.

    Parameters:
    port_in_times (List): The list of port in times.
    port_out_times (List): The list of port out times.

    Returns:
    Tuple: The corrected port in times and port out times lists.
    """
    # Initialize fixed flag as False
    fixed = False

    # If the lengths of port in times and port out times lists are not equal
    if len(port_in_times) != len(port_out_times):

        # If port in times list is longer than port out times list
        if len(port_in_times) > len(port_out_times):
            # Iterate over each item in the port out times list
            for i in range(len(port_out_times)):
                # If the port out time is later than the next port in time
                if port_out_times[i] >= port_in_times[i+1]:
                    # Insert a 'nan' at this position in the port out times list
                    port_out_times.insert(i, 'nan')
                    fixed = True

            # If the issue wasn't fixed by the above process, append 'nan' to port out times list
            if len(port_in_times) > len(port_out_times) and not fixed:
                port_out_times.append('nan')

        # If port out times list is longer than port in times list
        elif len(port_out_times) > len(port_in_times):
            # Iterate over each item in the port in times list
            for i in range(len(port_in_times)):
                # If the port in time is later than or equal to the port out time
                if port_in_times[i] >= port_out_times[i]:
                    # Insert a 'nan' at this position in the port in times list
                    port_in_times.insert(i, 'nan')
                    fixed = True

            # If the issue wasn't fixed by the above process, append 'nan' to port in times list
            if len(port_out_times) > len(port_in_times) and not fixed:
                port_in_times.append('nan')

    # If the lengths of port in times and port out times lists are still not equal
    if len(port_in_times) != len(port_out_times):
        print('Dropped event not fixed!!!!')

    return port_in_times, port_out_times

def remove_dropped_in_events(port_in_times: List, port_out_times: List, port_references: List) -> Tuple[List, List, List]:
    """
    Cleans up the data by removing 'nan' values from the lists of port in times, port out times, and port references.

    Parameters:
    port_in_times (List): The list of port in times.
    port_out_times (List): The list of port out times.
    port_references (List): The list of port references.

    Returns:
    Tuple: The cleaned port in times, port out times, and port references lists.
    """

    # Create a reversed list of indexes to remove in order to avoid index shifting during removal
    indexes_to_remove = [i for i, time in enumerate(port_in_times) if time == 'nan'][::-1]

    for index in indexes_to_remove:
        # Remove 'nan' entries from each list
        del port_in_times[index]
        del port_out_times[index]
        del port_references[index]

    return port_in_times, port_out_times, port_references

def sort_by_time(port_in_times: List, port_out_times: List, port_references: List) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sorts the data by port in times. If an out time is missing, it will be appended with 'nan'.

    Parameters:
    port_in_times (List): The list of port in times.
    port_out_times (List): The list of port out times.
    port_references (List): The list of port references.

    Returns:
    Tuple: The sorted port in times, port out times, and port references.
    """
    
    # Get the indices that would sort the in times
    sort_indices = np.argsort(port_in_times)

    # Apply the sorted indices to each list and convert them to numpy arrays
    sorted_in_times = np.array(port_in_times, dtype=float)[sort_indices]
    sorted_references = np.array(port_references)[sort_indices]
    
    # Check if the number of out times matches the number of sorted indices
    if len(sort_indices) == len(port_out_times):
        sorted_out_times = np.array(port_out_times, dtype=float)[sort_indices]
    else:
        # If they don't match, append a 'nan' to the out times before sorting
        sorted_out_times = np.array(port_out_times + [np.nan], dtype=float)[sort_indices]

    return sorted_in_times, sorted_out_times, sorted_references

def extract_reward_timestamps(behavior_data: Dict) -> List[float]:
    '''
    Extracts all reward timestamps across a session for each port.

    Parameters:
    behavior_data (Dict): The behavioral data dictionary.

    Returns:
    List[float]: A list containing all the reward timestamps for the session.
    '''
    # Initialize list to store all reward timestamps
    reward_timestamps = []

    # Iterate over each trial in the session
    for trial in range(behavior_data['SessionData']['nTrials']):
        
        # Check if the 'Reward' event exists in the trial data
        if 'Reward' in behavior_data['SessionData']['RawEvents']['Trial'][trial]['States']:
            
            # Calculate the timestamp of the reward event relative to the start of the trial
            trial_start_timestamp = behavior_data['SessionData']['TrialStartTimestamp'][trial]
            reward_time_offset = behavior_data['SessionData']['RawEvents']['Trial'][trial]['States']['Reward'][0]
            
            # Convert the reward timestamp to the session timeline
            reward_timestamp = trial_start_timestamp + reward_time_offset
            
            # Add the reward timestamp to the list of reward timestamps
            reward_timestamps.append(reward_timestamp)

    return reward_timestamps

def find_rewarded_event_indices(sorted_in_timestamps: List[float], 
                                sorted_port_references: List[int], 
                                reward_timestamps: List[float]) -> List[int]:
    '''
    Identifies the indices of rewarded events.

    Parameters:
    sorted_in_timestamps (List[float]): List of sorted poke in timestamps.
    sorted_port_references (List[int]): List of port references corresponding to the poke in timestamps.
    reward_timestamps (List[float]): List of reward timestamps.

    Returns:
    List[int]: Indices of rewarded events in sorted_in_timestamps and sorted_port_references.
    '''

    rewarded_event_indices = []  # Initialize the list to store indices of rewarded events
    reward_index = 0  # Initialize reward index counter

    # Iterate over sorted port references with their indices
    for event_index, port_number in enumerate(sorted_port_references):
        
        # Check if port number is 7 and there are reward timestamps left to process
        if port_number == 7 and reward_index < len(reward_timestamps):
            
            # Skip NaN timestamps
            while np.isnan(reward_timestamps[reward_index]):
                reward_index += 1

                # If there are no more reward timestamps, exit the loop
                if reward_index >= len(reward_timestamps):
                    break

            # If there are still reward timestamps left, check if the in time is greater than or equal to the current reward timestamp
            if reward_index < len(reward_timestamps) and sorted_in_timestamps[event_index] >= reward_timestamps[reward_index]:
                
                # If so, record the event index as a rewarded event
                rewarded_event_indices.append(event_index)
                
                # And move on to the next reward timestamp
                reward_index += 1

    return rewarded_event_indices

def align_trigger_to_index(triggers: List[float], 
                           trigger_indices: List[int], 
                           all_timestamps: List[float]) -> List[Union[float, str]]:
    '''
    Aligns triggers to their corresponding indices in the timestamp array.

    Parameters:
    triggers (List[float]): List of trigger timestamps.
    trigger_indices (List[int]): List of indices corresponding to trigger timestamps.
    all_timestamps (List[float]): List of all timestamps.

    Returns:
    List[Union[float, str]]: Array with triggers aligned to their indices, 
                              and 'NaN' for all other indices.
    '''

    # Initialize output array with 'NaN' for all indices
    aligned_triggers = ['NaN'] * len(all_timestamps)
    
    # Assign trigger values to their corresponding indices
    for trigger_value, trigger_index in zip(triggers, trigger_indices):
        aligned_triggers[trigger_index] = trigger_value

    return aligned_triggers

def extract_trial_timestamps(behavior_data):
    """
    Extracts trial timestamps from behavioral data.

    Args:
        behavior_data: The complete behavioral data dictionary.

    Returns:
        A list of trial timestamps.
    """
    trial_timestamps = []
    for trial in range(behavior_data['SessionData']['nTrials']):
        trial_start_timestamp = behavior_data['SessionData']['TrialStartTimestamp'][trial]
        trial_timestamps.append(trial_start_timestamp)
    return trial_timestamps

def extract_trial_end_times(behavior_data):
    """
    Extracts trial end times from behavioral data.

    Args:
        behavior_data: The complete behavioral data dictionary.

    Returns:
        A list of trial end times.
    """

    all_end_times = []
    for trial in range(behavior_data['SessionData']['nTrials']):
        if 'ExitSeq' in behavior_data['SessionData']['RawEvents']['Trial'][trial]['States']:
            trial_start_timestamp = behavior_data['SessionData']['TrialStartTimestamp'][trial]
            exit_time_offset = behavior_data['SessionData']['RawEvents']['Trial'][trial]['States']['ExitSeq'][-1]
            end_times = trial_start_timestamp + exit_time_offset
            all_end_times.append(end_times)
    return all_end_times

def determine_trial_id(sorted_port_in_times, trial_end_timestamps):
    """
    Determines the trial id for each port event.

    Args:
        sorted_port_in_times: Sorted list of port in times.
        trial_end_timestamps: List of trial end times.

    Returns:
        A list of trial ids for each port event.
    """

    trial_id = []
    trial_number = 1
    for time in sorted_port_in_times:
        if trial_number > len(trial_end_timestamps):
            trial_id.append(trial_number)
            continue
        if float(time) <= trial_end_timestamps[trial_number - 1]:
            trial_id.append(trial_number)
        else:
            trial_number += 1
            trial_id.append(trial_number)
    return trial_id

def find_trial_start_indices(trial_ids):
    """
    Determines the start indices for each trial.

    Args:
        trial_ids: List of trial ids for each port event.

    Returns:
        A list of start indices for each trial.
    """

    trial_start_indices = [0]
    for index, trial_id in enumerate(trial_ids[1:], 1):  # start enumerating from 1
        if trial_id != trial_ids[index-1]:
            trial_start_indices.append(index)
    return trial_start_indices

def align_trial_start_end_timestamps(trial_ids, trial_start_indices, trial_start_timestamps):
    """
    Aligns trial start and end timestamps.

    Args:
        trial_ids: List of trial ids for each port event.
        trial_start_indices: List of start indices for each trial.
        trial_start_timestamps: List of trial start times.

    Returns:
        A list of aligned trial start times.
    """

    aligned_trial_timestamps = []
    counter = 0
    for i in range(len(trial_ids)):
        if counter + 1 < len(trial_start_indices) and i == trial_start_indices[counter+1]:
            counter += 1
        if counter < len(trial_start_timestamps):
            aligned_trial_timestamps.append(trial_start_timestamps[counter])
        else:
            aligned_trial_timestamps.append(np.nan)

    if len(trial_start_timestamps) != len(trial_start_indices):
        difference = abs(len(trial_start_timestamps) - len(trial_start_indices))
        if difference > 2:
            warnings.warn(f"Difference between trial_start_timestamps and trial_start_indices exceeds 2: {difference}")

    return aligned_trial_timestamps

def align_data_to_trial_ids(trial_ids: List[int], data: List[int]) -> List[int]:
    """
    This function aligns the given data according to the trial ids.

    Args:
        trial_ids (List[int]): The list of trial ids.
        data (List[int]): The list of data to align.

    Returns:
        List[int]: The list of aligned data.
    """

    # Initialize the counter for executed trials and list for aligned trials
    data_counter = 0
    aligned_data = []

    # Iterate over the list of trial ids
    for index, trial_id in enumerate(trial_ids):
        # For the first trial, simply append the first data item
        if index == 0:
            aligned_data.append(data[data_counter])
        else:
            # If the current trial id is same as previous one, append the same data item
            if trial_id == trial_ids[index-1]:
                if data_counter < len(data):
                    aligned_data.append(data[data_counter])
                else:
                    aligned_data.append(float('nan'))
            else:
                # If the trial id has changed, increment the counter
                data_counter += 1
                # Check if data_counter has not exceeded the length of data
                if data_counter < len(data):
                    # Append the next data item
                    aligned_data.append(data[data_counter])
                else:
                    # If data_counter has exceeded the length of data, append NaN or any other suitable value
                    aligned_data.append(float('nan'))

    return aligned_data

def generate_map(GUI, GUIMeta):
    mapping = {}
    for key in GUI:
        if key in GUIMeta:
            if GUIMeta[key]['Style'] == 'popupmenu':
                index = GUI[key] - 1  # subtract 1 to adjust for 0-based indexing
                mapping[key] = GUIMeta[key]['String'][index]
            elif GUIMeta[key]['Style'] == 'checkbox':
                mapping[key] = bool(GUI[key])
        else:
            # if the value is an ndarray, convert to list
            if isinstance(GUI[key], np.ndarray):
                mapping[key] = GUI[key].tolist()
            else:
                mapping[key] = GUI[key]
    return mapping

### ---------------------------------------- ###
### handle data for optogenetics experiments ###
### ---------------------------------------- ###

def handle_opto_stim_data(behavior_data, trial_settings, session_index, trial_ids):
    """
    Aligns optostim data with trial data including pulse duration and delay.
    """
    # Initialize with NaNs upfront to ensure consistency in output lengths
    optotrials_aligned = [float('nan')] * len(trial_ids)
    optotrials_port_aligned = [float('nan')] * len(trial_ids)
    optotrials_pulse_duration = [float('nan')] * len(trial_ids)
    optotrials_pulse_delay = [float('nan')] * len(trial_ids)
    optotrials_variable_delay = [float('nan')] * len(trial_ids)

    # Initialize variables to None
    optotrials = port_stimulated_data = pulse_duration = pulse_delay = variable_delay = None

    if trial_settings['GUI'].get('OptoStim', 0) == 1:
        session_variables = behavior_data[session_index]['SessionData']['SessionVariables']
        # Assign data if available
        optotrials = session_variables.get('OptoStim', [float('nan')] * len(trial_ids))
        port_stimulated_data = session_variables.get('PortStimulated', [float('nan')] * len(trial_ids))
        pulse_duration = session_variables.get('OptoDuration', [float('nan')] * len(trial_ids))
        pulse_delay = session_variables.get('OptoDelay', [float('nan')] * len(trial_ids))
        variable_delay = trial_settings['GUI'].get('VariableTrainDelay', float('nan'))

        for i, trial_id in enumerate(trial_ids):
            # Assuming trial_ids are aligned with the indices for optotrials and other variables
            optotrials_aligned[i] = optotrials[i] if i < len(optotrials) else float('nan')
            optotrials_port_aligned[i] = port_stimulated_data[i] if i < len(port_stimulated_data) else float('nan')
            optotrials_pulse_duration[i] = pulse_duration[i] if i < len(pulse_duration) else float('nan')
            optotrials_pulse_delay[i] = pulse_delay[i] if i < len(pulse_delay) else float('nan')
            # Assuming variable_delay is a session-wide setting, not trial-specific
            optotrials_variable_delay[i] = variable_delay

    # Debugging prints removed for clarity
    return optotrials_aligned, optotrials_port_aligned, optotrials_pulse_duration, optotrials_pulse_delay, optotrials_variable_delay

### ------------------------------ ###
### transition data preprocessing  ###
### ------------------------------ ###

def determine_transition_times_and_types(all_port_in_times_sorted: np.ndarray,
                                         all_port_out_times_sorted: np.ndarray,
                                         all_port_references_sorted: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List, List]:
    """
    Determines transition times and types given sorted port in/out times and sorted port references.

    Args:
        all_port_in_times_sorted (np.ndarray): Array of sorted port in times.
        all_port_out_times_sorted (np.ndarray): Array of sorted port out times.
        all_port_references_sorted (np.ndarray): Array of sorted port references.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, List, List]: Tuple containing arrays of out-in transitions, in-in transitions, 
        transition types, and lists of out-in and in-in transition references.
    """
    
    out_in_transitions = []
    in_in_transitions = []
    transition_types = []
    out_in_transition_references = []
    in_in_transition_references = []

    for index, port in enumerate(all_port_references_sorted):
        if index > 0:
            # Calculate out-in transition and reference
            out_in_transitions.append(all_port_in_times_sorted[index] - all_port_out_times_sorted[index-1])
            out_in_transition_references.append(all_port_out_times_sorted[index-1])
            
            # Calculate in-in transition and reference
            in_in_transitions.append(all_port_in_times_sorted[index] - all_port_in_times_sorted[index-1])
            in_in_transition_references.append(all_port_in_times_sorted[index-1])
            
            # Determine transition type
            transition_types.append(int(str(all_port_references_sorted[index-1]) + str(port)))

    return (np.array(out_in_transitions), np.array(in_in_transitions), np.array(transition_types), out_in_transition_references, in_in_transition_references)

def get_start_end_port_id(transition_types: np.ndarray, start_end_arg: int) -> List[int]:
    """
    Returns the start or end port id from the transition types.

    Args:
        transition_types (np.ndarray): Array of transition types.
        start_end_arg (int): Indicator of whether to return start or end port id.
                             0 for start, 1 for end.

    Returns:
        List[int]: List of start or end port ids depending on the start_end_arg.
    """
    
    output_ids = []
    for transition in transition_types:
        transition_str = str(transition)
        output_ids.append(int(transition_str[start_end_arg]))

    return output_ids

def determine_repeat_port_events(start_port_ids: List[int], end_port_ids: List[int]) -> List[int]:
    """
    Determines repeat port events.

    Args:
        start_port_ids (List[int]): List of start port ids.
        end_port_ids (List[int]): List of end port ids.

    Returns:
        List[int]: List of flags indicating if a port is repeated. 
                   0 for repeated port and 1 for non-repeated.
    """
    
    port_repeat_flags = []
    for start_id, end_id in zip(start_port_ids, end_port_ids):
        if start_id == end_id:
            port_repeat_flags.append(0)
        else: 
            port_repeat_flags.append(1)

    return port_repeat_flags

def filter_transitions_by_latency(transition_times: List[float], upper_limit: float) -> List[int]:
    """
    Filters transitions based on their latency times.

    Args:
        transition_times (List[float]): List of transition times.
        upper_limit (float): The upper limit to filter the transitions.

    Returns:
        List[int]: List of flags indicating whether a transition time is less than the upper limit. 
                   1 for less than the limit and 0 otherwise.
    """
    
    filtered_transitions = []
    for time in transition_times:
        if time < upper_limit:
            filtered_transitions.append(1)
        else:
            filtered_transitions.append(0)

    return filtered_transitions

def create_sequences_by_time_and_port(
    transition_types: List[int],
    transition_times: List[float],
    port1: int,
    transition_reference_time: List[float],
    transition_filter_time: float
) -> Tuple[List[List[int]], List[List[float]], List[List[float]]]:
    """
    Reorder transitions into sequences relevant to time and port events. 

    Args:
        transition_types (List[int]): List of transition types.
        transition_times (List[float]): List of transition times.
        port1 (int): Identifier for port1.
        transition_reference_time (List[float]): List of reference times for transitions.
        transition_filter_time (float): Time filter for transitions.

    Returns:
        TimeFiltered_ids (List[List[int]]): Nested list of filtered transition ids.
        TimeFiltered_times (List[List[float]]): Nested list of filtered transition times.
        Reference_times (List[List[float]]): Nested list of reference times for the filtered transitions.
    """
    
    sequence_index = 0
    filtered_transition_ids = [[]]
    filtered_transition_times = [[]]
    reference_times = [[]]

    for index, transition in enumerate(transition_types):
        if 0.03 < transition_times[index] < transition_filter_time:  # if less than filter time and more than lower bound filter time (0.1s)
            if int(str(transition)[0]) == port1:  # check if first port matches filter port
                if filtered_transition_ids[sequence_index]:
                    sequence_index += 1
                    filtered_transition_ids.append([])
                    filtered_transition_times.append([])
                    reference_times.append([])
                filtered_transition_ids[sequence_index].append(transition)
                filtered_transition_times[sequence_index].append(transition_times[index])
                reference_times[sequence_index].append(transition_reference_time[index])
            else: 
                filtered_transition_ids[sequence_index].append(transition)
                filtered_transition_times[sequence_index].append(transition_times[index])
                reference_times[sequence_index].append(transition_reference_time[index])

        elif filtered_transition_ids[sequence_index]:  # if not empty 
            sequence_index += 1
            filtered_transition_ids.append([])
            filtered_transition_times.append([])
            reference_times.append([])

    return filtered_transition_ids, filtered_transition_times, reference_times

def number_of_rewarded_events(aligned_reward_timestamps: List) -> int:
    """
    Function to count the number of non-NaN items in a list.

    Parameters:
    aligned_reward_timestamps (List): A list of numerical values or strings that may include NaNs.

    Returns:
    int: The count of non-NaN items in the list.
    
    """
    # Use a generator expression with the sum function to count the non-NaN items
    # The isinstance() function checks if an item is a float or integer
    # The math.isnan() function checks if a numerical item is a NaN
    return sum(1 for item in aligned_reward_timestamps if isinstance(item, (float, int)) and not math.isnan(item))

###################################################################################################
###Functions to handle video data ###
###################################################################################################
def convert_and_uncycle_timestamps(timestamps: np.ndarray) -> np.ndarray:
    """
    Converts an array of timestamps to seconds and uncycles the time array.

    Args:
        timestamps (np.ndarray): The timestamps to be converted and uncycled.

    Returns:
        np.ndarray: The converted and uncycled time array in seconds.
    """
    # Convert timestamps to seconds
    cycle1 = (timestamps >> 12) & 0x1FFF
    cycle2 = (timestamps >> 25) & 0x7F
    time_in_seconds = cycle2 + cycle1 / 8000.0

    # Uncycle timestamps
    cycles = np.insert(np.diff(time_in_seconds) < 0, 0, False)
    cycle_index = np.cumsum(cycles)
    uncycled_time_seconds = time_in_seconds + cycle_index * 128

    return uncycled_time_seconds

def find_trigger_indices(camera_timestamps: pd.DataFrame) -> tuple:
    """
    Finds indices for 'pin_state_bool' transitions: the index immediately before the transition to high,
    the index of the transition to high, and the index of the subsequent transition back to low, grouping them into
    separate lists.

    Args:
        camera_timestamps (pd.DataFrame): The camera timestamps dataframe.

    Returns:
        tuple: A tuple of lists with meaningful names:
               (indices_before_high, high_indices, low_indices)
    """

    # Ensure the dataframe is sorted by time
    camera_timestamps = camera_timestamps.sort_values(by='time_seconds')

    indices_before_high = []
    high_indices = []
    low_indices = []

    previous_state_was_high = False
    previous_high_index = None
    index_before_high = None

    for i, (index, row) in enumerate(camera_timestamps.iterrows()):
        current_state_is_high = row['pin_state_bool']

        if i == 0:
            previous_state_was_high = current_state_is_high
            continue

        if not previous_state_was_high and current_state_is_high:
            # Transition from low to high
            index_before_high = camera_timestamps.index[i - 1]  # Capture the index immediately before the transition to high
            previous_high_index = index
        elif previous_state_was_high and not current_state_is_high and previous_high_index is not None:
            # Capturing the transition from high to low immediately following a high
            indices_before_high.append(index_before_high)
            high_indices.append(previous_high_index)
            low_indices.append(index)
            previous_high_index = None  # Reset for the next transition

        previous_state_was_high = current_state_is_high
    
    # modify indices_before_high to remove the first index as it's just a recording without behaviour start
    # and add the last index as the last high does not have a corresponding low
        
    indices_before_high.pop(0)
    indices_before_high.append(camera_timestamps.index[-1])    

    return (indices_before_high, high_indices, low_indices)

def video_data_exists(camera_directory, animal_id, date): 
    """
    Checks if video data exists for a given session date.
    """
    camera_path = os.path.join(camera_directory, animal_id)
    if not os.path.exists(camera_path):
        return False, ""
    
    camera_files = os.listdir(camera_path)
    timestamp_files = [file for file in camera_files if file.endswith('.csv')]

    date_file_dict = {}
    for file in timestamp_files:
        file_date = re.findall(r'\d{4}-\d{2}-\d{2}', file)
        if file_date:
            date_file_dict[file_date[0]] = file
    
    file_name = date_file_dict.get(date, "")
    if not file_name:
        return False, ""
    
    file_name = os.path.join(camera_path, file_name)
    return True, file_name

def handle_camera_data(absolute_timestamp_filepath, trial_ids, sorted_port_in_times, sorted_port_out_times, sorted_port_references,
                       aligned_trial_start_timestamps, aligned_trial_end_timestamps) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Processes camera data to align it with behavioral trial data, including trial start and end times, and port in and out times.
    
    Parameters:
    - absolute_timestamp_filepath (str): Path to the file containing raw camera timestamps.
    - trial_ids (List[int]): Identifiers for each trial in the session.
    - sorted_port_in_times (List[float]): Timestamps for when an entity entered a port, sorted chronologically.
    - sorted_port_out_times (List[float]): Timestamps for when an entity exited a port, sorted chronologically.
    - sorted_port_references (List[int]): References to the port involved in each timestamp, indicating a specific event or condition.
    - aligned_trial_start_timestamps (List[float]): Start times for each trial, aligned with the camera's timestamps.
    - aligned_trial_end_timestamps (List[float]): End times for each trial, aligned with the camera's timestamps.
    
    Returns:
    - Tuple[List[float], List[float], List[float], List[float]]: A tuple containing the aligned camera times for port in events, port out events, trial start events, and trial end events.
    """
    try: 
        # Initialize lists to hold the camera timestamps aligned with behavioral events
        camera_port_in_times = []
        camera_port_out_times = []
        camera_start_times = []
        camera_end_times = []

        # Load and preprocess camera timestamps from the CSV file
        camera_timestamps = pd.read_csv(absolute_timestamp_filepath, delimiter=' ', header=None)
        camera_timestamps = camera_timestamps.drop(columns=[3])  # Drop the last column
        camera_timestamps.columns = ['pin_state', 'timestamp', 'frame_number']
        camera_timestamps['pin_state_bool'] = camera_timestamps['pin_state'] != camera_timestamps['pin_state'][0]

        # Convert timestamps to seconds and uncycle
        timestamp_array = camera_timestamps['timestamp'].values
        timestamps_in_seconds_uncycled = convert_and_uncycle_timestamps(timestamp_array)
        camera_timestamps['time_seconds'] = timestamps_in_seconds_uncycled

        # Find TTL times within the camera timestamps
        indices_before_high, high_indices, low_indices = find_trigger_indices(camera_timestamps)

        
        behaviour_df = pd.DataFrame({
            'trial_id': trial_ids,
            'trial_start': aligned_trial_start_timestamps,
            'trial_end': aligned_trial_end_timestamps, 
            'port_in': sorted_port_in_times,
            'port_out': sorted_port_out_times,
            'port_reference': sorted_port_references
        })

        for trial_id in behaviour_df['trial_id'].unique():
            trial_df = behaviour_df[behaviour_df['trial_id'] == trial_id].copy()
            first_port_in_index = trial_df[trial_df['port_reference'] == 2].index[0]

            # get camera_start_time for the trial by camera_timestamps['time_seconds'][low_indices[trial_id - 1]]
            trial_start_camera = camera_timestamps['time_seconds'][low_indices[trial_id - 1]]
            # repeat for len(trial_df) times to get the correct length
            camera_start_times.extend([trial_start_camera] * len(trial_df))
            
            # get camera_end_time for the trial by camera_timestamps['time_seconds'][indices_before_high[trial_id - 1]]
            trial_end_camera = camera_timestamps['time_seconds'][indices_before_high[trial_id - 1]]
            # repeat for len(trial_df) times to get the correct length
            camera_end_times.extend([trial_end_camera] * len(trial_df))

            first_poke_in_time = trial_df.loc[first_port_in_index, 'port_in']
            first_poke_in_camera = camera_timestamps['time_seconds'][low_indices[trial_id - 1]]
            port_in_times_trial = trial_df['port_in'].values
            relative_port_in_times = port_in_times_trial - first_poke_in_time
            camera_port_in_time = first_poke_in_camera + relative_port_in_times
            camera_port_in_times.extend(camera_port_in_time)

            port_out_times_trial = trial_df['port_out'].values
            relative_port_out_times = port_out_times_trial - port_in_times_trial
            camera_port_out_time = camera_port_in_time + relative_port_out_times
            camera_port_out_times.extend(camera_port_out_time)          
            
        return camera_port_in_times, camera_port_out_times, camera_start_times, camera_end_times
    except Exception as e:
        print(f"Error: {e}", 'returning empty lists')
        # determine the length of the trial ids and return empty lists of the same length
        length = len(trial_ids)
        return [np.nan] * length, [np.nan] * length, [np.nan] * length, [np.nan] * length


def get_video_timestamp_and_frame(absolute_timestamp_filepath, trial_ids, aligned_trial_start_timestamps, aligned_trial_end_timestamps) -> Tuple[List, List, List, List]:
    """
    Aligns trial start and end times with camera data timestamps.
    # Important distinction from the previous function is that this function gets aligned start and end timestamps/ frames for the video data.
    # the length is identical to unique trial ids, and is to be used for video splitting/ analysis
    
    This function processes raw camera timestamps to align them with the start and end times of behavioral trials, 
    focusing only on these specific events for further analysis.
    
    Args:
        absolute_timestamp_filepath (str): Path to the CSV file containing raw camera timestamps.
        trial_ids (list): A list of unique identifiers for each trial.
        aligned_trial_start_timestamps (list): Start times for each trial, aligned with the camera's timestamps.
        aligned_trial_end_timestamps (list): End times for each trial, aligned with the camera's timestamps.
    
    Returns:
        Tuple[List, List]: A tuple containing two lists:
            - The first list contains aligned trial start times with the camera data.
            - The second list contains aligned trial end times with the camera data.
    """
    try:
        # Load the raw camera timestamps from the specified file
        camera_timestamps = pd.read_csv(absolute_timestamp_filepath, delimiter=' ', header=None)
        camera_timestamps = camera_timestamps.drop(columns=[3])  # Assume the last column is not needed
        camera_timestamps.columns = ['pin_state', 'timestamp', 'frame_number']
        camera_timestamps['pin_state_bool'] = camera_timestamps['pin_state'] != camera_timestamps['pin_state'][0]

        # Convert the raw timestamps to a more usable format (e.g., seconds) and handle any cycling issues
        timestamp_array = camera_timestamps['timestamp'].values
        timestamps_in_seconds_uncycled = convert_and_uncycle_timestamps(timestamp_array)
        camera_timestamps['time_seconds'] = timestamps_in_seconds_uncycled

        # Extract times when the TTL signal was high or low from the camera timestamps
        indices_before_high, high_indices, low_indices = find_trigger_indices(camera_timestamps)

        camera_trial_start_timestamps = []
        camera_trial_end_timestamps = []
        camera_trial_start_frames = []
        camera_trial_end_frames = []
        # unique trial ids
        unique_trial_ids = np.unique(trial_ids)

        # Align trial start and end times with camera timestamps
        for trial_id in unique_trial_ids:
            
            # high_indices are indices when the trial starts
            trial_start_index = high_indices[trial_id - 1]
            trial_start_time = camera_timestamps['time_seconds'][trial_start_index]
            trial_start_frame = camera_timestamps['frame_number'][trial_start_index]

            # indices_before_high are indices when the trial ends
            trial_end_index = indices_before_high[trial_id - 1]
            trial_end_time = camera_timestamps['time_seconds'][trial_end_index]
            trial_end_frame = camera_timestamps['frame_number'][trial_end_index]
            

            # Append the aligned times and frames to the respective lists
            camera_trial_start_timestamps.append(trial_start_time)
            camera_trial_end_timestamps.append(trial_end_time)
            camera_trial_start_frames.append(trial_start_frame)
            camera_trial_end_frames.append(trial_end_frame)
    
        return camera_trial_start_timestamps, camera_trial_end_timestamps, camera_trial_start_frames, camera_trial_end_frames
    except Exception as e:
        print(f"Error: {e}", 'returning empty lists')
        # determine the length of the trial ids and return empty lists of the same length
        length = len(np.unique(trial_ids))
        return [np.nan] * length, [np.nan] * length, [np.nan] * length, [np.nan] * length


###################################################################################################
###THIS IS THE MAIN FUNCTION THAT RUNS THE ANALYSIS###
###################################################################################################

def process_animal_data(
    animal_ids: List[str], 
    input_directory: str, 
    output_directory: str, 
    camera_directory: Optional[str] = None, 
    replace_existing: bool = False
    ) -> None:
    """
    Function to process data for each animal and each session.

    Args:
        animal_ids (List[str]): List of animal IDs.
        input_directory (str): Directory containing raw behavioral data for each animal.
        output_directory (str): Directory where processed data will be saved.
        camera_directory (Optional[str]): Directory containing the camera timestamp files for each animal, if available.
        replace_existing (bool): If True, existing processed data will be replaced. Defaults to False.

    Returns:
        None
    """

    # Iterate over each animal by its index and ID
    for animal_index, current_animal_id in enumerate(animal_ids):
        print ('Processing data for: ' + current_animal_id)

        # Construct the path for the current animal's data
        current_input_path = os.path.join(input_directory, current_animal_id, 'Sequence_Automated', 'Session Data/')

        # Load Behavioural data using the import_bpod_data_files function
        behavior_data, total_sessions, path, session_dates = import_bpod_data_files(current_input_path)

        # Initialize strings to store processed and skipped sessions
        processed_sessions = ''
        skipped_sessions = ''

        # Iterate over each session
        for session_index in range(total_sessions):

            # Create unique identifier for the session
            session_date = session_dates[session_index] + '_' + str(behavior_data[session_index]['__header__'])[-25:-22]

            # Set the save path depending on the session number
            if session_index < 10:
                save_path = os.path.join(output_directory, current_animal_id, 'Preprocessed', f'0{session_index}_{session_date}')
            else:
                save_path = os.path.join(output_directory, current_animal_id, 'Preprocessed', f'{session_index}_{session_date}')

            if os.path.isdir(save_path):
                print(f"Directory {save_path} exists.")
                if replace_existing:
                    print(f"Replace flag is set. Removing and recreating directory {save_path}.")
                    shutil.rmtree(save_path)
                    os.makedirs(save_path)  # recreate the directory
                    should_process = True  # Set should_process to True after removing directory
                else:
                    # If replace_existing is False, print a message and continue to the next session
                    print(f"Replace flag is not set. Skipping directory {save_path}.")
                    should_process = False  # Set should_process to False as we're not replacing existing data
            else:
                # If directory does not exist, create it and set should_process to True
                print(f"Directory {save_path} does not exist. Creating it.")
                os.makedirs(save_path)
                should_process = True  # Set should_process to True as directory is new and we need to process data

                
            # If processing flag is True, convert the data to a Python-friendly format
            if should_process:                
                # Calculate final reward amount for the session
                final_reward_amounts = []
                for item in behavior_data[session_index]['SessionData']['SessionVariables']['TLevel']:
                    training_level = item
                    final_reward_amounts.append(behavior_data[0]['SessionData']['SessionVariables']['TrainingLevels'][training_level-1][4])

                # save out training levels on their own
                filename = 'PreProcessed_TrainingLevels' 
                with open(save_path + '/'+ filename, 'wb') as fp:
                    pickle.dump(behavior_data[session_index]['SessionData']['SessionVariables']['TLevel'], fp)

                # fetch trial_settings 
                trial_settings = behavior_data[session_index]['SessionData']['TrialSettings'][0]
                # save trial settings
                current_trial_settings = generate_map(trial_settings['GUI'], trial_settings['GUIMeta'])
                os.path.join(save_path, 'settings.json')

                with open(os.path.join(save_path, 'settings.json'), 'w') as outfile:
                    json.dump(current_trial_settings, outfile)         

                # Save out LED intensities and reward amounts on their own:
                led_intensities = pd.DataFrame({
                    'Port2': behavior_data[session_index]['SessionData']['SessionVariables']['LEDIntensitys']['port2'],
                    'Port3': behavior_data[session_index]['SessionData']['SessionVariables']['LEDIntensitys']['port3'],
                    'Port4': behavior_data[session_index]['SessionData']['SessionVariables']['LEDIntensitys']['port4'],
                    'Port5': behavior_data[session_index]['SessionData']['SessionVariables']['LEDIntensitys']['port5']
                })

                # Save out LED intensities and reward amounts on their own:
                led_intensities.to_csv(save_path + '/PreProcessed_LED_Intensities.csv')

                # Create a DataFrame for reward amounts for each port:41
                reward_amounts = pd.DataFrame({
                    'Port1': behavior_data[session_index]['SessionData']['SessionVariables']['RewardAmount']['port1'],
                    'Port2': behavior_data[session_index]['SessionData']['SessionVariables']['RewardAmount']['port2'],
                    'Port3': behavior_data[session_index]['SessionData']['SessionVariables']['RewardAmount']['port3'],
                    'Port4': behavior_data[session_index]['SessionData']['SessionVariables']['RewardAmount']['port4']
                })
                
                # Save out reward amounts on their own:
                reward_amounts.to_csv(save_path + '/PreProcessed_Reward_Amounts.csv')

                # Extract PortIn times for each port and check for errors
                port_in_times, port_out_times, port_references = extract_poke_times(behavior_data[session_index])

                # Remove 'nan' values (these represent times when part of the event was dropped by Bpod for some reason)
                fixed_port_in_times, fixed_port_out_times, fixed_port_references = remove_dropped_in_events(port_in_times, port_out_times, port_references)

                # Resort these times for consistent chronology
                sorted_port_in_times, sorted_port_out_times, sorted_port_references = sort_by_time(fixed_port_in_times, fixed_port_out_times, fixed_port_references)

                
                # Extract reward timestamps:
                reward_timestamps = extract_reward_timestamps(behavior_data[session_index])


                # Find indices corresponding to rewarded events and align them to poke events:
                rewarded_event_indices = find_rewarded_event_indices(sorted_port_in_times, sorted_port_references, reward_timestamps)

                # Remove 'NaN' entries from reward timestamps:
                reward_timestamps = np.asarray(reward_timestamps)
                reward_timestamps = reward_timestamps[np.logical_not(np.isnan(reward_timestamps))]
                reward_timestamps = list(reward_timestamps)

                # Align reward timestamps to the corresponding poke events:
                aligned_reward_timestamps = align_trigger_to_index(reward_timestamps, rewarded_event_indices, sorted_port_references)

                # Extract trial start timestamps:
                trial_start_timestamps = extract_trial_timestamps(behavior_data[session_index])

                # Extract trial end times:
                trial_end_timestamps = extract_trial_end_times(behavior_data[session_index])

                # Determine trial IDs:
                trial_ids = determine_trial_id(sorted_port_in_times, trial_end_timestamps)

                # Find trial start indices:
                trial_start_indices = find_trial_start_indices(trial_ids)

                # Align trial start timestamps to poke events:
                aligned_trial_start_timestamps = align_trial_start_end_timestamps(trial_ids, trial_start_indices, trial_start_timestamps)

                # Align trial end timestamps to poke events:
                aligned_trial_end_timestamps = align_trial_start_end_timestamps(trial_ids, trial_start_indices, trial_end_timestamps)
                
                # handle video data
                # initialize camera_port_in_times, camera_port_out_times, camera_trial_start_times, camera_trial_end_times as lists of same length as trial_ids
                camera_port_in_times = [np.nan] * len(trial_ids)
                camera_port_out_times = [np.nan] * len(trial_ids)
                camera_trial_start_times = [np.nan] * len(trial_ids)
                camera_trial_end_times = [np.nan] * len(trial_ids)

                # initialize camera_trial_start_timestamps, camera_trial_end_timestamps, camera_trial_start_frames, camera_trial_end_frames
                # same size as len(unique(trial_ids))

                camera_trial_start_timestamps = [np.nan] * len(np.unique(trial_ids))
                camera_trial_end_timestamps = [np.nan] * len(np.unique(trial_ids))
                camera_trial_start_frames = [np.nan] * len(np.unique(trial_ids))
                camera_trial_end_frames = [np.nan] * len(np.unique(trial_ids))

                
                # first prepare to prepare input adjust the regex pattern to match YYYYMMDD format
                regex_session_date = re.findall(r'\d{4}\d{2}\d{2}', session_dates[session_index])[0]
                # format the date in YYYY-MM-DD format after extracting
                formatted_session_date = f"{regex_session_date[:4]}-{regex_session_date[4:6]}-{regex_session_date[6:]}"

                # check if camera_directory is provided as an input and exists               
                if camera_directory:
                    video_exists, camera_file_path = video_data_exists(camera_directory, current_animal_id, formatted_session_date)
                    if video_exists:
                        camera_port_in_times, camera_port_out_times, camera_trial_start_times, camera_trial_end_times = handle_camera_data(camera_file_path, trial_ids, sorted_port_in_times, sorted_port_out_times, sorted_port_references, aligned_trial_start_timestamps, aligned_trial_end_timestamps)
                        camera_trial_start_timestamps, camera_trial_end_timestamps, camera_trial_start_frames, camera_trial_end_frames = get_video_timestamp_and_frame(camera_file_path, trial_ids, aligned_trial_start_timestamps, aligned_trial_end_timestamps)
                    else:
                        print(f"No video data found for {current_animal_id} on {formatted_session_date}. Skipping video data processing, setting timestamps to nan.")
                else:
                    print("No camera directory provided. Skipping video data processing, setting timestamps to nan.")

                                   
                
                
                # handle optogenetic stimulation
                optotrials_aligned, optotrials_port_aligned, optotrials_pulse_duration, optotrials_pulse_delay, optotrials_variable_delay = handle_opto_stim_data(behavior_data, trial_settings, 
                                                                                                                                                                  session_index, trial_ids)

                # Create empty lists to store intermediate rewards and LED intensities data for each trial
                intermediate_rewards_data = []
                led_intensities_data = []

                # Iterate over 'TLevel' items in SessionVariables
                for tlevel_item in behavior_data[session_index]['SessionData']['SessionVariables']['TLevel']:
                    tlevel = tlevel_item
                    # Append intermediate rewards and LED intensities data for the current trial
                    intermediate_rewards_data.append(
                        list(behavior_data[session_index]['SessionData']['SessionVariables']['TrainingLevels'][tlevel-1][0:4])
                    )
                    led_intensities_data.append(
                        list(behavior_data[session_index]['SessionData']['SessionVariables']['TrainingLevels'][tlevel-1][6:10])
                    )

                # Align intermediate rewards and LED intensities data with trial start indices
                aligned_led_intensities = align_trial_start_end_timestamps(trial_ids, trial_start_indices, led_intensities_data)
                aligned_intermediate_rewards = align_trial_start_end_timestamps(trial_ids, trial_start_indices, intermediate_rewards_data)


                # Align training level for each trial
                training_levels = align_data_to_trial_ids(trial_ids, behavior_data[session_index]['SessionData']['SessionVariables']['TLevel'])


                PortIn_df = pd.DataFrame(
                    {
                        'trial_id': trial_ids,
                        'trial_start_time': aligned_trial_start_timestamps,
                        'poke_port': sorted_port_references,
                        'poke_in_timestamp': sorted_port_in_times,
                        'poke_out_timestamp': sorted_port_out_times,
                        'reward_timestamps': aligned_reward_timestamps,
                        'trial_end_time': aligned_trial_end_timestamps,
                        'led_intensities_ports_2_3_4_5': aligned_led_intensities,
                        'reward_amounts_ports_1_2_3_4': aligned_intermediate_rewards,
                        'opto_condition': optotrials_aligned,
                        'opto_stimulated_port': optotrials_port_aligned,
                        'variable_delay': optotrials_variable_delay,
                        'opto_pulse_duration': optotrials_pulse_duration,
                        'opto_pulse_delay': optotrials_pulse_delay,
                        'training_level': training_levels
                    }
                )

                
                #Save Data
                PortIn_df.to_csv(save_path +'/PreProcessed_RawPokeData.csv')

                # PART 2: Transitions
                # Determine Transition times and types for all events 
                out_in_transition_times, in_in_transition_times, transition_types, out_in_transition_reference, in_in_transition_reference = determine_transition_times_and_types(sorted_port_in_times, sorted_port_out_times, sorted_port_references)

                # Split transition types into first and last ports: 
                start_port_ids = get_start_end_port_id(transition_types, 0)
                end_port_ids = get_start_end_port_id(transition_types, 1)

                # Align start and end port times
                end_port_in_time = sorted_port_in_times[1:]
                start_port_in_time = sorted_port_in_times[:-1]
                end_port_out_time = sorted_port_out_times[1:]
                start_port_out_time = sorted_port_out_times[:-1]

                # Find Port repeat events (double pokes)
                non_port_repeat = determine_repeat_port_events(start_port_ids, end_port_ids)

                # Determine which transitions are good: less than 2s
                out_in_filtered_transitions = filter_transitions_by_latency(out_in_transition_times, upper_limit=2)
                in_in_filtered_transitions = filter_transitions_by_latency(in_in_transition_times, upper_limit=2)

                # Create DataFrame
                transition_df = pd.DataFrame(
                    {
                        'trial_id': trial_ids[:-1],
                        'transition_type': transition_types,
                        'start_poke_port': start_port_ids,
                        'end_poke_port': end_port_ids,
                        'start_poke_in_timestamp': start_port_in_time,
                        'start_poke_out_timestamp': start_port_out_time,
                        'end_poke_in_timestamp': end_port_in_time,
                        'end_poke_out_timestamp': end_port_out_time,
                        'out_in_latency': out_in_transition_times,
                        'in_in_latency': in_in_transition_times,
                        'repeat_filter': non_port_repeat,
                        '2s_time_filter_out_in': out_in_filtered_transitions,
                        '2s_time_filter_in_in': in_in_filtered_transitions,
                        'opto_condition': optotrials_aligned[:-1],
                        'opto_stimulated_port': optotrials_port_aligned[:-1],
                        'variable_delay': optotrials_variable_delay[:-1],
                        'opto_pulse_duration': optotrials_pulse_duration[:-1],
                        'opto_pulse_delay': optotrials_pulse_delay[:-1],
                        'training_level': training_levels[:-1],
                        'led_intensities_ports_2_3_4_5': aligned_led_intensities[:-1],
                        'reward_amounts_ports_1_2_3_4': aligned_intermediate_rewards[:-1],
                        'camera_port_in_times': camera_port_in_times[:-1],
                        'camera_port_out_times': camera_port_out_times[:-1],
                        'camera_trial_start_times': camera_trial_start_times[:-1],
                        'camera_trial_end_times': camera_trial_end_times[:-1]
                    }
                )

                # Replace all NaN values with string 'NaN', this is to account for some NaNs show up as blank cells in excel
                transition_df = transition_df.fillna('NaN')

                # Save Data
                transition_df.to_csv(os.path.join(save_path, 'PreProcessed_TransitionData.csv'))

                # create dataframe for camera timestamps
                camera_timestamps_df = pd.DataFrame({
                    'trial_id': np.unique(trial_ids),
                    'camera_trial_start_timestamps': camera_trial_start_timestamps,
                    'camera_trial_end_timestamps': camera_trial_end_timestamps,
                    'camera_trial_start_frames': camera_trial_start_frames,
                    'camera_trial_end_frames': camera_trial_end_frames
                })

                # Replace all NaN values with string 'NaN', this is to account for some NaNs show up as blank cells in excel
                camera_timestamps_df = camera_timestamps_df.fillna('NaN')

                # Save Data
                camera_timestamps_df.to_csv(os.path.join(save_path, 'PreProcessed_CameraTimestamps.csv'), index=False)

                # Define useful port/sequence related information
                port1 = behavior_data[session_index]['SessionData']['TrialSequence'][0][0]
                port2 = behavior_data[session_index]['SessionData']['TrialSequence'][0][1]
                port3 = behavior_data[session_index]['SessionData']['TrialSequence'][0][2]
                port4 = behavior_data[session_index]['SessionData']['TrialSequence'][0][3]
                port5 = behavior_data[session_index]['SessionData']['TrialSequence'][0][4]

                sequence1 = int(f"{port1}{port2}")
                sequence2 = int(f"{port2}{port3}")
                sequence3 = int(f"{port3}{port4}")
                sequence4 = int(f"{port4}{port5}")

                # Filter transitions into sequences for each port
                transition_filter_time = 2.0
                port1_time_filtered_ids, port1_time_filtered_times, port1_ref_times = create_sequences_by_time_and_port(
                    transition_types, in_in_transition_times, port1, in_in_transition_reference, transition_filter_time
                )

                port2_time_filtered_ids, port2_time_filtered_times, port2_ref_times = create_sequences_by_time_and_port(
                    transition_types, in_in_transition_times, port2, in_in_transition_reference, transition_filter_time
                )

                port3_time_filtered_ids, port3_time_filtered_times, port3_ref_times = create_sequences_by_time_and_port(
                    transition_types, in_in_transition_times, port3, in_in_transition_reference, transition_filter_time
                )

                port4_time_filtered_ids, port4_time_filtered_times, port4_ref_times = create_sequences_by_time_and_port(
                    transition_types, in_in_transition_times, port4, in_in_transition_reference, transition_filter_time
                )

                port5_time_filtered_ids, port5_time_filtered_times, port5_ref_times = create_sequences_by_time_and_port(
                    transition_types, in_in_transition_times, port5, in_in_transition_reference, transition_filter_time
                )

                # Filter transitions into sequences that are within the transition filter time (not filtered to start at first poke):
                time_filtered_ids, time_filtered_times, reference_times = create_sequences_by_time_and_port(
                    transition_types, out_in_transition_times, port1, in_in_transition_reference, transition_filter_time
                )

                # Make dataframes
                sequence_df_time_filtered_port1_aligned = pd.DataFrame({
                    'sequence_ids': port1_time_filtered_ids,
                    'sequence_times': port1_time_filtered_times,
                    'session_time_reference': port1_ref_times
                })

                sequence_df_time_filtered_port2_aligned = pd.DataFrame({
                    'sequence_ids': port2_time_filtered_ids,
                    'sequence_times': port2_time_filtered_times,
                    'session_time_reference': port2_ref_times
                })

                sequence_df_time_filtered_port3_aligned = pd.DataFrame({
                    'sequence_ids': port3_time_filtered_ids,
                    'sequence_times': port3_time_filtered_times,
                    'session_time_reference': port3_ref_times
                })

                sequence_df_time_filtered_port4_aligned = pd.DataFrame({
                    'sequence_ids': port4_time_filtered_ids,
                    'sequence_times': port4_time_filtered_times,
                    'session_time_reference': port4_ref_times
                })

                sequence_df_time_filtered_port5_aligned = pd.DataFrame({
                    'sequence_ids': port5_time_filtered_ids,
                    'sequence_times': port5_time_filtered_times,
                    'session_time_reference': port5_ref_times
                })

                # Save Data
                sequence_df_time_filtered_port1_aligned.to_csv(save_path + '/PreProcessed_Sequence_df_timefiltered_port1aligned.csv')
                sequence_df_time_filtered_port2_aligned.to_csv(save_path + '/PreProcessed_Sequence_df_timefiltered_port2aligned.csv')
                sequence_df_time_filtered_port3_aligned.to_csv(save_path + '/PreProcessed_Sequence_df_timefiltered_port3aligned.csv')
                sequence_df_time_filtered_port4_aligned.to_csv(save_path + '/PreProcessed_Sequence_df_timefiltered_port4aligned.csv')
                sequence_df_time_filtered_port5_aligned.to_csv(save_path + '/PreProcessed_Sequence_df_timefiltered_port5aligned.csv')

                # Make final session information dataframe
                training_levels = list(trial_settings['GUIMeta']['TrainingLevel']['String'])
                session_level = trial_settings['GUI']['TrainingLevel']
                no_rewarded_events = number_of_rewarded_events(aligned_reward_timestamps)

                # experiment or training session:
                if trial_settings['GUI']['ExperimentType'] == 2:
                    experiment = 1
                else:
                    experiment = 0

                session_information = pd.DataFrame({
                    'port1': [port1],
                    'port2': [port2],
                    'port3': [port3],
                    'port4': [port4],
                    'port5': [port5],
                    'transition1': sequence1,
                    'transition2': sequence2,
                    'transition3': sequence3,
                    'transition4': sequence4,
                    'n_trials': [trial_ids[-1]],
                    'n_rewards': [no_rewarded_events],
                    'final_reward_amount': [final_reward_amounts],
                    'session_level': [training_levels],
                    'experiment': [experiment],
                })


                # Save Data
                session_information.to_csv(save_path + '/PreProcessed_SessionInfo.csv')
                processed_sessions = ", ".join(str(session) for session in processed_sessions)
                            
            else:
                skipped_sessions += f"{session_index}, "  # Append session_index to the list of skipped sessions

    print(f'Already Processed so skipped: {skipped_sessions.rstrip(", ")}')  # Remove trailing comma and space
    print(f'Processed: {processed_sessions.rstrip(", ")}')  # Remove trailing comma and space
    print('preprocessing finished')
