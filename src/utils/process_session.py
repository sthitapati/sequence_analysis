# following are functions to handle preprocessed data and convert them into useful format for cross-session and cross-animal analysis
# they also have functions to look into sessions and parse the data according to the user's needs (e.g. OtpoStim, ExperimentType, etc.)
# saves processed data into a csv file for each animal in each session and across all sessions
# Future upgrade would be to save the data across all animals in a single csv file (or some other format)

#####################################################################################################################################
# Notes: These processes assume that the data has already been preprocessed and is in the correct format.
# # metadata should be in the following format:
# Experiment = "test_data"
# Animals = ["SP111", "SP112"]
# Group = ["Control", "Control"]
# Path_To_Raw_Data = "/home/sthitapati/Documents/sequence_data/bpod_raw_data"
# Camera_Folder = "/home/sthitapati/Documents/sequence_data/SP_FlyCap"
# Output_Folder = "/home/sthitapati/Documents/sequence_data/output"

## for now there is no check on whether the file was run before and preprocessing is properly done (may be implemented later)
# it will replace the existing files since no flag is set to check if the file was run before

# import libraries
from datetime import datetime
import pandas as pd
import numpy as np
import os
import json

#####################################################################################################################################
# get settings for each session and save them in a dataframe ######
#####################################################################################################################################


def get_session_details(output_folder, current_animal_id):
    """
    This function takes a list of filenames and extracts the session id, 
    date, file number, and day from each filename. It returns a dataframe 
    sorted by session id. This function also appends opto_session and 
    experiment_type from settings.json files in the respective session folders.
    
    Parameters: 
    sessions (list): List of filenames. Each filename is expected to contain 
    session id, date, file number and day separated by underscores ('_'). 
    Date is expected to be in the format 'YYYYMMDD'.
    output_folder (str): The base output folder path.
    current_animal_id (str): The ID of the current animal.
    
    Returns: 
    DataFrame: A pandas dataframe containing session id, session, date, 
    day, file number, opto_session, and experiment_type.
    """
    # Initializing empty lists to store extracted information
    session_id_list = []
    date_list = []
    file_number_list = []
    day_list = []

    # list all the sessions for the current animal
    sessions = os.listdir(os.path.join(output_folder, current_animal_id, 'Preprocessed'))

    # Loop through each filename and extract the information
    for session in sessions:
        parts = session.split('_')
        session_id = int(parts[0])  # Convert the session ID to an integer
        date_str = parts[1]
        file_number = int(parts[2])  # Convert the file number to an integer
        day = parts[3]
        
        # Convert date_str to datetime format
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        
        # Append the extracted information to respective lists
        session_id_list.append(session_id)
        date_list.append(date_obj)
        file_number_list.append(file_number)
        day_list.append(day)

    # Create a dataframe with the sessions, session_id_list, date_list, and day_list
    # And sort it by session_id_list
    sessions_df = pd.DataFrame({
        'session_id': session_id_list, 
        'session': sessions, 
        'date': date_list, 
        'file_number': file_number_list,
        'day': day_list
    })
    sessions_df = sessions_df.sort_values(by=['session_id'])
    sessions_df = sessions_df.reset_index(drop=True)

    # Append opto_session and experiment_type to sessions_df
    for session_id in range (0, len(sessions_df)):
        session = sessions_df.loc[session_id]['session']
        session_path = os.path.join(output_folder, current_animal_id, 'Preprocessed', session)

        settings_file_path = os.path.join(session_path, 'settings.json')
        with open(settings_file_path, 'r') as f:
            current_settings = json.load(f)

        if current_settings['OptoStim'] == 1:
            sessions_df.loc[session_id, 'opto_session'] = True
        else:
            sessions_df.loc[session_id, 'opto_session'] = False

        sessions_df.loc[session_id, 'experiment_type'] = current_settings['ExperimentType']

    return sessions_df

# for testing
# Output_Folder = "/home/sthitapati/Documents/sequence_data/output"
# Animals = ["SP111", "SP112"]
# current_animal_id = Animals[1]
# # sessions = os.listdir(os.path.join(Output_Folder, current_animal_id, 'Preprocessed'))

# sessions_df = get_session_settings(Output_Folder, current_animal_id)
# print(sessions_df)



# TODO - add a function to get the sessions with any arbitrary condition (e.g. experiment_type = 2_Experiment, or date = 2023-04-21, etc.)

#####################################################################################################################################
# function to convert trial_ids to cumulative trial_ids ######
#####################################################################################################################################

def calculate_cumulative_trial_id(trial_ids, previous_cumulative_trial_id=0):
    """
    Calculate cumulative trial ids within each session.

    Given a NumPy array containing 'trial_ids' for a single session, this function calculates the cumulative
    trial ids. The 'trial_ids' array is structured such that it increments for each unique trial within a session.

    Parameters:
        trial_ids (np.array): NumPy array containing 'trial_ids' for a single session.
        previous_cumulative_trial_id (int): Cumulative trial id from the previous session.

    Returns:
        np.array: A new NumPy array representing the cumulative trial ids within the session.
    """
    cumulative_trial_ids = np.zeros_like(trial_ids)
    cumulative_trial_id = previous_cumulative_trial_id

    for i, trial_id in enumerate(trial_ids):
        if trial_id != trial_ids[i - 1]:
            cumulative_trial_id += 1
        cumulative_trial_ids[i] = cumulative_trial_id

    return cumulative_trial_ids


#####################################################################################################################################
# function to calculate cumulative trial_ids for all sessions ######
#####################################################################################################################################

def process_transition_data(sessions_df, output_folder, current_animal_id, current_group, calculate_cumulative_trial_id):
    """
    This function processes all 'transition_data' files for the current animal.
    It adds cumulative trial ids, session ids, dates, days, animal id, and group to the data,
    saves each processed transition data file with a unique name, and finally concatenates all
    processed data into one final DataFrame which is also saved with a unique name.

    Parameters: 
    sessions_df (DataFrame): The dataframe containing session information.
    output_folder (str): The output folder path.
    current_animal_id (str): The ID of the current animal.
    current_group (str): The group of the current animal.
    calculate_cumulative_trial_id (function): A function to calculate cumulative trial ids.

    Returns: 
    DataFrame: A dataframe containing all processed transition data.
    """
    all_transition_data = []
    cumulative_trial_id = 0

    for session in sessions_df['session']:
        transition_data_file_path = os.path.join(output_folder, current_animal_id, 'Preprocessed', session, 'PreProcessed_TransitionData.csv')
        transition_data = pd.read_csv(transition_data_file_path)

        trial_ids_numpy = transition_data['trial_id'].to_numpy()
        cumulative_trial_ids = calculate_cumulative_trial_id(trial_ids_numpy, cumulative_trial_id)

        transition_data['cumulative_trial_id'] = cumulative_trial_ids

        current_session_info = sessions_df.loc[sessions_df['session'] == session]
        session_id = current_session_info['session_id'].values[0]
        date = current_session_info['date'].values[0]
        day = current_session_info['day'].values[0]

        transition_data['session_id'] = session_id
        transition_data['date'] = date
        transition_data['day'] = day
        transition_data['animal_id'] = current_animal_id
        transition_data['group'] = current_group
        
        save_path = os.path.join(output_folder, current_animal_id, 'Preprocessed', session, 'Processed_TransitionData.csv')
        # Replace all NaN values with string 'NaN', this is to account for some NaNs show up as blank cells in excel
        transition_data = transition_data.fillna('NaN')
        transition_data.to_csv(save_path, index=False)

        cumulative_trial_id = cumulative_trial_ids[-1]
        all_transition_data.append(transition_data)

    transition_data_DF = pd.concat(all_transition_data, ignore_index=True)

    save_path_all_sessions = os.path.join(output_folder, current_animal_id, f'{current_animal_id}_transition_data_all_sessions.csv')
    # Replace all NaN values with string 'NaN', this is to account for some NaNs show up as blank cells in excel
    transition_data_DF = transition_data_DF.fillna('NaN')
    transition_data_DF.to_csv(save_path_all_sessions, index=False)

    return transition_data_DF


#####################################################################################################################################
# function to loop through all animals and analyse the data given the metadata ######
#####################################################################################################################################

def process_sessions(animal_ids, groups, output_directory):
    """
    This function processes data for all animals in the given list.
    It retrieves session settings for each animal, processes transition data and saves the output.

    Parameters:
    animal_ids (list): List of animal IDs.
    group_info (list): Corresponding list of groups for the animals.
    raw_data_directory (str): The path of the directory where the raw data resides.
    processed_data_directory (str): The path of the directory where the processed data will be saved.
    camera_data_directory (str): The path of the directory where the camera data resides.
    """
    for animal_id, group in zip(animal_ids, groups):
        # Print a message to indicate progress
        print(f"processing all sessions of animal {animal_id} from group {group}.")

        # Get session details
        session_info_df = get_session_details(output_directory, animal_id)

        # Process transition data for the current animal
        all_transition_data_df = process_transition_data(session_info_df, 
                                                          output_directory, 
                                                          animal_id, 
                                                          group, 
                                                          calculate_cumulative_trial_id)

        # Print a message to indicate completion
        print(f"combining all sessions for animal {animal_id} from group {group} and adding useful columns.")

    print("Processing of all animals is complete.")

