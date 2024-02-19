
### --------------------------------------------------------------------- ###
### These functions are used to handle the data for the camera timestamps ###
### --------------------------------------------------------------------- ###

def handle_camera_data(session_date, camera_directory, current_animal_id, trial_ids, trial_start_indices, sorted_port_references, save_path):
    """
    Handle the processing of camera timestamps for a specific animal and session.

    Args:
        session_date (str): Session date.
        camera_directory (str): Directory path for camera data.
        current_animal_id (str): ID of the animal being processed.
        trial_ids (list): List of trial IDs.
        trial_start_indices (list): List of trial start indices.
        sorted_port_references (list): List of sorted port references.
        save_path (str): Path to save the preprocessed camera data.

    Returns:
        Tuple of aligned start, end, and first poke trial timestamps.
    """

    # Determine if camera timestamps exist for the session
    do_timestamps_exist, timestamp_file_path = find_camera_timestamps(session_date, camera_directory, current_animal_id)

    # Initialize the arrays with 'NaN'
    aligned_start_trial_timestamps = ['NaN'] * len(trial_ids)
    aligned_end_trial_timestamps = ['NaN'] * len(trial_ids)
    aligned_first_poke_timestamps = ['NaN'] * len(trial_ids)

    if do_timestamps_exist:
        print('Timestamps found for session.')
        
        # Load camera timestamps
        raw_camera_timestamps_df = load_camera_timestamps_from_file(input_file_path=timestamp_file_path)
        
        # Convert to seconds and uncycle timestamps
        camera_timestamps = convert_and_uncycle_timestamps(camera_timestamps_df=raw_camera_timestamps_df)
        
        # Check for dropped frames
        check_for_dropped_frames(timestamps=camera_timestamps, expected_frame_rate=60)

        # reset index to start from 0
        raw_camera_timestamps_df = raw_camera_timestamps_df.reset_index(drop=True)
      
        # Find trigger states
        camera_trigger_states = determine_trigger_states_from_raw_timestamps(raw_camera_timestamps_df=raw_camera_timestamps_df)
        
        # Check if triggers are working
        are_triggers_broken = np.max(camera_trigger_states) == np.min(camera_trigger_states)
        
        if not are_triggers_broken:
            # Construct camera dataframe
            camera_dataframe = pd.DataFrame(
                {
                    'timestamps': camera_timestamps,
                    'trigger_states': camera_trigger_states,
                    'datapath': [timestamp_file_path] * len(camera_timestamps)
                }
            )

            # Save the dataframe
            camera_dataframe.to_csv(os.path.join(save_path, 'preprocessed_cameradata.csv'))

            # Find camera indices for trial start and first poke
            trial_start_camera_indices, first_poke_indices = find_trial_start_and_poke1_camera_indices(camera_trigger_states=camera_trigger_states)
            
            # Align behavioural data (trial starts) with camera timestamps
            aligned_start_trial_timestamps = align_trial_start_end_timestamps(
                trial_ids=trial_ids,
                trial_start_indices=trial_start_indices,
                trial_start_timestamps=camera_timestamps[trial_start_indices]
            )
            
            # Align behavioural data (trial ends) with camera timestamps
            aligned_end_trial_timestamps = generate_aligned_trial_end_camera_timestamps(
                trial_start_camera_indices=trial_start_camera_indices,
                trial_start_indices=trial_start_indices,
                trial_ids=trial_ids,
                camera_timestamps=camera_timestamps
            )
            
            # Align behavioural data (first poke in port1) with camera timestamps
            aligned_first_poke_timestamps = align_firstpoke_camera_timestamps(
                trial_ids=trial_ids,
                trial_start_indices=trial_start_indices,
                trial_start_timestamps=camera_timestamps[first_poke_indices],
                all_port_references_sorted=sorted_port_references,
            )
        else:
            print('Camera trigger malfunction detected in session.')
    else:
        print('No camera timestamps found for the given session.')
    
    return do_timestamps_exist, aligned_start_trial_timestamps, aligned_end_trial_timestamps, aligned_first_poke_timestamps


def find_trial_start_and_poke1_camera_indices(camera_trigger_states: np.ndarray) -> Tuple[List[int], List[int]]:
    """
    Find indices in the camera timestamps where the trial starts and the first poke happens.

    Args:
        camera_trigger_states (np.ndarray): Array of trigger states from the camera.

    Returns:
        Tuple[List[int], List[int]]: Lists of indices where trial starts and the first poke happens.
    """
    ttl_change_indices = list(np.where(np.roll(camera_trigger_states, 1) != camera_trigger_states)[0])
    if ttl_change_indices[0] == 0:
        ttl_change_indices = ttl_change_indices[1:]

    poke1_camera_indices = ttl_change_indices[1::2]
    trial_start_camera_indices = ttl_change_indices[0::2]

    return trial_start_camera_indices, poke1_camera_indices


def generate_aligned_trial_end_camera_timestamps(trial_start_camera_indices: List[int], trial_ids: List[int], trial_start_indices: List[int], camera_timestamps: np.ndarray) -> List[Union[float, str]]:
    """
    Generate aligned timestamps for the end of trials based on camera timestamps.

    Args:
        trial_start_camera_indices (List[int]): List of indices where each trial starts.
        trial_ids (List[int]): List of trial ids for each port event.
        trial_start_indices (List[int]): List of start indices for each trial.
        camera_timestamps (np.ndarray): Array of camera timestamps.

    Returns:
        List[Union[float, str]]: List of aligned trial end timestamps.
    """
    end_indices = [item for index, item in enumerate(trial_start_camera_indices) if index > 0]
    aligned_trial_end_timestamps = align_trial_start_end_timestamps(trial_ids, trial_start_indices, camera_timestamps[end_indices])

    last_trial_length = len(trial_ids) - trial_start_indices[-1]
    if len(aligned_trial_end_timestamps) == len(trial_ids):
        del aligned_trial_end_timestamps[-last_trial_length:]

    aligned_trial_end_timestamps += ['NaN'] * last_trial_length
    return aligned_trial_end_timestamps


def align_firstpoke_camera_timestamps(trial_ids: List[int], trial_start_indices: List[int], trial_start_timestamps: List[float], all_port_references_sorted: List[float]) -> List[Union[float, str]]:
    """
    Align the timestamps of the first poke with the camera timestamps.

    Args:
        trial_ids (List[int]): List of trial ids for each port event.
        trial_start_indices (List[int]): List of start indices for each trial.
        trial_start_timestamps (List[float]): List of trial start timestamps.
        all_port_references_sorted (List[float]): Sorted list of all port references.

    Returns:
        List[Union[float, str]]: List of aligned first poke timestamps.
    """
    trial_timestamps_aligned = []
    counter = 0
    for index, item in enumerate(trial_ids):
        if all_port_references_sorted[index] == 2.0:
            if item > counter:
                counter += 1
                if counter - 1 < len(trial_start_timestamps):
                    trial_timestamps_aligned.append(trial_start_timestamps[counter-1])
                else:
                    trial_timestamps_aligned.append('NaN')
            else:
                trial_timestamps_aligned.append('NaN')
        else:
            trial_timestamps_aligned.append('NaN')
    return trial_timestamps_aligned

def find_camera_timestamps(session_date: str, camera_directory: str, animal_id: str) -> Tuple[bool, Union[str, None]]:
    """
    Searches for timestamp files for a given animal and session date in the camera directory.
    
    Args:
        session_date (str): The date of the session, in 'yyyymmddHHMMSS' format.
        camera_directory (str): The path to the directory where camera files are stored.
        animal_id (str): The ID of the animal.
    
    Returns:
        Tuple[bool, Union[str, None]]: A tuple with a boolean indicating whether the timestamp file exists,
        and the path to the timestamp file, if it exists. If no timestamp file is found, the path is None.
    """
    timestamps_exist = False
    timestamp_file_path = None

    # Parse the session_date string into a datetime object
    session_datetime = datetime.datetime.strptime(session_date, '%Y%m%d_%H%M%S_%a')

    # Check if the camera directory for the animal exists
    animal_camera_directory = os.path.join(camera_directory, animal_id)
    if not os.path.isdir(animal_camera_directory):
        return timestamps_exist, timestamp_file_path

    # Look for timestamp file in the animal_camera_directory
    for filename in os.listdir(animal_camera_directory):
        # Check if the file is a csv file
        if filename.endswith('.csv'):
            # Extract timestamp from filename
            file_timestamp = filename[6:-4]

            # Parse the file timestamp string into a datetime object
            file_datetime = datetime.datetime.strptime(file_timestamp, '%Y-%m-%dT%H_%M_%S')

            # Check if the file timestamp is before the session start time
            if file_datetime < session_datetime:
                timestamps_exist = True
                timestamp_file_path = os.path.join(animal_camera_directory, filename)
                break

    return timestamps_exist, timestamp_file_path

### Timestamp preprocessing:

def load_camera_timestamps_from_file(input_file_path: str) -> pd.DataFrame:
    """
    Loads camera timestamps from a file and returns them as a DataFrame.

    Args:
        input_file_path (str): The path of the file containing camera timestamps.

    Returns:
        pd.DataFrame: A dataframe containing camera timestamps.
    """
    camera_timestamps_df = pd.read_csv(input_file_path, delim_whitespace=True, header=None, names=['Trigger', 'Timestamp', 'blank'])
    camera_timestamps_df = camera_timestamps_df.set_index('blank')
    camera_timestamps_df.index.name = None

    return camera_timestamps_df

def convert_timestamp_to_seconds(timestamp: int) -> float:
    """
    Converts the timestamp into seconds.

    Args:
        timestamp (int): The timestamp to be converted.

    Returns:
        float: The timestamp converted into seconds.
    """
    cycle1 = (timestamp >> 12) & 0x1FFF
    cycle2 = (timestamp >> 25) & 0x7F
    time_in_seconds = cycle2 + cycle1 / 8000.0
    return time_in_seconds


def uncycle_timestamps(time_array: np.ndarray) -> np.ndarray:
    """
    Uncycles the time array.

    Args:
        time_array (np.ndarray): The time array to be uncycled.

    Returns:
        np.ndarray: The uncycled time array.
    """
    cycles = np.insert(np.diff(time_array) < 0, 0, False)
    cycle_index = np.cumsum(cycles)
    return time_array + cycle_index * 128


def convert_and_uncycle_timestamps(camera_timestamps_df: pd.DataFrame) -> np.ndarray:
    """
    Converts the timestamps into seconds and then uncycles them.

    Args:
        camera_timestamps_df (pd.DataFrame): DataFrame containing camera timestamps.

    Returns:
        np.ndarray: Uncycled timestamps in seconds.
    """
    timestamps_in_seconds = []
    for index, row in camera_timestamps_df.iterrows():
        if row.Trigger > 0: 
            timestamp_in_seconds = convert_timestamp_to_seconds(camera_timestamps_df.at[index, 'Timestamp'])
            timestamps_in_seconds.append(timestamp_in_seconds)
        else:    
            raise ValueError('Timestamps are broken')

    uncycled_timestamps = uncycle_timestamps(timestamps_in_seconds)
    uncycled_timestamps = uncycled_timestamps - uncycled_timestamps[0]  # make first timestamp 0 and the others relative to this 
    return uncycled_timestamps


def check_for_dropped_frames(timestamps: np.ndarray, expected_frame_rate: int) -> None:
    """
    Checks for dropped frames in the timestamps.

    Args:
        timestamps (np.ndarray): The array of timestamps.
        expected_frame_rate (int): The expected frame rate in frames per second.
    """
    frame_gaps = 1 / np.diff(timestamps)
    dropped_frames_count = np.sum((frame_gaps < expected_frame_rate - 5) | (frame_gaps > expected_frame_rate + 5))
    
    print(f'Frames dropped = {dropped_frames_count}')
    plt.suptitle(f'Frame rate = {expected_frame_rate}fps', color = 'red')
    plt.hist(frame_gaps, bins=100)
    plt.xlabel('Frequency')
    plt.ylabel('Number of frames')


def determine_trigger_states_from_raw_timestamps(raw_camera_timestamps_df: pd.DataFrame) -> np.ndarray:
    """
    Determines the trigger states from the raw camera timestamps.

    Args:
        raw_camera_timestamps_df (pd.DataFrame): DataFrame containing raw camera timestamps.

    Returns:
        np.ndarray: An array of trigger states.
    """

    down_state = raw_camera_timestamps_df['Trigger'].iloc[0]
    down_state_times = np.where(raw_camera_timestamps_df['Trigger'] == down_state)
    temporary_trigger_states = np.ones(len(raw_camera_timestamps_df['Trigger']))
    temporary_trigger_states[down_state_times] = 0
    return temporary_trigger_states

def calculate_port_events_in_camera_time(trial_start_timestamps: List[float], start_port_times: List[float], camera_start_timestamps: List[str]) -> List[float]:
    """
    Calculate the camera timestamps for port events.

    Args:
        trial_start_timestamps (List[float]): List of trial start timestamps.
        start_port_times (List[float]): List of start port times.
        camera_start_timestamps (List[str]): List of camera start timestamps.

    Returns:
        List[float]: List of port events in camera time.
    """
    
    # Convert camera_start_timestamps elements to floats
    camera_start_timestamps = list(map(float, camera_start_timestamps))
    
    port_camera_timestamps = []
    for index, start_time in enumerate(trial_start_timestamps[:-1]):
        time_difference = start_port_times[index] - start_time
        port_camera_timestamps.append(camera_start_timestamps[index] + time_difference)

    return port_camera_timestamps