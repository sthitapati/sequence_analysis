import pandas as pd

#####################################################################################################################################
# fucntion to get the sessions with opto_stim True ######
#####################################################################################################################################

def get_opto_sessions(sessions_df):
    """
    This function takes the sessions dataframe, filters for rows where 'opto_session' is True 
    and 'experiment_type' is '2_Experiment', and returns a list of session names.

    Parameters: 
    sessions_df (DataFrame): The dataframe containing session information.

    Returns: 
    list: A list of session names where 'opto_session' is True and 'experiment_type' is '2_Experiment'.
    """
    opto_session_ids = sessions_df[(sessions_df['opto_session'] == True) & 
                                (sessions_df['experiment_type'] == '2_Experiment')]['session_id'].tolist()

    return opto_session_ids