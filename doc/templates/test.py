# print("this is a test file")
# add the current directory to the PYTHONPATH
import sys
import os
import argparse
import yaml

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

def main():
    """
    Main function that processes animal data.
    """
    # Parse command line arguments
    arguments = parse_arguments()
    
    # Create a ConfigParser object and parse the metadata file
    metadata = ConfigParser(metadata_file=arguments['metadata'])
    metadata.parse_metadata()
    
    # print all the parsed metadata
    print(f"Experiment: {metadata.experiment}")
    print(f"Animal IDs: {metadata.animal_ids}")
    print(f"Group: {metadata.group}")
    print(f"Sessions to plot: {metadata.sessions_to_plot}")
    print(f"Input directory: {metadata.input_directory}")
    print(f"Output directory: {metadata.output_directory}")
    print(f"Camera directory: {metadata.camera_directory}")
    

if __name__ == "__main__":
    """
    Entry point of the script. The script can be called using the command 
    'python analyse_sequence_task.py -md (or --metadata) path_to_metadata_file -r (or --replace for if replace_existing is true)'
    """
    # Execute main function
    main()
