from utils.preprocesser import *
from utils.process_session import *

def main():
    """
    Main function that processes animal data.
    """
    # Parse command line arguments
    arguments = parse_arguments()
    
    # Create a ConfigParser object and parse the metadata file
    metadata = ConfigParser(metadata_file=arguments['metadata'])
    metadata.parse_metadata()
    
    # Process the animal data
    process_animal_data(animal_ids= metadata.animal_ids,
                        input_directory=metadata.input_directory,
                        output_directory=metadata.output_directory,
                        camera_directory=metadata.camera_directory,
                        replace_existing=arguments['replace']
                        )
    
    # combine the preprocessed data and add additional variables from settings to make it analysis friendly
    process_sessions(animal_ids = metadata.animal_ids, 
                     groups = metadata.group, 
                     output_directory = metadata.output_directory
                     )

if __name__ == "__main__":
    """
    Entry point of the script. The script can be called using the command 
    'python analyse_sequence_task.py -md (or --metadata) path_to_metadata_file -r (or --replace for if replace_existing is true)'
    """
    # Execute main function
    main()
