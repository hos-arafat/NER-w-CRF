# Helper file used to test "implementation.py" locally
# This file reads a tsv dataset file and returns it
# as a list of lists to be fed to the "predict" function
# in "implementation.py"

import os


def read_data(path):
    """
    Function that reads data file and returns list of lists 
    representing the input sentences and labels in the file

    Args: path to .tsv file
    
    Returns:
        list_l_tokens: list of lists: [ ["This", "is", "the", "first", "homework"], ["He", "is", "John"] ]
        list_l_labels: list of lists: [ ["O", "O", "O", "O", "O"], ["O", "O", "PER"] ]
                    
    """ 
    print("Model will be tested on the following files {:}...".format(path))

    list_l_tokens = []
    list_l_labels = []

    # Read file content
    with open(path, 'r', encoding='utf-8') as f:
        f_content_1 = f.readlines()
        for line in f_content_1:
            
            # All sentences start with the '#' token  
            if line[0] == "#":
                
                list_l_tokens.append(line[1:].split())
            
                # Labels for each line = number of words in the line
                line_length = len(line[1:].split())


                list_labels = []
                list_tokens = []
                
            # Line for each label starts with a digit 
            # and has the following form '23	Mediterranean	LOC' 
            # Add all labels for sentence to list of labels
            elif line[0].isdigit():
                num = int(line.split()[0])
                
                token = line.split()[1]
                label = line.split()[-1]

                list_labels.append(label)

                if num == (line_length-1):
                    list_l_labels.append(list_labels)

    return list_l_tokens, list_l_labels