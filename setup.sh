#!/bin/bash

# Get the current directory
current_dir=$(pwd)

# Ask the user for confirmation
read -p "Do you want to add the current directory ($current_dir) to PYTHONPATH? (y/n): " user_response

# Check the user's response
if [[ "$user_response" == "y" || "$user_response" == "Y" ]]; then
    # Add the current directory to PYTHONPATH
    echo "export PYTHONPATH=\"$PYTHONPATH:$current_dir\"" >> ~/.bashrc
    source ~/.bashrc
    echo "The directory has been added to PYTHONPATH."
    echo "Current PYTHONPATH: $PYTHONPATH"
else
    echo "The directory was not added to PYTHONPATH."
fi