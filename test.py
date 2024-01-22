import sys
import os

# Get the directory of your current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Go two levels up to reach the 'services' folder
parent_directory = os.path.dirname(os.path.dirname(current_directory))

# Path to the 'services' folder
services_path = os.path.join(parent_directory, 'services')
print(services_path)

# Add the 'services' directory to sys.path
sys.path.append(services_path)