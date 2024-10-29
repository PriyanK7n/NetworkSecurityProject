"""
This setup.py file is used for packaging and distributing Python projects. 
It uses setuptools to define project configurations, such as metadata, dependencies, and other settings.
The `find_packages()` function scans for folders containing __init__.py files, which mark directories as Python packages thereby creating 
NetworkSecurityProject.egg-info folder
"""


from setuptools import find_packages, setup
from typing import List

def get_requirements()->List[str]:
    """
        Reads and returns a list of package dependencies from the requirements.txt file.
        
        This function reads each line in 'requirements.txt', removes any empty lines and the '-e .' entry,
        and returns a list of valid requirements for the project.
        
        Returns:
            List[str]: A list containing all the required package names as strings.
    """
    requirement_lst: List[str]=[]
    try:
        with open('requirements.txt','r') as file:
            #Read lines from the file
            lines=file.readlines()
            ## Process each line to build the requirements list
            for line in lines:
                requirement=line.strip()
                ## ignore empty lines and -e . which is used for editable installs
                if requirement and requirement != '-e .':
                    requirement_lst.append(requirement)
    
    except FileNotFoundError:
        print("requirements.txt file not found")

    return requirement_lst

# Configure and initialize the setup function for package distribution
setup(
        name="NetworkSecurityProject",  # Name of the project

        version="0.0.1",  # Initial project version

        author="Priyank Negi",  # Author's name

        author_email="peiyank99@gmail.com",  # Author's email

        packages=find_packages(),  # Automatically find and include all packages

        install_requires=get_requirements()  # Populate dependencies from requirements.txt
)