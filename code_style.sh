# sort imports
isort --quiet --profile=black src deliverables/D4_4_Software_Executable_for_Data_Reduction/autoencoder deliverables/D4_4_Software_Executable_for_Data_Reduction/gain_clean notebooks
# Black code style
black --line-length 127 src deliverables/D4_4_Software_Executable_for_Data_Reduction/autoencoder deliverables/D4_4_Software_Executable_for_Data_Reduction/gain_clean notebooks
# flake8 standards
flake8 src deliverables/D4_4_Software_Executable_for_Data_Reduction/autoencoder deliverables/D4_4_Software_Executable_for_Data_Reduction/gain_clean notebooks --max-complexity=10 --max-line-length=127 --ignore=E203,E266,E501,E722,F401,F403,F405,W503,C901,F811
# mypy
# mypy src deliverables/D4_4_Software_Executable_for_Data_Reduction/autoencoder deliverables/D4_4_Software_Executable_for_Data_Reduction/gain_clean notebooks --ignore-missing-imports --no-strict-optional