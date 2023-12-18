# ------------------------------------------------------------------------------
# Created on Sat Nov 25 18:46:10 2023
# @author: Pablo Aguirre
# GitHub: https://github.com/pabloagn
# Website: https://pabloagn.com
# Contact: https://pabloagn.com/contact
# Part of Blog Article: the-state-of-our-world-in-2023
# ------------------------------------------------------------------------------

# This file will update CSV indicator list from Excel file and commit any changes in Guided Project.

# Define paths
$gitRepoPath = "D:/OneDrive/Documents/Professional Projects/A Journey Through Data Science/Guided Projects/data-science"
$pythonScriptPath = "$gitRepoPath/the-state-of-our-world-in-2023/resources/indicator_list_updater.py"

# Navigate to the Git repository directory
Push-Location -Path $gitRepoPath

# Run the Python script to update the CSV
python $pythonScriptPath

# Perform Git operations
git add *
git commit -m "Updated Guided Project"
git push -u origin master

# Return to location
Pop-Location