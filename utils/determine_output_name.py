import os
from datetime import datetime


def determine_output_name() -> str:  # OUTPUT IN FORMAT MONDD_RUN_N.json
    folder_path = './output/'
    today = datetime.now()
    file_names = [f for f in os.listdir(
        folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    today_in_format = f'{today.strftime("%B").upper()[:3]}{today.day}_RUN_'
    runs_today = 0
    for file_name in file_names:
        if today_in_format in file_name:
            runs_today += 1
    return today_in_format + str(runs_today+1) + '.json'


"""
a = determine_output_name()
print(a)
"""
