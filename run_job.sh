#!/bin/bash
    
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd pretext_task
python pretext_main.py
cd ../downstream_task
python downstream_main.py
json_file="downstream_config_mimic_iii.json"
old_line='"classifier_name": "MLP"'
new_value="CNN-LSTM"
new_line='"classifier_name": "'"$new_value"'"'
sed -i "s/$old_line/$new_line/g" "$json_file"
python downstream_main.py