# Drowsiness-detection-codes

https://forums.developer.nvidia.com/t/how-to-install-pyqt5-jetson-nano/260936
https://chatgpt.com/s/t_6902142ebed08191828bd13fa77b2f94


import os
import pandas as pd

file_dir = r"F:\\biosignal data very important"
sessions = ['3AM', 'AML', 'APL1', 'APL2', 'BL', 'evening', 'meal']
initial_files = {session: os.path.join(file_dir, f'GSR_stats_{session}.csv') for session in sessions}
participants = [f'V{i}' for i in range(1, 101)]

def extract_participant_and_session(filename):
    # Example filename: V1_3AM_processed.xlsx
    base = os.path.basename(filename)
    parts = base.split('_')
    participant = parts[0]  # 'V1'
    session = parts[1]  # '3AM', 'AML', etc.
    return participant, session

def load_all_freq_stats():
    dfs = []
    for p in participants:
        for session in sessions:
            file_path = os.path.join(file_dir, f'{p}_{session}_processed.xlsx')
            if os.path.exists(file_path):
                df = pd.read_excel(file_path)
                
                # Add columns for participant and session from filename
                df['vol'] = p
                df['session'] = session
                
                dfs.append(df)
            else:
                print(f'Missing frequency file: {file_path}')
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return None

# Load all frequency stats combined
freq_all = load_all_freq_stats()
if freq_all is None:
    raise ValueError("No frequency stats files loaded.")

# For each session, load the initial CSV, merge with freq stats for that session, and save
for session in sessions:
    # Load session CSV (which contains 'volunteer' column for participant)
    init_csv_path = initial_files[session]
    session_df = pd.read_csv(init_csv_path)

    # Filter freq stats for current session
    freq_session_df = freq_all[freq_all['session'] == session].copy()
    
    # Ensure participant/volunteer column name matches for merging
    session_df['volunteer'] = session_df['Volunteer'].astype(str).str.strip()
    freq_session_df['vol'] = freq_session_df['vol'].astype(str).str.strip()
    
    # Merge on volunteer/participant_id
    merged_df = pd.merge(session_df, freq_session_df, left_on='volunteer', right_on='vol', how='left')
    
    # Drop the duplicate participant_id and session columns if not needed
    merged_df.drop(columns=['vol', 'session'], inplace=True)
    
    # Save extended CSV
    out_path = os.path.join(file_dir, f'extended_biosignal_stats_{session}.csv')
    merged_df.to_csv(out_path, index=False)
    print(f'Extended CSV saved for session: {session} at {out_path}')
