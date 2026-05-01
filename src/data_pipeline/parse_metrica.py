import pandas as pd
import numpy as np
import os

def parse_and_process_metrica(input_path: str, output_path: str):
    # Load raw data, skipping the first header row as requested
    df = pd.read_csv(input_path, skiprows=1)
    
    # Identify X and Y columns for players
    player_xs = df.filter(regex=r'Player.*_X', axis=1)
    player_ys = df.filter(regex=r'Player.*_Y', axis=1)
    
    # Fallback to lowercase or generic _x/_y if first regex doesn't match
    if player_xs.empty:
        player_xs = df.filter(regex=r'.*_x', axis=1)
        player_ys = df.filter(regex=r'.*_y', axis=1)
        
    # We want exactly 11 players for the 22-dimensional feature vector.
    # We'll take the first 11 valid player columns.
    player_xs = player_xs.iloc[:, :11]
    player_ys = player_ys.iloc[:, :11]
    
    # Convert from [0, 1] normalized grid to actual meters
    # Pitch dimensions: 105m x 68m, centered at (0, 0)
    player_xs = (player_xs - 0.5) * 105.0
    player_ys = (player_ys - 0.5) * 68.0
    
    # Calculate Team Centroid (mean X/Y of active players, ignoring NaNs)
    centroid_x = player_xs.mean(axis=1)
    centroid_y = player_ys.mean(axis=1)
    
    # Subtract the Team Centroid from every player's coordinate in that frame
    # This "normalizes" the formation to the center
    rel_xs = player_xs.sub(centroid_x, axis=0)
    rel_ys = player_ys.sub(centroid_y, axis=0)
    
    # Fill any NaNs with 0.0 (for missing players or inactive frames)
    rel_xs = rel_xs.fillna(0.0)
    rel_ys = rel_ys.fillna(0.0)
    
    # Determine Frame ID column
    frame_col = 'Frame' if 'Frame' in df.columns else df.columns[1] if len(df.columns) > 1 else np.arange(len(df))
    frame_ids = df[frame_col] if isinstance(frame_col, str) else frame_col
    
    # Construct output DataFrame with 22 columns (11 players * 2 coordinates)
    out_dict = {'Frame_ID': frame_ids}
    
    for i in range(11):
        if i < len(rel_xs.columns):
            out_dict[f'Player{i+1}_X'] = rel_xs.iloc[:, i]
            out_dict[f'Player{i+1}_Y'] = rel_ys.iloc[:, i]
        else:
            out_dict[f'Player{i+1}_X'] = 0.0
            out_dict[f'Player{i+1}_Y'] = 0.0
            
    out_df = pd.DataFrame(out_dict)
    
    # Ensure output directory exists and save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    input_file = "data/raw/Sample_Game_1_RawTrackingData_Home_Team.csv"
    output_file = "data/processed/home_team_features.csv"
    parse_and_process_metrica(input_file, output_file)
    print(f"Successfully processed {input_file} and saved 22 formation features to {output_file}")
