import pandas as pd
import numpy as np
import os

def parse_and_process_metrica(input_path: str, output_path: str):
    # Load raw data, skipping the first header row as requested
    df = pd.read_csv(input_path, skiprows=1)
    
    # Identify X and Y columns for players
    # Assuming columns like 'Player1_X', 'Player2_X', etc.
    player_xs = df.filter(regex=r'Player.*_X', axis=1)
    player_ys = df.filter(regex=r'Player.*_Y', axis=1)
    
    # Fallback to lowercase or generic _x/_y if first regex doesn't match
    if player_xs.empty:
        player_xs = df.filter(regex=r'.*_x', axis=1)
        player_ys = df.filter(regex=r'.*_y', axis=1)
        
    # Convert from [0, 1] normalized grid to actual meters
    # Pitch dimensions: 105m x 68m, centered at (0, 0)
    # X: -52.5 to 52.5
    # Y: -34.0 to 34.0
    player_xs = (player_xs - 0.5) * 105.0
    player_ys = (player_ys - 0.5) * 68.0
    
    # Calculate Team Centroid (mean X/Y of active players, ignoring NaNs)
    centroid_x = player_xs.mean(axis=1)
    centroid_y = player_ys.mean(axis=1)
    
    # Calculate Team Spread (max X/Y - min X/Y, ignoring NaNs)
    spread_x = player_xs.max(axis=1) - player_xs.min(axis=1)
    spread_y = player_ys.max(axis=1) - player_ys.min(axis=1)
    
    # Determine Frame ID column
    frame_col = 'Frame' if 'Frame' in df.columns else df.columns[1] if len(df.columns) > 1 else np.arange(len(df))
    frame_ids = df[frame_col] if isinstance(frame_col, str) else frame_col
    
    # Construct output DataFrame
    out_df = pd.DataFrame({
        'Frame_ID': frame_ids,
        'Centroid_X': centroid_x,
        'Centroid_Y': centroid_y,
        'Spread_X': spread_x,
        'Spread_Y': spread_y
    })
    
    # Handle NaN values by filling with 0.0
    out_df = out_df.fillna(0.0)
    
    # Ensure output directory exists and save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    input_file = "data/raw/Sample_Game_1_RawTrackingData_Home_Team.csv"
    output_file = "data/processed/home_team_features.csv"
    parse_and_process_metrica(input_file, output_file)
    print(f"Successfully processed {input_file} and saved features to {output_file}")
