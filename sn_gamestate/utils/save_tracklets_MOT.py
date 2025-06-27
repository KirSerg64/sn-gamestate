import pandas as pd
from pathlib import Path
from sn_gamestate.utils.frame_extractor import frame_generator
import cv2

def save_tracklab_df_to_mot(tracklets_df: pd.DataFrame, output_dir: str):
    """
    Saves each tracklet sequence from a tracklab-formatted DataFrame to a separate text file in MOT format.

    Args:
        tracklets_df (pd.DataFrame): The DataFrame containing tracking results.
        output_dir (str): The directory to save output text files.
    """
    # Security checks for inputs
    if not isinstance(tracklets_df, pd.DataFrame):
        raise TypeError("tracklets_df must be a pandas DataFrame.")
    required_columns = {'image_id', 'track_id', 'bbox_ltwh', 'bbox_conf'}
    if not required_columns.issubset(tracklets_df.columns):
        missing = required_columns - set(tracklets_df.columns)
        raise ValueError(f"tracklets_df is missing required columns: {missing}")
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ValueError("output_dir must be a non-empty string.")

    tracklets_dir = Path(output_dir) / "tracklets"
    tracklets_dir.mkdir(parents=True, exist_ok=True)

    # Group by track_id and save each group to a separate file
    for track_id, group in tracklets_df.groupby('track_id'):
        file_path = tracklets_dir / f"seq_{track_id}.txt"
        with open(file_path, 'w') as f:
            for _, row in group.iterrows():
                frame_id = int(row['image_id'])
                bbox = row['bbox_ltwh'].tolist()
                if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                    raise ValueError("Each 'bbox_ltwh' must be a list or tuple of length 4.")
                bb_left, bb_top, bb_width, bb_height = bbox
                conf = float(row['bbox_conf'])
                f.write(
                    f'{frame_id},{track_id},{bb_left:.2f},{bb_top:.2f},{bb_width:.2f},{bb_height:.2f},{conf:.5f},-1,-1,-1\n'
                )
