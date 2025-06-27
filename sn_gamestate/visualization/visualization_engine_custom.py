import os
from pathlib import Path
import cv2
import logging
import pandas as pd

from tracklab.visualization.visualization_engine import VisualizationEngine
from tracklab.callbacks import Progressbar

from sn_gamestate.utils.frame_extractor import frame_generator  # Make sure this path is correct

log = logging.getLogger(__name__)


class VisualizationEngineCustom(VisualizationEngine):
    def __init__(self, *args, **kwargs):
        # Just pass all arguments to the parent VisualizationEngine
        super().__init__(*args, **kwargs)

    def on_video_loop_end(self, engine, video_metadata, video_idx, detections, image_pred):
        """
        Visualize only the frames that were processed, using a generator to extract them efficiently.
        """
        if not (self.save_videos or self.save_images):
            return

        progress = engine.callbacks.get("progress", Progressbar(dummy=True))
        tracker_state = engine.tracker_state

        # Get all processed frame IDs for this video
        processed_ids = list(detections["image_id"].unique())
        video_path = video_metadata.iloc[video_idx]["name"]
        video_width = video_metadata.iloc[video_idx]["width"]
        video_height = video_metadata.iloc[video_idx]["height"]
        video_dir, video_name = os.path.split(video_path)
        save_dir = Path(video_dir) / "outputs"

        # Prepare video writer if needed
        video_writer = None
        if self.save_videos:
            filepath = save_dir / "videos_res" / f"{video_name}.mp4"
            filepath.parent.mkdir(parents=True, exist_ok=True)
            video_writer = cv2.VideoWriter(
                str(filepath),
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(self.video_fps),
                (video_width, video_height),
            )

        progress.init_progress_bar("vis", "Visualization", len(processed_ids))

        # Use the frame_generator to yield frames by processed_ids
        tracklet_image_ids = {int(track_id):0 for track_id in detections["track_id"].unique()}
        for image_id, frame in frame_generator(video_path, processed_ids):
            # Prepare detection and prediction data for this frame
            detections_pred = detections[detections.image_id == image_id] if len(detections) else None
            image_pred_row = image_pred.loc[image_id] if image_pred is not None and image_id in image_pred.index else None

            # Save original image if required
            if self.save_images:
                for track_id in detections[detections.image_id == image_id]["track_id"]:
                    track_id = int(track_id)
                    filepath = save_dir / "images" / f"seq_{track_id}" / "img1" / f"{tracklet_image_ids[track_id]:06d}.jpg"
                    tracklet_image_ids[track_id] += 1
                    filepath.parent.mkdir(parents=True, exist_ok=True)
                    assert cv2.imwrite(str(filepath), frame)
   
            # Draw frame using visualizers
            for visualizer in self.visualizers.values():
                try:
                    visualizer.draw_frame(frame, detections_pred, pd.DataFrame([]), image_pred_row, pd.DataFrame([]))
                except Exception as e:
                    log.warning(f"Visualizer {visualizer} raised error : {e} during drawing.")

            # Write to video if required
            if self.save_videos and video_writer is not None:
                video_writer.write(frame)

            progress.on_module_step_end(None, "vis", None, None)

        if video_writer is not None:
            video_writer.release()
        progress.on_module_end(None, "vis", None)