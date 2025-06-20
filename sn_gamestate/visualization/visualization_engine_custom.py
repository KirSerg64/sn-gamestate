from pathlib import Path
import cv2
import logging

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
        image_metadatas = tracker_state.image_metadatas[tracker_state.image_metadatas.video_id == video_idx]
        processed_ids = list(image_metadatas.index)
        video_name = tracker_state.video_metadatas.loc[video_idx]["name"]
        video_path = video_name  # Adjust if you need full path

        # Prepare video writer if needed
        video_writer = None
        if self.save_videos and len(image_metadatas) > 0:
            first_image = image_metadatas.iloc[0]
            first_frame = cv2.imread(first_image.file_path)
            filepath = self.save_dir / "videos" / f"{video_name}.mp4"
            filepath.parent.mkdir(parents=True, exist_ok=True)
            video_writer = cv2.VideoWriter(
                str(filepath),
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(self.video_fps),
                (first_frame.shape[1], first_frame.shape[0]),
            )

        progress.init_progress_bar("vis", "Visualization", len(processed_ids))

        # Use the frame_generator to yield frames by processed_ids
        for image_id, frame in frame_generator(video_path, processed_ids):
            # Prepare detection and prediction data for this frame
            detections_pred = detections[detections.image_id == image_id] if len(detections) else None
            image_pred_row = image_pred.loc[image_id] if image_pred is not None and image_id in image_pred.index else None
            image_gt_row = tracker_state.image_gt.loc[image_id] if hasattr(tracker_state, "image_gt") and image_id in tracker_state.image_gt.index else None
            detections_gt = tracker_state.detections_gt[tracker_state.detections_gt.image_id == image_id] if hasattr(tracker_state, "detections_gt") and tracker_state.detections_gt is not None else None

            # Draw frame using visualizers
            for visualizer in self.visualizers.values():
                try:
                    visualizer.draw_frame(frame, detections_pred, detections_gt, image_pred_row, image_gt_row)
                except Exception as e:
                    log.warning(f"Visualizer {visualizer} raised error : {e} during drawing.")

            # Save image if required
            if self.save_images:
                filepath = self.save_dir / "images" / str(video_name) / f"{image_id}.jpg"
                filepath.parent.mkdir(parents=True, exist_ok=True)
                assert cv2.imwrite(str(filepath), frame)

            # Write to video if required
            if self.save_videos and video_writer is not None:
                video_writer.write(frame)

            progress.on_module_step_end(None, "vis", None, None)

        if video_writer is not None:
            video_writer.release()
        progress.on_module_end(None, "vis", None)