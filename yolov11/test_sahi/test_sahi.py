from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import cv2
from IPython.display import Image

model_path="/home/c515/lmx/ultralytics-8.3.35/runs/detect/train4/weights/best.pt"

detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path=model_path,
    confidence_threshold=0.3,
    device="cuda:0",  # or 'cuda:0'
)

result = get_sliced_prediction(
    "/home/c515/lmx/ultralytics-8.3.35/refer/refer6/result_0.jpg",

    detection_model,
    slice_height=320,
    slice_width=180,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,

)
result.export_visuals(export_dir="/home/c515/lmx/ultralytics-8.3.35/test_sahi/data_exp/exp2")
# Image("/home/c515/lmx/ultralytics-8.3.35/test_sahi/data_exp/prediction_visual.png")