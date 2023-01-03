# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
# import os, json, cv2_imshow
import os, glob, cv2
# from PIL import Image

# # import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
# , DatasetCatalog

IMAGE_SIZE = 224

cfg = get_cfg()
cfg.MODEL.DEVICE = 'gpu'
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
PREDICTOR = DefaultPredictor(cfg)
# print(im.shape)

# im = cv2.imread("input.png")
def mark_rois(img_path):
    filename = os.path.basename(img_path)

    im = cv2.imread(img_path)
    im = cv2.resize(im, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
    # cv2.imshow('image', im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    outputs = PREDICTOR(im)
    boxes = [[int(i) for i in b.tolist()] for b in outputs['instances'].pred_boxes]
    print(outputs["instances"].pred_classes)
    print(boxes)

    with open("rois.csv", "a") as f:
        f.write(f"{img_path};{boxes}\n")

    # We can use `Visualizer` to draw the predictions on the image.
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # cv2_imshow()
    # outputname = filename.split(".")[0] + "-roi.png"

    output_filepath = "roi-" + img_path.replace("images", "output").replace(filename,
        f"roi-{filename.replace('.jpg', '.png')}")
    print(output_filepath)
    if not os.path.exists(os.path.dirname(output_filepath)):
        os.makedirs(os.path.dirname(output_filepath))

    cv2.imwrite(output_filepath, out.get_image()[:, :, ::-1])

# images = glob.glob("images_full/imagenet_images/**/*.jpg", recursive=True)
images = glob.glob("images_full/*.jpg")
with open("rois.csv") as f:
    readed = set( [ l.split(";")[0] for l in f.readlines() ] )

for image in images:
    if image in readed:
        continue
    mark_rois(image)
