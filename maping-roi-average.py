import torch
import cv2
import matplotlib.pyplot as plt
import torchvision.models as models
from torchvision import transforms
from gradcam import GradCAM
from utils import visualize_cam
import json
import glob
from collections import defaultdict
import pickle

with open("classes.json") as json_file:
    CLASSES = json.load(json_file)

CLASSES = { int(k): v.split(",")[0].replace(" ", "_") for k,v in CLASSES.items() }

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

NET_TYPE = "resnet18"
net = models.__dict__[NET_TYPE](pretrained=True)

net.eval()
net.to(device)

crop = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224))
    ])

normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def heatmap_in_roi(heatmap, rois):
    if not rois:
        return 999

    entire = torch.sum(heatmap[0])
    most_fitted = 0
    # with open("lol.txt", "w") as f:
    #     f.write(str(heatmap[0].tolist()))
    for (x1, y1, x2, y2) in rois:
        inner = 0.0
        outer = 0.0
        for h, row in enumerate(heatmap[0]):
            for w, pixel in enumerate(row):
                if h > y1 and h < y2:
                    # print(f"h={h} between {y1}|{y2} = {pixel}")
                    if w > x1 and w < x2:
                        inner += pixel
                        continue
                outer += pixel
        if (inner / entire) > most_fitted:
            most_fitted = inner / entire

    return int(most_fitted * 100)

FITTINGS = defaultdict(lambda: [], {})

def map_image(img_path, gradcam, rois=[]):
    if not rois:
        print(f"Skipping {img_path} as there are no rois")
        return

    global FITTINGS

    imagename_jpg = img_path.split("/")[-1]

    img = cv2.imread(img_path)
    resize_img = crop(img)
    img = normalize(resize_img).to(device)
    img = img.unsqueeze(0)

    mask, logit = gradcam(img)
    classified_as_id = int(torch.argmax(logit))

    class_name = CLASSES[classified_as_id]
    heatmap, cam_result = visualize_cam(mask.cpu(), img.cpu())
    fit = heatmap_in_roi(heatmap, rois)

    FITTINGS[class_name].append(fit)

    roi_filepath = "roi-" + img_path.replace("images", "output").replace(imagename_jpg,
        f"roi-{imagename_jpg.replace('.jpg', '.png')}")

    output_filepath = img_path.replace("images", "output").replace(imagename_jpg,
        f"{fit:02}-roi-{class_name}-{imagename_jpg.replace('.jpg', '.png')}")

    # img_roi = cv2.imread("/" + roi_name)
    print(f"{output_filepath} as {class_name}")
    # img = torch.squeeze(img, 0)
    print(roi_filepath)
    plt.imshow(cv2.imread(roi_filepath))
    # plt.imshow(cam_result.permute(1,2,0))
    plt.imshow(heatmap.permute(1,2,0), alpha=0.5)
    plt.savefig(output_filepath)

    plt.clf()
    plt.cla()
    plt.close()


# my_images = sorted(glob.glob("images/**/*.jpg", recursive=True))
# for i, f in enumerate(my_images):
#     print(f"Processing {i}/{len(my_images)}: {f}")
#     map_image(f, net)

# my_images = sorted(glob.glob("images/*.jpg"))

# accuracy = 0
# my_images = glob.glob("images_im9/imagenet_images/**/*.jpg", recursive=True)
# for i, f in enumerate(my_images):
#     print(f"Processing {i}/{len(my_images)}: {f}")
#     if classify_image(f, net):
#         accuracy += 1

for param in net.parameters():
    param.requires_grad = True


with open("rois.csv") as f:
    entries = [ e.split(";") for e in f.readlines()]

# GradCAM setup
model_dict = dict(type='resnet', arch=net, layer_name='features', input_size=(224, 224))
gradcam = GradCAM(model_dict)

for i, (f, rois) in enumerate(entries):
    print(f"Processing {i}: {f} with rois: {rois}")
    map_image(f, gradcam, eval(rois))
    if i % 100 == 0:
        print(f"Saving at {i}")
        with open("fittings.pickle", "wb") as f:
            pickle.dump(FITTINGS, f)


# print(f"Accuracy {accuracy}/{len(my_images)}")
