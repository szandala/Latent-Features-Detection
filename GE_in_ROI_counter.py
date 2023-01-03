from gradual_extrapolation import GradualExtrapolator
from torchray.attribution.grad_cam import grad_cam
from torch.nn.functional import interpolate

import torch
from torchvision import transforms
import cv2
from torchvision import models
from icecream import ic
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import patches
import json
import pprint
from statistics import variance
# READ roi csv
# load & prepare model
# itarate over roi's images
# do GE
# Eval map in roi
FIX="-im9"
source_file = f"rois{FIX}.csv"
output_file = f"rois{FIX}_outputs.csv"

open(output_file,'a+').close()

with open(source_file) as f:
    entries = [ e.split(";") for e in f.readlines()]
with open(output_file) as f:
    mapped = set([ e.split(";")[0] for e in f.readlines()])


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

NET_TYPE = "vgg16"
net = models.__dict__[NET_TYPE](pretrained=True)
# net = torch.load("vgg16_im_9_236.0783.pth")
net.eval()
# for param in net.parameters():
#     param.requires_grad = False

net.to(device)
GradualExtrapolator.register_hooks(net)

crop = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224))
])

normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
CLASS_NAMES = {
    "bird": 0,
    "camel": 1,
    "dog": 2,
    "fish": 3,
    "insect": 4,
    "musical_instrument": 5,
    "ship": 6,
    "snowboard": 7,
    "wheeled_vehicle": 8,
    "roulette": 8,
    "x": 8
    }
with open("classes.json") as json_file:
    CLASSES = json.load(json_file)

# # CLASSES_IDs = { int(k): v.split(",")[0].replace(" ", "_") for k,v in CLASSES.items() }
# CLASS_NAMES = { v.split(",")[0].replace(" ", "_").lower(): int(k) for k,v in CLASSES.items() }

def get_class_id(img_path):
    return CLASS_NAMES[img_path.lower().split("/")[1]]

def visualize_cam(mask, img):
    """Make heatmap from mask and synthesize GradCAM result image using heatmap and img.
    Args:
        mask (torch.tensor): mask shape of (1, 1, H, W) and each element has value in range [0, 1]
        img (torch.tensor): img shape of (1, 3, H, W) and each pixel value is in range [0, 1]

    Return:
        heatmap (torch.tensor): heatmap img shape of (3, H, W)
        result (torch.tensor): synthesized GradCAM result of same shape with heatmap.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask.squeeze()), cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b])

    # result = heatmap+img.cpu()
    # result = result.div(result.max()).squeeze()
    img = img.cpu()
    result = img.div(img.max()).squeeze()
    # ic(heatmap.histogram())
    return heatmap, result


def heatmap_in_roi(heatmap, rois):
    if not rois:
        var = heatmap.flatten().var()
        return 999999, "{:.2f}".format(var)#, heatmap.flatten().tolist()

    # entire = torch.sum(heatmap[0])
    most_fitted = 100000
    # with open("lol.txt", "w") as f:
    #     f.write(str(heatmap[0].tolist()))
    # ic(np.quantile(heatmap.flatten(), [0,0.25,0.5,0.75]))
    # median = heatmap.flatten().quantile(dim=0, q=0.3)
    biggest_square = 0
    var = 0
# [178, 137, 223, 223]
    # best_variances = []
    for (x1, y1, x2, y2) in rois:
        variancable = []
        roi_area = (x2-x1) * (y2-y1) #x2*y2
        if roi_area < biggest_square:
            continue
        biggest_square = roi_area
        inner = 0.0
        outer = 0.0
        for h, row in enumerate(heatmap[0]):
            for w, pixel in enumerate(row):
                if h > y1 and h < y2:
                    # print(f"h={h} between {y1}|{y2} = {pixel}")
                    if w > x1 and w < x2:
                        if pixel > 0.1:
                            inner += 1
                        variancable.append(float(pixel))
                        # continue
                # outer += pixel
        # ic(inner)

        if inner > 0 and (inner / roi_area) < most_fitted:
            most_fitted = inner / roi_area
            var = variance(variancable)
            best_variances = variancable[:]
    # print(f"inner = {inner}, outer = {outer}, entire = {entire}, sum = {inner + outer}")
    return int(most_fitted * 100), "{:.2f}".format(var)#, best_variances

def map_image(img_path, net, rois=[]):
    imagename_jpg = img_path.split("/")[-1]

    # img = cv2.imread(img_path)
    # resize_img = crop(img)
    # img = normalize(resize_img).to(device)
    # img = img.unsqueeze(0)

    # ic(img_path)

    img = torch.stack([normalize(crop(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)))]).to(device)
    img.requires_grad=True

    saliency_grad_cam = grad_cam(
        net,
        img,
        [get_class_id(img_path)],
        saliency_layer="features.28", #28 for vgg16
    )
    saliency_grad_cam = (saliency_grad_cam - saliency_grad_cam.min())/saliency_grad_cam.max()
    # ic(saliency_grad_cam.min(), saliency_grad_cam.max())
    saliency_gradual_grad_cam = GradualExtrapolator.get_smooth_map(saliency_grad_cam)
    # saliency_gradual_grad_cam = saliency_gradual_grad_cam.clip(saliency_gradual_grad_cam.flatten().quantile(dim=0, q=0.99), 1)
    heatmap, cam_result = visualize_cam(saliency_gradual_grad_cam.cpu(), img.cpu())
    fit, var = heatmap_in_roi(heatmap, rois)


    # output_filepath = img_path.replace("imagenet_images", "im_images_output").replace(imagename_jpg,
    #     f"roi-{fit:02}-{img_path.split('/')[1]}-{imagename_jpg.replace('.jpg', '.png')}")
    if FIX:
        output_filepath = img_path.replace("imagenet_9", "im_9_output").replace(imagename_jpg,
        f"roi-{fit:02}-{img_path.split('/')[1]}-{imagename_jpg.replace('.jpg', '.png')}")
    else:
        output_filepath = img_path.replace("imagenet_images", "im_images_output").replace(imagename_jpg,
        f"roi-{fit:02}-{img_path.split('/')[1]}-{imagename_jpg.replace('.jpg', '.png')}")
    # img_roi = cv2.imread("/" + roi_name)
    print(f"{output_filepath} as {img_path.split('/')[1]}")
    # img = torch.squeeze(img, 0)
    # print(roi_filepath)
    # plt.imshow(cv2.imread(roi_filepath))
    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax[0][0].imshow(cam_result.permute(1,2,0).detach().cpu().numpy())
    # saliency_grad_cam = interpolate(
    #             saliency_grad_cam,
    #             size=(224, 224),
    #             mode="lanczos",
    #             align_corners=False,
    #         )
    # bitmap = saliency_grad_cam[0].expand(3, 224, 224).permute(1, 2, 0).cpu().numpy()
    ax[1][0].imshow(
            saliency_grad_cam[0].permute(1,2,0).detach().cpu().numpy(), interpolation="lanczos", cmap="jet")
    ax[0][1].set_visible(False)
    # ax[0][1].hist(mapa, bins=10)
    # ax[1][0].imshow(saliency_grad_cam.permute(1,2,0).detach().cpu().numpy())
    ax[1][1].imshow(heatmap.permute(1,2,0).detach().cpu().numpy())
    # plt.imshow(cam_result.permute(1,2,0).detach().cpu().numpy())
    # plt.imshow(heatmap.permute(1,2,0).detach().cpu().numpy(), alpha=0.25)

    for roi in rois:
        rect = patches.Rectangle((roi[0], roi[1]), roi[2]-roi[0], roi[3]-roi[1], linewidth=1, edgecolor='fuchsia', facecolor='none')

        rect3 = patches.Rectangle((roi[0], roi[1]),roi[2]-roi[0], roi[3]-roi[1], linewidth=1, edgecolor='fuchsia', facecolor='none')

        # Add the patch to the Axes
        ax[0][0].add_patch(rect)

        ax[1][1].add_patch(rect3)

#ax[0].gca().add_patch(rect)
        # ax[1].gca().add_patch(rect)
    if not os.path.exists(os.path.dirname(output_filepath)):
        os.makedirs(os.path.dirname(output_filepath))

    fig.savefig(output_filepath) # bylo plt

    plt.clf()
    plt.cla()
    plt.close()

    with open(output_file, "a") as f:
        f.write(f"{img_path};{output_filepath};{fit};{var}\n")

# print(net)
# with torch.no_grad():
for f, rois in entries:
    if "roulette" not in f:
        if f in mapped:
            continue

    # if "black_widow" in f:
    print(f"Processing {f} with rois: {rois}")
    map_image(f, net, eval(rois))
