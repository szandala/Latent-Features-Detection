import torch
import torchvision
import glob
import cv2
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from random import shuffle
import torchvision.models as models
from torchvision import transforms
import time
from icecream import ic

torch.cuda.empty_cache()
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

NET_TYPE = "vgg16"
net = models.__dict__[NET_TYPE](pretrained=True)
print(net)

# net = torch.load("resnet50_im_new_3.1313.pth")
# net.eval()
# net.to(device)

# disable training for hidden layers
for param in net.parameters():
    param.requires_grad = True

# reduce amount of features:
num_ftrs = net.classifier[0].in_features
net.fc = nn.Linear(num_ftrs, 9)

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")
net.to(device)

##########################
### loading train_set
##########################

addrs = glob.glob("imagenet_9/**/*.jpg", recursive=True)


def class_namer(address):
    return address.split("/")[1]

classifier_names = {
    "bird": 0,
    "camel": 1,
    "dog": 2,
    "fish": 3,
    "insect": 4,
    "musical_instrument": 5,
    "ship": 6,
    "snowboard": 7,
    "wheeled_vehicle": 8
}

labels = []
for address in addrs:
    l = 9*[0]
    # ic(address)
    # ic(class_namer(address))
    # ic(classifier_names.get(class_namer(address)))
    l[classifier_names.get(class_namer(address))] = 1
    labels.append(torch.Tensor(l))

##########################
### training
##########################
crop = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224))
    ])


normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

img_data = []
for i in range(len(addrs)):
    if i % 100 == 0 and i > 1:
        print ('Train data: {}/{}'.format(i, len(addrs)))

    img = cv2.imread(addrs[i])
    resize_img = crop(img)
    img = normalize(resize_img).to(device)
    img = img.unsqueeze(0)
    img_data.append([img.to(device), labels[i].to(device)])
shuffle(img_data)

training = img_data

############################################################
############################################################
############################################################

####
# https://discuss.pytorch.org/t/valueerror-expected-input-batch-size-324-to-match-target-batch-size-4/24498
####
torch.set_printoptions(sci_mode=False)

criterian = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.0001, momentum = 0.9)
ALL_START = time.time()
for epoch in range(5):
    running_loss  = 0.0
    start = time.time()
    for i, (X, Y) in enumerate(training):
        if i % 100 == 0:
            print(f"Image {i}/{len(training)}")
        # print(Y.argmax())
        optimizer.zero_grad()
        output = net(X)
        loss = criterian(output, Y.argmax().unsqueeze(0))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"\n\n{epoch}: {running_loss:.4f} in {((time.time()-start) / 60):.2f} minutes\n\n")

print(f"Done in {((time.time()-ALL_START) / 60):.2f} minutes")
torch.save(net, f"{NET_TYPE}_im_9_{running_loss:.4f}.pth")
