# 用来生成所需要的 annotion.txt和labele.txt
import os
import glob

dict = {'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}


image_dir = "/mnt/hdd-4t/data/hanyue/mobileNet/flower_for_cnn/val/"
assert os.path.exists(image_dir), "image dir does not exist..."

img_list = glob.glob(os.path.join(image_dir, "*", "*.jpg"))
assert len(img_list) > 0, "No images(.jpg) were found in image dir..."

classes_info = os.listdir(image_dir)
classes_info.sort()
classes_dict = {}

# create label file
with open(
    "/home/yuesang/Pythonproject/MobileNetV3/onnxinference/int8inference/my_labels.txt",
    "w",
) as lw:
    # 注意，没有背景时，index要从0开始
    for index, c in enumerate(classes_info, start=0):
        txt = "{}:{}".format(index, c)
        if index != len(classes_info):
            txt += "\n"
        lw.write(txt)
        classes_dict.update({c: str(dict.get(c))})

print("create my_labels.txt successful...")

# create annotation file
with open(
    "/home/yuesang/Pythonproject/MobileNetV3/onnxinference/int8inference/my_annotation.txt",
    "w",
) as aw:
    for img in img_list:
        str_list = img.split('/')
        txt = "{} {}".format(str_list[-1], dict.get(str_list[-2]))
        txt += "\n"
        aw.write(txt)
    # for img in img_list:
    #     img_classes = classes_dict[img.split("/")[-2]]
    #     txt = "{} {}".format(img, img_classes)
    #     if index != len(img_list):
    #         txt += "\n"
    # aw.write(txt)
print("create my_annotation.txt successful...")
