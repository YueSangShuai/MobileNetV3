import os
import cv2


def get_filelist(dir):
    Filelist = []

    for home, dirs, files in os.walk(dir):
        for filename in files:
            # 文件名列表，包含完整路径

            Filelist.append(os.path.join(home, filename))

            # # 文件名列表，只包含文件名

            # Filelist.append( filename)

    return Filelist

    img = cv2.imread(path, 0)

    # kernel = np.ones((6, 6), np.uint8)
    # kerne2 = np.ones((3, 3), np.uint8)
    # img = cv2.erode(img, kernel)
    # img=cv2.dilate(img, kerne2)
    cv2.imwrite(imwrite_name, img)


if __name__ == "__main__":
    imread_dir = "/mnt/hdd-4t/data/hanyue/mobileNet/flower_for_cnn/val/"
    files = get_filelist(imread_dir)
    imwrite_dir = "/mnt/hdd-4t/data/hanyue/mobileNet/flower"
    # print(len(files))
    for file in files:
        img = cv2.imread(file)
        # print(os.path.join(imwrite_dir, os.path.split(file)[1]))
        cv2.imwrite(os.path.join(imwrite_dir, os.path.split(file)[-1]), img)
