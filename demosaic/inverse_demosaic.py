import imageio
import os
import numpy as np
import cv2
import argparse


def reconstruct_bayer_image(RGB):
    # uint8 -> uint16
    RAW_combined = RGB.astype(np.uint16) * 4

    # Extract individual channels
    ch_B = RAW_combined[:, :, 0]
    ch_G = RAW_combined[:, :, 1]
    ch_R = RAW_combined[:, :, 2]

    # Get the original Bayer image dimensions
    H, W = ch_B.shape
    raw = np.zeros((H * 2, W * 2), dtype=np.uint16)

    # Reconstruct the Bayer pattern
    raw[1::2, 1::2] = ch_B  # Blue
    raw[0::2, 1::2] = ch_G  # Green (Gb)
    raw[0::2, 0::2] = ch_R  # Red
    raw[1::2, 0::2] = ch_G  # Green (Gr)

    return raw.astype(np.uint16)


def process_image(path, save, file):
    raw_image = np.asarray(imageio.imread(path))
    mosaic = reconstruct_bayer_image(raw_image)
    save_path = "{}/{}".format(save, file)
    print("Saving", save_path)
    cv2.imwrite(save_path, mosaic)


def batch_process(root, save):
    if not os.path.exists(save):
        os.makedirs(save)
    idx = 0
    for file in os.listdir(root):
        path = os.path.join(root, file)
        print("Loading", path)
        idx += 1
        process_image(path, save, str(idx) + ".png")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="For generating input for validation_4channel"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="/Documents/wzry/full_dataset/lq_sub",
        help="data directory of your raw images",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="/Documents/wzry/full_dataset/lq_sub_mosaic",
        help="save image folder",
    )
    args = parser.parse_args()
    batch_process(args.data, args.save)
