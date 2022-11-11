import cv2 as cv2
import numpy as np
import os
from skvideo import io
from glob import glob1
from tqdm import tqdm


# Flir dataset contains more rgb images
def clear_flir(rgb_path, gray_path):
    rgbs = sorted(glob1(rgb_path, "*.jpg"))
    grays = sorted(glob1(gray_path, "*.jpeg"))
    removed = 0
    for frame in grays.copy():
        fname = os.path.basename(frame).split('.')[0]
        if str(fname + '.jpg') not in rgbs:
            removed += 1
            grays.remove(frame)
    return rgbs, grays


def create_dataset(vids_frames, size, output):
    """
    Creates specified dataset for further training, type=[video, images]
    vids_frames: (rgb_vid, gray_vid, frames_to_take, type)
    size: (width, height)
    output: output file
    seg: segmentation function handle (should return segmentated image)
    """
    source = []
    target = []

    for (rgb_path, gray_path, frames, type) in vids_frames:
        print('Processing: ', rgb_path, gray_path)
        if type == "video":
            rgb_vid, gray_vid = cv2.VideoCapture(rgb_path), cv2.VideoCapture(gray_path)
            total_length = int(rgb_vid.get(cv2.CAP_PROP_FRAME_COUNT) - 10)
        else:
            rgb_vid, gray_vid = clear_flir(rgb_path, gray_path)
            total_length = len(rgb_vid) - 1

        selection = np.linspace(10, total_length, frames, dtype=int)
        for i in tqdm(selection):
            # Load
            if type == "video":
                rgb_vid.set(cv2.CAP_PROP_POS_FRAMES, i)
                gray_vid.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, rgb_frame = rgb_vid.read()
                ret, gray_frame = gray_vid.read()
            else:
                rgb_frame, gray_frame = cv2.imread(os.path.join(rgb_path, rgb_vid[i])), cv2.imread(os.path.join(gray_path, gray_vid[i]))

            if rgb_frame is not None and gray_frame is not None:
                # Convert
                rgb_frame, gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB), cv2.cvtColor(gray_frame, cv2.COLOR_BGR2RGB)
                # Resize
                rgb_frame = cv2.resize(rgb_frame, size)
                gray_frame = cv2.resize(gray_frame, size)
                # Save
                source.append(rgb_frame)
                target.append(gray_frame)
            else:
                print('Problem with reading frame at position: ', i)

    source = np.asarray(source, dtype=np.uint8)
    target = np.asarray(target, dtype=np.uint8)
    # Save as videos
    io.vwrite(output + '_source.mp4', source)
    io.vwrite(output + '_target.mp4', target)
    # Save as numpy array
    np.savez_compressed(output, A=source, B=target[:, :, :, 0:1])
    # Save multiple arrays for pytorch implementation
    if not os.path.exists(output):
        os.makedirs(output)
        for i in range(len(source)):
            np.savez_compressed(f'{output}/{i}', A=source[i], B=target[i, :, :, 0:1])


if __name__ == '__main__':
    flir_path = '/home/user3574/PycharmProjects/ai_robotrain/FLIR/'
    rgb_path = '/home/user3574/PycharmProjects/ai_robotrain/Utils/Origs/RGB/'
    ir_path = '/home/user3574/PycharmProjects/ai_robotrain/Utils/Origs/IR/'
    create_dataset([(f'{rgb_path}/rgb.mkv', f'{ir_path}/ir.mkv', 300, 'video'),
                    (f'{rgb_path}/rgb-1.mp4', f'{ir_path}/ir-1.mp4', 300, 'video'),
                    (f'{rgb_path}/rgb_1.mp4', f'{ir_path}/ir_1.mp4', 10, 'video'),
                    (f'{rgb_path}/rgb_2.mp4', f'{ir_path}/ir_2.mp4', 10, 'video'),
                    (f'{rgb_path}/rgb_3.mp4', f'{ir_path}/ir_3.mp4', 10, 'video'),
                    (f'{rgb_path}/rgb_4.mp4', f'{ir_path}/ir_4.mp4', 10, 'video'),
                    (f'{rgb_path}/rgb_5.mp4', f'{ir_path}/ir_5.mp4', 10, 'video'),
                    (f'{flir_path}/RGB', f'{flir_path}/thermal_8_bit', 350, 'image')], (256, 256), 'dataset_650_350')
