import os

SUPPORTED_EXTENSIONS = [
   '.npz'
]


def is_supported_file(filename):
    return any(filename.endswith(extension) for extension in SUPPORTED_EXTENSIONS)


# walks through a folder and gets list of files inside
# fixme: bloat
def load_frames(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            path = os.path.join(root, fname)
            images.append(path)
    return images[:min(max_dataset_size, len(images))]