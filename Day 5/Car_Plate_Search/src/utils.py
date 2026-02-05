import os

def get_images_from_dir(directory):
    exts = (".jpg", ".jpeg", ".png")
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(exts)
    ]
