import json

def generate_metadata(image_paths, detector):
    metadata = {}

    for img in image_paths:
        detections = detector.predict(img)
        metadata[img] = {
            "plate_count": len(detections),
            "detections": detections
        }

    return metadata


def save_metadata(metadata, path):
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)


def load_metadata(path):
    with open(path, "r") as f:
        return json.load(f)
