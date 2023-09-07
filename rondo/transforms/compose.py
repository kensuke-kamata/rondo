class Compose:
    """Composes several transforms together."""
    def __init__(self, transforms=[]):
        self.transforms = []

    def __call__(self, img):
        if not self.transforms:
            return img
        for t in self.transforms:
            img = t(img)
        return img
