import rondo

class ImageNet(rondo.Dataset):
    def __init__(self):
        NotImplemented

    @staticmethod
    def labels():
        url = 'https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt'
        path = rondo.utils.download(url)
        with open(path, 'r') as f:
            labels = eval(f.read())
        return labels
