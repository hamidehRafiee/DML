import torch
from PIL import Image
from sklearn.cluster import KMeans
from torchvision import transforms


# torch.backends.cudnn.benchmark = True

class Dataset(torch.utils.data.Dataset):
    """"Characterizes a dataset for PyTorch
    """

    def __init__(self, image_dict, opt):
        # 'Initialization'
        # self.labels = labels
        self.image_dict = image_dict
        self.avail_classes = sorted(list(self.image_dict.keys()))
        self.image_dict = {i: self.image_dict[key] for i, key in enumerate(self.avail_classes)}

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transf_list = []
        transf_list.extend([transforms.Resize(256),
                            transforms.CenterCrop(
                                224) if opt.arch == 'resnet50' else transforms.CenterCrop(227)])
        transf_list.extend([transforms.ToTensor(), normalize])

        self.transform = transforms.Compose(transf_list)

        # Convert Image-Dict to list of (image_path, image_class).
        # Allows for easier direct sampling.
        self.image_list = [[(x, key) for x in self.image_dict[key]] for key in
                           self.image_dict.keys()]
        self.image_list = [x for y in self.image_list for x in y]

    def ensure_3dim(self, img):
        """
        Function that ensures that the input img is three-dimensional.

        Args:
            img: PIL.Image, image which is to be checked for three-dimensionality (i.e. if some images are black-and-white in an otherwise coloured dataset).
        Returns:
            Checked PIL.Image img.
        """
        if len(img.size) == 2:
            img = img.convert('RGB')
        return img

    def __getitem__(self, idx):
        """
        Args:
            idx: Sample idx for training sample
        Returns:
            tuple of form (path of image , torch.Tensor() of input image)
        """
        # print(self.image_list[idx][0])
        return self.image_list[idx][0], self.transform(
            self.ensure_3dim(Image.open(self.image_list[idx][0])))

    def __len__(self):
        return len(self.image_list)


def clustering(image_dict, model, opt):
    image_dataset = Dataset(image_dict, opt)
    print("image list len ==  ", len(image_dataset.image_list))
    print("avail classes len ==  ", len(image_dataset.image_dict.keys()))
    params = {'batch_size': 8,
              'shuffle': False,
              'num_workers': 8}
    training_generator = torch.utils.data.DataLoader(image_dataset, **params, pin_memory=True)
    embedding = []
    paths = []
    with torch.no_grad():
        for i, (image_path, image_tensor) in enumerate(training_generator):
            # print(image_path)
            # print(len(image_path))
            # print(type(image_path))
            # print(image_tensor.shape)
            features = model(image_tensor.to(opt.device))
            embedding.extend(features)
            paths.extend(image_path)

    # print(len(embedding))
    print("_______________________________features calculated ___________________________________")
    embedding = [i.detach().cpu().numpy() for i in embedding]
    kmeans = KMeans(n_clusters=100, random_state=0).fit(embedding)
    labels = kmeans.labels_
    # print(len(labels))
    # print(len(embedding))
    # print(len(paths))
    data_list = []
    [data_list.append((labels[i], paths[i])) for i in range(len(labels))]
    # [image_list[i].append(labels[i]) for i in range(len(image_list))]
    cluster_dict = {}
    for key, img_path in data_list:

        if not key in cluster_dict.keys():
            cluster_dict[key] = []
        cluster_dict[key].append(img_path)
    # for changing labels of cluster classes
    cluster_dict_new = {}
    for i in cluster_dict.keys():
        cluster_dict_new[i + 100] = cluster_dict[i]

    return cluster_dict_new
