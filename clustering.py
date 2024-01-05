import torch
from PIL import Image
from sklearn.cluster import KMeans
from torchvision import transforms


class ClusterDataset(torch.utils.data.Dataset):

    """"
    Characterizes a dataset for PyTorch
    for forwarding data to model calculate embedding just for clustering
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
                            transforms.CenterCrop(224)])
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
        # print(self.image_list[idx][0]) it is path of a sample
        # print(self.image_list[idx][-1]) it is label of a sample

        # (label, path, image tensor)
        return self.image_list[idx][-1], self.image_list[idx][0], self.transform(
            self.ensure_3dim(Image.open(self.image_list[idx][0])))

    def __len__(self):
        return len(self.image_list)


def clustering(image_dict, model, opt):
    print("befor clustering .... ", opt.num_cluster)
    image_dataset = ClusterDataset(image_dict, opt)
    # print("image list len ==  ", len(image_dataset.image_list))
    # print("avail classes len ==  ", len(image_dataset.image_dict.keys()))
    params = {'batch_size': 8,
              'shuffle': False,
              'num_workers': 8}
    training_generator = torch.utils.data.DataLoader(image_dataset, **params, pin_memory=True)
    embedding = []
    paths = []
    real_label=[]
    with torch.no_grad():
        for i, (image_label, image_path, image_tensor) in enumerate(training_generator):
            features = model(image_tensor.to(opt.device))
            embedding.extend(features[0])
            paths.extend(image_path)
            real_label.extend(image_label)

    print("_____________features calculated ______________")
    embedding = [i.detach().cpu().numpy() for i in embedding]
    real_label=[i.numpy().tolist() for i in real_label]
    sub_classes_label = [[] for _ in set(real_label)]
    sub_classes_path = [[] for _ in set(real_label)]
    sub_classes_embedding = [[] for _ in set(real_label)]
    for i, label in enumerate(real_label):
        sub_classes_label[label].append(label)
        sub_classes_path[label].append(paths[i])
        sub_classes_embedding[label].append(embedding[i])
    # we should clustering all classes sepraitly
    cluster_dict = {}
    for c_label, (c_embedding, c_path) in enumerate(zip(sub_classes_embedding,sub_classes_path)):

        kmeans = KMeans(n_clusters=opt.num_cluster, random_state=0).fit(c_embedding)
        labels = kmeans.labels_
        # print(labels)
        tmp_dict = {i: [] for i in set(labels)}
        # print(tmp_dict)

        for idx, (l, p) in enumerate(zip(labels, c_path)):
            tmp_dict[l].append((c_label, p))

        for key, item in tmp_dict.items():
            cluster_dict[len(cluster_dict)] = item
    print(cluster_dict.keys())
    print(len(cluster_dict.keys()))
    return cluster_dict
