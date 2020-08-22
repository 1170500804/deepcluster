from __future__ import print_function, division

import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import skimage
eps = np.finfo(float).eps #small number to avoid zeros
#from attention_utils.utils import sliding_window
class cluster_year_built_dataset(Dataset):
    def __init__(self, attribute_name, csv_path, img_path, transform=None, regression=False, mask_buildings=False, softmask=False,steps=10):
        if (attribute_name != 'year_built' and attribute_name != 'effective_year_built:') or regression:
            raise ValueError('Wrong attribute or training type for this dataset: {}'.format(attribute_name))

        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.regression = regression
        self.attribute_name = attribute_name
        self.mask_buildings = mask_buildings
        self.softmask=softmask
        min_year = 1913
        max_year = 2012
        self.img_path = img_path
        self.classes = []
        for year in self.df[self.attribute_name].unique():
            self.classes.append(year)
        self.classes = sorted(self.classes)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.df.iloc[idx]['filename']
        image = Image.open(os.path.join(self.img_path, img_name))

        if self.mask_buildings:
            image = np.array(image)
            if self.softmask:
                mask_filename = self.df.iloc[idx]['filename'].replace('.jpg', '-softmask.npy')
                mask = np.load(os.path.join(self.img_path,mask_filename))
                mask = np.array(mask)
                image = np.array(np.stack(
                    (image[:, :, 0] * mask, image[:, :, 1] * mask, image[:, :, 2] * mask), 2),
                         dtype=np.uint8)
                #plt.imshow(image)
                #plt.show()
            else:
                mask_filename = self.df.iloc[idx]['filename'].replace('jpg', 'png')
                mask = Image.open(os.path.join(self.img_path, mask_filename))
                mask = np.array(mask)
                # Filter building labels
                mask[np.where((mask != 25) & (mask != 1))] = 0
                image[mask == 0, :] = 0
                #plt.imshow(image)
                #plt.show()
            image = Image.fromarray(np.uint8(image))

        label = self.df.iloc[idx][self.attribute_name]
        # try:
        #     label = self.label_lookup[int(label)] # Translate to coarse class
        # except KeyError:
        #     # year not in class. That can happen when validation classes are not in training
        #     # We choose the most appropriate class instead
        #     for i, class_range in enumerate(self.classes):
        #         if int(label) > class_range[0] and int(label) < class_range[-1]:
        #             label = i
        #             break


        if (self.transform):
            image = self.transform(image)

        return (image, label, []) # [] for compatibility

class Rolling_Window_Year_Built_Dataset(Dataset):
    '''
    Generic Dataset to access building type information. Possible values are
    'building_address_full',
       'first_floor_elevation_ft', 'assessment_type', 'year_built',
       'effective_year_built', 'roof_shape', 'roof_cover', 'wall_cladding',
       'number_of_stories', 'building_address_full_cleaned'
    '''

    def __init__(self, attribute_name, csv_path, img_path, transform=None, regression=False, mask_buildings=False, softmask=False,steps=10):
        if (attribute_name != 'year_built' and attribute_name != 'effective_year_built:') or regression:
            raise ValueError('Wrong attribute or training type for this dataset: {}'.format(attribute_name))

        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.regression = regression
        self.attribute_name = attribute_name
        self.mask_buildings = mask_buildings
        self.softmask=softmask
        self.steps = steps

        #min_year = self.df[self.attribute_name].min()
        #max_year = self.df[self.attribute_name].max()
        #max_year += int((max_year - min_year) % 10)+2 # padd to full 10 year intervals
        min_year = 1913
        max_year = 2012 + steps + 1 # Not all datasets have all years so this needs to be hard set

        #classes = sliding_window(np.array(range(int(min_year),int(max_year))), size=10, stepsize=10)
        self.classes = skimage.util.view_as_windows(np.array(range(int(min_year),int(max_year))),steps,step=steps)
        classes = self.classes
        self.class_names = [(str(start) + '-' + str(end)) for start, end in zip(classes[:, 0], classes[:, -1])]
        self.label_lookup = {}
        for year in self.df[self.attribute_name].unique():
            for i in range(len(classes)):
                if int(year) in classes[i]:
                    self.label_lookup[int(year)] = i
                    break


        self.img_path = img_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.df.iloc[idx]['filename']
        image = Image.open(os.path.join(self.img_path, img_name))

        if self.mask_buildings:
            image = np.array(image)
            if self.softmask:
                mask_filename = self.df.iloc[idx]['filename'].replace('.jpg', '-softmask.npy')
                mask = np.load(os.path.join(self.img_path,mask_filename))
                mask = np.array(mask)
                image = np.array(np.stack(
                    (image[:, :, 0] * mask, image[:, :, 1] * mask, image[:, :, 2] * mask), 2),
                         dtype=np.uint8)
                #plt.imshow(image)
                #plt.show()
            else:
                mask_filename = self.df.iloc[idx]['filename'].replace('jpg', 'png')
                mask = Image.open(os.path.join(self.img_path, mask_filename))
                mask = np.array(mask)
                # Filter building labels
                mask[np.where((mask != 25) & (mask != 1))] = 0
                image[mask == 0, :] = 0
                #plt.imshow(image)
                #plt.show()
            image = Image.fromarray(np.uint8(image))

        label = self.df.iloc[idx][self.attribute_name]
        try:
            label = self.label_lookup[int(label)] # Translate to coarse class
        except KeyError:
            # year not in class. That can happen when validation classes are not in training
            # We choose the most appropriate class instead
            for i, class_range in enumerate(self.classes):
                if int(label) > class_range[0] and int(label) < class_range[-1]:
                    label = i
                    break


        if (self.transform):
            image = self.transform(image)

        return (image, label, []) # [] for compatibility

class First_Floor_Binary(Dataset):
    '''
    Generic Dataset to access building type information. Possible values are
    'building_address_full',
       'first_floor_elevation_ft', 'assessment_type', 'year_built',
       'effective_year_built', 'roof_shape', 'roof_cover', 'wall_cladding',
       'number_of_stories', 'building_address_full_cleaned'
    '''
    def __init__(self, attribute_name, csv_path, img_path, transform=None, regression=False, mask_buildings=False):
        if regression:
            raise ValueError('This dataset does not support regression')
        if attribute_name != 'first_floor_elevation_ft':
            raise ValueError('Wrong attribute type for this dataset')

        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.regression = regression
        self.attribute_name = attribute_name
        self.mask_buildings = mask_buildings

        if not regression:
            self.class_names = ['Small','Big']
        else:
            self.class_names = None
        self.train_labels = np.array(self.df[attribute_name].values > 8,dtype=np.uint8)
        self.img_path = img_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.df.iloc[idx]['filename']
        image = Image.open(os.path.join(self.img_path, img_name))

        if self.mask_buildings:
            image = np.array(image)
            mask_filename = self.df.iloc[idx]['filename'].replace('jpg', 'png')
            mask = Image.open(os.path.join(self.img_path, mask_filename))
            mask = np.array(mask)
            # Filter building labels
            mask[np.where((mask != 25) & (mask != 1))] = 0
            image[mask == 0, :] = 0
            image = Image.fromarray(np.uint8(image))

        label = self.df.iloc[idx][self.attribute_name]

        # Convert feet numbers into discrete classes
        label = label >= 8.0
        label = int(label)

        if (self.transform):
            image = self.transform(image)

        return (image, label, idx)

class Building_Information_Dataset(Dataset):
    '''
    Generic Dataset to access building type information. Possible values are
    'building_address_full',
       'first_floor_elevation_ft', 'assessment_type', 'year_built',
       'effective_year_built', 'roof_shape', 'roof_cover', 'wall_cladding',
       'number_of_stories', 'building_address_full_cleaned'
    '''
    def __init__(self, attribute_name, csv_path, img_path, transform=None, regression=False, mask_buildings=False, softmask=False):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.regression = regression
        self.attribute_name = attribute_name
        self.mask_buildings = mask_buildings
        self.softmask=softmask
        if not regression:
            self.class_names = np.sort(self.df[attribute_name].unique())
        else:
            self.class_names = None


        self.img_path = img_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.df.iloc[idx]['filename']
        image = Image.open(os.path.join(self.img_path, img_name))

        if self.mask_buildings:
            image = np.array(image)
            if self.softmask:
                mask_filename = self.df.iloc[idx]['filename'].replace('.jpg', '-softmask.npy')
                mask = np.load(os.path.join(self.img_path,mask_filename))
                mask = np.array(mask)
                image = np.array(np.stack(
                    (image[:, :, 0] * mask, image[:, :, 1] * mask, image[:, :, 2] * mask), 2),
                         dtype=np.uint8)
                #plt.imshow(image)
                #plt.show()
            else:
                mask_filename = self.df.iloc[idx]['filename'].replace('jpg', 'png')
                mask = Image.open(os.path.join(self.img_path, mask_filename))
                mask = np.array(mask)
                # Filter building labels
                mask[np.where((mask != 25) & (mask != 1))] = 0
                image[mask == 0, :] = 0
                #plt.imshow(image)
                #plt.show()
            image = Image.fromarray(np.uint8(image))

        label = self.df.iloc[idx][self.attribute_name]

        if self.regression:
            label = torch.from_numpy(np.asarray(label))
        else:
            # Convert feet numbers into discrete classes
            label = np.flatnonzero(
                np.array(self.class_names) == label)  # Flatnonzero is the sane version of np.where which does not return weird tuples
            label = label.squeeze()

        if (self.transform):
            image = self.transform(image)

        return (image, label, []) # [] to make it compatible to other datasets

class No_Of_Stories(Dataset):
    def __init__(self, csv_path, img_path, transform=None, regression=False):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.regression = regression

        # self.classes = np.sort(self.df['first_floor_elevation_ft'].unique())
        self.ffe_classes = [0., 1., 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5,
                            4., 4.5, 5., 6., 7., 8., 8.5, 9., 10.,
                            11., 12., 13., 14.]  # No dataset alone has all the clases therefore hard define this here
        self.nos_classes = [0., 1., 1.5, 2., 3., 4., 5., 14.]

        self.img_path = img_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.df.iloc[idx]['filename']
        image = Image.open(os.path.join(self.img_path, img_name))
        elevation = self.df.iloc[idx]['first_floor_elevation_ft']
        number_of_stories = self.df.iloc[idx]['number_of_stories']
        if self.regression:
            ffe_label = torch.from_numpy(np.asarray(elevation))
            nos_label = torch.from_numpy(np.asarray(number_of_stories))
        else:
            # Convert feet numbers into discrete classes
            ffe_label = np.flatnonzero(
                self.ffe_classes == elevation)  # Flatnonzero is the sane version of np.where which does not return weird tuples
            ffe_label = ffe_label.squeeze()

        if (self.transform):
            image = self.transform(image)

        return (image, ffe_label, nos_label)

class Floor_Ele(Dataset):
    def __init__(self, csv_path, img_path, transform=None, regression=False):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.regression = regression

        # self.classes = np.sort(self.df['first_floor_elevation_ft'].unique())
        self.ffe_classes = [0., 1., 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5,
                            4., 4.5, 5., 6., 7., 8., 8.5, 9., 10.,
                            11., 12., 13., 14.]  # No dataset alone has all the clases therefore hard define this here
        self.nos_classes = [0., 1., 1.5, 2., 3., 4., 5., 14.]

        self.img_path = img_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.df.iloc[idx]['filename']
        image = Image.open(os.path.join(self.img_path, img_name))
        elevation = self.df.iloc[idx]['first_floor_elevation_ft']
        number_of_stories = self.df.iloc[idx]['number_of_stories']
        if self.regression:
            ffe_label = torch.from_numpy(np.asarray(elevation))
            nos_label = torch.from_numpy(np.asarray(number_of_stories))
        else:
            # Convert feet numbers into discrete classes
            ffe_label = np.flatnonzero(
                self.ffe_classes == elevation)  # Flatnonzero is the sane version of np.where which does not return weird tuples
            ffe_label = ffe_label.squeeze()

        if (self.transform):
            image = self.transform(image)

        return (image, ffe_label, nos_label)

class Number_of_Stories(Dataset):
    def __init__(self, csv_path, img_path, transform=None, regression=False):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.regression = regression

        self.classes = [0., 1., 1.5, 2., 3., 4., 5., 14.]

        self.img_path = img_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.df.iloc[idx]['filename']
        image = Image.open(os.path.join(self.img_path, img_name))

        number_of_stories = self.df.iloc[idx]['number_of_stories']
        if self.regression:
            label = torch.from_numpy(np.asarray(number_of_stories))
        else:
            # Convert feet numbers into discrete classes
            label = np.flatnonzero(
                self.classes == number_of_stories)  # Flatnonzero is the sane version of np.where which does not return weird tuples
            label = label.squeeze()

        if (self.transform):
            image = self.transform(image)

        return (image, label)


class Fine_Grained_Dataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.classes = self.df.loc[:, 'class'].unique()

        self.labels = {}
        for c in self.classes:
            self.labels[c] = int(c) - 5001

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.df.loc[idx, 'dir']
        image = Image.open(img_name)
        label = self.labels[self.df.loc[idx, 'class']]
        if (self.transform):
            image = self.transform(image)
        return (image, label)


class Coarse_Grained_Dataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.classes = list(range(2))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.df.loc[idx, 'dir']
        image = Image.open(img_name)

        # class_id = self.df.loc[idx, 'class']
        # if class_id in [5001, 5005]:
        #     label = 0
        # elif class_id in [5004, 5006]:
        #     label = 1
        # elif class_id in [5002, 5003]:
        #     label = 2
        # else:

        class_id = self.df.loc[idx, 'class']
        if class_id in [5001, 5005, 5002, 5003]:
            label = torch.FloatTensor([0, 1])
        elif class_id in [5004, 5006]:
            label = torch.FloatTensor([1, 0])
        else:
            print("Something is wrong with the class ids")
            exit()

        if (self.transform):
            image = self.transform(image)
        return (image, label)



# ############ The specific NPID datasets are deprecated and functionality is included in a different way
# class NPID_Feat_Dataset(Dataset):
#     '''
#     Generic Dataset to access building type information and return saved NPID features. Possible values are
#     'building_address_full',
#        'first_floor_elevation_ft', 'assessment_type', 'year_built',
#        'effective_year_built', 'roof_shape', 'roof_cover', 'wall_cladding',
#        'number_of_stories', 'building_address_full_cleaned'
#     '''
#
#     def __init__(self, attribute_name, csv_path, img_path, transform=None, regression=False, mask_buildings=False, softmask=False):
#
#         self.df = pd.read_pickle(csv_path)
#         self.transform = transform
#         self.regression = regression
#         self.attribute_name = attribute_name
#         self.mask_buildings = mask_buildings
#         self.softmask=softmask
#
#         self.class_names = np.sort(self.df[attribute_name].unique())
#
#         class_label_histogram = np.array([len(self.df[self.df[attribute_name] == num]) for num in self.df[attribute_name].unique()]) # numpy histogram has weird effects for some classes
#         sort_order = self.df[attribute_name].unique().argsort()
#         unique_labels_sorted = self.df[attribute_name].unique()[sort_order]
#         class_label_histogram = class_label_histogram[sort_order]
#         class_label_weights = 1/(class_label_histogram+eps)
#         self.instance_weights = []
#
#         min_dist = self.df['distance'].min()
#         max_dist = self.df['distance'].max()
#         dist_threshold = ((max_dist-min_dist) / 2.) + min_dist
#
#         for i,row in self.df.iterrows():
#             if row['distance'] < dist_threshold: # filter noise
#                 self.df.drop(self.df.iloc[i])
#                 continue
#             label = row[self.attribute_name]
#             self.instance_weights.append(class_label_weights[np.where(unique_labels_sorted == label)])
#
#         self.instance_weights = np.array(self.instance_weights).squeeze()
#
#         self.img_path = img_path
#
#     def __len__(self):
#         return len(self.df)
#
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#
#         label = self.df.iloc[idx][self.attribute_name]
#
#         if not self.regression:
#             label = np.flatnonzero(
#                 np.array(self.class_names) == label)  # Flatnonzero is the sane version of np.where which does not return weird tuples
#             label = label.squeeze()
#
#         npid_vec =  self.df.iloc[idx]['features']
#
#         return (npid_vec, label)
#
#
# class Rolling_Window_Year_NPID_Feat_Dataset(Dataset):
#     '''
#     Generic Dataset to access building type information. Possible values are
#     'building_address_full',
#        'first_floor_elevation_ft', 'assessment_type', 'year_built',
#        'effective_year_built', 'roof_shape', 'roof_cover', 'wall_cladding',
#        'number_of_stories', 'building_address_full_cleaned'
#     '''
#
#     def __init__(self, attribute_name, csv_path, img_path, transform=None, regression=False, mask_buildings=False, softmask=False):
#         if (attribute_name != 'year_built' and attribute_name != 'effective_year_built:') or regression:
#             raise ValueError('Wrong attribute or training type for this dataset')
#
#         self.df = pd.read_pickle(csv_path)
#         self.transform = transform
#         self.regression = regression
#         self.attribute_name = attribute_name
#         self.mask_buildings = mask_buildings
#         self.softmask=softmask
#
#         #min_year = self.df[self.attribute_name].min()
#         #max_year = self.df[self.attribute_name].max()
#         #max_year += int((max_year - min_year) % 10)+2 # padd to full 10 year intervals
#         min_year = 1913
#         max_year = 2023 # Not all datasets have all years so this needs to be hard set
#
#         classes = skimage.util.view_as_windows(np.array(range(int(min_year),int(max_year))),10,step=10)
#         self.class_names = [(str(start) + '-' + str(end)) for start, end in zip(classes[:, 0], classes[:, -1])]
#         self.label_lookup = {}
#         for year in self.df[self.attribute_name].unique():
#             for i in range(len(classes)):
#                 if int(year) in classes[i]:
#                     self.label_lookup[int(year)] = i
#                     break
#         class_label_histogram = np.histogram(np.fromiter(self.label_lookup.values(), dtype=float),bins=len(classes))[0]
#         class_label_weights = 1/(class_label_histogram+eps)
#         self.instance_weights = []
#         for i,row in self.df.iterrows():
#             label = row[self.attribute_name]
#             self.instance_weights.append(class_label_weights[self.label_lookup[int(label)]])
#
#         self.instance_weights = np.array(self.instance_weights)
#
#         self.img_path = img_path
#
#     def __len__(self):
#         return len(self.df)
#
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#
#         label = self.df.iloc[idx][self.attribute_name]
#         label = self.label_lookup[int(label)] # Translate to coarse class
#
#         npid_vec =  self.df.iloc[idx]['features']
#
#         return (npid_vec, label)