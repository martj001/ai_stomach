import numpy as np
from matplotlib import pyplot as plt
from skimage.morphology import dilation, erosion
from skimage.morphology import label, remove_small_holes, remove_small_objects
from skimage.color import gray2rgb, grey2rgb
import SimpleITK as sitk
from radiomics import featureextractor

# for lesion_long calculation
from skimage.morphology import skeletonize
from skan import Skeleton, summarize
from collections import defaultdict

from algorithm.config import *
from algorithm.io import load_ct_data
from algorithm.feature import *


class CT:
    def __init__(self, ct_id: str):
        self.ct_id = ct_id

        self.image, self.scale_xy = load_ct_data(ct_id)
        self.label, label_scale_xy = load_ct_data(ct_id + '-label')
        
        # check img and label dimension
        if self.image.shape != self.label.shape:
            error_data = {
                'pid': self.ct_id,
                'image_dim': str(self.image.shape),
                'label_dim': str(self.label.shape)
            }
            raise ValueError({
                'message': ERROR_DIM_MISMATCH,
                'data': error_data
            })
        
        # check img and label scale
        assert(self.scale_xy == label_scale_xy)
        
        # build lesion object
        self.lesion_id_list = self.get_unique_lesion_id() 
        
        # [TEMP OVERRIDE] lesion id list limit to [1,2]
        if (1 in self.lesion_id_list) and (2 in self.lesion_id_list):
            self.lesion_id_list = [1,2]
            
        self.lesion_list = self.get_lesion_data()

        # feature related
        self.ct_features_dict = {}

    def get_unique_lesion_id(self) -> list:
        unique_label = np.unique(self.label.flatten())
        unique_label = list(np.delete(unique_label, 0))
        label_count = len(unique_label)
        if label_count != 4:
            print('WARNING - only ' + str(label_count) + ' lesion detected for CT: ' + self.ct_id)
            # raise ValueError('Lesion label Count is not 4')

        return unique_label

    def get_lesion_data(self) -> list:
        lesion_list = []
        for lesion_id in self.lesion_id_list:
            lesion = Lesion(self, lesion_id)
            lesion_list.append(lesion)
        return lesion_list

    def extract_features(self, feature_type: str) -> dict:
        assert(feature_type in ['radiomics', 'long'])
        ct_feature_dict = {}
        for i, lesion in enumerate(self.lesion_list):
            if feature_type == 'radiomics':
                lesion_feature_dict = lesion.extract_features_radiomics()
            if feature_type == 'long':
                lesion_feature_dict = lesion.extract_features_long()
            
            lesion_id = 'l' + str(i + 1)
            lesion_feature_dict = add_prefix_to_dict(lesion_feature_dict, lesion_id)
            ct_feature_dict = {**ct_feature_dict, **lesion_feature_dict}

        self.ct_features_dict = ct_feature_dict
        return ct_feature_dict


class Lesion:
    def __init__(self, ct: CT, lesion_id: int):
        self.ct = ct
        self.ct_id = ct.ct_id
        self.lesion_id = lesion_id
        self.lesion_full_id = self.ct_id + '-' + str(lesion_id)
        self.lesion_layer = None

        # lesion image
        self.global_image = None
        self.global_lesion_image = None
        self.global_lesion_mask = None
        self.global_background = None

        # roi image
        self.roi_coordinate = {
            'x_min': None,
            'x_max': None,
            'y_min': None,
            'y_max': None,
        }
        self.roi_image = None
        self.roi_lesion_image = None
        self.roi_lesion_mask = None
        self.roi_background = None
        self.roi_boundary_image = None
        self.roi_boundary_mask = None

        # lesion pixels
        self.lesion_pixels = None
        self.boundary_pixels = None

        # lesion properties
        self.lesion_size = None

        # apply pre-processing
        self.pre_processing()

        # feature extraction
        self.lesion_feature_dict = {}

    def pre_processing(self):
        lesion_coordinates = np.where(self.ct.label == self.lesion_id)
        lesion_layers = np.unique(lesion_coordinates[0])
        if len(lesion_layers) != 1:
            # print('WARNING: same label occur in more than one layer: ' + self.lesion_full_id)
            error_data = {
                'lesion_id': self.lesion_full_id,
                'layers': lesion_layers
            }
            raise ValueError({
                'message': ERROR_LABEL_IN_TWO_LAYER,
                'data': error_data
            })
        self.lesion_layer = lesion_layers[0]

        # global image
        self.global_image = self.ct.image[self.lesion_layer]
        
        # global mask + remove dots
        global_lesion_mask = self.ct.label[self.lesion_layer] / self.lesion_id
        mask_labelled = label(global_lesion_mask)
        mask = remove_small_objects(mask_labelled, min_size=5)
        self.global_lesion_mask = (mask != 0).astype(int)

        # lesion / background image
        self.global_lesion_image = \
            (self.global_lesion_mask * self.global_image) + (self.global_lesion_mask - 1) * 1024
        self.global_background = \
            np.abs(self.global_lesion_mask - 1) * self.global_image - self.global_lesion_mask * 1024

        # lesion roi
        self.calculate_roi_coordinate()
        self.roi_image = crop_image(self.global_image, self.roi_coordinate)
        self.roi_lesion_image = crop_image(self.global_lesion_image, self.roi_coordinate)
        
        # roi mask + remove holes
        self.roi_lesion_mask = crop_image(self.global_lesion_mask, self.roi_coordinate)
        mask = remove_small_holes(self.roi_lesion_mask)
        self.roi_lesion_mask = mask.astype(int)

        # lesion properties
        self.lesion_size = len(lesion_coordinates[0])

    def calculate_roi_coordinate(self):
        x_range = np.where(self.global_lesion_mask != 0)[0]
        y_range = np.where(self.global_lesion_mask != 0)[1]

        self.roi_coordinate['x_min'] = np.min(x_range) - ROI_EXPANSION
        self.roi_coordinate['x_max'] = np.max(x_range) + ROI_EXPANSION
        self.roi_coordinate['y_min'] = np.min(y_range) - ROI_EXPANSION
        self.roi_coordinate['y_max'] = np.max(y_range) + ROI_EXPANSION

    def calculate_lesion_long(self, enable_skeleton_path=False):
        # [ToDo] Thickness: https://stackoverflow.com/questions/53808511/finding-distance-between-skeleton-and-boundary-using-opencv-python
        self.roi_skeleton = skeletonize(self.roi_lesion_mask)
        scale_xy = self.ct.scale_xy

        skan_obj = Skeleton(self.roi_skeleton, spacing=1)
        df_skeleton_branch = summarize(skan_obj)

        if len(df_skeleton_branch) == 1:
            # Case: skeleton only has one possible path
            row = df_skeleton_branch.loc[0]
            #assert(row['branch-type'] == 0)

            self.lesion_long = row['branch-distance']*scale_xy
            max_path = [ row['node-id-src'], row['node-id-dst'] ]
        else:
            # Case: skeleton has multiple possible path
            node1_list = df_skeleton_branch['node-id-src']
            node2_list = df_skeleton_branch['node-id-dst']
            distance_list = df_skeleton_branch['branch-distance']

            dict_dist = construct_dict_distance(node1_list, node2_list, distance_list)
            G = construct_G_dict(node1_list, node2_list)

            all_nodes = list(G.keys())
            list_max_distance = []
            list_path_max_distance = []
            for node in all_nodes:
                all_paths = DFS(G, node) # longest path started from current node
                path_max_dist, max_dist = get_max_distance(all_paths, dict_dist)

                list_max_distance.append(max_dist)
                list_path_max_distance.append(path_max_dist)
                
            max_distance = max(list_max_distance)
            self.lesion_long = max_distance*scale_xy
            
            idx_path_max_distance = np.where(max_distance == np.array(list_max_distance))[0][0]
            max_path = list_path_max_distance[idx_path_max_distance]            

        if enable_skeleton_path:
            self.max_path_coordinate = get_path_coordinate(max_path, df_skeleton_branch, skan_obj)
            self.get_roi_skeleton_with_path()

    def get_roi_skeleton_with_path(self):
        path_coordinate = self.max_path_coordinate.astype(int)

        scatter_x = path_coordinate[:,1]
        scatter_y = path_coordinate[:,0]

        plt.figure(figsize=(10, 10))
        mask = self.roi_lesion_mask.astype(int)/2
        for point in path_coordinate:
            mask[point[0], point[1]] = 1

        self.roi_skeleton_with_path = mask

    def display_tool_1x2(self, image_dict: dict, cmap='gray'):
        assert (len(image_dict) == 2)
        fig = plt.figure(figsize=(12, 6))
        subtitle = 'Lesion: ' + self.lesion_full_id
        fig.suptitle(subtitle, fontsize=20)

        for i, (title, image) in enumerate(image_dict.items()):
            ax = fig.add_subplot(1, 2, i + 1)
            ax.set_title(title)
            ax.imshow(image, cmap=cmap)

        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
        
    def display_tool_1x3(self, image_dict: dict, cmap='gray'):
        assert (len(image_dict) == 3)
        fig = plt.figure(figsize=(18, 6), facecolor='white')
        subtitle = 'Lesion: ' + self.lesion_full_id + ' | CT long: ' + str((self.lesion_long/10).round(3)) 
        fig.suptitle(subtitle, fontsize=20)

        for i, (title, image) in enumerate(image_dict.items()):
            ax = fig.add_subplot(1, 3, i + 1)
            ax.set_title(title)
            ax.imshow(image, cmap=cmap)

        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
        
    def display_global(self, cmap='gray'):
        img = np.array(self.global_image.copy().transpose())
        mask = self.global_lesion_mask.transpose()
        
        img[img>=250] = 250
        img[img<=-250] = -250

        img = (img + 250)*(255/500)
        img = img.astype(np.uint8)

        img_rgb = gray2rgb(img)
        mask_loc = np.where(mask)
        img_rgb[mask_loc] = list(map(lambda x: [255, x[1], x[2]], img_rgb[mask_loc]))
        
        image_dict = {
            'Image': img,
            'Lesion': img_rgb,
        }
        self.display_tool_1x2(image_dict, cmap=cmap)
        
    def display_roi_skeleton(self, title_str: str, cmap='gray'):
        img = np.array(self.roi_image.copy().transpose())
        mask = self.roi_lesion_mask.transpose()

        img[img>=250] = 250
        img[img<=-250] = -250

        img = (img + 250)*(255/500)
        img = img.astype(np.uint8)

        img_rgb = gray2rgb(img)
        mask_loc = np.where(mask)
        img_rgb[mask_loc] = list(map(lambda x: [255, x[1], x[2]], img_rgb[mask_loc]))

        fig = plt.figure(figsize=(18, 6), facecolor='white')
        subtitle = 'Lesion: ' + self.lesion_full_id + \
            ' | CT long: ' + str((self.lesion_long/10).round(3)) + title_str
        fig.suptitle(subtitle, fontsize=20)

        image_dict = {
            'Image': img,
            'Lesion': img_rgb,
            'Lesion long path': self.roi_skeleton_with_path.transpose()
        }
        self.display_tool_1x3(image_dict, cmap=cmap)
    
    def extract_features_radiomics(self):
        # extract_features_radiomics
        n = -1
        image = self.roi_lesion_image.copy()

        image = sitk.GetImageFromArray(np.array(image))
        mask = self.roi_lesion_mask.astype(int)
        if n < 0:
            mask = multiple_erosion(mask, -n)
        elif n > 0:
            mask = multiple_dilation(mask, n)
        mask = sitk.GetImageFromArray(np.array(mask))

        extractor = featureextractor.RadiomicsFeatureExtractor()

        extractor.enableAllFeatures() 
        #extractor.enableFeatureClassByName('firstorder')

        featureVector = extractor.execute(image, mask)

        radiomics_key_list = list(featureVector.keys())
        radiomics_key_list = [x for x in radiomics_key_list if x.startswith(('original'))]
        lesion_feature_dict = dict((key.replace('original_', ''), featureVector[key]) for key in radiomics_key_list)
        self.lesion_feature_dict = lesion_feature_dict
        
        return lesion_feature_dict
    
    def extract_features_long(self):
        # extract_features_lesion_long
        lesion_feature_dict = {}
        
        if self.lesion_id < 3:
            self.calculate_lesion_long()
            lesion_feature_dict['lesion_long'] = self.lesion_long
        
        self.lesion_feature_dict = lesion_feature_dict
        
        return lesion_feature_dict


def crop_image(image, coordinate):
    cropped_image = image[coordinate['x_min']:coordinate['x_max'], coordinate['y_min']:coordinate['y_max']].copy()
    return cropped_image


def multiple_dilation(image, n):
    dilated_image = image.copy()
    for i in range(n):
        dilated_image = dilation(dilated_image)
    return dilated_image


def multiple_erosion(image, n):
    erosion_image = image.copy()
    for i in range(n):
        erosion_image = erosion(erosion_image)
    return erosion_image


def construct_G_dict(node1_list, node2_list):
    edges = list(zip(node1_list, node2_list))
    edges = np.array(edges).astype(str)
    edges = edges.tolist()

    G = defaultdict(list)
    for (s,t) in edges:
        G[s].append(t)
        G[t].append(s)
        
    return G


def DFS(G,v,seen=None,path=None):
    # https://stackoverflow.com/questions/29320556/finding-longest-path-in-a-graph/29321323
    if seen is None: seen = []
    if path is None: path = [v]

    seen.append(v)

    paths = []
    for t in G[v]:
        if t not in seen:
            t_path = path + [t]
            paths.append(tuple(t_path))
            paths.extend(DFS(G, t, seen[:], t_path))
    return paths


def construct_dict_distance(node1_list, node2_list, distance_list):
    dict1 = dict(zip(list(zip(node1_list, node2_list)), distance_list))
    dict2 = dict(zip(list(zip(node2_list, node1_list)), distance_list))

    dict_dist = {**dict1, **dict2}
    
    return dict_dist


def get_max_distance(all_paths, dict_dist):
    list_path_distance = []
    for path in all_paths:
        total_distance = 0
        for i in range(len(path)-1):
            current_node = (int(path[i]), int(path[i+1]))
            total_distance += dict_dist[current_node]

        list_path_distance.append(total_distance)

    max_distance = max(list_path_distance)
    idx_max_distance = np.where(max_distance == np.array(list_path_distance))[0][0]
    max_path = all_paths[idx_max_distance]
    
    return max_path, max_distance


def get_path_coordinate(path, df_skeleton_branch, skan_obj):
    node_path_max = []
    for i in range(len(path) - 1):
        node1 = int(path[i])
        node2 = int(path[i+1])

        vec1 = (df_skeleton_branch['node-id-src'] == node1) & (df_skeleton_branch['node-id-dst'] == node2)
        vec2 = (df_skeleton_branch['node-id-src'] == node2) & (df_skeleton_branch['node-id-dst'] == node1)

        if sum(vec1):
            vec = vec1
        else:
            vec = vec2

        idx_path = np.where(vec)[0][0]
        node_path_max = node_path_max + list(skan_obj.path(idx_path))

    path_coordinate = skan_obj.coordinates[node_path_max]
    
    return path_coordinate
