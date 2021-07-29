#import pose_match
#import parse_openpose_json
import collections
import numpy as np
import numpy
import json
import logging
from . import prepocessing
from . import draw_humans
from . import pose_comparison
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("pose_match")
import warnings
warnings.filterwarnings("ignore")
logging.getLogger('matplotlib.font_manager').disabled = True
#json_data_path = 'data/json_data/'
#images_data_path = 'data/image_data/'
import cv2

'''
-------------------------------- SINGLE PERSON -------------------------------------------
Read openpose output and parse body-joint points into an 2D arr ay of 18 rows
Elke entry is een coordinatenkoppel(joint-point) in 3D , z-coordinaat wordt nul gekozen want we werken in 2D
'''

def feature_scaling(input):
    #logger.info("inn: %s" , str(input))
    # We accept the presence of (0,0) points in the input poses (undetected body-parts)
    # But we don't want them to influence our normalisation

    # Here it's assumed that (0,y) and (x,0) don't occur
    # Is a acceptable assumption because the chance is sooooo small
    #   that a feature is positioned just right on the x or y axis
  
    #input = np.array(input)
    xmax = max(input[:, 0])
    ymax = max(input[:, 1])

    xmin = np.min(input[np.nonzero(input[:,0])]) #np.nanmin(input[:, 0])
    ymin = np.min(input[np.nonzero(input[:,1])]) #np.nanmin(input[:, 1])

    sec_x = (input[:, 0]-xmin)/(xmax-xmin)
    sec_y = (input[:, 1]-ymin)/(ymax-ymin)

    output = np.vstack([sec_x, sec_y]).T
    output[output<0] = 0
    #logger.info("out: %s", str(output))
    return output

def parse_pts(arr):
        array = numpy.zeros((18,2))
        index = 0
        for i in range(0,len(arr),3):
            array[index][0] = arr[i]
            array[index][1] = arr[i+1]
            index +=1
        return array
class OpenPoseinitializer():
    
    
    def __init__(self,db_img_path='',user_img_path='',db_key_points=None,user_keypoints =None, save_folder = None):
        self.model_image = db_img_path
        self.user_img_path = user_img_path
        self.model_features = parse_pts(db_key_points)
        self.input_features = parse_pts(user_keypoints)
        self.save_folder = save_folder
        self.MatchResult = collections.namedtuple("MatchResult", ["match_bool", "error_score", "input_transformation"])
    
    
    def find_transformation(self,model_features,input_features):
        
        pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])  # horizontaal stacken
        unpad = lambda x: x[:, :-1]
        nan_indices = []

        Y = pad(self.model_features)
        X = pad(self.input_features)

        # Solve the least squares problem X * A = Y
        # to find our transformation matrix A and then we can display the input on the model = Y'
        A, res, rank, s = np.linalg.lstsq(X, Y)
        transform = lambda x: unpad(np.dot(pad(x), A))
        input_transform = transform(input_features)

        input_transform_list  = input_transform.tolist()
        for index in nan_indices:
            input_transform_list.insert(index, [0,0])
        input_transform = np.array(input_transform_list)


        A[np.abs(A) < 1e-10] = 0  # set really small values to zero

        return (input_transform, A)

    def single_person(self):

        # Filter the undetected features and mirror them in the other pose
        (input_features_copy, model_features_copy) = prepocessing.handle_undetected_points(self.input_features, self.model_features)

        model_features_copy = feature_scaling(model_features_copy)
        input_features_copy = feature_scaling(input_features_copy)

        #Split features in three parts
        (model_face, model_torso, model_legs) = prepocessing.split_in_face_legs_torso(model_features_copy)
        (input_face, input_torso, input_legs) = prepocessing.split_in_face_legs_torso(input_features_copy)

        # Zoek transformatie om input af te beelden op model
        # Returnt transformatie matrix + afbeelding/image van input op model
        (input_transformed_face, transformation_matrix_face) = self.find_transformation(model_face, input_face)
        (input_transformed_torso, transformation_matrix_torso) = self.find_transformation(model_torso, input_torso)
        (input_transformed_legs, transformation_matrix_legs) = self.find_transformation(model_legs, input_legs)

        # Wrapped the transformed input in one whole pose
        input_transformation = prepocessing.unsplit(input_transformed_face, input_transformed_torso, input_transformed_legs)

        # In case of no normalisation, return here (ex; plotting)
        # Without normalisation the thresholds don't say anything
        #   -> so comparison is useless
        # if(not normalise):
        #     result = MatchResult(None,
        #                         error_score=0,
        #                         input_transformation=input_transformation)
        #     return result

        max_euclidean_error_face = pose_comparison.max_euclidean_distance(model_face, input_transformed_face)
        max_euclidean_error_torso = pose_comparison.max_euclidean_distance(model_torso, input_transformed_torso)
        max_euclidean_error_legs = pose_comparison.max_euclidean_distance(model_legs, input_transformed_legs)

        max_euclidean_error_shoulders = pose_comparison.max_euclidean_distance_shoulders(model_torso, input_transformed_torso)

        ######### THE THRESHOLDS #######
        eucl_dis_tresh_torso = 0.11 #0.065  of 0.11 ??
        rotation_tresh_torso = 40
        eucl_dis_tresh_legs = 0.055
        rotation_tresh_legs = 40

        eucld_dis_shoulders_tresh = 0.063
        ################################

        result_torso = pose_comparison.decide_torso_shoulders_incl(max_euclidean_error_torso, transformation_matrix_torso,
                                                    eucl_dis_tresh_torso, rotation_tresh_torso,
                                                    max_euclidean_error_shoulders, eucld_dis_shoulders_tresh)

        result_legs = pose_comparison.decide_legs(max_euclidean_error_legs, transformation_matrix_legs,
                                                eucl_dis_tresh_legs, rotation_tresh_legs)

        #TODO: construct a solid score algorithm
        error_score = (max_euclidean_error_torso + max_euclidean_error_legs)/2.0
        
        result = self.MatchResult((result_torso and result_legs),
                            error_score=error_score,
                            input_transformation=input_transformation)
        return result,error_score

    def plot_single_person(self, input_title = "input",  model_title="model",
                        transformation_title="transformed input -incl. split()"):

        # Filter the undetected features and mirror them in the other pose
        (input_features_copy, model_features_copy) = prepocessing.handle_undetected_points(self.input_features,self.model_features)

        # plot vars
        model_image_name = self.model_image
        input_image_name = self.user_img_path
        #Load images
        # model_image = plt.imread(model_image_name)
        # input_image = plt.imread(input_image_name)

        # Split features in three parts
        (model_face, model_torso, model_legs) = prepocessing.split_in_face_legs_torso(model_features_copy)
        (input_face, input_torso, input_legs) = prepocessing.split_in_face_legs_torso(input_features_copy)

        # Zoek transformatie om input af te beelden op model
        # Returnt transformatie matrix + afbeelding/image van input op model
        (input_transformed_face, transformation_matrix_face) = self.find_transformation(model_face,input_face)
        (input_transformed_torso, transformation_matrix_torso) = self.find_transformation(model_torso,input_torso)
        (input_transformed_legs, transformation_matrix_legs) = self.find_transformation(model_legs,input_legs)


        whole_input_transform = prepocessing.unsplit(input_transformed_face, input_transformed_torso,
                                                    input_transformed_legs)

        

        model_image = plt.imread(model_image_name) #png.read_png_int(model_image_name) #plt.imread(model_image_name)
        input_image = plt.imread(input_image_name) #png.read_png_int(input_image_name) #plt.imread(input_image_name)
        
       
        
        model_image = draw_humans.draw_humans(model_image, self.model_features, True,user=False)  # plt.imread(model_image_name)
        input_image = draw_humans.draw_humans(input_image, self.input_features, True,user=True)  # plt.imread(input_image_name)

        input_trans_image = draw_humans.draw_humans(plt.imread(input_image_name), self.input_features,user=True)
        # input_trans_image = draw_humans.draw_square(plt.imread(model_image_name), self.model_features)
        input_trans_image = draw_humans.draw_humans(input_trans_image, whole_input_transform,
                                                    True,user=False)  # plt.imread(input_image_name) png.read_png_int(model_image_name)

        f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(14, 6))
        implot = ax1.imshow(model_image)
        plt.axis('off')
        #ax1.set_title(model_image_name + ' (model)')
        ax1.set_title(model_title)
        ax1.axis('off')

        ax2.set_title(input_image_name + ' (input)')
        ax2.set_title(input_title)
        ax2.axis('off')
        ax2.imshow(input_image)

        ax3.set_title(transformation_title)
        ax3.axis('off')
        ax3.imshow(input_trans_image)

        plot_name = model_image_name.split("/")[-1] + "_" + input_image_name.split("/")[-1]
        
        # cv2.imwrite(self.save_folder+plot_name,input_trans_image)
        plt.savefig(self.save_folder + plot_name , bbox_inches='tight')

        print("Ploting Done")
        # return error_score

        
    def call_model(self):
        match_result,error_score = self.single_person()
        logger.info("--Match or not: %s  score=%f ", str(match_result.match_bool), match_result.error_score)
        self.plot_single_person(input_title = "input",  model_title="model",
                        transformation_title="transformed input -incl. split()")
        return error_score

##################################################################################################################################


'''
Calculate match fo real (incl. normalizing)
'''
#TODO: edit return tuple !!

##################################################################################################################################


    #plt.show(block=False)

'''
Calculate match + plot the whole thing
'''
# Reload features bc model_features is a immutable type  -> niet meer nodig want er wordt een copy gemaalt in single_psoe()
# and is changed in single_pose in case of undetected bodyparts

# plot_single_person(model_features, input_features, model_image, input_image,save_folder)




