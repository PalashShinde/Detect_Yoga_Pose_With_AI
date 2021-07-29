import imghdr
import cv2
import os
import numpy as np
import torch
from tqdm import tqdm
import natsort
from scipy.spatial import distance
from scipy import spatial
import shutil
from flask_pymongo import PyMongo
from werkzeug.utils import secure_filename
from detector.apis import get_detector
from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.detector import DetectionLoader
from alphapose.utils.transforms import flip,get_func_heatmap_to_coord
from types import SimpleNamespace
from flask import render_template,request,redirect,url_for,abort,send_from_directory,g,Flask
from process import main_single_person

# FLASK_APP=app.py FLASK_ENV=development flask run --host=0.0.0.0  --port=5000

app = Flask(__name__)
Mongo_URL = "mongodb://127.0.0.1:27017/"
DB_Name = "user_db"

app.config['MONGO_URI'] = os.path.join(Mongo_URL,DB_Name)
mongo = PyMongo(app)

dir_path = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = dir_path +'/user_images/'
os.makedirs(UPLOAD_FOLDER,exist_ok=True)
db_img_path = dir_path+'/db_images/'

app.config['UPLOAD_PATH'] = UPLOAD_FOLDER
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.jpeg','.JPG']

global pose_model,args,cfg
args = SimpleNamespace(cfg=dir_path+'/configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml', checkpoint=dir_path+'/pretrained_models/fast_res50_256x192.pth' \
    , debug=False, detbatch=5, detector='yolo', detfile='', device='cpu', eval=False, flip=False, format=None, gpus=[-1], inputimg='', inputlist='', inputpath='', min_box_area=0, pose_flow=False, pose_track=False, posebatch=80, profile=False, qsize=1024, save_img=False, save_video=False, showbox=False, sp=False, tracking=False, video='', vis=False, vis_fast=False, webcam=-1)
cfg = update_config(args.cfg)
pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
## Initializing the Alphose model loading in RAM
print('Loading pose model')
pose_model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
pose_model.to(args.device)
pose_model.eval()
print('AlphaPose model loaded')

torch.multiprocessing.set_start_method('forkserver', force=True)
torch.multiprocessing.set_sharing_strategy('file_system')

def check_input(inputpath):
    if len(inputpath):
        if len(inputpath) and inputpath != '/':
            for _,_, files in os.walk(inputpath):
                im_names = files
            im_names = natsort.natsorted(im_names)
        return 'image', im_names
    else:
        raise NotImplementedError

def remove_folder_content(root_dir_path):
    for root, dirs, files in os.walk(root_dir_path):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))

def validate_image(stream):
    header = stream.read(512)
    stream.seek(0) 
    format = imghdr.what(None, header)
    if not format:
        return None
    return '.' + (format if format != 'jpeg' else 'jpg')

def remove_scores(input_list):
    for scores in input_list[2::3]:
        input_list.remove(scores)
    return input_list

def compare_two_arrays(user_key_points,db_key_points):
    user_array = np.array(user_key_points)
    db_array = np.array(db_key_points)
    cosineSimilarity = 1 - spatial.distance.cosine(db_array, user_array)
    cosine_result =  np.sqrt(2 * (1 - cosineSimilarity))
    eculine_dst = distance.euclidean(db_array,user_array)
    ling_np_dst = np.linalg.norm(db_array-user_array)
    print(f' From The image {"stand1_ideal"}"eculine_dst is =="{eculine_dst} ||| "ling_np_dst is =={ling_np_dst}" ||| "Distance is =="{cosine_result} ' )
    return cosine_result


@app.route('/check',methods=['GET'])
def check_endpoint():
    if request.method == 'GET':
        return "Hello There"

@app.route('/upload',methods=['POST','GET'])
def upload():
    # Get method redirects to frontpage
    if request.method == 'GET':
        return render_template('index_new.html')    
    elif request.method == 'POST':
        # Getting user form data 
        data = request.form.to_dict(flat=True)
        global html_username,eculine_dst,message
        username = request.form['username']
        html_username = username
        assan_name = request.form['assan_name']
        uploaded_file = request.files['image']

        # Convert user filename in std secure_filename format 
        filename = secure_filename(uploaded_file.filename)
        if filename != '':
            file_ext = os.path.splitext(filename)[1]
            if file_ext not in app.config['UPLOAD_EXTENSIONS'] or \
                file_ext != validate_image(uploaded_file.stream):
                return "Invalid image", 400
        
        if username is not None:
            # Checking if User already exsists in DB if new user updating the same in DB and saving the user img in dir 
            if not mongo.db.usertable.find_one({"username":username}):
                mongo.db.usertable.insert_one(data)
                input_user_dir = os.path.join(app.config['UPLOAD_PATH']+username,"input_img")
                os.makedirs(input_user_dir,exist_ok=True)
                uploaded_file.save(os.path.join(input_user_dir,filename))
            if os.path.exists(os.path.join(app.config['UPLOAD_PATH']+username,"input_img")):
                input_user_dir = os.path.join(app.config['UPLOAD_PATH']+username,"input_img")
                remove_folder_content(os.path.join(input_user_dir))
                uploaded_file.save(os.path.join(input_user_dir,filename))
            else:
                input_user_dir = os.path.join(app.config['UPLOAD_PATH']+username,"input_img")
                os.makedirs(input_user_dir,exist_ok=True)
                uploaded_file.save(os.path.join(input_user_dir,filename))
        else:
            return "Please enter different username"

        args.inputpath = input_user_dir
        
        cv2.imwrite(os.path.join(input_user_dir,filename), cv2.resize(cv2.imread(os.path.join(input_user_dir,filename)),(1024,1024)))
        mode, input_source = check_input(args.inputpath)

        # Initializing Alphapose DetectionLoader model for pose Keypoints predictions  
        det_loader = DetectionLoader(input_source, detector = get_detector(args),opt=args,cfg=cfg, batchSize=args.detbatch, mode=mode, queueSize=args.qsize)
        det_loader.start()
        data_len = det_loader.length
        im_names_desc = tqdm(range(data_len), dynamic_ncols=True)
        heatmap_to_coord = get_func_heatmap_to_coord(cfg)
        batchSize = args.posebatch
        keypoints_list = []
        try:
            for i in im_names_desc:
                with torch.no_grad():
                    (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes) = det_loader.read()
                    # Pose Estimation
                    inps = inps.to(args.device)
                    datalen = inps.size(0)
                    leftover = 0
                    if (datalen) % batchSize:
                        leftover = 1
                    num_batches = datalen // batchSize + leftover
                    hm = []
                    for j in range(num_batches):
                        inps_j = inps[j * batchSize:min((j + 1) * batchSize, datalen)]
                        hm_j = pose_model(inps_j)
                        hm.append(hm_j)
                    hm = torch.cat(hm)
                    hm = hm.cpu()
                    data = dict()
                    for i in range(hm.shape[0]):
                        bbox = cropped_boxes[i].tolist()
                        pose_coords, pose_scores = heatmap_to_coord(hm[i], bbox)
                        keypoints = np.concatenate((pose_coords, pose_scores), axis=1)
                        keypoints = keypoints.reshape(-1).tolist()
                        keypoints_list.append(keypoints)
        except Exception as e:
            print(repr(e))
            print('An error as above occurs when processing the images, please check it')
            pass

        ## OPen format keypointlts in tmp['pose_keypoints_2d']
        result={}
        tmp={'pose_keypoints_2d':[]}
        result['keypoints'] = keypoints_list[0].copy()
        result['keypoints'].append((result['keypoints'][15]+result['keypoints'][18])/2)
        result['keypoints'].append((result['keypoints'][16]+result['keypoints'][19])/2)
        result['keypoints'].append((result['keypoints'][17]+result['keypoints'][20])/2)
        indexarr=[0,51,18,24,30,15,21,27,36,42,48,33,39,45,6,3,12,9]
        for i in indexarr:
            tmp['pose_keypoints_2d'].append(result['keypoints'][i])
            tmp['pose_keypoints_2d'].append(result['keypoints'][i+1])
            tmp['pose_keypoints_2d'].append(result['keypoints'][i+2])
        
        # user_keypoints
        user_keypoints_list = keypoints_list[0]

        from_db_key = mongo.db.yogaimgs.find_one({"image_id":assan_name})

        # temp code
        if "mean" in from_db_key:
            mean_score = from_db_key["mean"]+0.05
        else:
            mean_score = 0.1

        db_key_points_list = from_db_key['keypoints']
    
        save_folder = os.path.join(app.config['UPLOAD_PATH']+username,"processed_img/")
        os.makedirs(save_folder,exist_ok=True)
        remove_folder_content(os.path.join(save_folder))

        db_img_list = os.listdir(db_img_path)
        for db_imgs in db_img_list:
            if assan_name in db_imgs:
                open_model = main_single_person.OpenPoseinitializer(db_img_path=os.path.join(db_img_path,db_imgs) ,user_img_path=os.path.join(input_user_dir,filename),db_key_points=db_key_points_list,user_keypoints =user_keypoints_list, save_folder = save_folder)
                error_score = open_model.call_model()

        # Calculating the similarity score for two keypoints array using CosineSimilarity
        eculine_dst = compare_two_arrays(user_keypoints_list,db_key_points_list)
        matching_percentage = round((1-eculine_dst)*100,2)
        message = f"Congrats your yoga pose is a exact match with Ideal assan {assan_name} Pose By Similarity factor of {matching_percentage}" if eculine_dst <= mean_score else f"Please try with correct Pose,your current similarity level is {matching_percentage}"
        return redirect(url_for('get_gallery'))

@app.route('/upload/<filename>')
def send_image(filename):
    dir_name = os.path.join(app.config['UPLOAD_PATH']+html_username,"processed_img")
    return send_from_directory(dir_name, filename)

@app.route('/gallery')
def get_gallery():
    image_names = os.listdir(os.path.join(app.config['UPLOAD_PATH']+html_username,"processed_img"))
    print("image_names",image_names,eculine_dst)
    return render_template("index.html", image_names=image_names,value=message)

if __name__ == '__main__':
    app.run(threaded=True, port=5000)