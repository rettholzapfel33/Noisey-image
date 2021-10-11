import os
import urllib
from bs4 import BeautifulSoup
import requests
from tqdm import tqdm

def downloadMITWeight(filename:str):
    HOST_URL = "http://sceneparsing.csail.mit.edu/model/pytorch/"
    new_url = "%s/%s"%(HOST_URL, filename)
    
    # get contents of directory:
    try:
        _content = requests.get(new_url).text
        p_content = BeautifulSoup(_content, 'html.parser')
        weight_files = [node.get('href') for node in p_content.find_all('a') if '.pt' in node.get('href')]
    except Exception as e:
        print("Error in parsing web directory. Make sure %s is available!"%(new_url))
        print(e)

    try:
        for _file in weight_files:
            file_url = "%s/%s"%(new_url, _file)
            _folder = './src/ckpt/%s'%(filename)
            if not os.path.exists(_folder): os.mkdir(_folder)
            response = requests.get(file_url)
            with open(os.path.join(_folder, _file), "wb") as handle:
                for data in tqdm(response.iter_content()):
                    handle.write(data)
    except Exception as e:
        print("Failed in grabbing needed files: %s"%(str(weight_files)))
        print(e)

def downloadYOLOv3Weights(save_path, host='https://pjreddie.com/media/files/yolov3.weights'):
    response = requests.get(host)
    with open(save_path, "wb") as handle:
        for data in tqdm(response.iter_content()):
            handle.write(data)

def checkWeightsExists(path_dict:dict):
    # path_dict: key=model_name; val=model_type
    print("Checking weights...")
    for path in path_dict.items():
        if path[0] == 'mit_semseg':
            _path_base = os.path.join('./src/ckpt/', path[1])
            print(_path_base)
            if not os.path.exists(_path_base):
                print("MIT Segmentation weights (%s) not found. Attempting to download..."%(path[1]))
                downloadMITWeight(path[1])
        elif path[0] == 'yolov3':
            _path_base = os.path.join('./src/obj_detector/weights', path[1])
            print(_path_base)
            if not os.path.exists(_path_base):
                print("YOLOv3 COCO weights not found. Attempting to download...")
                downloadYOLOv3Weights(_path_base)


#if __name__ == '__main__':
#    downloadMITWeight("ade20k-resnet18dilated-c1_deepsup")