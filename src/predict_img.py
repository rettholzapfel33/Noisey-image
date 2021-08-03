import os
import cv2
import sys
import yaml
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

##################### model stuff #####################
# System libs
import os, csv, torch, numpy, scipy.io, PIL.Image, torchvision.transforms
# Our libs
from src.mit_semseg.models import ModelBuilder, SegmentationModule
from src.mit_semseg.utils import colorEncode

# class_list from names
def new_visualize_result(pred, img, name=None, index=None):
    class_list = {}
    with open(str(Path(__file__).parent.absolute()) + '/data/object150_info.csv') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            class_list[int(row[0])] = row[5].split(";")[0]

    switch_names = {y:x for x,y in class_list.items()}
    if name is not None:
        try:
            index = switch_names[name] - 1
        except:
            print("class not exist!")
            pass
    
    if index is not None:
        pred = pred.copy()
        pred[pred!=index] = -1
        # print(f'{names[i+1]}:')
        
    colors = scipy.io.loadmat(str(Path(__file__).parent.absolute()) + '/data/color150.mat')['colors']
    # colorize prediction
    pred_color = colorEncode(pred, colors).astype(numpy.uint8)

    
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    trans = transparent_overlays(img2, pred_color, alpha=0.6)

    return pred_color, trans


# pass in mode config(yaml file)
# return a dict for the file 
# return decoder and encoder weights path
def parse_model_config(path):
    with open(path) as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    
    #print(str(Path(__file__).parent.absolute()))
    data['DIR'] = str(Path(__file__).parent.absolute()) + '/' + data['DIR']
    #print(data['DIR'])

    encoder_path = None
    decoder_path = None

    for p in os.listdir(data['DIR']):
        if "encoder" in p.lower():
            encoder_path = "{}/{}".format(data['DIR'], p)
            continue
        if "decoder" in p.lower():
            decoder_path = "{}/{}".format(data['DIR'], p)
            continue

    if encoder_path==None or decoder_path==None:
        raise("model weights not found")
        
    return data, encoder_path, decoder_path

def visualize_result(img, pred, colors, index=None):
    # filter prediction class if requested
    if index is not None:
        pred = pred.copy()
        pred[pred != index] = -1
        print(f'{names[index+1]}:')

    # colorize prediction
    pred_color = colorEncode(pred, colors, mode='BGR').astype(numpy.uint8)

    # aggregate images and save
    im_vis = numpy.concatenate((img, pred_color), axis=1)
    #if show==True:
        #display(PIL.Image.fromarray(im_vis))
    #else:
    return pred_color, im_vis

def process_img(path=None, frame=None, cpu = 0):
    # Load and normalize one image as a singleton tensor batch
    pil_to_tensor = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], # These are RGB mean+std values
            std=[0.229, 0.224, 0.225])  # across a large photo dataset.
    ])
    # pil_image = PIL.Image.open('../ADE_val_00001519.jpg').convert('RGB')
    if path!=None:
        pil_image = PIL.Image.open(path).convert('RGB')
    else:
        pil_image = PIL.Image.fromarray(frame)

    img_original = numpy.array(pil_image)
    img_data = pil_to_tensor(pil_image)

    if torch.cuda.is_available() and cpu == 0:
        singleton_batch = {'img_data': img_data[None].cuda()}
    else:
        singleton_batch = {'img_data': img_data[None].cpu()}

    output_size = img_data.shape[1:]
    return (img_original, singleton_batch, output_size)

def predict_img(segmentation_module, singleton_batch, output_size):
    # Run the segmentation at the highest resolution.
    with torch.no_grad():
        scores = segmentation_module(singleton_batch, segSize=output_size)

    # Get the predicted scores for each pixel
    _, pred = torch.max(scores, dim=1)
    pred = pred.cpu()[0].numpy()
    return pred


def get_color_palette(pred, bar_height, names, colors, detectedNames):

    pred = np.int32(pred)
    pixs = pred.size

    top_left_y = 0
    bottom_right_y = 30
    uniques, counts = np.unique(pred, return_counts=True)

    # height = 0
    # for idx in np.argsort(counts)[::-1]:
    #     color_index = uniques[idx]
    #     name = names[color_index + 1]
    #     ratio = counts[idx] / pixs * 100
    #     if ratio > 0.1:
    #         height += 1 

    # Create a black image
    # bar_height = im_vis.shape[0]
    #img = np.zeros((height * 30,250,3), np.uint8)
    img = np.zeros((bar_height, 250, 3), np.uint8)

    for idx in np.argsort(counts)[::-1]:
        color_index = uniques[idx]
        name = names[color_index + 1]
        ratio = counts[idx] / pixs * 100
        if ratio > 0.1:
            print("{}  {}: {:.2f}% {}".format(color_index+1, name, ratio, colors[color_index]))
            img = cv2.rectangle(img, (0,top_left_y), (250,bottom_right_y), 
                       (int(colors[color_index][0]),int(colors[color_index][1]),int(colors[color_index][2])), -1)
            img = cv2.putText(img, "{}: {:.3f}%".format(name, ratio), (0,top_left_y+20), 5, 1, (255,255,255), 2, cv2.LINE_AA)
            top_left_y+=30
            bottom_right_y+=30
            detectedNames[name] = colors[color_index]
            
    return img


def transparent_overlays(image, annotation, alpha=0.5):
    img1 = image.copy()
    img2 = annotation.copy()

    # I want to put logo on top-left corner, So I create a ROI
    rows,cols,channels = img2.shape
    roi = img1[0:rows, 0:cols ]

    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Now black-out the area of logo in ROI
    # img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2,img2,mask = mask)

    # Put logo in ROI and modify the main image
    # dst = cv2.add(img1_bg, img2_fg)
    dst = cv2.addWeighted(image.copy(), 1-alpha, img2_fg, alpha, 0)
    img1[0:rows, 0:cols ] = dst
    return dst


def load_model_from_cfg(cfg):
    model_config, encoder_path, decoder_path = parse_model_config(cfg)
    print("encoder path: ", encoder_path)
    net_encoder = ModelBuilder.build_encoder(
        arch = model_config["MODEL"]['arch_encoder'],
        fc_dim = model_config['MODEL']['fc_dim'],
        weights = encoder_path)
    net_decoder = ModelBuilder.build_decoder(
        arch = model_config["MODEL"]['arch_decoder'],
        fc_dim = model_config['MODEL']['fc_dim'],
        num_class = model_config['DATASET']['num_class'],
        weights = decoder_path,
        use_softmax=True)
    
    crit = torch.nn.NLLLoss(ignore_index=-1)
    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
    return segmentation_module
    

def start_from_gui(img, save, progress, detectedNames, display = 1, alpha = 0.6):

    #cfg = str(Path(__file__).parent.absolute()) + "/config/ade20k-resnet50dilated-ppm_deepsup.yaml"
    cfg = str(Path(__file__).parent.absolute()) + "/config/ade20k-hrnetv2.yaml"

    # colors
    colors = scipy.io.loadmat(str(Path(__file__).parent.absolute()) + '/data/color150.mat')['colors']
    names = {}
    with open(str(Path(__file__).parent.absolute()) + '/data/object150_info.csv') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            names[int(row[0])] = row[5].split(";")[0]
    
    progress.emit(1)

    # Network Builders
    print("parsing {}".format(cfg))
    segmentation_module = load_model_from_cfg(cfg)
    
    segmentation_module.eval()

    try: 

        if torch.cuda.is_available():
            segmentation_module.cuda()
        
        progress.emit(2)

        # predict
        img_original, singleton_batch, output_size = process_img(frame = img)
        pred = predict_img(segmentation_module, singleton_batch, output_size)
    except:
        segmentation_module.cpu()
        progress.emit(2)
        print("Using cpu")

        # predict
        img_original, singleton_batch, output_size = process_img(frame = img, cpu = 1)
        pred = predict_img(segmentation_module, singleton_batch, output_size)


    progress.emit(3)
    # print(type(img_original))
    pred_color, org_pred_split = visualize_result(img_original, pred, colors)
    
    # color_palette
    color_palette = get_color_palette(pred, org_pred_split.shape[0], names, colors, detectedNames)
    
    # transparent pred on org
    dst = transparent_overlays(img_original, pred_color, alpha=alpha)
    
    # colored_pred + color_palette
    pred_color_palette = numpy.concatenate((color_palette, pred_color), axis=1)
    
    # transparent pred on org + color_palette
    pred_color_palette_dst = numpy.concatenate((color_palette, dst), axis=1)
    
    # org + colored_pred + color_palette
    pred_color_palette_all = numpy.concatenate((org_pred_split, color_palette), axis=1)

    progress.emit(4)
    
    cv2.imwrite("{}/color.png".format(save), cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR))
    cv2.imwrite("{}/org_pred_split.png".format(save), cv2.cvtColor(org_pred_split, cv2.COLOR_RGB2BGR))
    cv2.imwrite("{}/overlay.png".format(save), cv2.cvtColor(dst, cv2.COLOR_RGB2BGR))
    cv2.imwrite("{}/pred_color_palette.png".format(save), cv2.cvtColor(pred_color_palette, cv2.COLOR_RGB2BGR))
    cv2.imwrite("{}/pred_color_palette_dst.png".format(save), cv2.cvtColor(pred_color_palette_dst, cv2.COLOR_RGB2BGR))
    cv2.imwrite("{}/pred_color_palette_all.png".format(save), cv2.cvtColor(pred_color_palette_all, cv2.COLOR_RGB2BGR))
    cv2.imwrite("{}/color_palette.png".format(save), cv2.cvtColor(color_palette, cv2.COLOR_RGB2BGR))
    
    if (display==1):
        PIL.Image.fromarray(pred_color_palette_dst).show()
    else:
        print("results saved")

    progress.emit(5)

    img_rgb = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    pred_color_rgb = cv2.cvtColor(pred_color, cv2.COLOR_BGR2RGB)

    return img_rgb, pred_color_rgb, pred


if __name__ == '__main__':

    #print("Directory Path:", Path(__file__).parent.absolute())
    parser = argparse.ArgumentParser(description="PyTorch Semantic Segmentation Predict on image")
    parser.add_argument("-i", "--img", required=True, type=str, metavar='', help="an image path")
    parser.add_argument("-a", "--alpha", default=0.6, type=float, metavar='', help="transparent overlay level")
    parser.add_argument("-s", "--save", default="tmp_results/", type=str, metavar='', help="save prediction to")
    parser.add_argument("-d", "--display", default=1, type=int, metavar='', help="display real time prediction")
    
    parser.add_argument("--cfg", default=str(Path(__file__).parent.absolute()) + "/config/ade20k-resnet50dilated-ppm_deepsup.yaml", 
                        metavar="FILE", help="path to config file", type=str,)
    parser.add_argument("--gpu", default=0, type=int, metavar='', help="gpu id for evaluation")
    parser.add_argument("opts", help="Modify config options using the command-line", 
                        default=None, nargs=argparse.REMAINDER, metavar='')
    args = parser.parse_args()
        
    # print(args.save)
    
    # colors
    colors = scipy.io.loadmat(str(Path(__file__).parent.absolute()) + '/data/color150.mat')['colors']
    names = {}
    with open(str(Path(__file__).parent.absolute()) + '/data/object150_info.csv') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            names[int(row[0])] = row[5].split(";")[0]
    
    
    # Network Builders
    print("parsing {}".format(args.cfg))
    segmentation_module = load_model_from_cfg(args.cfg)
    
    '''
    model_config, encoder_path, decoder_path = parse_model_config(args.cfg)
    net_encoder = ModelBuilder.build_encoder(
        arch = model_config["MODEL"]['arch_encoder'],
        fc_dim = model_config['MODEL']['fc_dim'],
        weights = encoder_path)
    net_decoder = ModelBuilder.build_decoder(
        arch = model_config["MODEL"]['arch_decoder'],
        fc_dim = model_config['MODEL']['fc_dim'],
        num_class = model_config['DATASET']['num_class'],
        weights = decoder_path,
        use_softmax=True)
    
    crit = torch.nn.NLLLoss(ignore_index=-1)
    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
    '''
    segmentation_module.eval()
    
    if torch.cuda.is_available():
        segmentation_module.cuda()
    
    
    # predict
    img_original, singleton_batch, output_size = process_img(args.img)
    pred = predict_img(segmentation_module, singleton_batch, output_size)
    # print(type(img_original))
    pred_color, org_pred_split = visualize_result(img_original, pred)
    
    # color_palette
    color_palette = get_color_palette(pred, org_pred_split.shape[0])
    
    # transparent pred on org
    dst = transparent_overlays(img_original, pred_color, alpha=args.alpha)
    
    # colored_pred + color_palette
    pred_color_palette = numpy.concatenate((color_palette, pred_color), axis=1)
    
    # transparent pred on org + color_palette
    pred_color_palette_dst = numpy.concatenate((color_palette, dst), axis=1)
    
    # org + colored_pred + color_palette
    pred_color_palette_all = numpy.concatenate((org_pred_split, color_palette), axis=1)
    
    cv2.imwrite("{}/pred_color.png".format(args.save), cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR))
    cv2.imwrite("{}/org_pred_split.png".format(args.save), cv2.cvtColor(org_pred_split, cv2.COLOR_RGB2BGR))
    cv2.imwrite("{}/dst.png".format(args.save), cv2.cvtColor(dst, cv2.COLOR_RGB2BGR))
    cv2.imwrite("{}/pred_color_palette.png".format(args.save), cv2.cvtColor(pred_color_palette, cv2.COLOR_RGB2BGR))
    cv2.imwrite("{}/pred_color_palette_dst.png".format(args.save), cv2.cvtColor(pred_color_palette_dst, cv2.COLOR_RGB2BGR))
    cv2.imwrite("{}/pred_color_palette_all.png".format(args.save), cv2.cvtColor(pred_color_palette_all, cv2.COLOR_RGB2BGR))
    
    if (args.display)==1:
        PIL.Image.fromarray(pred_color_palette_dst).show()
    else:
        print("results saved")
