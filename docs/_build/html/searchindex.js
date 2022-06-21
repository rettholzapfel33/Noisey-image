Search.setIndex({docnames:["index","source/main_tutorial_guide","source/modules","source/src","source/src.mit_semseg","source/src.mit_semseg.config","source/src.mit_semseg.lib","source/src.mit_semseg.lib.nn","source/src.mit_semseg.lib.nn.modules","source/src.mit_semseg.lib.nn.modules.tests","source/src.mit_semseg.lib.nn.parallel","source/src.mit_semseg.lib.utils","source/src.mit_semseg.lib.utils.data","source/src.mit_semseg.models","source/src.obj_detector","source/src.obj_detector.utils","source/tutorials/abstract_model"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":3,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":2,"sphinx.domains.rst":2,"sphinx.domains.std":1,sphinx:56},filenames:["index.rst","source/main_tutorial_guide.rst","source/modules.rst","source/src.rst","source/src.mit_semseg.rst","source/src.mit_semseg.config.rst","source/src.mit_semseg.lib.rst","source/src.mit_semseg.lib.nn.rst","source/src.mit_semseg.lib.nn.modules.rst","source/src.mit_semseg.lib.nn.modules.tests.rst","source/src.mit_semseg.lib.nn.parallel.rst","source/src.mit_semseg.lib.utils.rst","source/src.mit_semseg.lib.utils.data.rst","source/src.mit_semseg.models.rst","source/src.obj_detector.rst","source/src.obj_detector.utils.rst","source/tutorials/abstract_model.rst"],objects:{"src.mit_semseg":{dataset:[4,0,0,"-"],lib:[6,0,0,"-"],models:[13,0,0,"-"],utils:[4,0,0,"-"]},"src.mit_semseg.dataset":{BaseDataset:[4,1,1,""],TestDataset:[4,1,1,""],TrainDataset:[4,1,1,""],ValDataset:[4,1,1,""],imresize:[4,3,1,""]},"src.mit_semseg.dataset.BaseDataset":{img_transform:[4,2,1,""],parse_input_list:[4,2,1,""],round2nearest_multiple:[4,2,1,""],segm_transform:[4,2,1,""]},"src.mit_semseg.lib":{nn:[7,0,0,"-"],utils:[11,0,0,"-"]},"src.mit_semseg.lib.nn":{modules:[8,0,0,"-"],parallel:[10,0,0,"-"]},"src.mit_semseg.lib.nn.modules":{batchnorm:[8,0,0,"-"],comm:[8,0,0,"-"],replicate:[8,0,0,"-"],tests:[9,0,0,"-"],unittest:[8,0,0,"-"]},"src.mit_semseg.lib.nn.modules.batchnorm":{SynchronizedBatchNorm1d:[8,1,1,""],SynchronizedBatchNorm2d:[8,1,1,""],SynchronizedBatchNorm3d:[8,1,1,""]},"src.mit_semseg.lib.nn.modules.batchnorm.SynchronizedBatchNorm1d":{affine:[8,4,1,""],eps:[8,4,1,""],momentum:[8,4,1,""],num_features:[8,4,1,""],track_running_stats:[8,4,1,""]},"src.mit_semseg.lib.nn.modules.batchnorm.SynchronizedBatchNorm2d":{affine:[8,4,1,""],eps:[8,4,1,""],momentum:[8,4,1,""],num_features:[8,4,1,""],track_running_stats:[8,4,1,""]},"src.mit_semseg.lib.nn.modules.batchnorm.SynchronizedBatchNorm3d":{affine:[8,4,1,""],eps:[8,4,1,""],momentum:[8,4,1,""],num_features:[8,4,1,""],track_running_stats:[8,4,1,""]},"src.mit_semseg.lib.nn.modules.comm":{FutureResult:[8,1,1,""],SlavePipe:[8,1,1,""],SyncMaster:[8,1,1,""]},"src.mit_semseg.lib.nn.modules.comm.FutureResult":{get:[8,2,1,""],put:[8,2,1,""]},"src.mit_semseg.lib.nn.modules.comm.SlavePipe":{run_slave:[8,2,1,""]},"src.mit_semseg.lib.nn.modules.comm.SyncMaster":{nr_slaves:[8,2,1,""],register_slave:[8,2,1,""],run_master:[8,2,1,""]},"src.mit_semseg.lib.nn.modules.replicate":{CallbackContext:[8,1,1,""],DataParallelWithCallback:[8,1,1,""],execute_replication_callbacks:[8,3,1,""],patch_replication_callback:[8,3,1,""]},"src.mit_semseg.lib.nn.modules.replicate.DataParallelWithCallback":{replicate:[8,2,1,""],training:[8,4,1,""]},"src.mit_semseg.lib.nn.modules.unittest":{TorchTestCase:[8,1,1,""],as_numpy:[8,3,1,""]},"src.mit_semseg.lib.nn.modules.unittest.TorchTestCase":{assertTensorClose:[8,2,1,""]},"src.mit_semseg.lib.nn.parallel":{data_parallel:[10,0,0,"-"]},"src.mit_semseg.lib.nn.parallel.data_parallel":{UserScatteredDataParallel:[10,1,1,""],async_copy_to:[10,3,1,""],user_scattered_collate:[10,3,1,""]},"src.mit_semseg.lib.nn.parallel.data_parallel.UserScatteredDataParallel":{scatter:[10,2,1,""],training:[10,4,1,""]},"src.mit_semseg.lib.utils":{data:[12,0,0,"-"],th:[11,0,0,"-"]},"src.mit_semseg.lib.utils.data":{dataloader:[12,0,0,"-"],dataset:[12,0,0,"-"],distributed:[12,0,0,"-"],sampler:[12,0,0,"-"]},"src.mit_semseg.lib.utils.data.dataloader":{DataLoader:[12,1,1,""],DataLoaderIter:[12,1,1,""],ExceptionWrapper:[12,1,1,""],default_collate:[12,3,1,""],pin_memory_batch:[12,3,1,""]},"src.mit_semseg.lib.utils.data.dataloader.DataLoaderIter":{next:[12,2,1,""]},"src.mit_semseg.lib.utils.data.dataset":{ConcatDataset:[12,1,1,""],Dataset:[12,1,1,""],Subset:[12,1,1,""],TensorDataset:[12,1,1,""],random_split:[12,3,1,""],randperm:[12,3,1,""]},"src.mit_semseg.lib.utils.data.dataset.ConcatDataset":{cummulative_sizes:[12,2,1,""],cumsum:[12,2,1,""]},"src.mit_semseg.lib.utils.data.distributed":{DistributedSampler:[12,1,1,""]},"src.mit_semseg.lib.utils.data.distributed.DistributedSampler":{set_epoch:[12,2,1,""]},"src.mit_semseg.lib.utils.data.sampler":{BatchSampler:[12,1,1,""],RandomSampler:[12,1,1,""],Sampler:[12,1,1,""],SequentialSampler:[12,1,1,""],SubsetRandomSampler:[12,1,1,""],WeightedRandomSampler:[12,1,1,""]},"src.mit_semseg.lib.utils.th":{as_numpy:[11,3,1,""],as_variable:[11,3,1,""],mark_volatile:[11,3,1,""]},"src.mit_semseg.models":{hrnet:[13,0,0,"-"],mobilenet:[13,0,0,"-"],models:[13,0,0,"-"],resnet:[13,0,0,"-"],resnext:[13,0,0,"-"],utils:[13,0,0,"-"]},"src.mit_semseg.models.hrnet":{hrnetv2:[13,3,1,""]},"src.mit_semseg.models.mobilenet":{mobilenetv2:[13,3,1,""]},"src.mit_semseg.models.models":{C1:[13,1,1,""],C1DeepSup:[13,1,1,""],MobileNetV2Dilated:[13,1,1,""],ModelBuilder:[13,1,1,""],PPM:[13,1,1,""],PPMDeepsup:[13,1,1,""],Resnet:[13,1,1,""],ResnetDilated:[13,1,1,""],SegmentationModule:[13,1,1,""],SegmentationModuleBase:[13,1,1,""],UPerNet:[13,1,1,""],conv3x3_bn_relu:[13,3,1,""]},"src.mit_semseg.models.models.C1":{forward:[13,2,1,""],training:[13,4,1,""]},"src.mit_semseg.models.models.C1DeepSup":{forward:[13,2,1,""],training:[13,4,1,""]},"src.mit_semseg.models.models.MobileNetV2Dilated":{forward:[13,2,1,""],training:[13,4,1,""]},"src.mit_semseg.models.models.ModelBuilder":{build_decoder:[13,2,1,""],build_encoder:[13,2,1,""],weights_init:[13,2,1,""]},"src.mit_semseg.models.models.PPM":{forward:[13,2,1,""],training:[13,4,1,""]},"src.mit_semseg.models.models.PPMDeepsup":{forward:[13,2,1,""],training:[13,4,1,""]},"src.mit_semseg.models.models.Resnet":{forward:[13,2,1,""],training:[13,4,1,""]},"src.mit_semseg.models.models.ResnetDilated":{forward:[13,2,1,""],training:[13,4,1,""]},"src.mit_semseg.models.models.SegmentationModule":{forward:[13,2,1,""],training:[13,4,1,""]},"src.mit_semseg.models.models.SegmentationModuleBase":{pixel_acc:[13,2,1,""],training:[13,4,1,""]},"src.mit_semseg.models.models.UPerNet":{forward:[13,2,1,""],training:[13,4,1,""]},"src.mit_semseg.models.resnet":{ResNet:[13,1,1,""],resnet101:[13,3,1,""],resnet18:[13,3,1,""],resnet50:[13,3,1,""]},"src.mit_semseg.models.resnet.ResNet":{forward:[13,2,1,""],training:[13,4,1,""]},"src.mit_semseg.models.resnext":{ResNeXt:[13,1,1,""],resnext101:[13,3,1,""]},"src.mit_semseg.models.resnext.ResNeXt":{forward:[13,2,1,""],training:[13,4,1,""]},"src.mit_semseg.models.utils":{load_url:[13,3,1,""]},"src.mit_semseg.utils":{AverageMeter:[4,1,1,""],NotSupportedCliException:[4,5,1,""],accuracy:[4,3,1,""],colorEncode:[4,3,1,""],find_recursive:[4,3,1,""],intersectionAndUnion:[4,3,1,""],parse_devices:[4,3,1,""],process_range:[4,3,1,""],setup_logger:[4,3,1,""],unique:[4,3,1,""]},"src.mit_semseg.utils.AverageMeter":{add:[4,2,1,""],average:[4,2,1,""],initialize:[4,2,1,""],update:[4,2,1,""],value:[4,2,1,""]},"src.models":{Model:[3,1,1,""],Segmentation:[3,1,1,""],YOLOv3:[3,1,1,""]},"src.models.Model":{deinitialize:[3,2,1,""],draw:[3,2,1,""],initialize:[3,2,1,""],outputFormat:[3,2,1,""],run:[3,2,1,""]},"src.models.Segmentation":{deinitialize:[3,2,1,""],draw:[3,2,1,""],initialize:[3,2,1,""],outputFormat:[3,2,1,""],run:[3,2,1,""]},"src.models.YOLOv3":{deinitialize:[3,2,1,""],draw:[3,2,1,""],initialize:[3,2,1,""],outputFormat:[3,2,1,""],run:[3,2,1,""]},"src.obj_detector":{detect:[14,0,0,"-"],models:[14,0,0,"-"],utils:[15,0,0,"-"]},"src.obj_detector.detect":{Colors:[14,1,1,""],detect:[14,3,1,""],detect_directory:[14,3,1,""],detect_image:[14,3,1,""],run:[14,3,1,""]},"src.obj_detector.detect.Colors":{hex2rgb:[14,2,1,""]},"src.obj_detector.models":{Darknet:[14,1,1,""],Upsample:[14,1,1,""],YOLOLayer:[14,1,1,""],create_modules:[14,3,1,""],load_model:[14,3,1,""]},"src.obj_detector.models.Darknet":{forward:[14,2,1,""],load_darknet_weights:[14,2,1,""],save_darknet_weights:[14,2,1,""],training:[14,4,1,""]},"src.obj_detector.models.Upsample":{forward:[14,2,1,""],training:[14,4,1,""]},"src.obj_detector.models.YOLOLayer":{forward:[14,2,1,""],training:[14,4,1,""]},"src.obj_detector.utils":{datasets:[15,0,0,"-"],logger:[15,0,0,"-"],loss:[15,0,0,"-"],parse_config:[15,0,0,"-"],transforms:[15,0,0,"-"],utils:[15,0,0,"-"]},"src.obj_detector.utils.datasets":{ImageFolder:[15,1,1,""],ListDataset:[15,1,1,""],pad_to_square:[15,3,1,""],resize:[15,3,1,""]},"src.obj_detector.utils.datasets.ListDataset":{collate_fn:[15,2,1,""]},"src.obj_detector.utils.logger":{Logger:[15,1,1,""]},"src.obj_detector.utils.logger.Logger":{list_of_scalars_summary:[15,2,1,""],scalar_summary:[15,2,1,""]},"src.obj_detector.utils.loss":{BCEBlurWithLogitsLoss:[15,1,1,""],FocalLoss:[15,1,1,""],QFocalLoss:[15,1,1,""],bbox_iou:[15,3,1,""],build_targets:[15,3,1,""],compute_loss:[15,3,1,""],smooth_BCE:[15,3,1,""]},"src.obj_detector.utils.loss.BCEBlurWithLogitsLoss":{forward:[15,2,1,""],training:[15,4,1,""]},"src.obj_detector.utils.loss.FocalLoss":{forward:[15,2,1,""],training:[15,4,1,""]},"src.obj_detector.utils.loss.QFocalLoss":{forward:[15,2,1,""],training:[15,4,1,""]},"src.obj_detector.utils.parse_config":{parse_data_config:[15,3,1,""],parse_model_config:[15,3,1,""]},"src.obj_detector.utils.transforms":{AbsoluteLabels:[15,1,1,""],ImgAug:[15,1,1,""],PadSquare:[15,1,1,""],RelativeLabels:[15,1,1,""],Resize:[15,1,1,""],ToTensor:[15,1,1,""]},"src.obj_detector.utils.utils":{ap_per_class:[15,3,1,""],bbox_iou:[15,3,1,""],bbox_wh_iou:[15,3,1,""],box_iou:[15,3,1,""],compute_ap:[15,3,1,""],get_batch_statistics:[15,3,1,""],load_classes:[15,3,1,""],non_max_suppression:[15,3,1,""],print_environment_info:[15,3,1,""],provide_determinism:[15,3,1,""],rescale_boxes:[15,3,1,""],to_cpu:[15,3,1,""],weights_init_normal:[15,3,1,""],worker_seed_set:[15,3,1,""],xywh2xyxy:[15,3,1,""],xywh2xyxy_np:[15,3,1,""]},"src.transforms":{AugDialog:[3,1,1,""],Augmentation:[3,1,1,""],AugmentationPipeline:[3,1,1,""],dim_intensity:[3,3,1,""],gaussian_blur:[3,3,1,""],gaussian_noise:[3,3,1,""],jpeg_comp:[3,3,1,""],letterbox_image:[3,3,1,""],normal_comp:[3,3,1,""],poisson_noise:[3,3,1,""],saltAndPapper_noise:[3,3,1,""],speckle_noise:[3,3,1,""]},"src.transforms.AugDialog":{demoAug:[3,2,1,""],pipelineChanged:[3,4,1,""],show:[3,2,1,""]},"src.transforms.Augmentation":{enabled:[3,2,1,""],exampleParam:[3,2,1,""],function_arg:[3,2,1,""],position:[3,2,1,""],setExampleParam:[3,2,1,""],setParam:[3,2,1,""],title:[3,2,1,""]},"src.transforms.AugmentationPipeline":{append:[3,2,1,""],clear:[3,2,1,""],exists:[3,2,1,""],load:[3,2,1,""],next:[3,2,1,""],remove:[3,2,1,""],save:[3,2,1,""]},src:{mit_semseg:[4,0,0,"-"],models:[3,0,0,"-"],obj_detector:[14,0,0,"-"],transforms:[3,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","function","Python function"],"4":["py","attribute","Python attribute"],"5":["py","exception","Python exception"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:function","4":"py:attribute","5":"py:exception"},terms:{"001":8,"100":8,"1000":13,"101":13,"1024":13,"150":13,"2048":13,"255":16,"256":13,"3x3":13,"4096":13,"416":[14,16],"512":13,"abstract":[3,8,12,16],"boolean":8,"case":[8,12,16],"class":[3,4,8,10,12,13,14,15,16],"default":[8,12,14],"final":16,"float":[3,8,14],"function":[3,8,12,13,14,15,16],"int":[3,8,12,14,16],"long":[12,16],"new":[0,1,12,14],"return":[3,8,12,13,14,15,16],"static":[3,12,13,14],"true":[8,12,13,15],"var":8,"while":[13,14,15],For:[8,16],NMS:15,The:[3,8,14,15,16],Use:14,Used:8,Useful:8,With:8,__data_parallel_replicate__:8,__getitem__:12,__init__:16,__iter__:12,__len__:12,_registri:16,_slavepipebas:8,_synchronizedbatchnorm:8,abc:3,about:15,absolutelabel:15,acceler:8,access:12,accuraci:4,across:[8,12],action:16,add:[3,4,8],added:8,affin:8,after:[8,12,16],afterward:[13,14,15],again:12,ahead:8,all:[8,12,13,14,15,16],along:12,alpha:15,also:8,although:[13,14,15],alwai:12,among:8,anchor:14,ani:8,anoth:12,ap_per_class:15,append:3,appli:[8,16],applic:16,arch:13,arg:[3,4,8,12,13,15],argument:[8,12,15],arrai:[3,14],array_lik:3,as_numpi:[8,11],as_vari:11,aspect:3,assembl:12,asserttensorclos:8,assign:8,assum:[12,16],async_copy_to:10,atol:8,aug:3,aug_titl:3,augdialog:3,auglist:3,augment:3,augmentationpipelin:3,autograd:[8,12],averag:[4,15],averagemet:4,back:8,base:[3,4,8,10,12,13,14,15,16],base_se:12,basedataset:4,batch:[8,10,12,14,15],batch_sampl:12,batch_siz:[8,12,14],batchnorm1d:8,batchnorm2d:8,batchnorm3d:8,batchnorm:[6,7],batchsampl:12,batchsiz:16,bbox:16,bbox_cl:16,bbox_iou:15,bbox_wh_iou:15,bceblurwithlogitsloss:15,becaus:8,been:16,befor:[12,14],behav:8,being:8,beta:8,between:14,bilinear:4,binari:16,bit:3,block:[13,14],boil:16,bool:[8,10,12,13,14,15],both:[12,15,16],bound:[12,15],box1:15,box2:15,box:15,box_iou:15,boxes1:15,boxes2:15,build_decod:13,build_encod:13,build_target:15,built:8,c1deepsup:13,calcul:8,call:[8,12,13,14,15],callabl:12,callback:8,callbackcontext:8,can:[8,12,16],cannot:12,care:[13,14,15],cdll:16,certain:16,cfg:[14,16],checkpoint:14,ciou:15,classmethod:3,clear:3,cls:[15,16],cnn:16,coco:16,code:15,collate_fn:[12,15],collect:[8,12],color:[4,14],colorencod:4,com:[13,15,16],combin:12,comm:[6,7],common:8,commun:[8,12],comput:[4,8,13,14,15],compute_ap:15,compute_loss:15,concatdataset:12,concaten:12,conf:15,conf_thr:[14,15,16],confid:14,config_path:14,configur:[14,15,16],conflict:16,conjunct:12,constant:12,construct:[13,14],conta:15,contain:[12,15],content:[4,14],context:8,conv3x3_bn_relu:13,conv_out:13,convert:[3,16],convolut:13,coordin:[14,16],copi:[8,12,16],copy_id:8,cpu:[8,12,14],creat:[3,8,16],create_modul:14,creation:16,crit:13,csail:4,ctx:8,cuda:12,cummulative_s:12,cumsum:12,current:[4,12],current_dim:15,curv:15,custom:8,cutoff:14,cv2:16,darknet:[14,16],data:[4,6,8,11,15],data_parallel:[6,7,8],data_sourc:12,data_tensor:12,dataload:[6,11,14],dataloaderit:12,dataparallel:8,dataparallelwithcallback:8,dataset:[4,6,11],deep_sup_scal:13,def:16,default_col:12,defaultparam:3,defin:[3,12,13,14,15],definit:[14,15],deiniti:[3,16],demoaug:3,denomin:8,deprec:14,depth:8,desir:[12,14],detail:8,detect:[14,15,16],detect_directori:14,detect_imag:[14,16],detector:[0,16],determin:8,dev:10,deviat:8,devic:[4,8,12],device_id:[8,10],dict:3,dictgatherdataparallel:10,dictionari:16,differ:[8,12],difficult:16,dilate_scal:13,dim:[3,8,10],dim_intens:3,dime:3,dimens:[8,12,14],dimimg:3,diou:15,directori:14,distribut:[6,8,11],distributed_rank:4,distributeddataparallel:12,distributedsampl:12,divis:12,doe:16,done:[8,12],down:16,draw:[3,12,16],drawn:[12,14],drop:12,drop_last:12,dtype:12,dure:[8,14],each:[8,12,14],easi:[8,16],element:[12,15],enabl:3,end_idx:4,enforc:3,entri:8,environ:15,epoch:12,eps:[8,15],epsilon:8,equival:8,eriklindernoren:16,especi:12,estim:8,etc:16,evalu:8,everi:[12,13,14,15],exactli:8,exampl:[1,8,12],exampleparam:3,exc_info:12,except:[4,12],exceptionwrapp:12,exclus:12,execut:8,execute_replication_callback:8,exist:[3,8,12,16],expect:[8,15],ext:4,factor:3,fals:[4,8,12,13,15],faster:15,fc_dim:13,feed_dict:13,field:12,file:[14,15,16],filenam:[3,4],find_recurs:4,first:[8,12],fit:3,fly:12,focalloss:15,follow:[13,16],form:12,format:[4,14,15],former:[13,14,15],forward:[8,13,14,15],four:3,fpn_dim:13,fpn_inplan:13,frac:8,framework:16,from:[8,12,13,14,15,16],function_arg:3,funtion:3,futur:8,futureresult:8,gamma:[8,15],gather:8,gaussian_blur:3,gaussian_nois:3,gener:[1,12,14],get:[8,16],get_batch_statist:15,giou:15,github:[13,15,16],give:[3,8],given:[12,14,15,16],going:16,gpu0:4,gpu1:4,gpu:8,group:13,guarante:8,gui:16,has:[12,16],have:[8,12,16],height:8,help:15,hex2rgb:14,hook:[13,14,15],how:[12,16],hrnetv2:13,http:[13,15,16],identifi:8,ignor:[13,14,15],imag:[3,14,15,16],imagefold:15,imagenet:13,img:[3,4,15,16],img_path:14,img_siz:[14,16],img_transform:4,imgaug:15,imlab:4,implement:[8,13,16],impr:4,imres:4,in_plan:13,inaccur:8,includ:8,incomplet:12,index:[0,12,15],indic:12,infer:[14,15,16],inferenc:3,info:15,inform:8,inherit:16,inhert:3,initi:[3,4,16],initial_se:12,inp:4,input:[3,4,8,10,12,14,16],input_devic:4,insert:16,instanc:[12,13,14,15],instead:[8,13,14,15],int64:12,integ:[3,12],integr:[0,1],intens:3,interfac:16,interp:4,intersect:15,intersectionandunion:4,invok:8,iou:[14,15],iou_thr:15,iou_threshold:15,isomorph:8,issu:[15,16],iter:12,its:[3,8,12,14],itself:8,jaccard:15,join:16,jpeg_comp:3,jpg:4,keep:8,kept:8,kernel_size_factor:3,keyword:12,kwarg:[3,10,13,16],kwd:[4,15],label:[4,12,13,15],labelmap:4,lambda:12,larg:12,last:12,later:16,latter:[13,14,15],layer:[8,13,14,15,16],layout:12,learnabl:8,len:12,length:12,less:12,let:16,letterbox:3,letterbox_imag:3,like:12,list:[12,14,15,16],list_of_scalars_summari:15,listdataset:15,listview:3,load:[3,12,14,15,16],load_class:[15,16],load_darknet_weight:14,load_model:[14,16],load_url:13,loader:12,log:[4,15],log_dir:15,log_hist:15,longtensor:12,look:16,loop:16,loss_fcn:15,mai:12,main:[3,8,12],main_stream:10,make:[15,16],mani:12,manner:12,map:16,map_loc:13,mark_volatil:11,master:8,master_callback:8,master_msg:8,matrix:15,max_sampl:4,maximum:[14,15],mean:[8,12],memori:12,merg:12,messag:8,method:[12,16],methodnam:8,metric:15,might:8,mini:[8,12],mit:0,mit_semseg:4,mobilenet_v2:13,mobilenetv2:13,mobilenetv2dil:13,mode:[4,14],model:[0,2,14,15,16],model_dir:13,model_path:14,modelbuild:13,modifi:13,modul:[0,2],module_def:14,momentum:8,monkei:8,msg:8,multi:[12,16],multipl:[8,12],must:[3,12],mutual:12,n_cpu:14,name:[3,14,16],nearest:14,necessari:12,need:[3,13,14,15,16],neg:12,net_dec:13,net_enc:13,network:[0,1,3,8],network_config:[3,16],neural:16,next:[3,12],nms_thre:[14,16],nois:3,non:[3,12,14,15],non_max_suppress:15,none:[8,10,12,13,14,15],normal:8,normal_comp:3,note:[8,16],notsupportedcliexcept:4,np_img:16,nr_slave:8,num_class:[13,14],num_featur:8,num_replica:12,num_sampl:12,num_work:12,number:[12,14],numclass:4,numer:[8,12],numpi:[3,12],nx6:15,nxm:15,obj:[10,11],obj_detector:[14,16],object:[0,3,4,8,12,13,15,16],obtain:8,odgt:4,onc:[12,16],one:[8,12,13,14,15],onli:[3,8,16],oper:12,option:[12,14,16],order:12,orig_net:13,orig_resnet:13,origin:[3,8,12,15,16],original_posit:3,original_shap:15,other:12,our:16,out:12,out_plan:13,outer:12,output:[8,12,14,15,16],output_devic:[8,10],output_path:14,outputformat:3,over:[8,12,15],overlap:12,overrid:12,overridden:[13,14,15],overview:1,pad:[3,14],pad_to_squar:15,pad_valu:15,padsquar:15,page:0,pairwis:15,parallel:[6,7,8,12],param:14,paramet:[3,8,14],pars:[4,14,15],parse_data_config:15,parse_devic:4,parse_input_list:4,parse_model_config:15,particip:12,particular:16,pass:[8,12,13,14,15],patch:8,patch_replication_callback:8,path:[14,15,16],peopl:15,pepper:3,per:[8,12,15],perfer:16,perform:[13,14,15],permut:12,pin:12,pin_memori:12,pin_memory_batch:12,pipe:8,pipelin:16,pipelinechang:3,pixel_acc:13,place:[8,13],plu:12,poisson_nois:3,pool_scal:13,port:16,posit:[3,12,15],possibl:12,ppm:13,ppm_deepsup:13,ppmdeepsup:13,pre:13,precis:15,pred:[3,4,13,15,16],pred_cl:15,predict:[15,16],pretrain:13,print:15,print_environment_info:15,printout:15,prob:3,probabl:[3,12],process:[12,16],process_rang:4,produc:12,project:3,properti:[3,8,12],provid:[12,14,16],provide_determin:15,pth:14,purpos:12,put:[8,12],pyqt5:[3,16],pytorch:[8,12,13,16],qdialog:3,qfocalloss:15,qt5:16,qtwidget:3,qualiti:3,queue:8,quit:16,rafaelpadilla:15,randn:8,random:12,random_split:12,randomli:12,randomsampl:12,randomst:3,randperm:12,rang:[3,12],rank:12,ratio:3,rbgirshick:15,rcnn:15,realli:16,reason:16,recal:15,receiv:8,recip:[13,14,15],record:12,rectangl:[3,16],reduc:8,regist:[8,13,14,15],register_slav:8,reimplement:16,reintegr:16,relativelabel:15,relu:13,remov:3,render:16,replac:[3,12],replic:[6,7],repositori:13,repres:12,requir:3,requires_grad:12,rescal:15,rescale_box:[14,15],reshuffl:12,resiz:[3,15],resnet101:13,resnet18:13,resnet50:13,resnet50dil:13,resnetdil:13,resnext101:13,restrict:12,result:[8,15],retriev:12,return_count:4,return_feature_map:13,return_index:4,return_invers:4,rgb:4,rng:12,root_dir:4,round2nearest_multipl:4,row:12,rtol:8,run:[3,8,13,14,15,16],run_mast:8,run_slav:8,running_mean:8,running_var:8,runtest:8,safe:8,sai:16,salt:3,saltandpapper_nois:3,same:[8,12,16],sampl:[8,12,15],sampler:[6,11],save:[3,14],save_darknet_weight:14,scalar:15,scalar_summari:15,scale:12,scale_factor:[3,14],scatter:10,score:15,search:0,see:[8,12],seed:[3,12,15],seen:8,segm:4,segm_transform:4,segment:[3,4,13,16],segmentationmodul:13,segmentationmodulebas:13,segsiz:13,self:[3,12,16],semant:[4,13,16],semseg:0,send:8,sens:16,sent:8,sequenc:[12,16],sequenti:12,sequentialsampl:12,set:[8,12,15],set_default_tensor_typ:12,set_epoch:12,setexampleparam:3,setparam:3,setup_logg:4,shape:[8,15],share:8,should:[8,12,13,14,15,16],show:3,shuffl:12,silent:[13,14,15],sinc:[13,14,15,16],singl:12,size:[3,4,8,12,14,15],skeleton:16,slave:8,slavepip:8,slice:8,smaller:12,smooth_bc:15,some:8,sourc:15,spatial:8,spatio:8,spawn:12,specifi:[3,12,14],speckle_nois:3,split:12,sqrt:8,src:[3,4,14],stabil:8,standard:[4,8],start:[12,16],start_idx:4,statist:8,std:3,stdx:3,stdy:3,step:15,store:[4,14],str:[4,14,16],strategi:12,stride:[12,13],structur:16,sub:8,subclass:[12,13,14,15],submodul:[0,2,4,6,7,14],subpackag:[4,14],subprocess:12,subset:12,subsetrandomsampl:12,sum:[8,12],support:12,suppress:[14,15],sync_bn:8,synchron:8,synchronizedbatchnorm1d:8,synchronizedbatchnorm2d:8,synchronizedbatchnorm3d:8,syncmast:8,system:[14,15],tag:15,tag_value_pair:15,take:[13,14,15,16],target:[12,15],target_cl:15,target_tensor:12,tempor:8,tensor:[8,12,14,15],tensordataset:12,terminolog:8,test:[7,8,14],test_numeric_batchnorm:[7,8],test_sync_batchnorm:[7,8],testcas:8,testdataset:4,text:16,than:12,thei:12,them:[12,13,14,15],thi:[3,8,12,13,14,15,16],thing:16,thread:[8,12,14],threshold:14,through:[8,16],time:12,timeout:12,titl:3,to_cpu:15,todo:16,tonylin:13,torch:[4,8,12,13,14,15],torchtestcas:8,totensor:15,traceback:12,track_running_stat:8,train:[8,10,12,13,14,15],traindataset:4,transform:[0,2,14,16],trigger:8,tupl:[3,16],turn:16,tutori:[0,16],two:[3,15],txt:4,type:[12,14],unchang:3,union:15,uniqu:4,unittest:[6,7],unpickl:12,unsign:3,updat:[4,16],upernet:13,upper:12,upsampl:14,url:13,usag:8,use:[3,12,14],use_softmax:13,used:[8,12],useful:12,user:4,user_scattered_col:10,userscattereddataparallel:10,uses:[8,12],using:[3,8,12,16],usual:8,util:[4,6,14,16],val:4,valdataset:4,valu:[4,8,12,15],variabl:[8,15,16],varianc:8,vector:8,veri:16,version:8,volumetr:8,wai:12,want:[8,16],weight:[4,12,13,14,16],weightedrandomsampl:12,weights_init:13,weights_init_norm:15,weights_path:14,were:8,wh1:15,wh2:15,when:[8,12,15],where:[8,12],which:[8,12,16],width:8,within:[12,13,14,15,16],without:[8,12],worker:12,worker_id:[12,15],worker_init_fn:12,worker_seed_set:15,would:[12,16],wrap:[8,12],x1y1x2y2:15,xpu:4,xywh2xyxi:15,xywh2xyxy_np:15,yield:12,yolo:[3,14,15,16],yololay:14,yolov3:[3,14,16],you:[8,12,16]},titles:["Welcome to Noisey-Image\u2019s documentation!","Tutorials","src","Submodules","Mit Semseg Module","src.mit_semseg.config package","src.mit_semseg.lib package","src.mit_semseg.lib.nn package","src.mit_semseg.lib.nn.modules package","src.mit_semseg.lib.nn.modules.tests package","src.mit_semseg.lib.nn.parallel package","src.mit_semseg.lib.utils package","src.mit_semseg.lib.utils.data package","src.mit_semseg.models package","Object Detector Module","src.obj_detector.utils package","Integrating New Network"],titleterms:{"default":5,"new":16,augment:15,batchnorm:8,comm:8,config:5,content:[5,6,7,8,9,10,11,12,13,15],data:12,data_parallel:10,dataload:12,dataset:[12,15],detector:14,distribut:12,document:0,exampl:16,gener:16,hrnet:13,imag:0,indic:0,integr:16,lib:[6,7,8,9,10,11,12],logger:15,loss:15,mit:4,mit_semseg:[5,6,7,8,9,10,11,12,13],mobilenet:13,model:[3,13],modul:[3,4,5,6,7,8,9,10,11,12,13,14,15],network:16,noisei:0,obj_detector:15,object:14,overview:16,packag:[5,6,7,8,9,10,11,12,13,15],parallel:10,parse_config:15,replic:8,resnet:13,resnext:13,sampler:12,semseg:4,src:[2,5,6,7,8,9,10,11,12,13,15],submodul:[3,5,8,9,10,11,12,13,15],subpackag:[6,7,8,11],tabl:0,test:9,test_numeric_batchnorm:9,test_sync_batchnorm:9,transform:[3,15],tutori:1,unittest:8,util:[11,12,13,15],welcom:0}})