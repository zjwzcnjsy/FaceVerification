folder = 'D:\jsy\res\face\images\lfw\lfw_original';
addpath('..');
image_list = get_image_list_in_folder(folder);
target_folder = 'D:\jsy\res\face\images\lfw\mtcnn_norm\lfw_original-align-wuxiang_256x256';
target_faceinfo_folder = 'D:\jsy\res\face\images\lfw\mtcnn_norm\lfw_original-mtcnn-result';
if exist(target_folder, 'dir')==0
    mkdir(target_folder);
end;
if exist(target_faceinfo_folder, 'dir')==0
    mkdir(target_faceinfo_folder);
end;
caffe_path = 'D:\jsy\bin\caffe\caffe\Build\x64\Release2\matcaffe';
addpath(genpath(caffe_path));

pdollar_toolbox_path='../toolbox';
addpath(genpath(pdollar_toolbox_path));

MTCNN_path = '../MTCNN_face_detection_alignment/code/codes/MTCNNv2';
addpath(genpath(MTCNN_path));
caffe_model_path=[MTCNN_path , '/model'];

coord5points = [30.2946, 65.5318, 48.0252, 33.5493, 62.7299; ...
                51.6963, 51.5014, 71.7366, 92.3655, 92.2041];
imgSize = [112, 96];
align_method = 'wuxiang_256x256';% wuxiang, wuxiang2, wuxiang_256x256 or yandong
            
%caffe.set_mode_cpu();
gpu_id=0;
caffe.set_mode_gpu();	
caffe.set_device(gpu_id);
caffe.reset_all();

%three steps's threshold
threshold=[0.6 0.7 0.7];

%◊Ó–°»À¡≥
minsize = 100;

%scale factor
factor=0.709;

%load caffe models
prototxt_dir =strcat(caffe_model_path,'/det1.prototxt');
model_dir = strcat(caffe_model_path,'/det1.caffemodel');
PNet=caffe.Net(prototxt_dir,model_dir,'test');
prototxt_dir = strcat(caffe_model_path,'/det2.prototxt');
model_dir = strcat(caffe_model_path,'/det2.caffemodel');
RNet=caffe.Net(prototxt_dir,model_dir,'test');	
prototxt_dir = strcat(caffe_model_path,'/det3.prototxt');
model_dir = strcat(caffe_model_path,'/det3.caffemodel');
ONet=caffe.Net(prototxt_dir,model_dir,'test');
prototxt_dir =  strcat(caffe_model_path,'/det4.prototxt');
model_dir =  strcat(caffe_model_path,'/det4.caffemodel');
LNet=caffe.Net(prototxt_dir,model_dir,'test');
faces=cell(0);	

for image_id = 1:length(image_list);
    [file_folder, file_name, file_ext] = fileparts(image_list{image_id});
    target_filename = strrep(image_list{image_id},folder, target_folder);
    target_faceinfo_filename = strrep(image_list{image_id},folder, target_faceinfo_folder);
    if exist(target_filename, 'file')
        continue;
    end;
    img = imread(image_list{image_id});
    if size(img, 3) < 3
       img(:,:,2) = img(:,:,1);
       img(:,:,3) = img(:,:,1);
    end
    
    assert(strcmp(target_filename, image_list{image_id})==0);
    [file_folder, file_name, file_ext] = fileparts(target_filename);
    if exist(file_folder,'dir')==0
        mkdir(file_folder);
    end;
    disp([num2str(image_id) '/' num2str(length(image_list)) ' ' target_filename]);
    target_faceinfo_filename = [target_faceinfo_filename '.txt'];
    if exist(target_faceinfo_filename)
        boundingboxes = textread(target_faceinfo_filename, '%f');
        if size(boundingboxes,1) > 0 && mod(size(boundingboxes,1), 15) == 0
            boundingboxes = reshape(boundingboxes, 15, [])';
            points = boundingboxes(:, 6:end)';
            boundingboxes = boundingboxes(:, 1:5);
        else
            [boundingboxes, points]=detect_face(img,min([minsize size(img,1) size(img,2)]),PNet,RNet,ONet,LNet,threshold,false,factor);
            fd = fopen(target_faceinfo_filename, 'w');
            for bb=1:size(boundingboxes,1)
                fprintf(fd, '%f %f %f %f %f ', boundingboxes(bb,:));
                fprintf(fd, '%f %f %f %f %f %f %f %f %f %f\n', points(:,bb)');
            end
            fclose(fd);
        end
    else
        [boundingboxes, points]=detect_face(img,min([minsize size(img,1) size(img,2)]),PNet,RNet,ONet,LNet,threshold,false,factor);
        fd = fopen(target_faceinfo_filename, 'w');
        for bb=1:size(boundingboxes,1)
            fprintf(fd, '%f %f %f %f %f ', boundingboxes(bb,:));
            fprintf(fd, '%f %f %f %f %f %f %f %f %f %f\n', points(:,bb)');
        end
        fclose(fd);
    end
    if isempty(boundingboxes)
        continue;
    end;
    
    default_face = 1;
    if size(boundingboxes,1) > 1
        for bb=2:size(boundingboxes,1)
            if abs((boundingboxes(bb,1) + boundingboxes(bb,3))/2 - size(img,2) / 2) + abs((boundingboxes(bb,2) + boundingboxes(bb,4))/2 - size(img,1) / 2) < ...
                    abs((boundingboxes(default_face,1) + boundingboxes(default_face,3))/2 - size(img,2) / 2) + abs((boundingboxes(default_face,2) + boundingboxes(default_face,4))/2 - size(img,1) / 2)
                default_face = bb;
            end;
        end;
    end;
    facial5points = double(reshape(points(:,default_face),[5 2])');
    if strcmp(align_method, 'wuxiang') > 0
        [res, eyec2, cropImg, resize_scale] = align_face_WX(img,facial5points',144,48,48);
        cropImg = uint8(cropImg);
    elseif  strcmp(align_method, 'wuxiang2') > 0
        [res, eyec2, cropImg, resize_scale] = align_face_WX2(img,facial5points',128,48,40);
        cropImg = uint8(cropImg);
    elseif  strcmp(align_method, 'wuxiang_256x256') > 0
        [res, eyec2, cropImg, resize_scale] = align_face_WX_256x256(img,facial5points',256,96,80);
        cropImg = uint8(cropImg);
    else
        Tfm =  cp2tform(facial5points', coord5points', 'similarity');
        cropImg = imtransform(img, Tfm, 'XData', [1 imgSize(2)],...
                                      'YData', [1 imgSize(1)], 'Size', imgSize);
    end;
    imwrite(cropImg, target_filename);
	% show detection result
% 	numbox=size(boundingboxes,1);
%     figure(1);
% 	imshow(img)
% 	hold on; 
% 	for j=1:numbox
% 		plot(points(1:5,j),points(6:10,j),'g.','MarkerSize',10);
% 		r=rectangle('Position',[boundingboxes(j,1:2) boundingboxes(j,3:4)-boundingboxes(j,1:2)],'Edgecolor','g','LineWidth',3);
%     end;
%     hold off;
%     figure(2);
%     imshow(cropImg);
% 	pause
end;
