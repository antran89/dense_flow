% Function that extracts optical flow for video frames of UCF101 dataset.
%
%    Copyright (C) 2015 An Tran
%
%    You can redistribute and/or modify this software for non-commercial use
%    under the terms of the GNU General Public License as published by
%    the Free Software Foundation, either version 3 of the License, or
%    (at your option) any later version.
%
%    This program is distributed in the hope that it will be useful,
%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%    GNU General Public License for more details.
%
%    You should have received a copy of the GNU General Public License
%    along with this program.  If not, see <http://www.gnu.org/licenses/>.
%
%    For commercial use, contact the author for licensing options.
%
%    Contact: tranlaman@gmail.com
%    This file is mainly cloned and modified from Fast Video Segment
%    software (http://groups.inf.ed.ac.uk/calvin/FastVideoSegmentation/)

function extractingDenseColorFlow_HMDB(index, device_id, type)
% path1 = '/nfs/lmwang/lmwang/Data/UCF101/ucf101_org/';
% if type ==0
%     path2 = '/media/sdb/lmwang/data/UCF101/ucf101_flow_img_farn_gpu_step_2/';
% elseif type ==1
%     path2 = '/media/sdb/lmwang/data/UCF101/ucf101_flow_img_tvl1_gpu_step_2/';
% else
%     path2 = '/media/sdb/lmwang/data/UCF101/ucf101_flow_img_brox_gpu_step_2/';
% end


path1 = '/home/tranlaman/Public/data/video/HMDB/Video/';
if type ==0
    path2 = '/media/data/ucf101_flow/hmdb_dense_color_flow_img_farn_gpu/';
elseif type ==1
    path2 = '/media/data/ucf101_flow/hdmb_dense_color_flow_img_tvl1_gpu/';
else
    path2 = '/media/data/ucf101_flow/hmdb_dense_color_flow_img_brox_gpu/';
end
folderlist = dir(path1);
foldername = {folderlist(:).name};
foldername = setdiff(foldername,{'.','..'});

for i = index
    if ~exist([path2,foldername{i}],'dir')
        mkdir([path2,foldername{i}]);
    end
    filelist = dir([path1,foldername{i},'/*.avi']);

    for j = 1:length(filelist)
        if ~exist([path2,foldername{i},'/',filelist(j).name(1:end-4)],'dir')
            mkdir([path2,foldername{i},'/',filelist(j).name(1:end-4)]);
        else
            continue;
        end 
        fprintf('extract flow of video %s\n', filelist(j).name);
        file1 = [path1,foldername{i},'/',filelist(j).name];
        file2 = [path2,foldername{i},'/',filelist(j).name(1:end-4),'/','flow'];
        cmd = sprintf('../src-build/dense_color_flow -f=''%s'' -x=''%s'' -b=20 -t=%d -d=%d -s=%d',...
            file1,file2,type,device_id,2);
        system(cmd);
    end
end
end