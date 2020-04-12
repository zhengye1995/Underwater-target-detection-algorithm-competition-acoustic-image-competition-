# Kesci 水下目标检测算法赛 代码提交

## author: rill

## 整体思路
   + detection algorithm: Cascade R-CNN 
   + backbone: ResNet101 + FPN + DCN
   + post process: soft nms
   + 基于[mmdetection](https://github.com/open-mmlab/mmdetection/)
   + 根据数据分为前视声呐和侧扫声呐的特点，分成两个分布来训练
   
## 代码环境及依赖

+ OS: Ubuntu16.10
+ GPU: 2080Ti * 4
+ python: python3.7
+ nvidia 依赖:
   - cuda: 10.0.130
   - cudnn: 7.5.1
   - nvidia driver version: 430.14


## 依赖安装及编译


- **依赖安装编译**

   1. 创建并激活虚拟环境
        conda create -n underwater python=3.7 -y
        conda activate underwater

   2. 安装 pytorch
        conda install pytorch=1.1.0 torchvision=0.3.0 cudatoolkit=10.0 -c pytorch
        
   3. 安装其他依赖
        pip install cython && pip --no-cache-dir install -r requirements.txt
   
   4. 编译cuda op等：
        python setup.py develop
   

## 模型预测
    
   - **数据准备**
   
   1. 将测试集中**前视声呐**图片放置于 data/b-test-image/image/Forward_looking_sonar_image 目录下
   
   2. 将测试集中**侧扫声呐**图片放置于 data/b-test-image/image/Side_scan_sonar_image 目录下

   3. 运行 python tools/data_process/generate_test_json.py 
   
   4. 从百度网盘： 链接： https://pan.baidu.com/s/11ObvDssM8sWbCZJdbhkKXA  提取码：bk1d
   
   下载模型文件forward_epoch12.pth 和side_epoch12.pth 分别放置于：
   
   work_dirs/forward/cas_r101_dcn_fpn_1x_fixbn_vflip/forward_epoch_12.pth 
   
   work_dirs/side_all/cas_r101_dcn_fpn_1x_vflip/side_epoch_12.pth

 
  

   - **预测**

    1. 运行:

        chmod +x tools/dist_test.sh

        ./tools/dist_test.sh configs/underwater/cas_r101/cas_r101_dcn_forward.py work_dirs/forward/cas_r101_dcn_fpn_1x_fixbn_vflip/forward_epoch_12.pth 4 --json_out results_testb/cas_r101_dcn_forward_vflip_flip_3scale.json

        (上面的4是我的gpu数量，请自行修改)
        
        ./tools/dist_test.sh configs/underwater/cas_r101/cas_r101_dcn_side_all.py work_dirs/side_all/cas_r101_dcn_fpn_1x_vflip/side_epoch_12.pth 4 --json_out results_testb/cas_r101_dcn_side_all_vflip_flip_3scale.json

        (上面的4是我的gpu数量，请自行修改)


    2. 预测结果文件会保存在 results_testb 目录下

    3. 转化mmd预测结果为提交csv格式文件：
       
       python tools/post_process/json2submit_testb.py --submit_file submit.csv
       

       最终符合官方要求格式的提交文件 submit.csv 位于 submit_testb 目录下
    

## Contact

    author：rill

    email：18813124313@163.com
