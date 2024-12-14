#! python3
# -*- encoding: utf-8 -*-
"""
@File    :   photos2video.py
@Time    :   24-12-14
@Version :   1.0
@Author  :   Tomas 
@Contact :   tomaswu@qq.com
@Desc    :   
"""

import cv2  
import tkinter as tk
import os 
import random
import numpy as np
import tqdm
from pydub import AudioSegment
from pydub.utils import make_chunks
import time
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool, Manager

def process_single_photo(args):
    photo, photo_dir, width, height, back_img, fps = args
    image_path = os.path.join(photo_dir, photo)
    curr_frame = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    
    # 保持原比例调整大小并添加黑色填充
    h, w = curr_frame.shape[:2]
    scale = min(width/w, height/h)
    new_w, new_h = int(w*scale), int(h*scale)
    
    # 调整图片大小
    curr_frame = cv2.resize(curr_frame, (new_w, new_h))
    
    # 创建空白画布
    frame_with_border = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 计算居中位置
    y_offset = (height - new_h) // 2
    x_offset = (width - new_w) // 2
    
    # 将调整后的图片放在画布中央
    frame_with_border[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = curr_frame
    
    # 根据相框的alpha通道混合图片和相框
    alpha = back_img[:,:,3] / 255.0
    alpha = np.stack([alpha, alpha, alpha], axis=-1)
    curr_frame = back_img[:,:,:3] * alpha + frame_with_border * (1 - alpha)
    curr_frame = curr_frame.astype(np.uint8)
    
    duration = random.uniform(3, 5)  # 随机生成3-5秒的时长
    return curr_frame, duration

def convert_photos_to_video(photo_dir, output_video_path, fps=30, music_files=[], back='./back.png'):
    # 获取照片目录下的所有图片文件
    photo_files = [f for f in os.listdir(photo_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    photo_files.sort()  # 按文件名排序
    
    if not photo_files:
        raise ValueError("未在目录中找到图片文件")
        
    # 读取第一张图片获取尺寸信息
    first_image = cv2.imdecode(np.fromfile(os.path.join(photo_dir, photo_files[0]), dtype=np.uint8), cv2.IMREAD_COLOR)
    height, width = first_image.shape[:2]
    
    # 读取相框背景图片
    back_img = cv2.imread(back, cv2.IMREAD_UNCHANGED)  # 读取带alpha通道的图片
    back_img = cv2.resize(back_img, (width, height))
    
    # 创建进程池
    pool = Pool(processes=mp.cpu_count())
    
    # 准备参数
    args = [(photo, photo_dir, width, height, back_img, fps) for photo in photo_files]
    
    # 创建视频写入器
    temp_video_path = "temp_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
    
    total_duration = 0  # 记录总时长
    prev_frame = None
    
    # 使用进程池处理图片
    with tqdm.tqdm(total=len(photo_files), desc="处理图片") as pbar:
        for curr_frame, duration in pool.imap(process_single_photo, args):
            # 如果存在前一帧,添加淡入淡出效果
            if prev_frame is not None:
                fade_frames = int(fps * 2)
                for i in range(fade_frames):
                    alpha = i / fade_frames
                    blended = cv2.addWeighted(prev_frame, 1-alpha, curr_frame, alpha, 0)
                    video_writer.write(blended)
                    total_duration += 1/fps
            
            # 写入当前帧
            for _ in range(int(fps * duration)):
                video_writer.write(curr_frame)
                total_duration += 1/fps
                
            prev_frame = curr_frame.copy()
            pbar.update(1)
    
    pool.close()
    pool.join()
    
    # 释放资源
    video_writer.release()

    # 处理音频
    if music_files:
        # 合并所有音频文件
        combined_audio = AudioSegment.empty()
        for music_file in music_files:
            music_file = os.path.abspath(music_file)
            audio = AudioSegment.from_file(music_file)
            combined_audio += audio

        # 如果音频时长不够，循环播放
        while len(combined_audio) < total_duration * 1000:  # pydub使用毫秒
            combined_audio += combined_audio

        # 如果音频过长，裁剪
        combined_audio = combined_audio[:int(total_duration * 1000)]

        # 添加音频渐弱效果（最后3秒）
        fade_duration = min(3000, len(combined_audio))
        combined_audio = combined_audio.fade_out(fade_duration)

        # 导出临时音频文件
        temp_audio_path = "temp_audio.mp3"
        combined_audio.export(temp_audio_path, format="mp3")

        # 使用ffmpeg合并视频和音频
        os.system(f'ffmpeg -y -i {temp_video_path} -i {temp_audio_path} -c:v copy -c:a aac {output_video_path}')

        # 删除临时文件
        os.remove(temp_video_path)
        os.remove(temp_audio_path)
    else:
        # 如果没有音乐文件，直接重命名临时视频文件
        os.rename(temp_video_path, output_video_path)

if __name__=='__main__':
    convert_photos_to_video(r'./photos', r'output.mp4',music_files = ['./music/gulou.mp3'])
