# config.py
"""
配置文件，包含：
- 模型参数
- 训练参数
- 数据路径
- 日志配置
例如：
"""
import math

body_length = 130
features_range = {
    'neck_length': {
        'min': body_length / 4,
        'max': body_length * 5 / 4
    },
    'head_length': {
        'min': body_length / 4,
        'max': body_length * 5 / 4
    },

    'leg_length': {
        'min': body_length / 4,
        'max': body_length * 5 / 4
    },
    'tail_length': {
        'min': body_length / 4,
        'max': body_length * 5 / 4
    },    
    'neck_angle': {
        'min': -math.pi / 3,
        'max': 0
    },
    'head_angle': {
        'min': 0,
        'max': math.pi / 3
    },
    'leg_angle': {
        'min': math.pi * 11 / 36,
        'max': math.pi * 17 / 36
    },
    'tail_angle': {
        'min': -math.pi / 3,
        'max': 0
    }
}

canvas_settings = {
    'Task1a' : {
        'canvas_height': 600,
        'canvas_width' : 1200
    },
    'Task1b' : {
        'canvas_height_max': 700,
        'canvas_height_min': 400,
        'canvas_width' : 780
    },
    'Task3c' : {
        'canvas_height': 600,
        'canvas_width' : 1200
    },
}

 

DATA_PATH = 'data/raw'