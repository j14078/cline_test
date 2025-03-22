#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
画像処理確認アプリの画像ユーティリティモジュール。

このモジュールは画像の読み込みと処理に関するユーティリティ関数を提供します。
"""

import os
import tkinter as tk
from tkinter import filedialog
import numpy as np
import cv2
from PIL import Image


def select_folder_simple():
    """シンプルなフォルダ選択ダイアログを表示する。

    Returns:
        選択されたフォルダのパス、キャンセルされた場合はNone
    """
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    folder_path = filedialog.askdirectory(parent=root)
    root.destroy()
    
    if folder_path:
        # パスを正規化して返す
        return os.path.normpath(folder_path)
    return None


def load_images_from_folder(folder_path, max_images=100):
    """フォルダから画像を読み込む。
    
    Args:
        folder_path: 画像フォルダのパス
        max_images: 読み込む最大画像数
    
    Returns:
        読み込まれた画像のリスト
    """
    images = []
    
    # フォルダパスのチェック
    if not folder_path or not os.path.exists(folder_path):
        return images
        
    # Windows環境での日本語パス対応
    if os.name == 'nt':
        folder_path = os.path.abspath(folder_path)
        folder_path = os.path.normpath(folder_path)
        
    # 画像ファイルのみをフィルタリング
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    image_files = []
    
    for filename in os.listdir(folder_path):
        _, ext = os.path.splitext(filename)
        if ext.lower() in image_extensions:
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                image_files.append(file_path)
    
    # 画像ファイルがない場合
    if not image_files:
        return images
        
    # 最大処理数の制限
    image_files = image_files[:max_images]
    
    # 各画像を読み込む
    for file_path in image_files:
        try:
            # PILで画像を読み込み
            pil_img = Image.open(file_path)
            img = np.array(pil_img)
            
            # RGBからBGRに変換（OpenCVはBGR形式）
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
            if img is not None and img.size > 0:
                images.append(img)
        except Exception:
            # 読み込みに失敗した画像はスキップ
            continue
            
    return images


def navigate_images(current_index, images, direction):
    """画像を前後に切り替える。
    
    Args:
        current_index: 現在の画像インデックス
        images: 画像のリスト
        direction: 'next' または 'prev'
    
    Returns:
        新しい画像インデックス
    """
    if len(images) <= 1:
        return current_index
        
    if direction == "next":
        return (current_index + 1) % len(images)
    elif direction == "prev":
        return (current_index - 1) % len(images)
    
    return current_index