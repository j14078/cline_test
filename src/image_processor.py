#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
画像処理モジュール。

このモジュールは画像処理確認アプリで使用される画像処理機能を提供します。
"""

import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import scipy.ndimage as ndi
from skimage import feature, filters, segmentation, morphology, measure
import matplotlib.pyplot as plt

# タイトル
st.title("画像処理確認アプリ")


class ImageProcessor:
    """画像処理を行うクラス。"""
    
    def __init__(self, library="opencv"):
        """初期化メソッド。

        Args:
            library: 使用する画像処理ライブラリ（"opencv", "pillow", "scikit-image"など）
        """
        self.library = library
        self.supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
    def load_image(self, path):
        """画像を読み込む。

        Args:
            path: 画像ファイルのパス

        Returns:
            読み込まれた画像オブジェクト
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"画像ファイルが見つかりません: {path}")
            
        _, ext = os.path.splitext(path)
        if ext.lower() not in self.supported_extensions:
            raise ValueError(f"サポートされていないファイル形式です: {ext}")
            
        if self.library == "opencv":
            # 日本語パスに対応するため、PILで読み込んでからOpenCVに変換
            try:
                # 直接読み込みを試みる
                img = cv2.imread(path)
                if img is None:
                    # 失敗した場合はPILを使用
                    pil_img = Image.open(path)
                    img = np.array(pil_img)
                    # RGBからBGRに変換（OpenCVはBGR形式）
                    if len(img.shape) == 3 and img.shape[2] == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                return img
            except Exception as e:
                # 両方の方法が失敗した場合はエラーを表示
                raise IOError(f"画像の読み込みに失敗しました: {path} - {e}")
        elif self.library == "pillow":
            return Image.open(path)
        else:
            # デフォルトはOpenCV（日本語パス対応）
            try:
                img = cv2.imread(path)
                if img is None:
                    pil_img = Image.open(path)
                    img = np.array(pil_img)
                    if len(img.shape) == 3 and img.shape[2] == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                return img
            except Exception as e:
                raise IOError(f"画像の読み込みに失敗しました: {path} - {e}")
        
    def save_image(self, path, image):
        """画像を保存する。

        Args:
            path: 保存先のパス
            image: 保存する画像オブジェクト
        """
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        if self.library == "opencv":
            cv2.imwrite(path, image)
        elif self.library == "pillow":
            image.save(path)
        else:
            # デフォルトはOpenCV
            cv2.imwrite(path, image)
        
    def apply_processing(self, image, process_type, params=None):
        """指定された処理を画像に適用する。

        Args:
            image: 処理対象の画像
            process_type: 処理の種類
            params: 処理のパラメータ

        Returns:
            処理後の画像
        """
        if params is None:
            params = {}
            
        # 処理タイプに応じて適切なメソッドを呼び出す
        if process_type == "grayscale":
            return self.grayscale(image)
        elif process_type == "invert":
            return self.invert(image)
        elif process_type == "rotate":
            angle = params.get("angle", 90)
            return self.rotate(image, angle)
        elif process_type == "resize":
            width = params.get("width", image.shape[1] // 2)
            height = params.get("height", image.shape[0] // 2)
            return self.resize(image, width, height)
        elif process_type == "blur":
            kernel_size = params.get("kernel_size", 5)
            blur_type = params.get("blur_type", "gaussian")
            return self.blur(image, kernel_size, blur_type)
        elif process_type == "sharpen":
            return self.sharpen(image)
        elif process_type == "edge_detection":
            method = params.get("method", "canny")
            return self.edge_detection(image, method)
        elif process_type == "brightness":
            value = params.get("value", 1.5)
            return self.adjust_brightness(image, value)
        elif process_type == "contrast":
            value = params.get("value", 1.5)
            return self.adjust_contrast(image, value)
        elif process_type == "saturation":
            value = params.get("value", 1.5)
            return self.adjust_saturation(image, value)
        elif process_type == "sepia":
            return self.sepia(image)
        elif process_type == "emboss":
            return self.emboss(image)
        elif process_type == "mosaic":
            block_size = params.get("block_size", 10)
            return self.mosaic(image, block_size)
        elif process_type == "threshold":
            thresh_value = params.get("thresh_value", 127)
            return self.threshold(image, thresh_value)
        elif process_type == "dilate":
            kernel_size = params.get("kernel_size", 5)
            return self.dilate(image, kernel_size)
        elif process_type == "erode":
            kernel_size = params.get("kernel_size", 5)
            return self.erode(image, kernel_size)
        elif process_type == "flip_horizontal":
            return self.flip_horizontal(image)
        elif process_type == "flip_vertical":
            return self.flip_vertical(image)
        else:
            raise ValueError(f"サポートされていない処理タイプです: {process_type}")
        
    # 基本処理メソッド
    def grayscale(self, image):
        """グレースケール変換。

        Args:
            image: 入力画像

        Returns:
            グレースケール変換された画像
        """
        if self.library == "opencv":
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif self.library == "pillow":
            try:
                image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
            except:
                pass
            return Image.fromarray(image).convert('L')
        else:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    def invert(self, image):
        """画像の反転。

        Args:
            image: 入力画像

        Returns:
            反転された画像
        """
        if self.library == "opencv":
            return cv2.bitwise_not(image)
        elif self.library == "pillow":
            try:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except:
                pass
            img = Image.fromarray(image)
            return Image.eval(img, lambda x: 255 - x)
        else:
            return cv2.bitwise_not(image)
        
    def rotate(self, image, angle):
        """画像の回転。

        Args:
            image: 入力画像
            angle: 回転角度（度）

        Returns:
            回転された画像
        """
        if self.library == "opencv":
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            return cv2.warpAffine(image, rotation_matrix, (width, height))
        elif self.library == "pillow":
            try:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except:
                pass
            img = Image.fromarray(image)
            return img.rotate(angle)
        else:
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            return cv2.warpAffine(image, rotation_matrix, (width, height))
        
    def resize(self, image, width, height):
        """画像のリサイズ。

        Args:
            image: 入力画像
            width: 新しい幅
            height: 新しい高さ

        Returns:
            リサイズされた画像
        """
        if self.library == "opencv":
            return cv2.resize(image, (width, height))
        elif self.library == "pillow":
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            return img.resize((width, height))
        else:
            return cv2.resize(image, (width, height))
        
    # フィルタ処理メソッド
    def blur(self, image, kernel_size, blur_type="gaussian"):
        """ぼかし処理。

        Args:
            image: 入力画像
            kernel_size: カーネルサイズ
            blur_type: ぼかしの種類（"gaussian", "median", "box", "bilateral"）

        Returns:
            ぼかし処理された画像
        """
        if self.library == "opencv":
            if blur_type == "gaussian":
                return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
            elif blur_type == "median":
                return cv2.medianBlur(image, kernel_size)
            elif blur_type == "box":
                return cv2.blur(image, (kernel_size, kernel_size))
            elif blur_type == "bilateral":
                return cv2.bilateralFilter(image, kernel_size, 75, 75)
            else:
                return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        elif self.library == "pillow":
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            # PillowではImageFilterを使用
            if blur_type == "gaussian":
                return img.filter(ImageFilter.GaussianBlur(radius=kernel_size))
            elif blur_type == "median":
                return img.filter(ImageFilter.MedianFilter(size=kernel_size))
            elif blur_type == "box":
                return img.filter(ImageFilter.BoxBlur(radius=kernel_size))
            else:
                return img.filter(ImageFilter.GaussianBlur(radius=kernel_size))
        else:
            if blur_type == "gaussian":
                return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
            elif blur_type == "median":
                return cv2.medianBlur(image, kernel_size)
            elif blur_type == "box":
                return cv2.blur(image, (kernel_size, kernel_size))
            elif blur_type == "bilateral":
                return cv2.bilateralFilter(image, kernel_size, 75, 75)
            else:
                return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
            
    def flip_horizontal(self, image):
        """画像を水平方向に反転する。

        Args:
            image: 入力画像

        Returns:
            水平反転された画像
        """
        if self.library == "opencv":
            return cv2.flip(image, 1)  # 1は水平方向の反転
        elif self.library == "pillow":
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            return cv2.flip(image, 1)
            
    def flip_vertical(self, image):
        """画像を垂直方向に反転する。

        Args:
            image: 入力画像

        Returns:
            垂直反転された画像
        """
        if self.library == "opencv":
            return cv2.flip(image, 0)
        elif self.library == "pillow":
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        else:
            return cv2.flip(image, 0)
        
    def sharpen(self, image):
        """シャープ化処理。

        Args:
            image: 入力画像

        Returns:
            シャープ化された画像
        """
        if self.library == "opencv":
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            return cv2.filter2D(image, -1, kernel)
        elif self.library == "pillow":
            enhancer = ImageEnhance.Sharpness(image)
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            return enhancer.enhance(2.0)
        else:
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            return cv2.filter2D(image, -1, kernel)
        
    def edge_detection(self, image, method="canny"):
        """エッジ検出処理。

        Args:
            image: 入力画像
            method: エッジ検出方法（"canny", "sobel", "prewitt"）

        Returns:
            エッジ検出された画像
        """
        if self.library == "opencv":
            if method == "canny":
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image
                return cv2.Canny(gray, 100, 200)
            elif method == "sobel":
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
                return cv2.magnitude(sobelx, sobely)
            else:
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image
                return cv2.Laplacian(gray, cv2.CV_64F)
        elif self.library == "pillow":
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            return img.filter(ImageFilter.FIND_EDGES)
        else:
            if method == "canny":
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image
                return cv2.Canny(gray, 100, 200)
            elif method == "sobel":
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
                return cv2.magnitude(sobelx, sobely)
            else:
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image
                return cv2.Laplacian(gray, cv2.CV_64F)
        
    # 色調調整メソッド
    def adjust_brightness(self, image, value):
        """明るさ調整。

        Args:
            image: 入力画像
            value: 明るさの調整値（1.0が元の明るさ）

        Returns:
            明るさ調整された画像
        """
        if self.library == "opencv":
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            v = cv2.multiply(v, value)
            v = np.clip(v, 0, 255).astype(np.uint8)
            hsv = cv2.merge([h, s, v])
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        elif self.library == "pillow":
            enhancer = ImageEnhance.Brightness(image)
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            return enhancer.enhance(value)
        else:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            v = cv2.multiply(v, value)
            v = np.clip(v, 0, 255).astype(np.uint8)
            hsv = cv2.merge([h, s, v])
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
    def adjust_contrast(self, image, value):
        """コントラスト調整。

        Args:
            image: 入力画像
            value: コントラストの調整値（1.0が元のコントラスト）

        Returns:
            コントラスト調整された画像
        """
        if self.library == "opencv":
            alpha = value  # コントラスト制御（1.0-3.0）
            beta = 0       # 明るさ制御（0-100）
            return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        elif self.library == "pillow":
            enhancer = ImageEnhance.Contrast(image)
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            return enhancer.enhance(value)
        else:
            alpha = value  # コントラスト制御（1.0-3.0）
            beta = 0       # 明るさ制御（0-100）
            return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        
    def adjust_saturation(self, image, value):
        """彩度調整。

        Args:
            image: 入力画像
            value: 彩度の調整値（1.0が元の彩度）

        Returns:
            彩度調整された画像
        """
        if self.library == "opencv":
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            s = cv2.multiply(s, value)
            s = np.clip(s, 0, 255).astype(np.uint8)
            hsv = cv2.merge([h, s, v])
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        elif self.library == "pillow":
            enhancer = ImageEnhance.Color(image)
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            return enhancer.enhance(value)
        else:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            s = cv2.multiply(s, value)
            s = np.clip(s, 0, 255).astype(np.uint8)
            hsv = cv2.merge([h, s, v])
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
    # 特殊効果メソッド
    def sepia(self, image):
        """セピア効果。

        Args:
            image: 入力画像

        Returns:
            セピア効果が適用された画像
        """
        if self.library == "opencv":
            # セピアカラーマトリックス
            sepia_kernel = np.array([[0.272, 0.534, 0.131],
                                     [0.349, 0.686, 0.168],
                                     [0.393, 0.769, 0.189]])
            sepia_img = cv2.transform(image, sepia_kernel)
            sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
            return sepia_img
        elif self.library == "pillow":
            # Pillowでのセピア実装
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            return img.convert('L').convert('RGB')
        else:
            # セピアカラーマトリックス
            sepia_kernel = np.array([[0.272, 0.534, 0.131],
                                     [0.349, 0.686, 0.168],
                                     [0.393, 0.769, 0.189]])
            sepia_img = cv2.transform(image, sepia_kernel)
            sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
            return sepia_img
        
    def emboss(self, image):
        """エンボス効果。

        Args:
            image: 入力画像

        Returns:
            エンボス効果が適用された画像
        """
        if self.library == "opencv":
            kernel = np.array([[-2, -1, 0],
