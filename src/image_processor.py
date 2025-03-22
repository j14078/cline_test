#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
画像処理確認アプリの画像処理モジュール。

このモジュールは画像処理ロジックを提供します。
"""

import cv2
import numpy as np
import math
import random
import zlib


class ImageProcessor:
    """画像処理クラス。"""
    
    def __init__(self):
        """初期化メソッド。"""
        # 顔検出のための分類器
        self.face_cascade = None
        self.eye_cascade = None
        
        # テキスト検出のためのEASTモデル
        self.text_detector = None
        
        # QRコード検出用の検出器
        self.qr_detector = cv2.QRCodeDetector()
    
    def apply_processing(self, image, process_type, params=None):
        """画像に指定された処理を適用する。

        Args:
            image: 処理する画像
            process_type: 処理タイプ
            params: 処理パラメータ（オプション）

        Returns:
            処理された画像
        """
        if params is None:
            params = {}
            
        # 元画像のコピーを作成
        result = image.copy()
        
        # 処理タイプに応じて処理を適用
        if process_type == "grayscale":
            result = self._apply_grayscale(result)
        elif process_type == "invert":
            result = self._apply_invert(result)
        elif process_type == "rotate":
            result = self._apply_rotate(result, params)
        elif process_type == "resize":
            result = self._apply_resize(result, params)
        elif process_type == "blur":
            result = self._apply_blur(result, params)
        elif process_type == "sharpen":
            result = self._apply_sharpen(result)
        elif process_type == "edge_detection":
            result = self._apply_edge_detection(result, params)
        elif process_type == "brightness":
            result = self._apply_brightness(result, params)
        elif process_type == "contrast":
            result = self._apply_contrast(result, params)
        elif process_type == "saturation":
            result = self._apply_saturation(result, params)
        elif process_type == "sepia":
            result = self._apply_sepia(result)
        elif process_type == "emboss":
            result = self._apply_emboss(result)
        elif process_type == "mosaic":
            result = self._apply_mosaic(result, params)
        elif process_type == "threshold":
            result = self._apply_threshold(result, params)
        elif process_type == "dilate":
            result = self._apply_dilate(result, params)
        elif process_type == "erode":
            result = self._apply_erode(result, params)
        elif process_type == "flip_horizontal":
            result = self._apply_flip_horizontal(result)
        elif process_type == "flip_vertical":
            result = self._apply_flip_vertical(result)
        elif process_type == "gamma":
            result = self._apply_gamma(result, params)
        elif process_type == "denoise":
            result = self._apply_denoise(result, params)
        elif process_type == "equalize_hist":
            result = self._apply_equalize_hist(result)
        elif process_type == "adjust_red":
            result = self._apply_adjust_color(result, 2, params)  # OpenCV: BGR
        elif process_type == "adjust_green":
            result = self._apply_adjust_color(result, 1, params)  # OpenCV: BGR
        elif process_type == "adjust_blue":
            result = self._apply_adjust_color(result, 0, params)  # OpenCV: BGR
        elif process_type == "segmentation":
            result = self._apply_segmentation(result, params)
        elif process_type == "contour":
            result = self._apply_contour(result, params)
        elif process_type == "posterize":
            result = self._apply_posterize(result, params)
        elif process_type == "watercolor":
            result = self._apply_watercolor(result, params)
        elif process_type == "oil_painting":
            result = self._apply_oil_painting(result, params)
        elif process_type == "sketch":
            result = self._apply_sketch(result, params)
        # モルフォロジー変換系
        elif process_type == "opening":
            result = self._apply_opening(result, params)
        elif process_type == "closing":
            result = self._apply_closing(result, params)
        elif process_type == "tophat":
            result = self._apply_tophat(result, params)
        elif process_type == "blackhat":
            result = self._apply_blackhat(result, params)
        elif process_type == "morphology_gradient":
            result = self._apply_morphology_gradient(result, params)
        # 直線検出系
        elif process_type == "hough_lines":
            result = self._apply_hough_lines(result, params)
        elif process_type == "probabilistic_hough_lines":
            result = self._apply_prob_hough_lines(result, params)
        elif process_type == "hough_circles":
            result = self._apply_hough_circles(result, params)
        # その他の処理
        elif process_type == "perspective_transform":
            result = self._apply_perspective_transform(result, params)
        elif process_type == "template_matching":
            result = self._apply_template_matching(result, params)
        elif process_type == "feature_detection":
            result = self._apply_feature_detection(result, params)
        elif process_type == "face_detection":
            result = self._apply_face_detection(result, params)
        elif process_type == "text_detection":
            result = self._apply_text_detection(result, params)
        elif process_type == "watershed":
            result = self._apply_watershed(result, params)
        # 新しい処理
        elif process_type == "add_gaussian_noise":
            result = self._apply_gaussian_noise(result, params)
        elif process_type == "add_salt_pepper_noise":
            result = self._apply_salt_pepper_noise(result, params)
        elif process_type == "cartoon_effect":
            result = self._apply_cartoon_effect(result, params)
        elif process_type == "vignette_effect":
            result = self._apply_vignette_effect(result, params)
        elif process_type == "color_temperature":
            result = self._apply_color_temperature(result, params)
        elif process_type == "color_extraction":
            result = self._apply_color_extraction(result, params)
        elif process_type == "lens_distortion":
            result = self._apply_lens_distortion(result, params)
        elif process_type == "lens_correction":
            result = self._apply_lens_correction(result, params)
        elif process_type == "miniature_effect":
            result = self._apply_miniature_effect(result, params)
        elif process_type == "document_scan":
            result = self._apply_document_scan(result, params)
        elif process_type == "barcode_detection":
            result = self._apply_barcode_detection(result, params)
        elif process_type == "glitch_effect":
            result = self._apply_glitch_effect(result, params)
        elif process_type == "old_photo_effect":
            result = self._apply_old_photo_effect(result, params)
        elif process_type == "neon_effect":
            result = self._apply_neon_effect(result, params)
        elif process_type == "pixelate":
            result = self._apply_pixelate(result, params)
        elif process_type == "cross_hatching":
            result = self._apply_cross_hatching(result, params)
        
        # 画像の型とデータ範囲を確保
        if result is not None:
            # 最大値と最小値によって値を0-255にスケーリング
            if result.dtype != np.uint8:
                if np.min(result) < 0 or np.max(result) > 255:
                    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
                result = result.astype(np.uint8)
        
        return result
    
    def _apply_grayscale(self, image):
        """グレースケール変換を適用する。

        Args:
            image: 元画像

        Returns:
            処理された画像
        """
        # 画像がすでにグレースケールの場合
        if len(image.shape) == 2:
            return image
            
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def _apply_invert(self, image):
        """色反転を適用する。

        Args:
            image: 元画像

        Returns:
            処理された画像
        """
        return cv2.bitwise_not(image)
    
    def _apply_rotate(self, image, params):
        """回転を適用する。

        Args:
            image: 元画像
            params: パラメータ辞書（angle: 回転角度）

        Returns:
            処理された画像
        """
        angle = params.get("angle", 90)
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, rotation_matrix, (width, height))
    
    def _apply_resize(self, image, params):
        """リサイズを適用する。

        Args:
            image: 元画像
            params: パラメータ辞書（width: 幅, height: 高さ）

        Returns:
            処理された画像
        """
        width = params.get("width", 300)
        height = params.get("height", 300)
        return cv2.resize(image, (width, height))
    
    def _apply_blur(self, image, params):
        """ぼかしを適用する。

        Args:
            image: 元画像
            params: パラメータ辞書（kernel_size: カーネルサイズ, blur_type: ぼかしの種類）

        Returns:
            処理された画像
        """
        kernel_size = params.get("kernel_size", 5)
        blur_type = params.get("blur_type", "gaussian")
        
        if blur_type == "gaussian":
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        elif blur_type == "median":
            return cv2.medianBlur(image, kernel_size)
        elif blur_type == "box":
            return cv2.blur(image, (kernel_size, kernel_size))
        elif blur_type == "bilateral":
            return cv2.bilateralFilter(image, kernel_size, 75, 75)
        
        # デフォルトはガウスぼかし
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def _apply_sharpen(self, image):
        """シャープ化を適用する。

        Args:
            image: 元画像

        Returns:
            処理された画像
        """
        kernel = np.array([[-1, -1, -1], 
                          [-1,  9, -1], 
                          [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)
    
    def _apply_edge_detection(self, image, params):
        """エッジ検出を適用する。

        Args:
            image: 元画像
            params: パラメータ辞書（method: 検出方法）

        Returns:
            処理された画像
        """
        method = params.get("method", "canny")
        
        # グレースケール変換
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        if method == "canny":
            edges = cv2.Canny(gray, 100, 200)
            return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR) if len(image.shape) == 3 else edges
        elif method == "sobel":
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
            sobelxy = cv2.magnitude(sobelx, sobely)
            sobelxy = cv2.normalize(sobelxy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            return cv2.cvtColor(sobelxy, cv2.COLOR_GRAY2BGR) if len(image.shape) == 3 else sobelxy
        elif method == "laplacian":
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            return cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR) if len(image.shape) == 3 else laplacian
        
        # デフォルトはCanny
        edges = cv2.Canny(gray, 100, 200)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR) if len(image.shape) == 3 else edges
    
    def _apply_brightness(self, image, params):
        """明るさ調整を適用する。

        Args:
            image: 元画像
            params: パラメータ辞書（value: 明るさ値）

        Returns:
            処理された画像
        """
        value = params.get("value", 1.5)
        
        # グレースケール画像の場合
        if len(image.shape) == 2:
            return cv2.multiply(image, value).clip(0, 255).astype(np.uint8)
        
        # HSV色空間で明るさ（V）を調整
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.multiply(v, value)
        v = np.clip(v, 0, 255).astype(np.uint8)
        hsv = cv2.merge([h, s, v])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def _apply_contrast(self, image, params):
        """コントラスト調整を適用する。

        Args:
            image: 元画像
            params: パラメータ辞書（value: コントラスト値）

        Returns:
            処理された画像
        """
        value = params.get("value", 1.5)
        return cv2.convertScaleAbs(image, alpha=value, beta=0)
    
    def _apply_saturation(self, image, params):
        """彩度調整を適用する。

        Args:
            image: 元画像
            params: パラメータ辞書（value: 彩度値）

        Returns:
            処理された画像
        """
        value = params.get("value", 1.5)
        
        # グレースケール画像は彩度調整できない
        if len(image.shape) == 2:
            return image
        
        # HSV色空間で彩度（S）を調整
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = cv2.multiply(s, value)
        s = np.clip(s, 0, 255).astype(np.uint8)
        hsv = cv2.merge([h, s, v])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def _apply_sepia(self, image):
        """セピア調を適用する。

        Args:
            image: 元画像

        Returns:
            処理された画像
        """
        # グレースケール画像の場合はRGBに変換
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        sepia_image = cv2.transform(image, kernel)
        return np.clip(sepia_image, 0, 255).astype(np.uint8)
    
    def _apply_emboss(self, image):
        """エンボス効果を適用する。

        Args:
            image: 元画像

        Returns:
            処理された画像
        """
        kernel = np.array([[-2, -1, 0],
                           [-1,  1, 1],
                           [ 0,  1, 2]])
        emboss_image = cv2.filter2D(image, -1, kernel) + 128
        return np.clip(emboss_image, 0, 255).astype(np.uint8)
    
    def _apply_mosaic(self, image, params):
        """モザイク効果を適用する。

        Args:
            image: 元画像
            params: パラメータ辞書（block_size: ブロックサイズ）

        Returns:
            処理された画像
        """
        block_size = params.get("block_size", 10)
        
        height, width = image.shape[:2]
        
        # 縮小してから拡大することでモザイク効果を実現
        small = cv2.resize(image, (width // block_size, height // block_size), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
    
    def _apply_threshold(self, image, params):
        """2値化を適用する。

        Args:
            image: 元画像
            params: パラメータ辞書（thresh_value: 閾値）

        Returns:
            処理された画像
        """
        thresh_value = params.get("thresh_value", 127)
        
        # グレースケール変換
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        _, binary = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY)
        
        # 元画像がカラーだった場合は、結果もカラーに変換
        if len(image.shape) == 3:
            return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        return binary
    
    def _apply_dilate(self, image, params):
        """膨張処理を適用する。

        Args:
            image: 元画像
            params: パラメータ辞書（kernel_size: カーネルサイズ）

        Returns:
            処理された画像
        """
        kernel_size = params.get("kernel_size", 5)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.dilate(image, kernel, iterations=1)
    
    def _apply_erode(self, image, params):
        """収縮処理を適用する。

        Args:
            image: 元画像
            params: パラメータ辞書（kernel_size: カーネルサイズ）

        Returns:
            処理された画像
        """
        kernel_size = params.get("kernel_size", 5)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.erode(image, kernel, iterations=1)
    
    def _apply_flip_horizontal(self, image):
        """左右反転を適用する。

        Args:
            image: 元画像

        Returns:
            処理された画像
        """
        return cv2.flip(image, 1)  # 1は水平方向の反転
    
    def _apply_flip_vertical(self, image):
        """上下反転を適用する。

        Args:
            image: 元画像

        Returns:
            処理された画像
        """
        return cv2.flip(image, 0)  # 0は垂直方向の反転
        
    def _apply_gamma(self, image, params):
        """ガンマ補正を適用する。

        Args:
            image: 元画像
            params: パラメータ辞書（gamma: ガンマ値）

        Returns:
            処理された画像
        """
        gamma = params.get("gamma", 1.5)
        
        # ガンマ補正のルックアップテーブルを作成
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
        
        # ルックアップテーブルを使用して画像を変換
        return cv2.LUT(image, table)
        
    def _apply_denoise(self, image, params):
        """ノイズ除去を適用する。

        Args:
            image: 元画像
            params: パラメータ辞書（strength: 強度）

        Returns:
            処理された画像
        """
        strength = params.get("strength", 10)
        
        # ノイズ除去の強度に基づいてパラメータを調整
        h = strength
        
        # 元画像がグレースケールかカラーかによって適用する関数を変える
        if len(image.shape) == 2:  # グレースケール
            return cv2.fastNlMeansDenoising(image, None, h, 7, 21)
        else:  # カラー
            return cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21)
        
    def _apply_equalize_hist(self, image):
        """ヒストグラム均等化を適用する。

        Args:
            image: 元画像

        Returns:
            処理された画像
        """
        # カラー画像の場合はHSVに変換してVチャンネルを均等化
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            v = cv2.equalizeHist(v)
            hsv = cv2.merge([h, s, v])
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        else:
            # グレースケール画像の場合は直接処理
            return cv2.equalizeHist(image)
            
    def _apply_adjust_color(self, image, channel_index, params):
        """特定の色チャンネルを調整する。

        Args:
            image: 元画像
            channel_index: 調整するチャンネルのインデックス（BGR: 0=Blue, 1=Green, 2=Red）
            params: パラメータ辞書（factor: 調整係数）

        Returns:
            処理された画像
        """
        # グレースケール画像の場合は処理できないので、そのまま返す
        if len(image.shape) == 2:
            return image
            
        factor = params.get("factor", 1.5)
        
        # 画像のチャンネルを分割
        b, g, r = cv2.split(image)
        
        # 指定されたチャンネルを調整
        channels = [b, g, r]
        channels[channel_index] = np.clip(channels[channel_index] * factor, 0, 255).astype(np.uint8)
        
        # チャンネルを結合
        return cv2.merge(channels)
        
    def _apply_segmentation(self, image, params):
        """セグメンテーションを適用する。

        Args:
            image: 元画像
            params: パラメータ辞書（clusters: クラスタ数）

        Returns:
            処理された画像
        """
        clusters = params.get("clusters", 5)
        
        # 画像をfloat32型に変換して、KMeansの入力形式に変形
        Z = image.reshape((-1, 3)).astype(np.float32)
        
        # 終了条件
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        
        # KMeansでクラスタリング
        _, labels, centers = cv2.kmeans(Z, clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # 各ピクセルをクラスタの中心の色に置き換える
        centers = np.uint8(centers)
        res = centers[labels.flatten()]
        
        # 元の画像の形状に戻す
        return res.reshape(image.shape)
        
    def _apply_contour(self, image, params):
        """輪郭抽出を適用する。

        Args:
            image: 元画像
            params: パラメータ辞書（thickness: 線の太さ）

        Returns:
            処理された画像
        """
        thickness = params.get("thickness", 2)
        
        # グレースケール変換
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # 2値化
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # 輪郭抽出
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # 元画像のコピーを作成
        if len(image.shape) == 2:
            result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            result = image.copy()
            
        # 輪郭を描画
        return cv2.drawContours(result, contours, -1, (0, 255, 0), thickness)
        
    def _apply_posterize(self, image, params):
        """ポスタリゼーション（色数を減らす）を適用する。

        Args:
            image: 元画像
            params: パラメータ辞書（levels: 色レベル数）

        Returns:
            処理された画像
        """
        levels = params.get("levels", 4)
        
        # 量子化レベルの計算
        div = 256 // levels
        quant = np.rint(image / div) * div
        
        return quant.astype(np.uint8)
        
    def _apply_watercolor(self, image, params):
        """水彩画風の効果を適用する。

        Args:
            image: 元画像
            params: パラメータ辞書（strength: 強度）

        Returns:
            処理された画像
        """
        strength = params.get("strength", 15)
        
        # バイラテラルフィルタで滑らかにする（エッジを保持）
        bilateral = cv2.bilateralFilter(image, 9, 75, 75)
        
        # エッジを強調
        gray = cv2.cvtColor(bilateral, cv2.COLOR_BGR2GRAY)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, strength)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # エッジを元画像に追加
        result = cv2.subtract(bilateral, edges)
        
        return result
        
    def _apply_oil_painting(self, image, params):
        """油絵風の効果を適用する。

        Args:
            image: 元画像
            params: パラメータ辞書（size: 効果の強さ）

        Returns:
            処理された画像
        """
        size = params.get("size", 7)
        
        # xDoGフィルタ（拡張差分ガウシアン）を適用
        # まず、2つの異なるサイズのガウシアンぼかしを適用
        blur1 = cv2.GaussianBlur(image, (size, size), 0)
        blur2 = cv2.GaussianBlur(image, (size*2+1, size*2+1), 0)
        
        # 差分を取り、コントラストを強化
        result = cv2.addWeighted(blur1, 1.5, blur2, -0.5, 0)
        
        # 彩度を上げる
        if len(result.shape) == 3:  # カラー画像の場合
            hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            s = cv2.add(s, 30)  # 彩度を上げる
            s = np.clip(s, 0, 255).astype(np.uint8)
            hsv = cv2.merge([h, s, v])
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return result
        
    def _apply_sketch(self, image, params):
        """スケッチ風の効果を適用する。

        Args:
            image: 元画像
            params: パラメータ辞書（ksize: カーネルサイズ）

        Returns:
            処理された画像
        """
        ksize = params.get("ksize", 17)
        sigma = params.get("sigma", 0)  # 0の場合、ksize から自動計算
        
        # グレースケール変換
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # 反転とぼかし
        inv = 255 - gray
        blur = cv2.GaussianBlur(inv, (ksize, ksize), sigma)
        
        # ドッジブレンド（グレースケール画像をぼかした反転画像で割る）
        result = cv2.divide(gray, 255 - blur, scale=256)
        
        # 元画像がカラーだった場合は、結果もカラーに変換
        if len(image.shape) == 3:
            return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        return result
        
    # --- モルフォロジー変換系の処理 ---
    
    def _apply_opening(self, image, params):
        """オープニング処理を適用する（先に収縮→次に膨張）。

        Args:
            image: 元画像
            params: パラメータ辞書（kernel_size: カーネルサイズ）

        Returns:
            処理された画像
        """
        kernel_size = params.get("kernel_size", 5)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        
    def _apply_closing(self, image, params):
        """クロージング処理を適用する（先に膨張→次に収縮）。

        Args:
            image: 元画像
            params: パラメータ辞書（kernel_size: カーネルサイズ）

        Returns:
            処理された画像
        """
        kernel_size = params.get("kernel_size", 5)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        
    def _apply_tophat(self, image, params):
        """トップハット変換を適用する（元画像 - オープニング処理）。

        Args:
            image: 元画像
            params: パラメータ辞書（kernel_size: カーネルサイズ）

        Returns:
            処理された画像
        """
        kernel_size = params.get("kernel_size", 5)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
        
    def _apply_blackhat(self, image, params):
        """ブラックハット変換を適用する（クロージング処理 - 元画像）。

        Args:
            image: 元画像
            params: パラメータ辞書（kernel_size: カーネルサイズ）

        Returns:
            処理された画像
        """
        kernel_size = params.get("kernel_size", 5)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
        
    def _apply_morphology_gradient(self, image, params):
        """モルフォロジー勾配を適用する（膨張 - 収縮）。

        Args:
            image: 元画像
            params: パラメータ辞書（kernel_size: カーネルサイズ）

        Returns:
            処理された画像
        """
        kernel_size = params.get("kernel_size", 5)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
        
    # --- 直線検出系の処理 ---
    
    def _apply_hough_lines(self, image, params):
        """ハフ変換による直線検出を適用する。

        Args:
            image: 元画像
            params: パラメータ辞書（threshold: 閾値, min_line_length: 最小線長, max_line_gap: 最大線間隔）

        Returns:
            処理された画像
        """
        threshold = params.get("threshold", 100)
        
        # グレースケール変換とエッジ検出
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # ハフ変換による直線検出
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold)
        
        # 結果画像の準備
        if len(image.shape) == 2:
            result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            result = image.copy()
            
        # 直線が検出されなかった場合
        if lines is None:
            return result
            
        # 検出された直線を描画
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
        return result
        
    def _apply_prob_hough_lines(self, image, params):
        """確率的ハフ変換による直線検出を適用する。

        Args:
            image: 元画像
            params: パラメータ辞書（threshold: 閾値, min_line_length: 最小線長, max_line_gap: 最大線間隔）

        Returns:
            処理された画像
        """
        threshold = params.get("threshold", 50)
        min_line_length = params.get("min_line_length", 50)
        max_line_gap = params.get("max_line_gap", 10)
        
        # グレースケール変換とエッジ検出
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # 確率的ハフ変換による直線検出
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
        
        # 結果画像の準備
        if len(image.shape) == 2:
            result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            result = image.copy()
            
        # 直線が検出されなかった場合
        if lines is None:
            return result
            
        # 検出された直線を描画
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
        return result
        
    def _apply_hough_circles(self, image, params):
        """ハフ変換による円検出を適用する。

        Args:
            image: 元画像
            params: パラメータ辞書（min_radius: 最小半径, max_radius: 最大半径）

        Returns:
            処理された画像
        """
        min_radius = params.get("min_radius", 10)
        max_radius = params.get("max_radius", 100)
        
        # グレースケール変換
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # ノイズ低減
        gray = cv2.medianBlur(gray, 5)
        
        # ハフ変換による円検出
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=50, param2=30, minRadius=min_radius, maxRadius=max_radius)
        
        # 結果画像の準備
        if len(image.shape) == 2:
            result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            result = image.copy()
            
        # 円が検出されなかった場合
        if circles is None:
            return result
            
        # 検出された円を描画
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # 円の中心
            cv2.circle(result, (i[0], i[1]), 2, (0, 255, 0), 3)
            # 円の輪郭
            cv2.circle(result, (i[0], i[1]), i[2], (0, 0, 255), 3)
            
        return result
        
    # --- その他の処理 ---
    
    def _apply_perspective_transform(self, image, params):
        """透視変換を適用する。

        Args:
            image: 元画像
            params: パラメータ辞書（angle: 角度）

        Returns:
            処理された画像
        """
        angle = params.get("angle", 30)
        height, width = image.shape[:2]
        
        # 元の4点の座標（左上、右上、右下、左下）
        pts1 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        
        # 変換後の4点の座標
        offset = int(width * math.tan(math.radians(angle)) / 4)
        pts2 = np.float32([[offset, 0], [width-offset, 0], [width, height], [0, height]])
        
        # 透視変換行列を取得
        M = cv2.getPerspectiveTransform(pts1, pts2)
        
        # 透視変換を適用
        result = cv2.warpPerspective(image, M, (width, height))
        
        return result
        
    def _apply_template_matching(self, image, params):
        """テンプレートマッチングを適用する。

        Args:
            image: 元画像
            params: パラメータ辞書（template: テンプレート画像）

        Returns:
            テンプレートマッチングの結果画像
        """
        # テンプレート画像がない場合は元画像の一部を使用
        height, width = image.shape[:2]
        
        # 中央部分をテンプレートとして使用
        template_size = min(width, height) // 4
        template_x = width // 2 - template_size // 2
        template_y = height // 2 - template_size // 2
        template = image[template_y:template_y+template_size, template_x:template_x+template_size]
        
        # 元画像のコピーを作成
        result = image.copy()
        
        # テンプレートマッチング
        method = cv2.TM_CCOEFF_NORMED
        res = cv2.matchTemplate(image, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        # 適合度の閾値
        threshold = 0.8
        
        # マッチング結果の描画
        locations = np.where(res >= threshold)
        h, w = template.shape[:2]
        
        for pt in zip(*locations[::-1]):
            cv2.rectangle(result, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
            
        # テンプレート自体を左上に表示
        h_temp, w_temp = template.shape[:2]
        result[10:10+h_temp, 10:10+w_temp] = template
        cv2.rectangle(result, (10, 10), (10+w_temp, 10+h_temp), (0, 0, 255), 2)
            
        return result
        
    def _apply_feature_detection(self, image, params):
        """特徴点検出を適用する。

        Args:
            image: 元画像
            params: パラメータ辞書（method: 特徴点検出方法）

        Returns:
            処理された画像
        """
        method = params.get("method", "sift")
        
        # グレースケール変換
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # 結果画像の準備
        if len(image.shape) == 2:
            result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            result = image.copy()
            
        # 特徴点検出
        if method == "sift":
            # SIFTの初期化
            sift = cv2.SIFT_create()
            keypoints = sift.detect(gray, None)
            # 特徴点の描画
            result = cv2.drawKeypoints(gray, keypoints, None, color=(0, 255, 0), 
                                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                
        elif method == "orb":
            # ORBの初期化
            orb = cv2.ORB_create()
            keypoints = orb.detect(gray, None)
            # 特徴点の描画
            result = cv2.drawKeypoints(gray, keypoints, None, color=(0, 255, 0), 
                                      flags=0)
                
        return result
        
    def _apply_face_detection(self, image, params):
        """顔検出を適用する。

        Args:
            image: 元画像
            params: パラメータ辞書

        Returns:
            処理された画像
        """
        # 元画像のコピーを作成
        result = image.copy()
        
        # グレースケール変換
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            # カラー画像に変換（検出結果の表示用）
            if len(result.shape) == 2:
                result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        # 顔検出器のロード
        if self.face_cascade is None:
            try:
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            except Exception:
                # ファイルが見つからない場合はエラーメッセージを表示
                return cv2.putText(result, "Cascade classifiers not found", (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 顔検出
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # 検出された顔に矩形を描画
        for (x, y, w, h) in faces:
            cv2.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # 顔領域内で目を検出
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = result[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            
            # 検出された目に円を描画
            for (ex, ey, ew, eh) in eyes:
                cv2.circle(roi_color, (ex + ew//2, ey + eh//2), ew//2, (0, 255, 0), 2)
                
        return result
        
    def _apply_text_detection(self, image, params):
        """テキスト検出を適用する。

        Args:
            image: 元画像
            params: パラメータ辞書

        Returns:
            処理された画像
        """
        # 元画像のコピーを作成
        result = image.copy()
        
        # シンプルなテキスト検出用にエッジ検出とモルフォロジー処理を使用
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            # カラー画像に変換（検出結果の表示用）
            if len(result.shape) == 2:
                result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        # 2値化
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        # モルフォロジー処理（テキストブロックの検出）
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # 輪郭検出
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 輪郭を描画（面積でフィルタリング）
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:  # 小さすぎる輪郭は無視
                x, y, w, h = cv2.boundingRect(cnt)
                if w > h:  # 幅が高さより大きい（テキストらしい）
                    cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
        return result
        
    def _apply_watershed(self, image, params):
        """マーカー付き分水嶺法を適用する。

        Args:
            image: 元画像
            params: パラメータ辞書

        Returns:
            処理された画像
        """
        # 元画像のコピーを作成
        result = image.copy()
        
        # グレースケール変換
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            # カラー画像に変換（結果表示用）
            result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # ノイズ除去と2値化
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # ノイズ除去
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # 確実な背景領域
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # 確実な前景領域を見つける（距離変換）
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        
        # 不明な領域を見つける
        sure_fg = sure_fg.astype(np.uint8)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # マーカーの作成
        _, markers = cv2.connectedComponents(sure_fg)
        
        # すべてのマーカーに1を加える
        markers = markers + 1
        
        # 不明な領域を0にマーク
        markers[unknown == 255] = 0
        
        # 分水嶺アルゴリズムの適用
        if len(image.shape) == 3:
            markers = cv2.watershed(image, markers)
        else:
            # グレースケール画像の場合、一時的にカラーに変換
            color_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            markers = cv2.watershed(color_img, markers)
        
        # 境界線を赤色でマーク
        result[markers == -1] = [0, 0, 255]
        
        return result
        
    # --- 新しく追加する処理 ---
    
    def _apply_gaussian_noise(self, image, params):
        """ガウシアンノイズを追加する。

        Args:
            image: 元画像
            params: パラメータ辞書（mean: 平均, sigma: 標準偏差）

        Returns:
            処理された画像
        """
        mean = params.get("mean", 0)
        sigma = params.get("sigma", 25)
        
        # ノイズの生成
        row, col = image.shape[:2]
        gauss = np.random.normal(mean, sigma, (row, col))
        
        # カラー画像の場合は3チャンネル分のノイズを生成
        if len(image.shape) == 3:
            gauss = np.random.normal(mean, sigma, (row, col, 3))
        
        # ノイズを加える
        noisy = image + gauss
        
        # 値を0-255に収める
        noisy = np.clip(noisy, 0, 255)
        
        return noisy.astype(np.uint8)
    
    def _apply_salt_pepper_noise(self, image, params):
        """ソルト＆ペッパーノイズを追加する。

        Args:
            image: 元画像
            params: パラメータ辞書（amount: ノイズの量）

        Returns:
            処理された画像
        """
        amount = params.get("amount", 0.05)
        
        # 元画像のコピーを作成
        noisy = image.copy()
        
        # ノイズの総数（画素数 × ノイズの割合）
        row, col = image.shape[:2]
        noise_pixels = int(amount * row * col)
        
        # ソルトノイズ（白点）の追加
        salt_coords = [np.random.randint(0, i-1, noise_pixels//2) for i in image.shape[:2]]
        if len(image.shape) == 3:
            noisy[salt_coords[0], salt_coords[1], :] = 255
        else:
            noisy[salt_coords[0], salt_coords[1]] = 255
            
        # ペッパーノイズ（黒点）の追加
        pepper_coords = [np.random.randint(0, i-1, noise_pixels//2) for i in image.shape[:2]]
        if len(image.shape) == 3:
            noisy[pepper_coords[0], pepper_coords[1], :] = 0
        else:
            noisy[pepper_coords[0], pepper_coords[1]] = 0
            
        return noisy
    
    def _apply_cartoon_effect(self, image, params):
        """カートゥーン(漫画)効果を適用する。

        Args:
            image: 元画像
            params: パラメータ辞書（edge_size: エッジの太さ）

        Returns:
            処理された画像
        """
        # グレースケールの場合はカラーに変換
        if len(image.shape) == 2:
            color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            color_image = image.copy()
            
        edge_size = params.get("edge_size", 7)
        
        # エッジ検出のためのグレースケール変換
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        
        # エッジ検出
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                     cv2.THRESH_BINARY, edge_size, 5)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # 色の量子化（色数を減らす）
        color = cv2.bilateralFilter(color_image, 9, 300, 300)
        
        # エッジと色を合成
        cartoon = cv2.bitwise_and(color, 255 - edges)
        
        return cartoon
    
    def _apply_vignette_effect(self, image, params):
        """ビネット効果（周辺減光）を適用する。

        Args:
            image: 元画像
            params: パラメータ辞書（strength: 効果の強さ）

        Returns:
            処理された画像
        """
        strength = params.get("strength", 0.5)
        
        # 画像の中心と半径を計算
        rows, cols = image.shape[:2]
        center_x, center_y = cols // 2, rows // 2
        
        # 中心からの距離マップを作成
        x, y = np.meshgrid(np.arange(cols), np.arange(rows))
        dist_map = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # 正規化した距離マップ
        max_dist = np.sqrt(center_x**2 + center_y**2)
        norm_dist_map = dist_map / max_dist
        
        # ビネットマスクを作成
        mask = 1 - strength * norm_dist_map
        mask = np.clip(mask, 0, 1)
        
        # マスクを適用
        if len(image.shape) == 3:
            # カラー画像の場合は3チャンネル用のマスクに拡張
            mask = np.dstack([mask] * 3)
        
        # マスクを適用して周辺を暗くする
        vignette = (image * mask).astype(np.uint8)
        
        return vignette
    
    def _apply_color_temperature(self, image, params):
        """色温度を調整する。

        Args:
            image: 元画像
            params: パラメータ辞書（temperature: 色温度（-100:青, 0:中間, 100:赤））

        Returns:
            処理された画像
        """
        # グレースケールの場合はカラーに変換
        if len(image.shape) == 2:
            return image
            
        temperature = params.get("temperature", 0)
        
        # 色温度に応じて各チャンネルの調整量を計算
        # 負の値は青色を強く、正の値は赤色を強くする
        b_factor = 1.0 - temperature / 200.0 if temperature > 0 else 1.0
        r_factor = 1.0 + temperature / 200.0 if temperature > 0 else 1.0
        g_factor = 1.0
        
        # 逆に青みを増す場合
        if temperature < 0:
            b_factor = 1.0 - temperature / 200.0  # 温度が負なので加算になる
            r_factor = 1.0
        
        # 画像のチャンネルを分離して調整
        b, g, r = cv2.split(image)
        
        # 各チャンネルを調整
        b = np.clip(b * b_factor, 0, 255).astype(np.uint8)
        g = np.clip(g * g_factor, 0, 255).astype(np.uint8)
        r = np.clip(r * r_factor, 0, 255).astype(np.uint8)
        
        # チャンネルを結合して結果を返す
        return cv2.merge([b, g, r])
    
    def _apply_color_extraction(self, image, params):
        """特定色抽出を適用する。

        Args:
            image: 元画像
            params: パラメータ辞書（hue: 色相(0-179), range: 抽出範囲）

        Returns:
            処理された画像
        """
        # グレースケールの場合はカラーに変換して処理
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
        hue = params.get("hue", 0)  # 0-179 (OpenCVのHSVは0-179)
        hue_range = params.get("range", 15)
        
        # HSV色空間に変換
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 色相の範囲を設定
        lower_bound = np.array([max(0, hue - hue_range), 50, 50])
        upper_bound = np.array([min(179, hue + hue_range), 255, 255])
        
        # マスクを作成
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # マスクを使って元画像から特定色を抽出
        result = cv2.bitwise_and(image, image, mask=mask)
        
        return result
    
    def _apply_lens_distortion(self, image, params):
        """レンズ歪みを追加する。

        Args:
            image: 元画像
            params: パラメータ辞書（strength: 歪みの強さ）

        Returns:
            処理された画像
        """
        strength = params.get("strength", 0.5)
        
        # 画像のサイズと中心
        height, width = image.shape[:2]
        center_x, center_y = width // 2, height // 2
        radius = min(center_x, center_y)
        
        # マップの初期化
        map_x = np.zeros((height, width), np.float32)
        map_y = np.zeros((height, width), np.float32)
        
        # 歪みマップを計算
        for y in range(height):
            for x in range(width):
                # 中心からの距離を計算
                dx = (x - center_x) / radius
                dy = (y - center_y) / radius
                r = np.sqrt(dx*dx + dy*dy)
                
                # 新しい座標を計算
                if r == 0:
                    map_x[y, x] = x
                    map_y[y, x] = y
                else:
                    # 樽型歪み
                    theta = np.arctan2(dy, dx)
                    r_new = r * (1 + strength * r * r)
                    map_x[y, x] = center_x + radius * r_new * np.cos(theta)
                    map_y[y, x] = center_y + radius * r_new * np.sin(theta)
        
        # 座標変換を適用
        return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
    
    def _apply_lens_correction(self, image, params):
        """レンズ歪み補正を適用する。

        Args:
            image: 元画像
            params: パラメータ辞書（strength: 補正の強さ）

        Returns:
            処理された画像
        """
        strength = params.get("strength", 0.5)
        
        # 画像のサイズと中心
        height, width = image.shape[:2]
        center_x, center_y = width // 2, height // 2
        radius = min(center_x, center_y)
        
        # マップの初期化
        map_x = np.zeros((height, width), np.float32)
        map_y = np.zeros((height, width), np.float32)
        
        # 歪み補正マップを計算
        for y in range(height):
            for x in range(width):
                # 中心からの距離を計算
                dx = (x - center_x) / radius
                dy = (y - center_y) / radius
                r = np.sqrt(dx*dx + dy*dy)
                
                # 新しい座標を計算
                if r == 0:
                    map_x[y, x] = x
                    map_y[y, x] = y
                else:
                    # 糸巻き型歪み補正（樽型歪みの逆）
                    theta = np.arctan2(dy, dx)
                    r_new = r / (1 + strength * r * r)
                    map_x[y, x] = center_x + radius * r_new * np.cos(theta)
                    map_y[y, x] = center_y + radius * r_new * np.sin(theta)
        
        # 座標変換を適用
        return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
    
    def _apply_miniature_effect(self, image, params):
        """ミニチュア効果を適用する（チルトシフト効果）。

        Args:
            image: 元画像
            params: パラメータ辞書（position: フォーカス位置(0-1), width: フォーカス幅）

        Returns:
            処理された画像
        """
        position = params.get("position", 0.5)  # 0-1
        blur_width = params.get("width", 0.2)  # 0-1
        
        # グレースケールの場合は複製してからカラーに変換（ぼかし計算用）
        result = image.copy()
        
        # 画像のサイズを取得
        height, width = image.shape[:2]
        
        # フォーカス領域の境界を計算
        focus_center = int(height * position)
        focus_top = max(0, int(focus_center - height * blur_width / 2))
        focus_bottom = min(height, int(focus_center + height * blur_width / 2))
        
        # 上側と下側の領域をぼかす
        if focus_top > 0:
            top_part = image[:focus_top]
            top_blurred = cv2.GaussianBlur(top_part, (31, 31), 0)
            result[:focus_top] = top_blurred
            
        if focus_bottom < height:
            bottom_part = image[focus_bottom:]
            bottom_blurred = cv2.GaussianBlur(bottom_part, (31, 31), 0)
            result[focus_bottom:] = bottom_blurred
            
        # 彩度を上げる
        if len(result.shape) == 3:
            hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            s = np.clip(s * 1.3, 0, 255).astype(np.uint8)
            hsv = cv2.merge([h, s, v])
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
        return result
    
    def _apply_document_scan(self, image, params):
        """文書スキャン最適化を適用する。

        Args:
            image: 元画像
            params: パラメータ辞書（threshold: 閾値）

        Returns:
            処理された画像
        """
        threshold = params.get("threshold", 150)
        
        # グレースケール変換
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # エッジ検出のためにグレースケール画像を平滑化
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # アダプティブしきい値処理で文字を強調
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # シャープ化
        kernel = np.array([[-1, -1, -1], 
                          [-1,  9, -1], 
                          [-1, -1, -1]])
        sharp = cv2.filter2D(binary, -1, kernel)
        
        # コントラスト強調
        alpha = 1.5  # コントラスト強調ファクター
        beta = 10    # 輝度調整
        adjusted = cv2.convertScaleAbs(sharp, alpha=alpha, beta=beta)
        
        # 元画像がカラーだった場合、結果もカラーに変換（スキャンの見た目を保持）
        if len(image.shape) == 3:
            # グレースケール強調画像を緑っぽい色調にする（通常のスキャン見た目）
            paper_color = np.ones_like(image) * np.array([235, 243, 235], dtype=np.uint8)
            
            # 白い部分をスキャン風の色に置き換え
            mask = adjusted == 255
            colored = np.zeros_like(image)
            colored[mask] = paper_color[mask]
            
            # 黒い文字はそのまま
            mask_inv = adjusted != 255
            colored[mask_inv] = 0
            
            return colored
        else:
            return adjusted
    
    def _apply_barcode_detection(self, image, params):
        """バーコード/QRコード検出を適用する。

        Args:
            image: 元画像
            params: パラメータ辞書

        Returns:
            処理された画像（検出結果表示）
        """
        # 元画像のコピーを作成
        result = image.copy()
        
        # グレースケール変換
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            if len(result.shape) == 2:
                result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        # QRコード検出
        ret_qr, points = self.qr_detector.detect(gray)
        decoded_info, points, _ = self.qr_detector.detectAndDecode(gray)
        
        # 検出された場合は枠を描画
        if ret_qr and points is not None:
            points = points.astype(np.int32)
            # QRコードを多角形で囲む
            cv2.polylines(result, [points], True, (0, 255, 0), 2)
            
            # デコードされた情報を表示
            if decoded_info:
                info_text = decoded_info if len(decoded_info) < 20 else decoded_info[:17] + "..."
                cv2.putText(result, f"QR: {info_text}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # バーコード検出（簡易的な実装 - エッジ検出とモルフォロジー処理による）
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 200, apertureSize=3)
        
        # 長方形検出用にモルフォロジー処理
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges.copy(), kernel, iterations=1)
        
        # 輪郭検出
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # バーコードらしい領域を検出
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # バーコードの特徴：幅が高さより大きく、ある程度のアスペクト比を持つ
            aspect_ratio = float(w) / h
            if 2.0 < aspect_ratio < 5.0 and w > 100:
                cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(result, "Barcode", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return result
    
    def _apply_glitch_effect(self, image, params):
        """グリッチ効果を適用する。

        Args:
            image: 元画像
            params: パラメータ辞書（strength: 効果の強さ）

        Returns:
            処理された画像
        """
        strength = params.get("strength", 0.5)
        
        # グレースケールの場合はカラーに変換
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
        # 元画像のコピーを作成
        result = image.copy()
        
        # チャンネルを分離
        b, g, r = cv2.split(result)
        
        # ランダムなグリッチを作成
        height, width = image.shape[:2]
        
        # グリッチの数を強度に応じて調整
        num_glitches = int(10 * strength)
        
        # 複数のグリッチを適用
        for _ in range(num_glitches):
            # ランダムな位置にグリッチを作成
            y_start = np.random.randint(0, height - 10)
            y_end = min(y_start + np.random.randint(5, 20), height - 1)
            x_offset = np.random.randint(-20, 20)
            
            # 選択された範囲をシフト
            if x_offset > 0:
                # 赤チャンネルを右にシフト
                r[y_start:y_end, x_offset:] = r[y_start:y_end, :-x_offset]
                r[y_start:y_end, :x_offset] = 0
            else:
                # 赤チャンネルを左にシフト
                x_offset = abs(x_offset)
                r[y_start:y_end, :-x_offset] = r[y_start:y_end, x_offset:]
                r[y_start:y_end, -x_offset:] = 0
                
            # 強度が高い場合は、青チャンネルもシフト
            if strength > 0.7:
                y_start = np.random.randint(0, height - 10)
                y_end = min(y_start + np.random.randint(5, 20), height - 1)
                x_offset = np.random.randint(-20, 20)
                
                if x_offset > 0:
                    b[y_start:y_end, x_offset:] = b[y_start:y_end, :-x_offset]
                    b[y_start:y_end, :x_offset] = 0
                else:
                    x_offset = abs(x_offset)
                    b[y_start:y_end, :-x_offset] = b[y_start:y_end, x_offset:]
                    b[y_start:y_end, -x_offset:] = 0
        
        # ランダムなノイズを追加
        if strength > 0.3:
            noise = np.random.randint(0, 255, (height, width), dtype=np.uint8)
            noise_mask = np.random.random((height, width)) < (0.05 * strength)
            r[noise_mask] = noise[noise_mask]
        
        # チャンネルを結合
        result = cv2.merge([b, g, r])
        
        return result
    
    def _apply_old_photo_effect(self, image, params):
        """古い写真効果を適用する。

        Args:
            image: 元画像
            params: パラメータ辞書（age: 古さの強度, noise: ノイズの量）

        Returns:
            処理された画像
        """
        age = params.get("age", 0.7)
        noise_amount = params.get("noise", 0.3)
        
        # グレースケールの場合はカラーに変換
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
        # セピア効果
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        sepia = cv2.transform(image, kernel)
        
        # コントラストを下げる
        contrast = 0.8 - age * 0.4  # ageが高いほどコントラストは低下
        brightness = 30  # 少し明るく
        faded = cv2.convertScaleAbs(sepia, alpha=contrast, beta=brightness)
        
        # ビネット効果（周辺を暗く）
        rows, cols = faded.shape[:2]
        center_x, center_y = cols // 2, rows // 2
        
        # 中心からの距離マップを作成
        x, y = np.meshgrid(np.arange(cols), np.arange(rows))
        dist_map = ((x - center_x)**2 + (y - center_y)**2).astype(np.float32)
        max_dist = max(center_x, center_y)**2
        vignette_mask = 1 - dist_map / max_dist * (0.5 * age)
        vignette_mask = np.clip(vignette_mask, 0, 1)
        
        # 3チャンネル用のマスクに拡張
        vignette_mask = np.dstack([vignette_mask] * 3)
        
        # ビネット効果を適用
        faded = (faded * vignette_mask).astype(np.uint8)
        
        # ランダムな傷を追加
        if age > 0.5:
            num_scratches = int(10 * age)
            for _ in range(num_scratches):
                x1 = np.random.randint(0, cols)
                y1 = np.random.randint(0, rows)
                length = np.random.randint(20, 100)
                angle = np.random.randint(0, 360)
                thickness = np.random.randint(1, 3)
                
                # 線の終点を計算
                x2 = int(x1 + length * np.cos(np.radians(angle)))
                y2 = int(y1 + length * np.sin(np.radians(angle)))
                
                # 画像の範囲内に収める
                x2 = min(max(0, x2), cols - 1)
                y2 = min(max(0, y2), rows - 1)
                
                # 傷（白い線）を描画
                cv2.line(faded, (x1, y1), (x2, y2), (230, 230, 210), thickness)
        
        # ノイズを追加
        if noise_amount > 0:
            noise = np.random.randint(0, 50, faded.shape, dtype=np.uint8)
            noise_mask = np.random.random(faded.shape[:2]) < (noise_amount * 0.3)
            noise_mask = np.dstack([noise_mask] * 3)
            faded[noise_mask] = noise[noise_mask]
        
        return faded
    
    def _apply_neon_effect(self, image, params):
        """ネオン効果を適用する。

        Args:
            image: 元画像
            params: パラメータ辞書（glow: 輝き強度）

        Returns:
            処理された画像
        """
        glow = params.get("glow", 0.7)
        
        # グレースケールの場合はカラーに変換
        if len(image.shape) == 2:
            color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            color_image = image.copy()
            
        # エッジ検出
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        
        # エッジをぼかす
        blur_size = int(5 + glow * 10)  # 輝き強度に応じてぼかしを調整
        if blur_size % 2 == 0:
            blur_size += 1  # 奇数にする
        blurred_edges = cv2.GaussianBlur(edges, (blur_size, blur_size), 0)
        
        # 色を追加
        colored_edges = np.zeros_like(color_image)
        
        # 画像を3つの領域に分け、それぞれに異なる色を適用
        h, w = edges.shape
        section1 = h // 3
        section2 = 2 * h // 3
        
        # 上部：青/シアン
        colored_edges[:section1][blurred_edges[:section1] > 0] = [255, 255, 0]  # シアン
        
        # 中部：マゼンタ/ピンク
        colored_edges[section1:section2][blurred_edges[section1:section2] > 0] = [255, 0, 255]  # マゼンタ
        
        # 下部：イエロー
        colored_edges[section2:][blurred_edges[section2:] > 0] = [0, 255, 255]  # イエロー
        
        # 暗い背景を作成
        dark_bg = cv2.convertScaleAbs(color_image, alpha=0.1, beta=0)
        
        # 輝きのある部分を重ねる
        neon = cv2.addWeighted(dark_bg, 1, colored_edges, 1, 0)
        
        return neon
    
    def _apply_pixelate(self, image, params):
        """ピクセル化効果を適用する。

        Args:
            image: 元画像
            params: パラメータ辞書（pixel_size: ピクセルサイズ）

        Returns:
            処理された画像
        """
        pixel_size = params.get("pixel_size", 10)
        
        height, width = image.shape[:2]
        
        # 元画像を小さくリサイズ
        temp = cv2.resize(image, (width // pixel_size, height // pixel_size), 
                         interpolation=cv2.INTER_LINEAR)
        
        # 元のサイズに戻す（ピクセル効果）
        return cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)
    
    def _apply_cross_hatching(self, image, params):
        """クロスハッチング効果を適用する。

        Args:
            image: 元画像
            params: パラメータ辞書（line_spacing: 線の間隔）

        Returns:
            処理された画像
        """
        line_spacing = params.get("line_spacing", 6)
        
        # グレースケール変換
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # 結果画像（白背景）
        if len(image.shape) == 3:
            result = np.ones_like(image) * 255
        else:
            result = np.ones_like(gray) * 255
            
        height, width = gray.shape
        
        # 明るさに応じた線を描画
        for y in range(0, height, line_spacing):
            for x in range(0, width, line_spacing):
                # 現在の位置の明るさを取得
                brightness = gray[y, x]
                
                # 明るさに応じてハッチングの種類を決定
                if brightness < 50:  # 暗い部分（複数方向の線）
                    if len(result.shape) == 3:
                        cv2.line(result, (x, y), (x + line_spacing, y + line_spacing), (0, 0, 0), 1)
                        cv2.line(result, (x + line_spacing, y), (x, y + line_spacing), (0, 0, 0), 1)
                        cv2.line(result, (x, y + line_spacing//2), (x + line_spacing, y + line_spacing//2), (0, 0, 0), 1)
                    else:
                        cv2.line(result, (x, y), (x + line_spacing, y + line_spacing), 0, 1)
                        cv2.line(result, (x + line_spacing, y), (x, y + line_spacing), 0, 1)
                        cv2.line(result, (x, y + line_spacing//2), (x + line_spacing, y + line_spacing//2), 0, 1)
                
                elif brightness < 100:  # やや暗い部分（斜め線2方向）
                    if len(result.shape) == 3:
                        cv2.line(result, (x, y), (x + line_spacing, y + line_spacing), (0, 0, 0), 1)
                        cv2.line(result, (x + line_spacing, y), (x, y + line_spacing), (0, 0, 0), 1)
                    else:
                        cv2.line(result, (x, y), (x + line_spacing, y + line_spacing), 0, 1)
                        cv2.line(result, (x + line_spacing, y), (x, y + line_spacing), 0, 1)
                
                elif brightness < 150:  # 中間（斜め線1方向）
                    if len(result.shape) == 3:
                        cv2.line(result, (x, y), (x + line_spacing, y + line_spacing), (0, 0, 0), 1)
                    else:
                        cv2.line(result, (x, y), (x + line_spacing, y + line_spacing), 0, 1)
                
                elif brightness < 200:  # やや明るい部分（水平線）
                    if len(result.shape) == 3:
                        cv2.line(result, (x, y + line_spacing//2), (x + line_spacing, y + line_spacing//2), (0, 0, 0), 1)
                    else:
                        cv2.line(result, (x, y + line_spacing//2), (x + line_spacing, y + line_spacing//2), 0, 1)
                
                # 明るい部分（200以上）は何も描画しない
        
        return result
