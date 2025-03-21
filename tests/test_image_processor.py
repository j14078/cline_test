#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ImageProcessorクラスのテスト。
"""

import os
import sys
import unittest
import numpy as np
import cv2
import pytest
from PIL import Image

# srcディレクトリをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.image_processor import ImageProcessor


class TestImageProcessor:
    """ImageProcessorクラスのテスト。"""
    
    def setup_method(self):
        """各テストメソッドの前に実行される。"""
        self.processor = ImageProcessor()
        
        # テスト用の画像を作成
        self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.test_image[25:75, 25:75] = [255, 255, 255]  # 白い四角形
        
        # テンプレート画像を保存
        self.template_path = "template.png"
        cv2.imwrite(self.template_path, self.test_image[25:75, 25:75])
        
    def teardown_method(self):
        """各テストメソッドの後に実行される。"""
        # テンプレート画像を削除
        if os.path.exists(self.template_path):
            os.remove(self.template_path)
        
    def test_init(self):
        """初期化のテスト。"""
        assert self.processor.library == "opencv"
        assert len(self.processor.supported_extensions) > 0
        
    def test_grayscale(self):
        """グレースケール変換のテスト。"""
        result = self.processor.grayscale(self.test_image)
        assert result.shape == (100, 100) or result.shape == (100, 100, 1)
        
    def test_invert(self):
        """画像反転のテスト。"""
        result = self.processor.invert(self.test_image)
        # 元画像の白い部分が黒くなっているか確認
        assert np.all(result[25:75, 25:75, 0] == 0)
        assert np.all(result[25:75, 25:75, 1] == 0)
        assert np.all(result[25:75, 25:75, 2] == 0)
        
    def test_rotate(self):
        """画像回転のテスト。"""
        result = self.processor.rotate(self.test_image, 90)
        assert result.shape == self.test_image.shape
        
    def test_resize(self):
        """リサイズのテスト。"""
        result = self.processor.resize(self.test_image, 50, 50)
        assert result.shape == (50, 50, 3)
        
    def test_blur(self):
        """ぼかし処理のテスト。"""
        result = self.processor.blur(self.test_image, 5)
        assert result.shape == self.test_image.shape
        
    def test_sharpen(self):
        """シャープ化処理のテスト。"""
        result = self.processor.sharpen(self.test_image)
        assert result.shape == self.test_image.shape
        
    def test_edge_detection(self):
        """エッジ検出処理のテスト。"""
        result = self.processor.edge_detection(self.test_image, "canny")
        assert result.shape == (100, 100) or result.shape == (100, 100, 1)
        
    def test_adjust_brightness(self):
        """明るさ調整のテスト。"""
        result = self.processor.adjust_brightness(self.test_image, 1.5)
        assert result.shape == self.test_image.shape
        
    def test_adjust_contrast(self):
        """コントラスト調整のテスト。"""
        result = self.processor.adjust_contrast(self.test_image, 1.5)
        assert result.shape == self.test_image.shape
        
    def test_adjust_saturation(self):
        """彩度調整のテスト。"""
        result = self.processor.adjust_saturation(self.test_image, 1.5)
        assert result.shape == self.test_image.shape
        
    def test_sepia(self):
        """セピア効果のテスト。"""
        result = self.processor.sepia(self.test_image)
        assert result.shape == self.test_image.shape
        
    def test_emboss(self):
        """エンボス効果のテスト。"""
        result = self.processor.emboss(self.test_image)
        assert result.shape == self.test_image.shape
        
    def test_mosaic(self):
        """モザイク効果のテスト。"""
        result = self.processor.mosaic(self.test_image, 10)
        assert result.shape == self.test_image.shape
        
    def test_threshold(self):
        """閾値処理のテスト。"""
        result = self.processor.threshold(self.test_image, 127)
        assert result.shape == (100, 100) or result.shape == (100, 100, 1)
        
    def test_dilate(self):
        """膨張処理のテスト。"""
        result = self.processor.dilate(self.test_image, 5)
        assert result.shape == self.test_image.shape
        
    def test_erode(self):
        """収縮処理のテスト。"""
        result = self.processor.erode(self.test_image, 5)
        assert result.shape == self.test_image.shape
        
    def test_template_matching(self):
        """テンプレートマッチングのテスト。"""
        result = self.processor.template_matching(self.test_image, self.template_path)
        assert result.shape == self.test_image.shape
        
    def test_feature_detection(self):
        """特徴点検出のテスト。"""
        result = self.processor.feature_detection(self.test_image, "SIFT")
        assert result.shape == self.test_image.shape
        
    def test_apply_processing(self):
        """処理適用のテスト。"""
        # グレースケール変換
        result = self.processor.apply_processing(self.test_image, "grayscale")
        assert result.shape == (100, 100) or result.shape == (100, 100, 1)
        
        # リサイズ
        result = self.processor.apply_processing(self.test_image, "resize", {"width": 50, "height": 50})
        assert result.shape == (50, 50, 3)
        
        # サポートされていない処理タイプ
        with pytest.raises(ValueError):
            self.processor.apply_processing(self.test_image, "unsupported_type")