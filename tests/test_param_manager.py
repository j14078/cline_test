#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
param_manager モジュールのテストコード。
"""

import pytest
import sys
import os

# プロジェクトのsrcディレクトリをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from param_manager import initialize_param_values


def test_initialize_param_values_default_processes():
    """基本的な処理タイプのパラメータ初期値をテストする。"""
    # 基本的な処理タイプをテスト
    assert initialize_param_values("rotate") == {"angle": 90}
    assert initialize_param_values("resize") == {"width": 300, "height": 300}
    assert initialize_param_values("blur") == {"kernel_size": 5, "blur_type": "gaussian"}
    assert initialize_param_values("brightness") == {"value": 1.5}
    assert initialize_param_values("contrast") == {"value": 1.5}
    assert initialize_param_values("saturation") == {"value": 1.5}
    assert initialize_param_values("mosaic") == {"block_size": 10}
    assert initialize_param_values("threshold") == {"thresh_value": 127}
    assert initialize_param_values("dilate") == {"kernel_size": 5}
    assert initialize_param_values("erode") == {"kernel_size": 5}
    
    # パラメータなしの処理タイプをテスト
    assert initialize_param_values("face_detection") == {}
    assert initialize_param_values("text_detection") == {}
    assert initialize_param_values("watershed") == {}
    assert initialize_param_values("barcode_detection") == {}
    
    # 存在しない処理タイプをテストして、空の辞書が返されることを確認
    assert initialize_param_values("non_existent_process") == {}


def test_initialize_param_values_morphology():
    """モルフォロジー変換系の処理タイプのパラメータ初期値をテストする。"""
    # 各モルフォロジー処理のパラメータをテスト
    for process_type in ["opening", "closing", "tophat", "blackhat", "morphology_gradient"]:
        assert initialize_param_values(process_type) == {"kernel_size": 5}


def test_initialize_param_values_hough():
    """ハフ変換系の処理タイプのパラメータ初期値をテストする。"""
    # ハフ変換（直線）
    assert initialize_param_values("hough_lines") == {"threshold": 100}
    
    # 確率的ハフ変換
    prob_hough_params = initialize_param_values("probabilistic_hough_lines")
    assert prob_hough_params["threshold"] == 50
    assert prob_hough_params["min_line_length"] == 50
    assert prob_hough_params["max_line_gap"] == 10
    
    # ハフ変換（円）
    hough_circles_params = initialize_param_values("hough_circles")
    assert hough_circles_params["min_radius"] == 10
    assert hough_circles_params["max_radius"] == 100


def test_initialize_param_values_new_processes():
    """新しく追加した処理タイプのパラメータ初期値をテストする。"""
    # ガウシアンノイズ
    gaussian_noise_params = initialize_param_values("add_gaussian_noise")
    assert gaussian_noise_params["mean"] == 0
    assert gaussian_noise_params["sigma"] == 25
    
    # ソルト＆ペッパーノイズ
    assert initialize_param_values("add_salt_pepper_noise") == {"amount": 0.05}
    
    # カートゥーン効果
    assert initialize_param_values("cartoon_effect") == {"edge_size": 7}
    
    # ビネット効果
    assert initialize_param_values("vignette_effect") == {"strength": 0.5}
    
    # 色温度調整
    assert initialize_param_values("color_temperature") == {"temperature": 0}
    
    # 特定色抽出
    color_extraction_params = initialize_param_values("color_extraction")
    assert color_extraction_params["hue"] == 0
    assert color_extraction_params["range"] == 15
    
    # レンズ歪み系
    assert initialize_param_values("lens_distortion") == {"strength": 0.5}
    assert initialize_param_values("lens_correction") == {"strength": 0.5}
    
    # ミニチュア効果
    miniature_params = initialize_param_values("miniature_effect")
    assert miniature_params["position"] == 0.5
    assert miniature_params["width"] == 0.2
    
    # 文書スキャン
    assert initialize_param_values("document_scan") == {"threshold": 150}
    
    # グリッチ効果
    assert initialize_param_values("glitch_effect") == {"strength": 0.5}
    
    # 古い写真効果
    old_photo_params = initialize_param_values("old_photo_effect")
    assert old_photo_params["age"] == 0.7
    assert old_photo_params["noise"] == 0.3
    
    # ネオン効果
    assert initialize_param_values("neon_effect") == {"glow": 0.7}
    
    # ピクセル化
    assert initialize_param_values("pixelate") == {"pixel_size": 10}
    
    # クロスハッチング
    assert initialize_param_values("cross_hatching") == {"line_spacing": 6}


if __name__ == "__main__":
    pytest.main()