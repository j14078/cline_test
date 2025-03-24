#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
画像処理確認アプリのパラメータ管理モジュール。

このモジュールは画像処理パラメータの管理と更新を担当します。
"""

import streamlit as st


def initialize_param_values(process_type):
    """処理タイプに基づいてパラメータの初期値を生成する。

    Args:
        process_type: 処理タイプ

    Returns:
        パラメータ辞書
    """
    if process_type == "rotate":
        return {"angle": 90}
    elif process_type == "resize":
        return {"width": 300, "height": 300}
    elif process_type == "blur":
        return {"kernel_size": 5, "blur_type": "gaussian"}
    elif process_type == "brightness":
        return {"value": 1.5}
    elif process_type == "contrast":
        return {"value": 1.5}
    elif process_type == "saturation":
        return {"value": 1.5}
    elif process_type == "mosaic":
        return {"block_size": 10}
    elif process_type == "threshold":
        return {"thresh_value": 127}
    elif process_type == "dilate" or process_type == "erode":
        return {"kernel_size": 5}
    elif process_type == "edge_detection":
        return {"method": "canny"}
    elif process_type == "gamma":
        return {"gamma": 1.5}
    elif process_type == "denoise":
        return {"strength": 10}
    elif process_type == "adjust_red" or process_type == "adjust_green" or process_type == "adjust_blue":
        return {"factor": 1.5}
    elif process_type == "segmentation":
        return {"clusters": 5}
    elif process_type == "contour":
        return {"thickness": 2}
    elif process_type == "posterize":
        return {"levels": 4}
    elif process_type == "watercolor":
        return {"strength": 15}
    elif process_type == "oil_painting":
        return {"size": 7}
    elif process_type == "sketch":
        return {"ksize": 17, "sigma": 0}
    # モルフォロジー変換系
    elif process_type in ["opening", "closing", "tophat", "blackhat", "morphology_gradient"]:
        return {"kernel_size": 5}
    # 直線検出系
    elif process_type == "hough_lines":
        return {"threshold": 100}
    elif process_type == "probabilistic_hough_lines":
        return {"threshold": 50, "min_line_length": 50, "max_line_gap": 10}
    elif process_type == "hough_circles":
        return {"min_radius": 10, "max_radius": 100}
    # その他
    elif process_type == "perspective_transform":
        return {"angle": 30}
    elif process_type == "template_matching":
        return {}  # テンプレートは画像から自動生成
    elif process_type == "feature_detection":
        return {"method": "sift"}
    elif process_type == "face_detection":
        return {}  # パラメータなし
    elif process_type == "text_detection":
        return {}  # パラメータなし
    elif process_type == "watershed":
        return {}  # パラメータなし
    # 新しい処理
    elif process_type == "add_gaussian_noise":
        return {"mean": 0, "sigma": 25}
    elif process_type == "add_salt_pepper_noise":
        return {"amount": 0.05}
    elif process_type == "cartoon_effect":
        return {"edge_size": 7}
    elif process_type == "vignette_effect":
        return {"strength": 0.5}
    elif process_type == "color_temperature":
        return {"temperature": 0}
    elif process_type == "color_extraction":
        return {"hue": 0, "range": 15}
    elif process_type == "lens_distortion":
        return {"strength": 0.5}
    elif process_type == "lens_correction":
        return {"strength": 0.5}
    elif process_type == "miniature_effect":
        return {"position": 0.5, "width": 0.2}
    elif process_type == "document_scan":
        return {"threshold": 150}
    elif process_type == "barcode_detection":
        return {}  # パラメータなし
    elif process_type == "glitch_effect":
        return {"strength": 0.5}
    elif process_type == "old_photo_effect":
        return {"age": 0.7, "noise": 0.3}
    elif process_type == "neon_effect":
        return {"glow": 0.7}
    elif process_type == "pixelate":
        return {"pixel_size": 10}
    elif process_type == "cross_hatching":
        return {"line_spacing": 6}
    else:
        return {}


def setup_rotate_params(current_params):
    """回転パラメータのUI要素を設定する。

    Args:
        current_params: 現在のパラメータ値

    Returns:
        更新されたパラメータ辞書
    """
    # スライダーの値変更時にテキスト入力も更新する関数
    def update_angle_text():
        st.session_state.param_values["rotate"]["angle"] = st.session_state.angle_slider
        
    # テキスト入力の値変更時にスライダーも更新する関数
    def update_angle_slider():
        try:
            value = int(st.session_state.angle_text)
            if -180 <= value <= 180:
                st.session_state.param_values["rotate"]["angle"] = value
        except ValueError:
            pass
    
    angle_value = current_params.get("angle", 90)
    col1, col2 = st.columns(2)
    with col1:
        st.slider("回転角度", -180, 180, angle_value, key="angle_slider", on_change=update_angle_text)
    with col2:
        st.text_input("回転角度（手動入力）", str(angle_value), key="angle_text", on_change=update_angle_slider)
    
    return {"angle": st.session_state.param_values["rotate"]["angle"]}


def setup_resize_params(current_params):
    """リサイズパラメータのUI要素を設定する。

    Args:
        current_params: 現在のパラメータ値

    Returns:
        更新されたパラメータ辞書
    """
    # スライダーの値変更時にテキスト入力も更新する関数
    def update_width_text():
        st.session_state.param_values["resize"]["width"] = st.session_state.width_slider
        
    def update_height_text():
        st.session_state.param_values["resize"]["height"] = st.session_state.height_slider
        
    # テキスト入力の値変更時にスライダーも更新する関数
    def update_width_slider():
        try:
            value = int(st.session_state.width_text)
            if 10 <= value <= 1000:
                st.session_state.param_values["resize"]["width"] = value
        except ValueError:
            pass
            
    def update_height_slider():
        try:
            value = int(st.session_state.height_text)
            if 10 <= value <= 1000:
                st.session_state.param_values["resize"]["height"] = value
        except ValueError:
            pass
    
    width_value = current_params.get("width", 300)
    height_value = current_params.get("height", 300)
    
    col1, col2 = st.columns(2)
    with col1:
        st.slider("幅", 10, 1000, width_value, key="width_slider", on_change=update_width_text)
        st.slider("高さ", 10, 1000, height_value, key="height_slider", on_change=update_height_text)
    with col2:
        st.text_input("幅（手動入力）", str(width_value), key="width_text", on_change=update_width_slider)
        st.text_input("高さ（手動入力）", str(height_value), key="height_text", on_change=update_height_slider)
    
    return {
        "width": st.session_state.param_values["resize"]["width"],
        "height": st.session_state.param_values["resize"]["height"]
    }


def setup_blur_params(current_params):
    """ぼかしパラメータのUI要素を設定する。

    Args:
        current_params: 現在のパラメータ値

    Returns:
        更新されたパラメータ辞書
    """
    # スライダーの値変更時にテキスト入力も更新する関数
    def update_kernel_text():
        st.session_state.param_values["blur"]["kernel_size"] = st.session_state.kernel_slider
        
    # テキスト入力の値変更時にスライダーも更新する関数
    def update_kernel_slider():
        try:
            value = int(st.session_state.kernel_text)
            if value % 2 == 1 and 1 <= value <= 31:  # カーネルサイズは奇数である必要がある
                st.session_state.param_values["blur"]["kernel_size"] = value
        except ValueError:
            pass
    
    kernel_size = current_params.get("kernel_size", 5)
    blur_type = current_params.get("blur_type", "gaussian")
    
    col1, col2 = st.columns(2)
    with col1:
        st.slider("カーネルサイズ", 1, 31, kernel_size, step=2, key="kernel_slider", on_change=update_kernel_text)
        blur_type = st.selectbox("ぼかしの種類", ["gaussian", "median", "box", "bilateral"], index=["gaussian", "median", "box", "bilateral"].index(blur_type))
        st.session_state.param_values["blur"]["blur_type"] = blur_type
    with col2:
        st.text_input("カーネルサイズ（手動入力）", str(kernel_size), key="kernel_text", on_change=update_kernel_slider)
    
    return {
        "kernel_size": st.session_state.param_values["blur"]["kernel_size"],
        "blur_type": st.session_state.param_values["blur"]["blur_type"]
    }


def setup_edge_detection_params(current_params):
    """エッジ検出パラメータのUI要素を設定する。

    Args:
        current_params: 現在のパラメータ値

    Returns:
        更新されたパラメータ辞書
    """
    method = current_params.get("method", "canny")
    method = st.selectbox("検出方法", ["canny", "sobel", "laplacian"], index=["canny", "sobel", "laplacian"].index(method))
    st.session_state.param_values["edge_detection"]["method"] = method
    
    return {"method": method}


def setup_value_params(process_type, current_params, label, min_val=0.1, max_val=3.0):
    """値を使うパラメータ（明るさ、コントラスト、彩度）のUI要素を設定する。

    Args:
        process_type: 処理タイプ
        current_params: 現在のパラメータ値
        label: UI上の表示ラベル
        min_val: スライダーの最小値
        max_val: スライダーの最大値

    Returns:
        更新されたパラメータ辞書
    """
    # スライダーの値変更時にテキスト入力も更新する関数
    def update_value_text():
        st.session_state.param_values[process_type]["value"] = getattr(st.session_state, f"{process_type}_slider")
        
    # テキスト入力の値変更時にスライダーも更新する関数
    def update_value_slider():
        try:
            value = float(getattr(st.session_state, f"{process_type}_text"))
            if min_val <= value <= max_val:
                st.session_state.param_values[process_type]["value"] = value
        except ValueError:
            pass
    
    value = current_params.get("value", 1.5)
    
    col1, col2 = st.columns(2)
    with col1:
        st.slider(label, min_val, max_val, value, key=f"{process_type}_slider", on_change=update_value_text)
    with col2:
        st.text_input(f"{label}（手動入力）", str(value), key=f"{process_type}_text", on_change=update_value_slider)
    
    return {"value": st.session_state.param_values[process_type]["value"]}


def setup_size_params(process_type, current_params, label, min_val=1, max_val=50, step=1):
    """サイズを使うパラメータ（モザイク、膨張、収縮）のUI要素を設定する。

    Args:
        process_type: 処理タイプ
        current_params: 現在のパラメータ値
        label: UI上の表示ラベル
        min_val: スライダーの最小値
        max_val: スライダーの最大値
        step: スライダーのステップ

    Returns:
        更新されたパラメータ辞書
    """
    # サイズのパラメータ名を取得
    param_name = "block_size" if process_type == "mosaic" else "kernel_size" if process_type in ["dilate", "erode", "opening", "closing", "tophat", "blackhat", "morphology_gradient"] else "thresh_value"
    
    # スライダーの値変更時にテキスト入力も更新する関数
    def update_size_text():
        st.session_state.param_values[process_type][param_name] = getattr(st.session_state, f"{process_type}_slider")
        
    # テキスト入力の値変更時にスライダーも更新する関数
    def update_size_slider():
        try:
            value = int(getattr(st.session_state, f"{process_type}_text"))
            if min_val <= value <= max_val:
                if param_name == "kernel_size" and value % 2 == 0:
                    # カーネルサイズは奇数である必要がある場合
                    value += 1
                st.session_state.param_values[process_type][param_name] = value
        except ValueError:
            pass
    
    size_value = current_params.get(param_name, 5)
    
    col1, col2 = st.columns(2)
    with col1:
        st.slider(label, min_val, max_val, size_value, step=step, key=f"{process_type}_slider", on_change=update_size_text)
    with col2:
        st.text_input(f"{label}（手動入力）", str(size_value), key=f"{process_type}_text", on_change=update_size_slider)
    
    return {param_name: st.session_state.param_values[process_type][param_name]}


# --- 新しいパラメータUIの設定関数 ---

def setup_gamma_params(current_params):
    """ガンマ補正パラメータのUI要素を設定する。

    Args:
        current_params: 現在のパラメータ値

    Returns:
        更新されたパラメータ辞書
    """
    def update_gamma_text():
        st.session_state.param_values["gamma"]["gamma"] = st.session_state.gamma_slider
        
    def update_gamma_slider():
        try:
            value = float(st.session_state.gamma_text)
            if 0.1 <= value <= 5.0:
                st.session_state.param_values["gamma"]["gamma"] = value
        except ValueError:
            pass
    
    gamma_value = current_params.get("gamma", 1.5)
    
    col1, col2 = st.columns(2)
    with col1:
        st.slider("ガンマ値", 0.1, 5.0, gamma_value, key="gamma_slider", on_change=update_gamma_text)
    with col2:
        st.text_input("ガンマ値（手動入力）", str(gamma_value), key="gamma_text", on_change=update_gamma_slider)
    
    return {"gamma": st.session_state.param_values["gamma"]["gamma"]}


def setup_denoise_params(current_params):
    """ノイズ除去パラメータのUI要素を設定する。

    Args:
        current_params: 現在のパラメータ値

    Returns:
        更新されたパラメータ辞書
    """
    def update_strength_text():
        st.session_state.param_values["denoise"]["strength"] = st.session_state.strength_slider
        
    def update_strength_slider():
        try:
            value = int(st.session_state.strength_text)
            if 1 <= value <= 30:
                st.session_state.param_values["denoise"]["strength"] = value
        except ValueError:
            pass
    
    strength_value = current_params.get("strength", 10)
    
    col1, col2 = st.columns(2)
    with col1:
        st.slider("ノイズ除去強度", 1, 30, strength_value, key="strength_slider", on_change=update_strength_text)
    with col2:
        st.text_input("ノイズ除去強度（手動入力）", str(strength_value), key="strength_text", on_change=update_strength_slider)
    
    return {"strength": st.session_state.param_values["denoise"]["strength"]}


def setup_color_adjust_params(process_type, current_params):
    """色調整パラメータのUI要素を設定する。

    Args:
        process_type: 処理タイプ（adjust_red, adjust_green, adjust_blue）
        current_params: 現在のパラメータ値

    Returns:
        更新されたパラメータ辞書
    """
    def update_factor_text():
        st.session_state.param_values[process_type]["factor"] = st.session_state.factor_slider
        
    def update_factor_slider():
        try:
            value = float(st.session_state.factor_text)
            if 0.0 <= value <= 3.0:
                st.session_state.param_values[process_type]["factor"] = value
        except ValueError:
            pass
    
    factor_value = current_params.get("factor", 1.5)
    
    # 色名の取得
    color_name = "赤" if process_type == "adjust_red" else "緑" if process_type == "adjust_green" else "青"
    
    col1, col2 = st.columns(2)
    with col1:
        st.slider(f"{color_name}色調整", 0.0, 3.0, factor_value, key="factor_slider", on_change=update_factor_text)
    with col2:
        st.text_input(f"{color_name}色調整（手動入力）", str(factor_value), key="factor_text", on_change=update_factor_slider)
    
    return {"factor": st.session_state.param_values[process_type]["factor"]}


def setup_segmentation_params(current_params):
    """セグメンテーションパラメータのUI要素を設定する。

    Args:
        current_params: 現在のパラメータ値

    Returns:
        更新されたパラメータ辞書
    """
    def update_clusters_text():
        st.session_state.param_values["segmentation"]["clusters"] = st.session_state.clusters_slider
        
    def update_clusters_slider():
        try:
            value = int(st.session_state.clusters_text)
            if 2 <= value <= 20:
                st.session_state.param_values["segmentation"]["clusters"] = value
        except ValueError:
            pass
    
    clusters_value = current_params.get("clusters", 5)
    
    col1, col2 = st.columns(2)
    with col1:
        st.slider("クラスタ数", 2, 20, clusters_value, key="clusters_slider", on_change=update_clusters_text)
    with col2:
        st.text_input("クラスタ数（手動入力）", str(clusters_value), key="clusters_text", on_change=update_clusters_slider)
    
    return {"clusters": st.session_state.param_values["segmentation"]["clusters"]}


def setup_contour_params(current_params):
    """輪郭抽出パラメータのUI要素を設定する。

    Args:
        current_params: 現在のパラメータ値

    Returns:
        更新されたパラメータ辞書
    """
    def update_thickness_text():
        st.session_state.param_values["contour"]["thickness"] = st.session_state.thickness_slider
        
    def update_thickness_slider():
        try:
            value = int(st.session_state.thickness_text)
            if 1 <= value <= 10:
                st.session_state.param_values["contour"]["thickness"] = value
        except ValueError:
            pass
    
    thickness_value = current_params.get("thickness", 2)
    
    col1, col2 = st.columns(2)
    with col1:
        st.slider("線の太さ", 1, 10, thickness_value, key="thickness_slider", on_change=update_thickness_text)
    with col2:
        st.text_input("線の太さ（手動入力）", str(thickness_value), key="thickness_text", on_change=update_thickness_slider)
    
    return {"thickness": st.session_state.param_values["contour"]["thickness"]}


def setup_posterize_params(current_params):
    """ポスタリゼーションパラメータのUI要素を設定する。

    Args:
        current_params: 現在のパラメータ値

    Returns:
        更新されたパラメータ辞書
    """
    def update_levels_text():
        st.session_state.param_values["posterize"]["levels"] = st.session_state.levels_slider
        
    def update_levels_slider():
        try:
            value = int(st.session_state.levels_text)
            if 2 <= value <= 16:
                st.session_state.param_values["posterize"]["levels"] = value
        except ValueError:
            pass
    
    levels_value = current_params.get("levels", 4)
    
    col1, col2 = st.columns(2)
    with col1:
        st.slider("色レベル数", 2, 16, levels_value, key="levels_slider", on_change=update_levels_text)
    with col2:
        st.text_input("色レベル数（手動入力）", str(levels_value), key="levels_text", on_change=update_levels_slider)
    
    return {"levels": st.session_state.param_values["posterize"]["levels"]}


def setup_effect_strength_params(process_type, current_params, label="強度"):
    """エフェクト強度パラメータのUI要素を設定する（水彩画風、油絵風）。

    Args:
        process_type: 処理タイプ
        current_params: 現在のパラメータ値
        label: UI上の表示ラベル

    Returns:
        更新されたパラメータ辞書
    """
    param_name = "strength" if process_type == "watercolor" else "size"
    min_val = 1
    max_val = 30 if process_type == "watercolor" else 15
    
    def update_strength_text():
        st.session_state.param_values[process_type][param_name] = getattr(st.session_state, f"{process_type}_slider")
        
    def update_strength_slider():
        try:
            value = int(getattr(st.session_state, f"{process_type}_text"))
            if min_val <= value <= max_val:
                if process_type == "oil_painting" and value % 2 == 0:
                    # 油絵風の場合、カーネルサイズは奇数が望ましい
                    value += 1
                st.session_state.param_values[process_type][param_name] = value
        except ValueError:
            pass
    
    strength_value = current_params.get(param_name, 15 if process_type == "watercolor" else 7)
    
    col1, col2 = st.columns(2)
    with col1:
        step = 2 if process_type == "oil_painting" else 1
        st.slider(label, min_val, max_val, strength_value, step=step, key=f"{process_type}_slider", on_change=update_strength_text)
    with col2:
        st.text_input(f"{label}（手動入力）", str(strength_value), key=f"{process_type}_text", on_change=update_strength_slider)
    
    return {param_name: st.session_state.param_values[process_type][param_name]}


def setup_sketch_params(current_params):
    """スケッチ風パラメータのUI要素を設定する。

    Args:
        current_params: 現在のパラメータ値

    Returns:
        更新されたパラメータ辞書
    """
    def update_ksize_text():
        st.session_state.param_values["sketch"]["ksize"] = st.session_state.ksize_slider
        
    def update_ksize_slider():
        try:
            value = int(st.session_state.ksize_text)
            if value % 2 == 1 and 3 <= value <= 31:  # カーネルサイズは奇数である必要がある
                st.session_state.param_values["sketch"]["ksize"] = value
        except ValueError:
            pass
    
    def update_sigma_text():
        st.session_state.param_values["sketch"]["sigma"] = st.session_state.sigma_slider
        
    def update_sigma_slider():
        try:
            value = float(st.session_state.sigma_text)
            if 0 <= value <= 10:
                st.session_state.param_values["sketch"]["sigma"] = value
        except ValueError:
            pass
    
    ksize_value = current_params.get("ksize", 17)
    sigma_value = current_params.get("sigma", 0)
    
    col1, col2 = st.columns(2)
    with col1:
        st.slider("カーネルサイズ", 3, 31, ksize_value, step=2, key="ksize_slider", on_change=update_ksize_text)
        st.slider("シグマ値", 0.0, 10.0, sigma_value, key="sigma_slider", on_change=update_sigma_text)
    with col2:
        st.text_input("カーネルサイズ（手動入力）", str(ksize_value), key="ksize_text", on_change=update_ksize_slider)
        st.text_input("シグマ値（手動入力）", str(sigma_value), key="sigma_text", on_change=update_sigma_slider)
    
    return {
        "ksize": st.session_state.param_values["sketch"]["ksize"],
        "sigma": st.session_state.param_values["sketch"]["sigma"]
    }
    
    
# --- 新しい処理タイプ用のパラメータUIセットアップ関数 ---

def setup_hough_lines_params(current_params):
    """ハフ変換（直線）パラメータのUI要素を設定する。

    Args:
        current_params: 現在のパラメータ値

    Returns:
        更新されたパラメータ辞書
    """
    def update_threshold_text():
        st.session_state.param_values["hough_lines"]["threshold"] = st.session_state.hough_threshold_slider
        
    def update_threshold_slider():
        try:
            value = int(st.session_state.hough_threshold_text)
            if 1 <= value <= 300:
                st.session_state.param_values["hough_lines"]["threshold"] = value
        except ValueError:
            pass
    
    threshold_value = current_params.get("threshold", 100)
    
    col1, col2 = st.columns(2)
    with col1:
        st.slider("検出閾値", 1, 300, threshold_value, key="hough_threshold_slider", on_change=update_threshold_text)
    with col2:
        st.text_input("検出閾値（手動入力）", str(threshold_value), key="hough_threshold_text", on_change=update_threshold_slider)
    
    return {"threshold": st.session_state.param_values["hough_lines"]["threshold"]}


def setup_prob_hough_lines_params(current_params):
    """確率的ハフ変換パラメータのUI要素を設定する。

    Args:
        current_params: 現在のパラメータ値

    Returns:
        更新されたパラメータ辞書
    """
    def update_threshold_text():
        st.session_state.param_values["probabilistic_hough_lines"]["threshold"] = st.session_state.phough_threshold_slider
        
    def update_threshold_slider():
        try:
            value = int(st.session_state.phough_threshold_text)
            if 1 <= value <= 300:
                st.session_state.param_values["probabilistic_hough_lines"]["threshold"] = value
        except ValueError:
            pass
            
    def update_min_line_length_text():
        st.session_state.param_values["probabilistic_hough_lines"]["min_line_length"] = st.session_state.min_line_length_slider
        
    def update_min_line_length_slider():
        try:
            value = int(st.session_state.min_line_length_text)
            if 1 <= value <= 300:
                st.session_state.param_values["probabilistic_hough_lines"]["min_line_length"] = value
        except ValueError:
            pass
    
    def update_max_line_gap_text():
        st.session_state.param_values["probabilistic_hough_lines"]["max_line_gap"] = st.session_state.max_line_gap_slider
        
    def update_max_line_gap_slider():
        try:
            value = int(st.session_state.max_line_gap_text)
            if 1 <= value <= 100:
                st.session_state.param_values["probabilistic_hough_lines"]["max_line_gap"] = value
        except ValueError:
            pass
    
    threshold_value = current_params.get("threshold", 50)
    min_line_length = current_params.get("min_line_length", 50)
    max_line_gap = current_params.get("max_line_gap", 10)
    
    col1, col2 = st.columns(2)
    with col1:
        st.slider("検出閾値", 1, 300, threshold_value, key="phough_threshold_slider", on_change=update_threshold_text)
        st.slider("最小線長", 1, 300, min_line_length, key="min_line_length_slider", on_change=update_min_line_length_text)
        st.slider("最大線間隔", 1, 100, max_line_gap, key="max_line_gap_slider", on_change=update_max_line_gap_text)
    with col2:
        st.text_input("検出閾値（手動入力）", str(threshold_value), key="phough_threshold_text", on_change=update_threshold_slider)
        st.text_input("最小線長（手動入力）", str(min_line_length), key="min_line_length_text", on_change=update_min_line_length_slider)
        st.text_input("最大線間隔（手動入力）", str(max_line_gap), key="max_line_gap_text", on_change=update_max_line_gap_slider)
    
    return {
        "threshold": st.session_state.param_values["probabilistic_hough_lines"]["threshold"],
        "min_line_length": st.session_state.param_values["probabilistic_hough_lines"]["min_line_length"],
        "max_line_gap": st.session_state.param_values["probabilistic_hough_lines"]["max_line_gap"]
    }


def setup_hough_circles_params(current_params):
    """ハフ変換（円）パラメータのUI要素を設定する。

    Args:
        current_params: 現在のパラメータ値

    Returns:
        更新されたパラメータ辞書
    """
    def update_min_radius_text():
        st.session_state.param_values["hough_circles"]["min_radius"] = st.session_state.min_radius_slider
        
    def update_min_radius_slider():
        try:
            value = int(st.session_state.min_radius_text)
            if 1 <= value <= 100:
                st.session_state.param_values["hough_circles"]["min_radius"] = value
        except ValueError:
            pass
            
    def update_max_radius_text():
        st.session_state.param_values["hough_circles"]["max_radius"] = st.session_state.max_radius_slider
        
    def update_max_radius_slider():
        try:
            value = int(st.session_state.max_radius_text)
            if 10 <= value <= 300:
                st.session_state.param_values["hough_circles"]["max_radius"] = value
        except ValueError:
            pass
    
    min_radius = current_params.get("min_radius", 10)
    max_radius = current_params.get("max_radius", 100)
    
    col1, col2 = st.columns(2)
    with col1:
        st.slider("最小半径", 1, 100, min_radius, key="min_radius_slider", on_change=update_min_radius_text)
        st.slider("最大半径", 10, 300, max_radius, key="max_radius_slider", on_change=update_max_radius_text)
    with col2:
        st.text_input("最小半径（手動入力）", str(min_radius), key="min_radius_text", on_change=update_min_radius_slider)
        st.text_input("最大半径（手動入力）", str(max_radius), key="max_radius_text", on_change=update_max_radius_slider)
    
    return {
        "min_radius": st.session_state.param_values["hough_circles"]["min_radius"],
        "max_radius": st.session_state.param_values["hough_circles"]["max_radius"]
    }


def setup_perspective_transform_params(current_params):
    """透視変換パラメータのUI要素を設定する。

    Args:
        current_params: 現在のパラメータ値

    Returns:
        更新されたパラメータ辞書
    """
    def update_angle_text():
        st.session_state.param_values["perspective_transform"]["angle"] = st.session_state.persp_angle_slider
        
    def update_angle_slider():
        try:
            value = int(st.session_state.persp_angle_text)
            if 0 <= value <= 80:
                st.session_state.param_values["perspective_transform"]["angle"] = value
        except ValueError:
            pass
    
    angle_value = current_params.get("angle", 30)
    
    col1, col2 = st.columns(2)
    with col1:
        st.slider("変換角度", 0, 80, angle_value, key="persp_angle_slider", on_change=update_angle_text)
    with col2:
        st.text_input("変換角度（手動入力）", str(angle_value), key="persp_angle_text", on_change=update_angle_slider)
    
    return {"angle": st.session_state.param_values["perspective_transform"]["angle"]}


def setup_feature_detection_params(current_params):
    """特徴点検出パラメータのUI要素を設定する。

    Args:
        current_params: 現在のパラメータ値

    Returns:
        更新されたパラメータ辞書
    """
    method = current_params.get("method", "sift")
    method = st.selectbox("検出方法", ["sift", "orb"], index=["sift", "orb"].index(method))
    st.session_state.param_values["feature_detection"]["method"] = method
    
    return {"method": method}


# --- 新しい処理用のパラメータUI設定関数 ---

def setup_gaussian_noise_params(current_params):
    """ガウシアンノイズパラメータのUI要素を設定する。

    Args:
        current_params: 現在のパラメータ値

    Returns:
        更新されたパラメータ辞書
    """
    def update_mean_text():
        st.session_state.param_values["add_gaussian_noise"]["mean"] = st.session_state.mean_slider
        
    def update_mean_slider():
        try:
            value = int(st.session_state.mean_text)
            if -50 <= value <= 50:
                st.session_state.param_values["add_gaussian_noise"]["mean"] = value
        except ValueError:
            pass
    
    def update_sigma_text():
        st.session_state.param_values["add_gaussian_noise"]["sigma"] = st.session_state.gnoise_sigma_slider
        
    def update_sigma_slider():
        try:
            value = int(st.session_state.gnoise_sigma_text)
            if 1 <= value <= 100:
                st.session_state.param_values["add_gaussian_noise"]["sigma"] = value
        except ValueError:
            pass
    
    mean_value = current_params.get("mean", 0)
    sigma_value = current_params.get("sigma", 25)
    
    col1, col2 = st.columns(2)
    with col1:
        st.slider("平均値", -50, 50, mean_value, key="mean_slider", on_change=update_mean_text)
        st.slider("標準偏差", 1, 100, sigma_value, key="gnoise_sigma_slider", on_change=update_sigma_text)
    with col2:
        st.text_input("平均値（手動入力）", str(mean_value), key="mean_text", on_change=update_mean_slider)
        st.text_input("標準偏差（手動入力）", str(sigma_value), key="gnoise_sigma_text", on_change=update_sigma_slider)
    
    return {
        "mean": st.session_state.param_values["add_gaussian_noise"]["mean"],
        "sigma": st.session_state.param_values["add_gaussian_noise"]["sigma"]
    }


def setup_salt_pepper_noise_params(current_params):
    """ソルト＆ペッパーノイズパラメータのUI要素を設定する。

    Args:
        current_params: 現在のパラメータ値

    Returns:
        更新されたパラメータ辞書
    """
    def update_amount_text():
        st.session_state.param_values["add_salt_pepper_noise"]["amount"] = st.session_state.amount_slider
        
    def update_amount_slider():
        try:
            value = float(st.session_state.amount_text)
            if 0.001 <= value <= 0.5:
                st.session_state.param_values["add_salt_pepper_noise"]["amount"] = value
        except ValueError:
            pass
    
    amount_value = current_params.get("amount", 0.05)
    
    col1, col2 = st.columns(2)
    with col1:
        st.slider("ノイズの量", 0.001, 0.5, amount_value, key="amount_slider", on_change=update_amount_text)
    with col2:
        st.text_input("ノイズの量（手動入力）", str(amount_value), key="amount_text", on_change=update_amount_slider)
    
    return {"amount": st.session_state.param_values["add_salt_pepper_noise"]["amount"]}


def setup_cartoon_effect_params(current_params):
    """カートゥーン効果パラメータのUI要素を設定する。

    Args:
        current_params: 現在のパラメータ値

    Returns:
        更新されたパラメータ辞書
    """
    def update_edge_size_text():
        st.session_state.param_values["cartoon_effect"]["edge_size"] = st.session_state.edge_size_slider
        
    def update_edge_size_slider():
        try:
            value = int(st.session_state.edge_size_text)
            if value % 2 == 1 and 3 <= value <= 31:  # 奇数である必要がある
                st.session_state.param_values["cartoon_effect"]["edge_size"] = value
        except ValueError:
            pass
    
    edge_size = current_params.get("edge_size", 7)
    
    col1, col2 = st.columns(2)
    with col1:
        st.slider("エッジの太さ", 3, 31, edge_size, step=2, key="edge_size_slider", on_change=update_edge_size_text)
    with col2:
        st.text_input("エッジの太さ（手動入力）", str(edge_size), key="edge_size_text", on_change=update_edge_size_slider)
    
    return {"edge_size": st.session_state.param_values["cartoon_effect"]["edge_size"]}


def setup_vignette_effect_params(current_params):
    """ビネット効果パラメータのUI要素を設定する。

    Args:
        current_params: 現在のパラメータ値

    Returns:
        更新されたパラメータ辞書
    """
    def update_strength_text():
        st.session_state.param_values["vignette_effect"]["strength"] = st.session_state.vignette_strength_slider
        
    def update_strength_slider():
        try:
            value = float(st.session_state.vignette_strength_text)
            if 0.0 <= value <= 1.0:
                st.session_state.param_values["vignette_effect"]["strength"] = value
        except ValueError:
            pass
    
    strength_value = current_params.get("strength", 0.5)
    
    col1, col2 = st.columns(2)
    with col1:
        st.slider("効果の強さ", 0.0, 1.0, strength_value, key="vignette_strength_slider", on_change=update_strength_text)
    with col2:
        st.text_input("効果の強さ（手動入力）", str(strength_value), key="vignette_strength_text", on_change=update_strength_slider)
    
    return {"strength": st.session_state.param_values["vignette_effect"]["strength"]}


def setup_color_temperature_params(current_params):
    """色温度パラメータのUI要素を設定する。

    Args:
        current_params: 現在のパラメータ値

    Returns:
        更新されたパラメータ辞書
    """
    def update_temperature_text():
        st.session_state.param_values["color_temperature"]["temperature"] = st.session_state.temperature_slider
        
    def update_temperature_slider():
        try:
            value = int(st.session_state.temperature_text)
            if -100 <= value <= 100:
                st.session_state.param_values["color_temperature"]["temperature"] = value
        except ValueError:
            pass
    
    temperature_value = current_params.get("temperature", 0)
    
    col1, col2 = st.columns(2)
    with col1:
        st.slider("色温度", -100, 100, temperature_value, key="temperature_slider", on_change=update_temperature_text, 
                 help="負の値は青色を強く、正の値は赤色を強くします")
    with col2:
        st.text_input("色温度（手動入力）", str(temperature_value), key="temperature_text", on_change=update_temperature_slider)
    
    return {"temperature": st.session_state.param_values["color_temperature"]["temperature"]}


def setup_color_extraction_params(current_params):
    """特定色抽出パラメータのUI要素を設定する。

    Args:
        current_params: 現在のパラメータ値

    Returns:
        更新されたパラメータ辞書
    """
    def update_hue_text():
        st.session_state.param_values["color_extraction"]["hue"] = st.session_state.hue_slider
        
    def update_hue_slider():
        try:
            value = int(st.session_state.hue_text)
            if 0 <= value <= 179:  # OpenCVのHSVは0-179
                st.session_state.param_values["color_extraction"]["hue"] = value
        except ValueError:
            pass
    
    def update_range_text():
        st.session_state.param_values["color_extraction"]["range"] = st.session_state.range_slider
        
    def update_range_slider():
        try:
            value = int(st.session_state.range_text)
            if 1 <= value <= 50:
                st.session_state.param_values["color_extraction"]["range"] = value
        except ValueError:
            pass
    
    hue_value = current_params.get("hue", 0)
    range_value = current_params.get("range", 15)
    
    # 主要な色相と名前のマッピング
    color_names = {
        0: "赤", 
        30: "オレンジ", 
        60: "黄色", 
        90: "黄緑", 
        120: "緑", 
        150: "シアン", 
        165: "水色",
        180-30: "青", 
        180-60: "青紫", 
        180-90: "紫", 
        180-120: "マゼンタ", 
        180-150: "ピンク"
    }
    
    # 最も近い色相の名前を表示
    closest_color = min(color_names.keys(), key=lambda x: abs(x - hue_value))
    color_name = color_names[closest_color]
    
    col1, col2 = st.columns(2)
    with col1:
        st.slider(f"色相 ({color_name})", 0, 179, hue_value, key="hue_slider", on_change=update_hue_text)
        st.slider("抽出範囲", 1, 50, range_value, key="range_slider", on_change=update_range_text)
    with col2:
        st.text_input("色相（手動入力）", str(hue_value), key="hue_text", on_change=update_hue_slider)
        st.text_input("抽出範囲（手動入力）", str(range_value), key="range_text", on_change=update_range_slider)
    
    return {
        "hue": st.session_state.param_values["color_extraction"]["hue"],
        "range": st.session_state.param_values["color_extraction"]["range"]
    }


def setup_lens_distortion_params(current_params):
    """レンズ歪み追加パラメータのUI要素を設定する。

    Args:
        current_params: 現在のパラメータ値

    Returns:
        更新されたパラメータ辞書
    """
    def update_strength_text():
        st.session_state.param_values["lens_distortion"]["strength"] = st.session_state.lens_dist_strength_slider
        
    def update_strength_slider():
        try:
            value = float(st.session_state.lens_dist_strength_text)
            if 0.0 <= value <= 2.0:
                st.session_state.param_values["lens_distortion"]["strength"] = value
        except ValueError:
            pass
    
    strength_value = current_params.get("strength", 0.5)
    
    col1, col2 = st.columns(2)
    with col1:
        st.slider("歪みの強さ", 0.0, 2.0, strength_value, key="lens_dist_strength_slider", on_change=update_strength_text)
    with col2:
        st.text_input("歪みの強さ（手動入力）", str(strength_value), key="lens_dist_strength_text", on_change=update_strength_slider)
    
    return {"strength": st.session_state.param_values["lens_distortion"]["strength"]}


def setup_lens_correction_params(current_params):
    """レンズ歪み補正パラメータのUI要素を設定する。

    Args:
        current_params: 現在のパラメータ値

    Returns:
        更新されたパラメータ辞書
    """
    def update_strength_text():
        st.session_state.param_values["lens_correction"]["strength"] = st.session_state.lens_corr_strength_slider
        
    def update_strength_slider():
        try:
            value = float(st.session_state.lens_corr_strength_text)
            if 0.0 <= value <= 2.0:
                st.session_state.param_values["lens_correction"]["strength"] = value
        except ValueError:
            pass
    
    strength_value = current_params.get("strength", 0.5)
    
    col1, col2 = st.columns(2)
    with col1:
        st.slider("補正強度", 0.0, 2.0, strength_value, key="lens_corr_strength_slider", on_change=update_strength_text)
    with col2:
        st.text_input("補正強度（手動入力）", str(strength_value), key="lens_corr_strength_text", on_change=update_strength_slider)
    
    return {"strength": st.session_state.param_values["lens_correction"]["strength"]}


def setup_miniature_effect_params(current_params):
    """ミニチュア効果パラメータのUI要素を設定する。

    Args:
        current_params: 現在のパラメータ値

    Returns:
        更新されたパラメータ辞書
    """
    def update_position_text():
        st.session_state.param_values["miniature_effect"]["position"] = st.session_state.position_slider
        
    def update_position_slider():
        try:
            value = float(st.session_state.position_text)
            if 0.0 <= value <= 1.0:
                st.session_state.param_values["miniature_effect"]["position"] = value
        except ValueError:
            pass
    
    def update_width_text():
        st.session_state.param_values["miniature_effect"]["width"] = st.session_state.mini_width_slider
        
    def update_width_slider():
        try:
            value = float(st.session_state.mini_width_text)
            if 0.01 <= value <= 0.5:
                st.session_state.param_values["miniature_effect"]["width"] = value
        except ValueError:
            pass
    
    position_value = current_params.get("position", 0.5)
    width_value = current_params.get("width", 0.2)
    
    col1, col2 = st.columns(2)
    with col1:
        st.slider("フォーカス位置", 0.0, 1.0, position_value, key="position_slider", on_change=update_position_text,
                 help="0が画像上部、1が画像下部")
        st.slider("フォーカス幅", 0.01, 0.5, width_value, key="mini_width_slider", on_change=update_width_text)
    with col2:
        st.text_input("フォーカス位置（手動入力）", str(position_value), key="position_text", on_change=update_position_slider)
        st.text_input("フォーカス幅（手動入力）", str(width_value), key="mini_width_text", on_change=update_width_slider)
    
    return {
        "position": st.session_state.param_values["miniature_effect"]["position"],
        "width": st.session_state.param_values["miniature_effect"]["width"]
    }


def setup_document_scan_params(current_params):
    """文書スキャン最適化パラメータのUI要素を設定する。

    Args:
        current_params: 現在のパラメータ値

    Returns:
        更新されたパラメータ辞書
    """
    def update_threshold_text():
        st.session_state.param_values["document_scan"]["threshold"] = st.session_state.doc_threshold_slider
        
    def update_threshold_slider():
        try:
            value = int(st.session_state.doc_threshold_text)
            if 50 <= value <= 250:
                st.session_state.param_values["document_scan"]["threshold"] = value
        except ValueError:
            pass
    
    threshold_value = current_params.get("threshold", 150)
    
    col1, col2 = st.columns(2)
    with col1:
        st.slider("閾値", 50, 250, threshold_value, key="doc_threshold_slider", on_change=update_threshold_text)
    with col2:
        st.text_input("閾値（手動入力）", str(threshold_value), key="doc_threshold_text", on_change=update_threshold_slider)
    
    return {"threshold": st.session_state.param_values["document_scan"]["threshold"]}


def setup_glitch_effect_params(current_params):
    """グリッチ効果パラメータのUI要素を設定する。

    Args:
        current_params: 現在のパラメータ値

    Returns:
        更新されたパラメータ辞書
    """
    def update_strength_text():
        st.session_state.param_values["glitch_effect"]["strength"] = st.session_state.glitch_strength_slider
        
    def update_strength_slider():
        try:
            value = float(st.session_state.glitch_strength_text)
            if 0.0 <= value <= 1.0:
                st.session_state.param_values["glitch_effect"]["strength"] = value
        except ValueError:
            pass
    
    strength_value = current_params.get("strength", 0.5)
    
    col1, col2 = st.columns(2)
    with col1:
        st.slider("効果の強さ", 0.0, 1.0, strength_value, key="glitch_strength_slider", on_change=update_strength_text)
    with col2:
        st.text_input("効果の強さ（手動入力）", str(strength_value), key="glitch_strength_text", on_change=update_strength_slider)
    
    return {"strength": st.session_state.param_values["glitch_effect"]["strength"]}


def setup_old_photo_effect_params(current_params):
    """古い写真効果パラメータのUI要素を設定する。

    Args:
        current_params: 現在のパラメータ値

    Returns:
        更新されたパラメータ辞書
    """
    def update_age_text():
        st.session_state.param_values["old_photo_effect"]["age"] = st.session_state.age_slider
        
    def update_age_slider():
        try:
            value = float(st.session_state.age_text)
            if 0.0 <= value <= 1.0:
                st.session_state.param_values["old_photo_effect"]["age"] = value
        except ValueError:
            pass
    
    def update_noise_text():
        st.session_state.param_values["old_photo_effect"]["noise"] = st.session_state.old_noise_slider
        
    def update_noise_slider():
        try:
            value = float(st.session_state.old_noise_text)
            if 0.0 <= value <= 1.0:
                st.session_state.param_values["old_photo_effect"]["noise"] = value
        except ValueError:
            pass
    
    age_value = current_params.get("age", 0.7)
    noise_value = current_params.get("noise", 0.3)
    
    col1, col2 = st.columns(2)
    with col1:
        st.slider("古さの強度", 0.0, 1.0, age_value, key="age_slider", on_change=update_age_text)
        st.slider("ノイズの量", 0.0, 1.0, noise_value, key="old_noise_slider", on_change=update_noise_text)
    with col2:
        st.text_input("古さの強度（手動入力）", str(age_value), key="age_text", on_change=update_age_slider)
        st.text_input("ノイズの量（手動入力）", str(noise_value), key="old_noise_text", on_change=update_noise_slider)
    
    return {
        "age": st.session_state.param_values["old_photo_effect"]["age"],
        "noise": st.session_state.param_values["old_photo_effect"]["noise"]
    }


def setup_neon_effect_params(current_params):
    """ネオン効果パラメータのUI要素を設定する。

    Args:
        current_params: 現在のパラメータ値

    Returns:
        更新されたパラメータ辞書
    """
    def update_glow_text():
        st.session_state.param_values["neon_effect"]["glow"] = st.session_state.glow_slider
        
    def update_glow_slider():
        try:
            value = float(st.session_state.glow_text)
            if 0.0 <= value <= 1.0:
                st.session_state.param_values["neon_effect"]["glow"] = value
        except ValueError:
            pass
    
    glow_value = current_params.get("glow", 0.7)
    
    col1, col2 = st.columns(2)
    with col1:
        st.slider("輝きの強さ", 0.0, 1.0, glow_value, key="glow_slider", on_change=update_glow_text)
    with col2:
        st.text_input("輝きの強さ（手動入力）", str(glow_value), key="glow_text", on_change=update_glow_slider)
    
    return {"glow": st.session_state.param_values["neon_effect"]["glow"]}


def setup_pixelate_params(current_params):
    """ピクセル化パラメータのUI要素を設定する。

    Args:
        current_params: 現在のパラメータ値

    Returns:
        更新されたパラメータ辞書
    """
    def update_pixel_size_text():
        st.session_state.param_values["pixelate"]["pixel_size"] = st.session_state.pixel_size_slider
        
    def update_pixel_size_slider():
        try:
            value = int(st.session_state.pixel_size_text)
            if 2 <= value <= 50:
                st.session_state.param_values["pixelate"]["pixel_size"] = value
        except ValueError:
            pass
    
    pixel_size_value = current_params.get("pixel_size", 10)
    
    col1, col2 = st.columns(2)
    with col1:
        st.slider("ピクセルサイズ", 2, 50, pixel_size_value, key="pixel_size_slider", on_change=update_pixel_size_text)
    with col2:
        st.text_input("ピクセルサイズ（手動入力）", str(pixel_size_value), key="pixel_size_text", on_change=update_pixel_size_slider)
    
    return {"pixel_size": st.session_state.param_values["pixelate"]["pixel_size"]}


def setup_cross_hatching_params(current_params):
    """クロスハッチングパラメータのUI要素を設定する。

    Args:
        current_params: 現在のパラメータ値

    Returns:
        更新されたパラメータ辞書
    """
    def update_line_spacing_text():
        st.session_state.param_values["cross_hatching"]["line_spacing"] = st.session_state.line_spacing_slider
        
    def update_line_spacing_slider():
        try:
            value = int(st.session_state.line_spacing_text)
            if 3 <= value <= 20:
                st.session_state.param_values["cross_hatching"]["line_spacing"] = value
        except ValueError:
            pass
    
    line_spacing_value = current_params.get("line_spacing", 6)
    
    col1, col2 = st.columns(2)
    with col1:
        st.slider("線の間隔", 3, 20, line_spacing_value, key="line_spacing_slider", on_change=update_line_spacing_text)
    with col2:
        st.text_input("線の間隔（手動入力）", str(line_spacing_value), key="line_spacing_text", on_change=update_line_spacing_slider)
    
    return {"line_spacing": st.session_state.param_values["cross_hatching"]["line_spacing"]}


def setup_params_ui(process_type):
    """処理タイプに対応するパラメータUI要素を表示する。

    Args:
        process_type: 処理タイプ

    Returns:
        更新されたパラメータ辞書
    """
    # セッション状態の確認と初期化
    if "param_values" not in st.session_state:
        st.session_state.param_values = {}
    
    # 現在の処理タイプのパラメータ値を確認
    if process_type not in st.session_state.param_values:
        st.session_state.param_values[process_type] = initialize_param_values(process_type)
    
    current_params = st.session_state.param_values[process_type]
    
    # 処理タイプに応じて対応するUI要素を表示
    if process_type == "rotate":
        return setup_rotate_params(current_params)
    elif process_type == "resize":
        return setup_resize_params(current_params)
    elif process_type == "blur":
        return setup_blur_params(current_params)
    elif process_type == "edge_detection":
        return setup_edge_detection_params(current_params)
    elif process_type in ["brightness", "contrast", "saturation"]:
        label = "明るさ" if process_type == "brightness" else "コントラスト" if process_type == "contrast" else "彩度"
        return setup_value_params(process_type, current_params, label)
    elif process_type == "mosaic":
        return setup_size_params(process_type, current_params, "ブロックサイズ", 2, 50)
    elif process_type == "threshold":
        return setup_size_params(process_type, current_params, "閾値", 0, 255)
    elif process_type in ["dilate", "erode", "opening", "closing", "tophat", "blackhat", "morphology_gradient"]:
        return setup_size_params(process_type, current_params, "カーネルサイズ", 1, 31, 2)
    elif process_type == "gamma":
        return setup_gamma_params(current_params)
    elif process_type == "denoise":
        return setup_denoise_params(current_params)
    elif process_type in ["adjust_red", "adjust_green", "adjust_blue"]:
        return setup_color_adjust_params(process_type, current_params)
    elif process_type == "segmentation":
        return setup_segmentation_params(current_params)
    elif process_type == "contour":
        return setup_contour_params(current_params)
    elif process_type == "posterize":
        return setup_posterize_params(current_params)
    elif process_type == "watercolor":
        return setup_effect_strength_params(process_type, current_params, "エフェクト強度")
    elif process_type == "oil_painting":
        return setup_effect_strength_params(process_type, current_params, "エフェクト強度")
    elif process_type == "sketch":
        return setup_sketch_params(current_params)
    elif process_type == "hough_lines":
        return setup_hough_lines_params(current_params)
    elif process_type == "probabilistic_hough_lines":
        return setup_prob_hough_lines_params(current_params)
    elif process_type == "hough_circles":
        return setup_hough_circles_params(current_params)
    elif process_type == "perspective_transform":
        return setup_perspective_transform_params(current_params)
    elif process_type == "feature_detection":
        return setup_feature_detection_params(current_params)
    elif process_type == "template_matching":
        return {}  # テンプレートは自動生成
    elif process_type in ["face_detection", "text_detection", "watershed", "barcode_detection"]:
        return {}  # パラメータなし
    # 新しい処理タイプ
    elif process_type == "add_gaussian_noise":
        return setup_gaussian_noise_params(current_params)
    elif process_type == "add_salt_pepper_noise":
        return setup_salt_pepper_noise_params(current_params)
    elif process_type == "cartoon_effect":
        return setup_cartoon_effect_params(current_params)
    elif process_type == "vignette_effect":
        return setup_vignette_effect_params(current_params)
    elif process_type == "color_temperature":
        return setup_color_temperature_params(current_params)
    elif process_type == "color_extraction":
        return setup_color_extraction_params(current_params)
    elif process_type == "lens_distortion":
        return setup_lens_distortion_params(current_params)
    elif process_type == "lens_correction":
        return setup_lens_correction_params(current_params)
    elif process_type == "miniature_effect":
        return setup_miniature_effect_params(current_params)
    elif process_type == "document_scan":
        return setup_document_scan_params(current_params)
    elif process_type == "glitch_effect":
        return setup_glitch_effect_params(current_params)
    elif process_type == "old_photo_effect":
        return setup_old_photo_effect_params(current_params)
    elif process_type == "neon_effect":
        return setup_neon_effect_params(current_params)
    elif process_type == "pixelate":
        return setup_pixelate_params(current_params)
    elif process_type == "cross_hatching":
        return setup_cross_hatching_params(current_params)
    else:
        return {}
