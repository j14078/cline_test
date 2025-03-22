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
