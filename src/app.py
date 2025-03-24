#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
画像処理確認アプリケーション。

このアプリケーションは様々な画像処理をインタラクティブに試すことができます。
"""

import os
import sys
import json
import copy
import datetime
import streamlit as st
import numpy as np
import cv2
from PIL import Image

# 自作モジュールのインポート
from image_processor import ImageProcessor
import ui_components
import param_manager
from history_manager import HistoryManager
from code_generator import CodeGeneratorUI
import image_utils

# PyperClipがインストールされているか確認
try:
    import pyperclip
    PYPERCLIP_AVAILABLE = True
except ImportError:
    PYPERCLIP_AVAILABLE = False

# 処理タイプの英語名と日本語名の対応
PROCESS_TYPES = {
    "グレースケール": "grayscale",
    "色反転": "invert",
    "回転": "rotate",
    "リサイズ": "resize",
    "左右反転": "flip_horizontal",
    "上下反転": "flip_vertical",
    "ぼかし": "blur",
    "シャープ化": "sharpen",
    "エッジ検出": "edge_detection",
    "ノイズ除去": "denoise",
    "明るさ調整": "brightness",
    "コントラスト調整": "contrast",
    "彩度調整": "saturation",
    "ガンマ補正": "gamma_correction",
    "ヒストグラム均等化": "histogram_equalization",
    "赤色調整": "adjust_red",
    "緑色調整": "adjust_green",
    "青色調整": "adjust_blue",
    "エンボス効果": "emboss",
    "モザイク": "mosaic",
    "2値化": "threshold",
    "セピア調": "sepia",
    "水彩画風": "watercolor",
    "油絵風": "oil_painting",
    "スケッチ風": "sketch",
    "ポスタリゼーション": "posterize",
    "膨張処理": "dilate",
    "収縮処理": "erode",
    "オープニング": "opening",
    "クロージング": "closing",
    "トップハット変換": "tophat",
    "ブラックハット変換": "blackhat",
    "モルフォロジー勾配": "morphology_gradient",
    "マーカー付き分水嶺法": "watershed",
    "ハフ変換（直線）": "hough_lines",
    "確率的ハフ変換": "probabilistic_hough_lines",
    "ハフ変換（円）": "hough_circles",
    "輪郭抽出": "contour_detection",
    "セグメンテーション": "segmentation",
    "テンプレートマッチング": "template_matching",
    "特徴点検出": "feature_detection",
    "顔検出": "face_detection",
    "テキスト検出": "text_detection",
    "透視変換": "perspective_transform",
    "バーコード/QRコード検出": "barcode_detection",
    "ガウシアンノイズ追加": "add_gaussian_noise",
    "ソルト＆ペッパーノイズ追加": "add_salt_pepper_noise",
    "カートゥーン効果": "cartoon_effect",
    "ビネット効果": "vignette_effect",
    "色温度調整": "color_temperature",
    "特定色抽出": "color_extraction",
    "文書スキャン最適化": "document_scan",
    "レンズ歪み追加": "lens_distortion",
    "レンズ歪み補正": "lens_correction",
    "ミニチュア効果": "miniature_effect",
    "グリッチ効果": "glitch_effect",
    "古い写真効果": "old_photo_effect",
    "ネオン効果": "neon_effect",
    "ピクセル化": "pixelate",
    "クロスハッチング": "cross_hatching"
}

# 処理タイプの英語名と日本語名の対応（逆引き）
PROCESS_TYPES_REVERSE = {v: k for k, v in PROCESS_TYPES.items()}


class ImageProcessingApp:
    """画像処理アプリケーションのメインクラス。"""
    
    def __init__(self):
        """インスタンスを初期化する。"""
        self.image_processor = ImageProcessor()
        self.history_manager = HistoryManager()
        self.code_generator_ui = CodeGeneratorUI()
        
    def run(self):
        """アプリケーションを実行する。"""
        # セッション状態の初期化
        if 'images' not in st.session_state:
            st.session_state.images = []
        if 'original_images' not in st.session_state:
            st.session_state.original_images = []
        if 'current_image_index' not in st.session_state:
            st.session_state.current_image_index = 0
        if 'history' not in st.session_state:
            st.session_state.history = []
        if 'selected_history_index' not in st.session_state:
            st.session_state.selected_history_index = None
        if 'chain_count' not in st.session_state:
            st.session_state.chain_count = {}
        if 'output_folder' not in st.session_state:
            st.session_state.output_folder = ""
        if 'temp_processed_image' not in st.session_state:
            st.session_state.temp_processed_image = None
            
        # ヘッダー
        ui_components.show_header()
        
        # サイドバーとメインコンテンツの配置
        with st.sidebar:
            self._render_sidebar()
            
        # メインコンテンツの表示
        self._render_main_content()
        
    def _render_sidebar(self):
        """サイドバーの内容を表示する。"""
        st.header("設定")
        
        # セッション状態の初期化
        if 'input_folder_path' not in st.session_state:
            st.session_state.input_folder_path = ""
        if 'output_folder_path' not in st.session_state:
            st.session_state.output_folder_path = ""
        
        # フォルダ選択（テキスト入力）
        folder_path_input = st.text_input("画像フォルダのパス", value=st.session_state.input_folder_path)
        if folder_path_input and os.path.isdir(folder_path_input):
            if st.button("パスから読み込む"):
                try:
                    with st.spinner(f"フォルダから画像を読み込んでいます: {folder_path_input}"):
                        st.session_state.input_folder_path = folder_path_input
                        images = image_utils.load_images_from_folder(folder_path_input)
                        if images:
                            st.session_state.images = images
                            # オリジナル画像のバックアップを保存
                            st.session_state.original_images = copy.deepcopy(images)
                            st.session_state.current_image_index = 0
                            st.session_state.history = []  # 履歴をクリア
                            st.session_state.selected_history_index = None
                            st.session_state.chain_count = {}  # 処理連鎖カウントをリセット
                            st.session_state.temp_processed_image = None  # 一時的な処理結果をクリア
                            st.success(f"フォルダから{len(images)}枚の画像を読み込みました。")
                            
                            # 読み込み後にデフォルト処理を適用
                            self._apply_default_processing()
                        else:
                            st.error(f"フォルダ内に読み込み可能な画像ファイルがありません: {folder_path_input}")
                except Exception as e:
                    st.error(f"画像の読み込み中にエラーが発生しました: {str(e)}")
        
        # フォルダ選択（ダイアログ）
        if st.button("フォルダ選択"):
            try:
                folder_path = image_utils.select_folder_simple()
                if folder_path and os.path.isdir(folder_path):
                    st.session_state.input_folder_path = folder_path
                    with st.spinner(f"フォルダから画像を読み込んでいます: {folder_path}"):
                        images = image_utils.load_images_from_folder(folder_path)
                        if images:
                            st.session_state.images = images
                            # オリジナル画像のバックアップを保存
                            st.session_state.original_images = copy.deepcopy(images)
                            st.session_state.current_image_index = 0
                            st.session_state.history = []  # 履歴をクリア
                            st.session_state.selected_history_index = None
                            st.session_state.chain_count = {}  # 処理連鎖カウントをリセット
                            st.session_state.temp_processed_image = None  # 一時的な処理結果をクリア
                            st.success(f"フォルダから{len(images)}枚の画像を読み込みました。")
                            st.success("フォルダパスを更新するには、ページを手動で更新してください。")
                            
                            # 読み込み後にデフォルト処理を適用
                            self._apply_default_processing()
                        else:
                            st.error(f"フォルダ内に読み込み可能な画像ファイルがありません: {folder_path}")
            except Exception as e:
                st.error(f"フォルダ選択または画像読み込み中にエラーが発生しました: {str(e)}")
        
        # 出力フォルダ選択（テキスト入力）
        output_folder_input = st.text_input("出力フォルダのパス", value=st.session_state.output_folder_path)
        if output_folder_input:
            if st.button("出力フォルダを設定"):
                try:
                    if not os.path.exists(output_folder_input):
                        os.makedirs(output_folder_input)
                    st.session_state.output_folder = output_folder_input
                    st.session_state.output_folder_path = output_folder_input
                    st.success(f"出力フォルダを設定しました: {output_folder_input}")
                except Exception as e:
                    st.error(f"出力フォルダの設定中にエラーが発生しました: {str(e)}")
        
        # 出力フォルダ選択（ダイアログ）
        if st.button("出力フォルダ選択"):
            try:
                output_folder = image_utils.select_folder_simple()
                if output_folder:
                    st.session_state.output_folder = output_folder
                    st.session_state.output_folder_path = output_folder
                    st.success(f"出力フォルダを設定しました: {output_folder}")
                    st.success("フォルダパスを更新するには、ページを手動で更新してください。")
            except Exception as e:
                st.error(f"出力フォルダ選択中にエラーが発生しました: {str(e)}")
        
        # 処理タイプの選択
        st.header("処理を選択")
        
        # 処理タイプのカテゴリー分け
        categories = {
            "基本処理": ["グレースケール", "色反転", "回転", "リサイズ", "左右反転", "上下反転"],
            "フィルタ処理": ["ぼかし", "シャープ化", "エッジ検出", "ノイズ除去", "ガウシアンノイズ追加", "ソルト＆ペッパーノイズ追加",
                         "文書スキャン最適化", "レンズ歪み追加", "レンズ歪み補正"],
            "色調整": ["明るさ調整", "コントラスト調整", "彩度調整", "ガンマ補正", "ヒストグラム均等化", 
                     "赤色調整", "緑色調整", "青色調整", "セピア調", "色温度調整", "特定色抽出"],
            "エフェクト": ["エンボス効果", "モザイク", "2値化", "水彩画風", "油絵風", "スケッチ風", "ポスタリゼーション",
                        "カートゥーン効果", "ビネット効果", "ミニチュア効果", "グリッチ効果", "古い写真効果", "ネオン効果",
                        "ピクセル化", "クロスハッチング"],
            "モルフォロジー変換": ["膨張処理", "収縮処理", "オープニング", "クロージング", "トップハット変換", 
                              "ブラックハット変換", "モルフォロジー勾配", "マーカー付き分水嶺法"],
            "特徴検出": ["ハフ変換（直線）", "確率的ハフ変換", "ハフ変換（円）", "輪郭抽出", "セグメンテーション", 
                        "テンプレートマッチング", "特徴点検出", "顔検出", "テキスト検出", "透視変換", "バーコード/QRコード検出"]
        }
        
        # カテゴリー選択
        selected_category = st.selectbox("カテゴリー", list(categories.keys()))
        
        # 選択されたカテゴリーの処理タイプを表示
        process_type_jp = st.selectbox(
            "処理タイプ",
            categories[selected_category]
        )
        
        # 英語名に変換
        process_type = PROCESS_TYPES[process_type_jp]
        
        # 処理パラメータの設定
        params = param_manager.setup_params_ui(process_type)
        
        # 適用ボタン
        if st.button("適用"):
            self._apply_processing(process_type, params)
        
        # コード生成UI
        generated_code_ready, generated_code = self.code_generator_ui.render_ui(
            st.session_state.history, 
            PYPERCLIP_AVAILABLE
        )
                
    def _render_main_content(self):
        """メインコンテンツの内容を表示する。"""
        # 画像表示
        prev_button, next_button, use_processed_button, restore_original_button = ui_components.show_image_display(
            st.session_state.images,
            st.session_state.current_image_index,
            st.session_state.history,
            st.session_state.selected_history_index
        )
        
        # 画像ナビゲーションボタンの処理
        if prev_button:
            st.session_state.current_image_index = image_utils.navigate_images(
                st.session_state.current_image_index,
                st.session_state.images,
                "prev"
            )
            # 画像が変わったら処理も更新（履歴には追加しない）
            if len(st.session_state.history) > 0 and st.session_state.selected_history_index is not None:
                self._reapply_selected_processing()
        if next_button:
            st.session_state.current_image_index = image_utils.navigate_images(
                st.session_state.current_image_index,
                st.session_state.images,
                "next"
            )
            # 画像が変わったら処理も更新（履歴には追加しない）
            if len(st.session_state.history) > 0 and st.session_state.selected_history_index is not None:
                self._reapply_selected_processing()
        
        # 処理後画像を元画像として使用するボタンの処理
        if use_processed_button:
            self._use_processed_image_as_source()
            
        # オリジナル画像に戻すボタンの処理
        if restore_original_button:
            self._restore_original_image()
        
        # 履歴表示
        st.header("処理履歴")
        
        # 履歴ナビゲーションボタン
        prev_clicked, next_clicked = ui_components.show_history_navigation(
            st.session_state.history,
            st.session_state.selected_history_index
        )
        
        # 履歴ナビゲーションボタンの処理
        if prev_clicked and st.session_state.selected_history_index > 0:
            st.session_state.selected_history_index -= 1
        if next_clicked and st.session_state.selected_history_index < len(st.session_state.history) - 1:
            st.session_state.selected_history_index += 1
        
        # 履歴リスト
        selected_index = ui_components.show_history_list(
            st.session_state.history,
            st.session_state.selected_history_index,
            PROCESS_TYPES_REVERSE
        )
        
        if selected_index is not None:
            st.session_state.selected_history_index = selected_index
        
        # コード表示
        if "generated_code" in st.session_state:
            ui_components.show_code_display(st.session_state.generated_code)
            
    def _apply_processing(self, process_type, params, chain_info=None, source_history_index=None):
        """処理を適用する。
        
        Args:
            process_type: 処理タイプ
            params: 処理パラメータ
            chain_info: 処理連鎖の情報（オプション）
            source_history_index: 元となる履歴エントリのインデックス（オプション）
        
        Returns:
            bool: 処理が成功したかどうか
        """
        if len(st.session_state.images) == 0:
            st.error("画像が読み込まれていません。フォルダを選択してください。")
            return False
            
        try:
            with st.spinner("処理を適用中..."):
                current_image = st.session_state.images[st.session_state.current_image_index]
                processed_image = self.image_processor.apply_processing(current_image, process_type, params)
                
                # 処理連鎖情報を更新
                if chain_info is None:
                    chain_info = ""
                
                # 履歴に追加
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                history_entry = {
                    "timestamp": timestamp,
                    "process_type": process_type,
                    "params": params.copy(),  # paramsの値をコピーして保存
                    "processed_image": processed_image,
                    "image_index": st.session_state.current_image_index,  # 元画像のインデックスを保存
                    "chain_info": chain_info,  # 処理連鎖情報
                    "source_history_index": source_history_index  # 元となる履歴エントリのインデックス
                }
                st.session_state.history.append(history_entry)
                st.session_state.selected_history_index = len(st.session_state.history) - 1
                
                # 一時的な処理結果をクリア
                st.session_state.temp_processed_image = None
                
                # 日本語名で処理タイプを表示
                process_type_jp = PROCESS_TYPES_REVERSE.get(process_type, process_type)
                st.success(f"{process_type_jp}処理を適用しました。")
                return True
        except Exception as e:
            st.error(f"処理適用中にエラーが発生しました: {str(e)}")
            return False
            
    def _apply_default_processing(self):
        """デフォルトの処理を適用する。（グレースケール）"""
        if len(st.session_state.images) > 0 and len(st.session_state.history) == 0:
            process_type = "grayscale"
            params = {}
            self._apply_processing(process_type, params)
            
    def _reapply_selected_processing(self):
        """選択中の処理を現在の画像に再適用する（履歴に追加せず表示のみ）。"""
        if len(st.session_state.history) == 0 or st.session_state.selected_history_index is None:
            return
            
        try:
            # 選択中の履歴エントリから処理情報を取得
            selected_entry = st.session_state.history[st.session_state.selected_history_index]
            process_type = selected_entry["process_type"]
            params = selected_entry["params"]
            
            # 現在の画像に処理を適用するが、履歴には追加しない
            current_image = st.session_state.images[st.session_state.current_image_index]
            processed_image = self.image_processor.apply_processing(current_image, process_type, params)
            
            # 一時的な処理結果を保存
            st.session_state.temp_processed_image = processed_image
        except Exception as e:
            st.error(f"処理の再適用中にエラーが発生しました: {str(e)}")
        
    def _use_processed_image_as_source(self):
        """処理後画像を元画像として使用する。"""
        if len(st.session_state.history) == 0 or st.session_state.selected_history_index is None:
            st.error("処理履歴がありません。処理を適用してください。")
            return
            
        try:
            # 選択中の履歴エントリから処理後画像を取得
            history_index = st.session_state.selected_history_index
            history_entry = st.session_state.history[history_index]
            
            # 一時的な処理結果がある場合はそれを使用
            if hasattr(st.session_state, 'temp_processed_image') and st.session_state.temp_processed_image is not None:
                processed_image = st.session_state.temp_processed_image
            else:
                processed_image = history_entry["processed_image"]
            
            # 現在のインデックスを保存
            current_index = st.session_state.current_image_index
            
            # 画像を複製して元画像リストに追加
            st.session_state.images[current_index] = processed_image.copy()
            
            # 処理連鎖カウントを更新
            if current_index not in st.session_state.chain_count:
                st.session_state.chain_count[current_index] = 1
            else:
                st.session_state.chain_count[current_index] += 1
                
            chain_level = st.session_state.chain_count[current_index]
            
            # 新しい処理連鎖情報を作成
            chain_info = f"連鎖レベル {chain_level}"
            
            # 処理連鎖が深すぎる場合は警告
            if chain_level > 5:
                st.warning(f"処理連鎖が深くなっています（レベル {chain_level}）。処理の累積によって画質が劣化する可能性があります。")
            
            st.success(f"処理後画像を元画像として設定しました。連鎖レベル: {chain_level}")
            
            # 一時的な処理結果をクリア
            st.session_state.temp_processed_image = None
            
            # 再レンダリングのフラグを設定
            st.session_state.needs_rerender = True
            st.rerun()
            
        except Exception as e:
            st.error(f"処理後画像の設定中にエラーが発生しました: {str(e)}")
            
    def _restore_original_image(self):
        """オリジナル画像に戻す。"""
        if len(st.session_state.original_images) == 0:
            st.error("オリジナル画像がありません。")
            return
            
        try:
            # 現在のインデックスを保存
            current_index = st.session_state.current_image_index
            
            # オリジナル画像を元画像リストに戻す
            if current_index < len(st.session_state.original_images):
                st.session_state.images[current_index] = st.session_state.original_images[current_index].copy()
                
                # 処理連鎖カウントをリセット
                if current_index in st.session_state.chain_count:
                    st.session_state.chain_count[current_index] = 0
                
                # 一時的な処理結果をクリア
                st.session_state.temp_processed_image = None
                
                st.success("オリジナル画像に戻しました。")
                
                # 再レンダリングのフラグを設定
                st.session_state.needs_rerender = True
                st.rerun()
            else:
                st.error("インデックスが範囲外です。")
        except Exception as e:
            st.error(f"オリジナル画像の復元中にエラーが発生しました: {str(e)}")


def main():
    """アプリケーションのエントリーポイント。"""
    app = ImageProcessingApp()
    app.run()


if __name__ == "__main__":
    main()