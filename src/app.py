#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
画像処理確認アプリのメインモジュール。

このモジュールはStreamlitを使用したUIを提供し、アプリケーションのエントリーポイントとなります。
"""

import os
import sys
import tempfile
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import threading
import tkinter as tk
from tkinter import filedialog
import datetime

# pyperclipがインストールされている場合のみインポート
try:
    import pyperclip
    PYPERCLIP_AVAILABLE = True
except ImportError:
    PYPERCLIP_AVAILABLE = False

from image_processor import ImageProcessor
from history_manager import HistoryManager
from code_generator import CodeGenerator


class AppUI:
    """アプリケーションのUIを管理するクラス。"""
    
    def __init__(self):
        """初期化メソッド。"""
        self.image_processor = ImageProcessor()
        self.history_manager = HistoryManager()
        self.code_generator = CodeGenerator()
        self.current_image_index = 0
        self.output_folder = None
        
        # セッション状態の初期化
        if 'images' not in st.session_state:
            st.session_state.images = []
        if 'current_image_index' not in st.session_state:
            st.session_state.current_image_index = 0
        if 'history' not in st.session_state:
            st.session_state.history = []
        if 'selected_history_index' not in st.session_state:
            st.session_state.selected_history_index = None
        if 'output_folder' not in st.session_state:
            st.session_state.output_folder = None
        
    def render(self):
        """UIをレンダリングする。"""
        st.title("画像処理確認アプリ")
        
        # サイドバー
        with st.sidebar:
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
                    st.session_state.input_folder_path = folder_path_input
                    st.session_state.images = self._load_images_from_folder(folder_path_input)
                    st.session_state.current_image_index = 0
                    st.success(f"フォルダから{len(st.session_state.images)}枚の画像を読み込みました。")
            
            # フォルダ選択（ダイアログ）
            try:
                if st.button("フォルダ選択ダイアログを開く"):
                    folder_path = self.select_folder()
                    if folder_path:
                        st.session_state.input_folder_path = folder_path
                        st.session_state.images = self._load_images_from_folder(folder_path)
                        st.session_state.current_image_index = 0
                        st.success(f"フォルダから{len(st.session_state.images)}枚の画像を読み込みました。")
                        # テキストボックスを更新するために再実行（バージョンに応じて適切なメソッドを使用）
                        try:
                            st.rerun()  # 新しいバージョンのStreamlit
                        except AttributeError:
                            try:
                                st.experimental_rerun()  # 古いバージョンのStreamlit
                            except AttributeError:
                                st.success("フォルダパスを更新するには、ページを手動で更新してください。")
            except Exception as e:
                st.error(f"フォルダ選択ダイアログでエラーが発生しました: {e}")
                st.info("代わりにパス入力を使用してください。")
            
            # 出力フォルダ選択（テキスト入力）
            output_folder_input = st.text_input("出力フォルダのパス", value=st.session_state.output_folder_path)
            if output_folder_input:
                if st.button("出力フォルダを設定"):
                    if not os.path.exists(output_folder_input):
                        os.makedirs(output_folder_input)
                    st.session_state.output_folder = output_folder_input
                    st.session_state.output_folder_path = output_folder_input
                    st.success(f"出力フォルダを設定しました: {output_folder_input}")
            
            # 出力フォルダ選択（ダイアログ）
            try:
                if st.button("出力フォルダ選択ダイアログを開く"):
                    output_folder = self.select_output_folder()
                    if output_folder:
                        st.session_state.output_folder = output_folder
                        st.session_state.output_folder_path = output_folder
                        st.success(f"出力フォルダを設定しました: {output_folder}")
                        # テキストボックスを更新するために再実行（バージョンに応じて適切なメソッドを使用）
                        try:
                            st.rerun()  # 新しいバージョンのStreamlit
                        except AttributeError:
                            try:
                                st.experimental_rerun()  # 古いバージョンのStreamlit
                            except AttributeError:
                                st.success("フォルダパスを更新するには、ページを手動で更新してください。")
            except Exception as e:
                st.error(f"フォルダ選択ダイアログでエラーが発生しました: {e}")
                st.info("代わりにパス入力を使用してください。")
            
            # 処理タイプの選択
            st.header("処理を選択")
            process_type = st.selectbox(
                "処理タイプ",
                [
                    "grayscale", "invert", "rotate", "resize", "blur",
                    "sharpen", "edge_detection", "brightness", "contrast",
                    "saturation", "sepia", "emboss", "mosaic",
                    "threshold", "dilate", "erode", "flip_horizontal", "flip_vertical"
                ]
            )
            
            # 処理パラメータの設定
            params = {}
            if process_type == "rotate":
                col1, col2 = st.columns(2)
                with col1:
                    params["angle"] = st.slider("回転角度", -180, 180, 90)
                with col2:
                    angle_input = st.text_input("回転角度（手動入力）", "90")
                    try:
                        params["angle"] = int(angle_input)
                    except ValueError:
                        st.warning("回転角度には数値を入力してください")
            elif process_type == "resize":
                col1, col2 = st.columns(2)
                with col1:
                    params["width"] = st.slider("幅", 10, 1000, 300)
                    params["height"] = st.slider("高さ", 10, 1000, 300)
                with col2:
                    width_input = st.text_input("幅（手動入力）", "300")
                    height_input = st.text_input("高さ（手動入力）", "300")
                    try:
                        params["width"] = int(width_input)
                        params["height"] = int(height_input)
                    except ValueError:
                        st.warning("幅と高さには数値を入力してください")
            elif process_type == "blur":
                col1, col2 = st.columns(2)
                with col1:
                    params["kernel_size"] = st.slider("カーネルサイズ", 1, 31, 5, step=2)
                    params["blur_type"] = st.selectbox("ぼかしの種類", ["gaussian", "median", "box", "bilateral"])
                with col2:
                    kernel_input = st.text_input("カーネルサイズ（手動入力）", "5")
                    try:
                        kernel_size = int(kernel_input)
                        if kernel_size % 2 == 1:  # カーネルサイズは奇数である必要がある
                            params["kernel_size"] = kernel_size
                        else:
                            st.warning("カーネルサイズは奇数である必要があります")
                    except ValueError:
                        st.warning("カーネルサイズには数値を入力してください")
            elif process_type == "edge_detection":
                params["method"] = st.selectbox("検出方法", ["canny", "sobel", "laplacian"])
            elif process_type == "brightness":
                col1, col2 = st.columns(2)
                with col1:
                    params["value"] = st.slider("明るさ", 0.1, 3.0, 1.5)
                with col2:
                    value_input = st.text_input("明るさ（手動入力）", "1.5")
                    try:
                        params["value"] = float(value_input)
                    except ValueError:
                        st.warning("明るさには数値を入力してください")
            elif process_type == "contrast":
                col1, col2 = st.columns(2)
                with col1:
                    params["value"] = st.slider("コントラスト", 0.1, 3.0, 1.5)
                with col2:
                    value_input = st.text_input("コントラスト（手動入力）", "1.5")
                    try:
                        params["value"] = float(value_input)
                    except ValueError:
                        st.warning("コントラストには数値を入力してください")
            elif process_type == "saturation":
                col1, col2 = st.columns(2)
                with col1:
                    params["value"] = st.slider("彩度", 0.1, 3.0, 1.5)
                with col2:
                    value_input = st.text_input("彩度（手動入力）", "1.5")
                    try:
                        params["value"] = float(value_input)
                    except ValueError:
                        st.warning("彩度には数値を入力してください")
            elif process_type == "mosaic":
                col1, col2 = st.columns(2)
                with col1:
                    params["block_size"] = st.slider("ブロックサイズ", 1, 50, 10)
                with col2:
                    block_input = st.text_input("ブロックサイズ（手動入力）", "10")
                    try:
                        params["block_size"] = int(block_input)
                    except ValueError:
                        st.warning("ブロックサイズには数値を入力してください")
            elif process_type == "threshold":
                col1, col2 = st.columns(2)
                with col1:
                    params["thresh_value"] = st.slider("閾値", 0, 255, 127)
                with col2:
                    thresh_input = st.text_input("閾値（手動入力）", "127")
                    try:
                        params["thresh_value"] = int(thresh_input)
                    except ValueError:
                        st.warning("閾値には数値を入力してください")
            elif process_type == "dilate" or process_type == "erode":
                col1, col2 = st.columns(2)
                with col1:
                    params["kernel_size"] = st.slider("カーネルサイズ", 1, 31, 5, step=2)
                with col2:
                    kernel_input = st.text_input("カーネルサイズ（手動入力）", "5")
                    try:
                        kernel_size = int(kernel_input)
                        if kernel_size % 2 == 1:  # カーネルサイズは奇数である必要がある
                            params["kernel_size"] = kernel_size
                        else:
                            st.warning("カーネルサイズは奇数である必要があります")
                    except ValueError:
                        st.warning("カーネルサイズには数値を入力してください")
            
            # 適用ボタン
            if st.button("適用"):
                if len(st.session_state.images) > 0:
                    current_image = st.session_state.images[st.session_state.current_image_index]
                    processed_image = self.image_processor.apply_processing(current_image, process_type, params)
                    
                    # 履歴に追加
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    history_entry = {
                        "timestamp": timestamp,
                        "process_type": process_type,
                        "params": params,
                        "original_image": current_image,
                        "processed_image": processed_image
                    }
                    st.session_state.history.append(history_entry)
                    st.session_state.selected_history_index = len(st.session_state.history) - 1
                    
                    st.success("処理を適用しました。")
                else:
                    st.error("画像が読み込まれていません。フォルダを選択してください。")
            
            # コード生成
            st.header("コード生成")
            library = st.selectbox(
                "ライブラリ",
                ["opencv", "pillow", "scipy", "scikit-image", "numpy"]
            )
            
            if st.button("コード生成"):
                if st.session_state.selected_history_index is not None:
                    history_entry = st.session_state.history[st.session_state.selected_history_index]
                    self.code_generator = CodeGenerator(library=library)
                    code = self.code_generator.generate_code(
                        history_entry["process_type"],
                        history_entry["params"]
                    )
                    st.session_state.generated_code = code
                    st.success("コードを生成しました。")
                else:
                    st.error("履歴が選択されていません。")
            
            if st.button("コードをコピー"):
                if "generated_code" in st.session_state:
                    if PYPERCLIP_AVAILABLE:
                        pyperclip.copy(st.session_state.generated_code)
                        st.success("コードをクリップボードにコピーしました。")
                    else:
                        st.warning("pyperclipがインストールされていないため、クリップボードにコピーできません。pip install pyperclipでインストールしてください。")
                else:
                    st.error("生成されたコードがありません。")
            
            if st.button("コードをダウンロード"):
                if "generated_code" in st.session_state:
                    if st.session_state.output_folder:
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"image_processing_{timestamp}.py"
                        file_path = os.path.join(st.session_state.output_folder, filename)
                        self.code_generator.save_to_file(st.session_state.generated_code, file_path)
                        st.success(f"コードを保存しました: {file_path}")
                    else:
                        st.error("出力フォルダが設定されていません。")
                else:
                    st.error("生成されたコードがありません。")
        
        # メインコンテンツ
        col1, col2 = st.columns(2)
        
        # 画像表示
        if len(st.session_state.images) > 0:
            with col1:
                st.subheader("元画像")
                current_image = st.session_state.images[st.session_state.current_image_index]
                st.image(self._convert_to_pil(current_image), use_container_width=True)
                
                # 画像ナビゲーションボタン
                nav_col1, nav_col2 = st.columns(2)
                with nav_col1:
                    if st.button("前の画像"):
                        self.navigate_images("prev")
                with nav_col2:
                    if st.button("次の画像"):
                        self.navigate_images("next")
            
            with col2:
                st.subheader("処理後画像")
                if st.session_state.selected_history_index is not None:
                    history_entry = st.session_state.history[st.session_state.selected_history_index]
                    processed_image = history_entry["processed_image"]
                    st.image(self._convert_to_pil(processed_image), use_container_width=True)
                else:
                    st.info("処理を適用するか、履歴から選択してください。")
        else:
            st.info("フォルダを選択して画像を読み込んでください。")
        
        # 履歴表示
        st.header("処理履歴")
        if len(st.session_state.history) > 0:
            # 処理の1つ前に戻るボタン
            if st.button("処理の1つ前に戻る") and st.session_state.selected_history_index is not None:
                if st.session_state.selected_history_index > 0:
                    st.session_state.selected_history_index -= 1
                    st.success("1つ前の処理に戻りました。")
                else:
                    st.warning("これ以上前の処理はありません。")
            
            # 履歴選択
            history_options = [
                f"{i+1}. {entry['timestamp']} - {entry['process_type']}"
                for i, entry in enumerate(st.session_state.history)
            ]
            selected_history = st.selectbox(
                "履歴を選択",
                history_options,
                index=st.session_state.selected_history_index if st.session_state.selected_history_index is not None else 0
            )
            if selected_history:
                selected_index = history_options.index(selected_history)
                st.session_state.selected_history_index = selected_index
                
                # 選択された履歴の詳細を表示
                history_entry = st.session_state.history[selected_index]
                st.write(f"処理タイプ: {history_entry['process_type']}")
                st.write(f"パラメータ: {history_entry['params']}")
                
                # プログレスバー
                st.progress((selected_index + 1) / len(st.session_state.history))
        else:
            st.info("履歴がありません。")
        
        # コード表示
        if "generated_code" in st.session_state:
            st.header("生成されたコード")
            st.code(st.session_state.generated_code, language="python")
        
    def select_folder(self):
        """フォルダ選択ダイアログを表示する。

        Returns:
            選択されたフォルダのパス、キャンセルされた場合はNone
        """
        try:
            # フォルダ選択ダイアログをメインスレッドで実行
            root = tk.Tk()
            root.withdraw()
            root.wm_attributes('-topmost', 1)
            folder_path = None
            def ask_directory():
                nonlocal folder_path
                folder_path = filedialog.askdirectory()
            threading.Thread(target=ask_directory).start()
            root.mainloop()  # メインループを開始
            root.destroy()
            if folder_path:
                # パスを正規化して返す
                folder_path = os.path.normpath(folder_path)
            return folder_path if folder_path else None
        except Exception as e:
            st.error(f"フォルダ選択エラー: {type(e).__name__} - {e}")
            return None
        
    def select_output_folder(self):
        """出力フォルダ選択ダイアログを表示する。

        Returns:
            選択された出力フォルダのパス、キャンセルされた場合はNone
        """
        try:
            # 出力フォルダ選択ダイアログをメインスレッドで実行
            root = tk.Tk()
            root.withdraw()
            root.wm_attributes('-topmost', 1)
            folder_path = None
            def ask_directory():
                nonlocal folder_path
                folder_path = filedialog.askdirectory()
            threading.Thread(target=ask_directory).start()
            root.mainloop()  # メインループを開始
            root.destroy()
            if folder_path:
                # パスを正規化して返す
                folder_path = os.path.normpath(folder_path)
            return folder_path if folder_path else None
        except Exception as e:
            st.error(f"フォルダ選択エラー: {type(e).__name__} - {e}")
            return None
        
    def display_images(self, original, processed):
        """元画像と処理後画像を表示する。

        Args:
            original: 元画像
            processed: 処理後画像
        """
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("元画像")
            st.image(self._convert_to_pil(original), use_container_width=True)
        with col2:
            st.subheader("処理後画像")
            st.image(self._convert_to_pil(processed), use_container_width=True)
            
    def navigate_images(self, direction):
        """画像を前後に切り替える。
        
        Args:
            direction: 'next' または 'prev'
        """
        if len(st.session_state.images) == 0:
            return
            
        # 現在の画像インデックスを保存
        old_index = st.session_state.current_image_index
            
        if direction == "next":
            st.session_state.current_image_index = (st.session_state.current_image_index + 1) % len(st.session_state.images)
        elif direction == "prev":
            st.session_state.current_image_index = (st.session_state.current_image_index - 1) % len(st.session_state.images)
            
        # 画像が変更された場合、新しい画像に対して最後に適用された処理を適用
        if old_index != st.session_state.current_image_index and len(st.session_state.history) > 0:
            # 最後に適用された処理を取得
            latest_entry = st.session_state.history[-1]
            current_image = st.session_state.images[st.session_state.current_image_index]
            
            # 同じ処理を新しい画像に適用
            with st.spinner('処理を適用中...'):
                processed_image = self.image_processor.apply_processing(
                    current_image,
                    latest_entry["process_type"],
                    latest_entry["params"]
                )
            
            # 履歴に追加
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            history_entry = {
                "timestamp": timestamp,
                "process_type": latest_entry["process_type"],
                "params": latest_entry["params"],
                "original_image": current_image,
                "processed_image": processed_image
            }
            st.session_state.history.append(history_entry)
            st.session_state.selected_history_index = len(st.session_state.history) - 1
            
    def display_history(self):
        """履歴を表示する。"""
        st.header("処理履歴")
        if len(st.session_state.history) > 0:
            for i, entry in enumerate(st.session_state.history):
                if st.button(f"{i+1}. {entry['timestamp']} - {entry['process_type']}"):
                    st.session_state.selected_history_index = i
        else:
            st.info("履歴がありません。")
            
    def display_code(self, code):
        """生成されたコードを表示する。

        Args:
            code: 表示するコード
        """
        st.header("生成されたコード")
        st.code(code, language="python")
        
    def download_code(self, code, filename=None):
        """コードをファイルとしてダウンロードする。
        
        Args:
            code: ダウンロードするコード
            filename: 保存するファイル名（Noneの場合は自動生成）
        """
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"image_processing_{timestamp}.py"
            
        if st.session_state.output_folder:
            file_path = os.path.join(st.session_state.output_folder, filename)
            self.code_generator.save_to_file(code, file_path)
            st.success(f"コードを保存しました: {file_path}")
        else:
            st.error("出力フォルダが設定されていません。")
            
    def _load_images_from_folder(self, folder_path):
        """フォルダから画像を読み込む。

        Args:
            folder_path: 画像フォルダのパス

        Returns:
            読み込まれた画像のリスト
        """
        images = []
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            _, ext = os.path.splitext(filename)
            if ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
                try:
                    # OpenCVで直接読み込みを試みる
                    img = cv2.imread(file_path)
                    
                    # 読み込みに失敗した場合はPILを使用
                    if img is None:
                        try:
                            pil_img = Image.open(file_path)
                            img = np.array(pil_img)
                            # RGBからBGRに変換（OpenCVはBGR形式）
                            if len(img.shape) == 3 and img.shape[2] == 3:
                                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                            # 情報メッセージを削除（UIがクリアになるように）
                        except Exception as e:
                            st.error(f"PILでの画像読み込みに失敗しました: {filename} - {e}")
                            continue
                    
                    if img is not None:
                        images.append(img)
                    else:
                        st.error(f"画像の読み込みに失敗しました: {filename}")
                except Exception as e:
                    st.error(f"画像の読み込みに失敗しました: {filename} - {e}")
        
        if not images:
            st.warning(f"フォルダ内に読み込み可能な画像がありませんでした: {folder_path}")
        
        return images
        
    def _convert_to_pil(self, image):
        """OpenCV画像をPIL画像に変換する。

        Args:
            image: OpenCV画像

        Returns:
            PIL画像
        """
        if image is None:
            return None
            
        if len(image.shape) == 2:
            # グレースケール画像
            return Image.fromarray(image)
        else:
            # カラー画像（BGR -> RGB）
            return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def main():
    """アプリケーションのエントリーポイント。"""
    app = AppUI()
    app.render()


if __name__ == "__main__":
    main()