#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
画像処理確認アプリのコード生成UI管理モジュール。

このモジュールはコード生成に関連するUI要素とロジックを担当します。
"""

import os
import datetime
import streamlit as st


class CodeGeneratorUI:
    """コード生成UI機能を提供するクラス。"""
    
    def __init__(self, code_generator):
        """初期化メソッド。
        
        Args:
            code_generator: CodeGeneratorクラスのインスタンス
        """
        self.code_generator = code_generator
    
    def render_ui(self, history, pyperclip_available=False):
        """コード生成UIを表示する。
        
        Args:
            history: 処理履歴
            pyperclip_available: クリップボードコピー機能の有無
            
        Returns:
            (bool, str): コード生成されたか、生成されたコード
        """
        st.header("コード生成")
        
        generated_code = None
        
        if st.button("コード生成"):
            if len(history) > 0:
                try:
                    # 履歴を使用して複合処理コードを生成
                    generated_code = self.generate_combined_code(history)
                    st.session_state.generated_code = generated_code
                    st.success("コードを生成しました。")
                    return True, generated_code
                except Exception as e:
                    st.error(f"コード生成中にエラーが発生しました: {str(e)}")
            else:
                st.error("履歴がありません。処理を適用してください。")
        
        # コードのコピーとダウンロード機能
        if "generated_code" in st.session_state:
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("コードをコピー"):
                    if pyperclip_available:
                        import pyperclip
                        pyperclip.copy(st.session_state.generated_code)
                        st.success("コードをクリップボードにコピーしました。")
                    else:
                        st.warning("pyperclipがインストールされていないため、クリップボードにコピーできません。pip install pyperclipでインストールしてください。")
            
            with col2:
                if st.button("コードをダウンロード"):
                    if 'output_folder' in st.session_state and st.session_state.output_folder:
                        try:
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"image_processing_{timestamp}.py"
                            file_path = os.path.join(st.session_state.output_folder, filename)
                            self.save_to_file(st.session_state.generated_code, file_path)
                            st.success(f"コードを保存しました: {file_path}")
                        except Exception as e:
                            st.error(f"コード保存中にエラーが発生しました: {str(e)}")
                    else:
                        st.error("出力フォルダが設定されていません。")
            
            return False, st.session_state.generated_code
            
        return False, None
    
    def display_code(self, code):
        """生成されたコードを表示する。
        
        Args:
            code: 表示するコード
        """
        if code is not None:
            st.header("生成されたコード")
            st.code(code, language="python")
    
    def save_to_file(self, code, file_path):
        """コードをファイルに保存する。
        
        Args:
            code: 保存するコード
            file_path: 保存先ファイルパス
        """
        # ディレクトリが存在しない場合は作成
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(code)
    
    def generate_combined_code(self, history):
        """履歴から複合処理コードを生成する。
        
        Args:
            history: 処理履歴
            
        Returns:
            生成されたPythonコード
        """
        # コード生成に必要なインポート文
        imports = """
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter
"""
        
        # 画像読み込みコード
        load_code = """
def load_image(file_path):
    # 画像を読み込む関数
    # PILで画像を読み込み（日本語パス対応）
    pil_img = Image.open(file_path)
    img = np.array(pil_img)
    
    # RGBからBGRに変換（OpenCVはBGR形式）
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    return img

# 画像ファイルのパスを指定
image_path = 'path/to/your/image.jpg'  # ここを実際の画像パスに変更してください
original_image = load_image(image_path)
"""
        
        # 処理適用コード
        process_code = "# 画像処理を適用\n"
        process_code += "processed_image = original_image.copy()\n\n"
        
        # 履歴の各処理を適用するコード
        for i, entry in enumerate(history):
            process_type = entry["process_type"]
            params = entry["params"]
            
            process_code += f"# 処理 {i+1}: {process_type}\n"
            
            # 処理の種類に応じたコード生成
            if process_type == "grayscale":
                process_code += "processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)\n"
                # カラー画像のまま処理を続ける場合は、グレースケールを3チャンネルに戻す
                process_code += "if len(processed_image.shape) == 2:  # グレースケール画像の場合\n"
                process_code += "    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)\n"
                
            elif process_type == "invert":
                process_code += "processed_image = cv2.bitwise_not(processed_image)\n"
                
            elif process_type == "rotate":
                angle = params.get("angle", 90)
                process_code += f"""# 画像回転
height, width = processed_image.shape[:2]
center = (width // 2, height // 2)
rotation_matrix = cv2.getRotationMatrix2D(center, {angle}, 1.0)
processed_image = cv2.warpAffine(processed_image, rotation_matrix, (width, height))
"""
                
            elif process_type == "resize":
                width = params.get("width", 300)
                height = params.get("height", 300)
                process_code += f"processed_image = cv2.resize(processed_image, ({width}, {height}))\n"
                
            elif process_type == "blur":
                kernel_size = params.get("kernel_size", 5)
                blur_type = params.get("blur_type", "gaussian")
                
                if blur_type == "gaussian":
                    process_code += f"processed_image = cv2.GaussianBlur(processed_image, ({kernel_size}, {kernel_size}), 0)\n"
                elif blur_type == "median":
                    process_code += f"processed_image = cv2.medianBlur(processed_image, {kernel_size})\n"
                elif blur_type == "box":
                    process_code += f"processed_image = cv2.blur(processed_image, ({kernel_size}, {kernel_size}))\n"
                elif blur_type == "bilateral":
                    process_code += f"processed_image = cv2.bilateralFilter(processed_image, {kernel_size}, 75, 75)\n"
                
            elif process_type == "sharpen":
                process_code += """# シャープ化
kernel = np.array([[-1, -1, -1], 
                   [-1,  9, -1], 
                   [-1, -1, -1]])
processed_image = cv2.filter2D(processed_image, -1, kernel)
"""
                
            elif process_type == "edge_detection":
                method = params.get("method", "canny")
                
                if method == "canny":
                    process_code += """# エッジ検出（Canny法）
gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY) if len(processed_image.shape) == 3 else processed_image
edges = cv2.Canny(gray, 100, 200)
processed_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
"""
                elif method == "sobel":
                    process_code += """# エッジ検出（Sobel法）
gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY) if len(processed_image.shape) == 3 else processed_image
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
sobelxy = cv2.magnitude(sobelx, sobely)
# Sobel結果の正規化
sobelxy = cv2.normalize(sobelxy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
processed_image = cv2.cvtColor(sobelxy, cv2.COLOR_GRAY2BGR)
"""
                elif method == "laplacian":
                    process_code += """# エッジ検出（Laplacian法）
gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY) if len(processed_image.shape) == 3 else processed_image
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
laplacian = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
processed_image = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)
"""
                
            elif process_type == "brightness":
                value = params.get("value", 1.5)
                process_code += f"""# 明るさ調整
hsv = cv2.cvtColor(processed_image, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)
v = cv2.multiply(v, {value})
v = np.clip(v, 0, 255).astype(np.uint8)
hsv = cv2.merge([h, s, v])
processed_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
"""
                
            elif process_type == "contrast":
                value = params.get("value", 1.5)
                process_code += f"""# コントラスト調整
alpha = {value}  # コントラスト制御（1.0-3.0）
beta = 0  # 明るさ制御（0-100）
processed_image = cv2.convertScaleAbs(processed_image, alpha=alpha, beta=beta)
"""
                
            elif process_type == "saturation":
                value = params.get("value", 1.5)
                process_code += f"""# 彩度調整
hsv = cv2.cvtColor(processed_image, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)
s = cv2.multiply(s, {value})
s = np.clip(s, 0, 255).astype(np.uint8)
hsv = cv2.merge([h, s, v])
processed_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
"""
                
            elif process_type == "sepia":
                process_code += """# セピア効果
kernel = np.array([[0.272, 0.534, 0.131],
                   [0.349, 0.686, 0.168],
                   [0.393, 0.769, 0.189]])
processed_image = cv2.transform(processed_image, kernel)
processed_image = np.clip(processed_image, 0, 255).astype(np.uint8)
"""
                
            elif process_type == "emboss":
                process_code += """# エンボス効果
kernel = np.array([[-2, -1, 0],
                   [-1,  1, 1],
                   [ 0,  1, 2]])
processed_image = cv2.filter2D(processed_image, -1, kernel) + 128
processed_image = np.clip(processed_image, 0, 255).astype(np.uint8)
"""
                
            elif process_type == "mosaic":
                block_size = params.get("block_size", 10)
                process_code += f"""# モザイク効果
height, width = processed_image.shape[:2]
# 縮小してから拡大することでモザイク効果を実現
small = cv2.resize(processed_image, (width // {block_size}, height // {block_size}), interpolation=cv2.INTER_LINEAR)
processed_image = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
"""
                
            elif process_type == "threshold":
                thresh_value = params.get("thresh_value", 127)
                process_code += f"""# 閾値処理
gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY) if len(processed_image.shape) == 3 else processed_image
_, binary = cv2.threshold(gray, {thresh_value}, 255, cv2.THRESH_BINARY)
processed_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
"""
                
            elif process_type == "dilate":
                kernel_size = params.get("kernel_size", 5)
                process_code += f"""# 膨張処理
kernel = np.ones(({kernel_size}, {kernel_size}), np.uint8)
processed_image = cv2.dilate(processed_image, kernel, iterations=1)
"""
                
            elif process_type == "erode":
                kernel_size = params.get("kernel_size", 5)
                process_code += f"""# 収縮処理
kernel = np.ones(({kernel_size}, {kernel_size}), np.uint8)
processed_image = cv2.erode(processed_image, kernel, iterations=1)
"""
                
            elif process_type == "flip_horizontal":
                process_code += "processed_image = cv2.flip(processed_image, 1)  # 1は水平方向の反転\n"
                
            elif process_type == "flip_vertical":
                process_code += "processed_image = cv2.flip(processed_image, 0)  # 0は垂直方向の反転\n"
                
            process_code += "\n"
        
        # 結果表示コード
        display_code = """
# 結果の表示
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title('Original')
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Processed')
plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()

# 結果の保存
# output_path = 'processed_image.jpg'  # 保存先のパスを指定
# cv2.imwrite(output_path, processed_image)
"""
        
        # 最終的なコードを組み立て
        full_code = imports + load_code + process_code + display_code
        
        return full_code