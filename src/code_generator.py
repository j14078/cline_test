#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
コード生成モジュール。

このモジュールは画像処理確認アプリで使用される処理のPythonコード生成機能を提供します。
"""

import os
import textwrap
import streamlit as st
from typing import Dict, List, Any, Optional, Tuple


class CodeGenerator:
    """Pythonコードを生成するクラス。"""
    
    def __init__(self, library="opencv"):
        """初期化メソッド。

        Args:
            library: コード生成に使用するライブラリ（"opencv", "pillow", "scipy", "scikit-image", "numpy"）
        """
        self.library = library
        self.supported_libraries = ["opencv", "pillow", "scipy", "scikit-image", "numpy"]
        
        # 処理タイプと必要な前提条件・警告のマッピング
        self.process_prerequisites = {
            # モルフォロジー系（2値化が前提）
            "dilate": {"warning": "この処理は2値化画像に対して最も効果的です。先に2値化処理を適用することをお勧めします。", "prerequisite": "binary"},
            "erode": {"warning": "この処理は2値化画像に対して最も効果的です。先に2値化処理を適用することをお勧めします。", "prerequisite": "binary"},
            "opening": {"warning": "この処理は2値化画像に対して最も効果的です。先に2値化処理を適用することをお勧めします。", "prerequisite": "binary"},
            "closing": {"warning": "この処理は2値化画像に対して最も効果的です。先に2値化処理を適用することをお勧めします。", "prerequisite": "binary"},
            "tophat": {"warning": "この処理は2値化画像に対して最も効果的です。先に2値化処理を適用することをお勧めします。", "prerequisite": "binary"},
            "blackhat": {"warning": "この処理は2値化画像に対して最も効果的です。先に2値化処理を適用することをお勧めします。", "prerequisite": "binary"},
            "morphology_gradient": {"warning": "この処理は2値化画像に対して最も効果的です。先に2値化処理を適用することをお勧めします。", "prerequisite": "binary"},
            
            # 特徴検出系（グレースケールが前提、または推奨）
            "edge_detection": {"warning": "エッジ検出はグレースケール画像に対して最も効果的です。カラー画像の場合は自動的にグレースケール変換されます。", "prerequisite": "grayscale"},
            "hough_lines": {"warning": "ハフ変換はエッジ検出後の2値画像に対して最も効果的です。先にエッジ検出を適用することをお勧めします。", "prerequisite": "edges"},
            "probabilistic_hough_lines": {"warning": "確率的ハフ変換はエッジ検出後の2値画像に対して最も効果的です。先にエッジ検出を適用することをお勧めします。", "prerequisite": "edges"},
            "hough_circles": {"warning": "円検出はグレースケール画像に対して最も効果的です。カラー画像の場合は自動的にグレースケール変換されます。", "prerequisite": "grayscale"},
            "contour_detection": {"warning": "輪郭検出は2値化画像に対して最も効果的です。先に2値化処理を適用することをお勧めします。", "prerequisite": "binary"},
            
            # その他の特殊処理
            "histogram_equalization": {"warning": "ヒストグラム均等化はグレースケール画像に対して最も効果的です。カラー画像の場合は色空間を変換してから適用することをお勧めします。", "prerequisite": "grayscale"},
            "watershed": {"warning": "分水嶺法は正確なマーカーが必要です。マーカー作成のためにグレースケール変換や2値化などの前処理をお勧めします。", "prerequisite": "markers"},
            "template_matching": {"warning": "テンプレートマッチングはテンプレート画像と同じチャネル数の画像に対してのみ適用可能です。", "prerequisite": "same_channels"}
        }
        
        if library not in self.supported_libraries:
            raise ValueError(f"サポートされていないライブラリです: {library}")
        
    def generate_code(self, process_type, params=None, history=None, folder_processing=False):
        """処理に対応するPythonコードを生成する。

        Args:
            process_type: 処理の種類
            params: 処理のパラメータ
            history: 処理履歴（複数の処理を組み合わせる場合）
            folder_processing: フォルダ内の画像を一括処理するかどうか

        Returns:
            生成されたPythonコード
        """
        if params is None:
            params = {}
            
        # ライブラリに応じたインポート文を取得
        imports = self.get_library_imports()
        
        # 処理タイプに応じたコードを生成
        if folder_processing:
            # フォルダ内の画像を一括処理するコードを生成
            process_code = self._generate_folder_processing_code(process_type, params, history)
        elif history:
            # 履歴から複数の処理を組み合わせたコードを生成
            process_code = self._generate_code_from_history(history)
        else:
            # 単一の処理のコードを生成
            process_code = self._generate_code_for_process(process_type, params)
            
        # 最終的なコードを組み立て
        code = f"{imports}\n\n{process_code}"
        
        return self.format_code(code)
        
    def get_process_prerequisites(self, process_type):
        """処理タイプに対する前提条件と警告を取得する。

        Args:
            process_type: 処理の種類

        Returns:
            前提条件と警告のディクショナリ、該当なしの場合はNone
        """
        return self.process_prerequisites.get(process_type)
        
    def generate_comparison_code(self, image1_path, image2_path, metrics=None):
        """画像比較のコードを生成する。

        Args:
            image1_path: 比較する1つ目の画像のパス
            image2_path: 比較する2つ目の画像のパス
            metrics: 使用する比較指標のリスト（"psnr", "ssim"など）

        Returns:
            生成された画像比較コード
        """
        if metrics is None:
            metrics = ["psnr", "ssim"]
            
        # ライブラリに応じたインポート文を取得
        imports = self.get_library_imports()
        
        # 追加のインポート
        if "ssim" in metrics:
            imports += "\nfrom skimage.metrics import structural_similarity as ssim"
            
        # 画像読み込みコード
        load_code = self._generate_load_code(image1_path, "image1")
        load_code += "\n" + self._generate_load_code(image2_path, "image2")
        
        # 比較コード
        comparison_code = ""
        if "psnr" in metrics:
            if self.library == "opencv":
                comparison_code += "\n# PSNR（ピーク信号対雑音比）を計算\n"
                comparison_code += "psnr_value = cv2.PSNR(image1, image2)\n"
                comparison_code += "print(f\"PSNR: {psnr_value}\")\n"
            elif self.library == "scikit-image":
                comparison_code += "\n# PSNR（ピーク信号対雑音比）を計算\n"
                comparison_code += "from skimage.metrics import peak_signal_noise_ratio\n"
                comparison_code += "psnr_value = peak_signal_noise_ratio(image1, image2)\n"
                comparison_code += "print(f\"PSNR: {psnr_value}\")\n"
                
        if "ssim" in metrics:
            comparison_code += "\n# SSIM（構造的類似性指標）を計算\n"
            if self.library == "opencv":
                comparison_code += "# OpenCVでSSIMを計算するには、scikit-imageを使用\n"
            comparison_code += "ssim_value = ssim(image1, image2, multichannel=True)\n"
            comparison_code += "print(f\"SSIM: {ssim_value}\")\n"
            
        # 可視化コード
        visualization_code = "\n# 比較結果を可視化\n"
        visualization_code += "import matplotlib.pyplot as plt\n\n"
        visualization_code += "plt.figure(figsize=(12, 6))\n\n"
        visualization_code += "plt.subplot(1, 2, 1)\n"
        visualization_code += "plt.title('Image 1')\n"
        visualization_code += "plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))\n"
        visualization_code += "plt.axis('off')\n\n"
        visualization_code += "plt.subplot(1, 2, 2)\n"
        visualization_code += "plt.title('Image 2')\n"
        visualization_code += "plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))\n"
        visualization_code += "plt.axis('off')\n\n"
        visualization_code += "plt.tight_layout()\n"
        visualization_code += "plt.show()\n"
        
        # 最終的なコードを組み立て
        code = f"{imports}\n\n{load_code}\n{comparison_code}\n{visualization_code}"
        
        return self.format_code(code)

    def _generate_folder_processing_code(self, process_type, params, history=None):
        """フォルダ内の画像を一括処理するコードを生成する。

        Args:
            process_type: 処理の種類
            params: 処理のパラメータ
            history: 処理履歴（複数の処理を組み合わせる場合）

        Returns:
            生成されたコード
        """
        code = "import os\nimport glob\nimport matplotlib.pyplot as plt\nimport time\n\n"
        
        # 入力・出力フォルダ設定
        code += "# 入力・出力フォルダ設定\n"
        code += "input_folder = 'path/to/input/folder'  # 入力フォルダのパスを指定\n"
        code += "output_folder = 'path/to/output/folder'  # 出力フォルダのパスを指定\n\n"
        
        # 出力フォルダが存在しない場合は作成
        code += "# 出力フォルダが存在しない場合は作成\n"
        code += "if not os.path.exists(output_folder):\n"
        code += "    os.makedirs(output_folder)\n\n"
        
        # 画像ファイル一覧を取得
        code += "# 画像ファイル一覧を取得\n"
        code += "image_files = glob.glob(os.path.join(input_folder, '*.jpg')) + \\\n"
        code += "             glob.glob(os.path.join(input_folder, '*.jpeg')) + \\\n"
        code += "             glob.glob(os.path.join(input_folder, '*.png')) + \\\n"
        code += "             glob.glob(os.path.join(input_folder, '*.bmp'))\n\n"
        
        # 処理の前提条件をチェック
        prereq = self.get_process_prerequisites(process_type)
        if prereq:
            code += f"# 注意: {prereq['warning']}\n\n"
        
        # 処理関数を定義
        code += "# 処理関数を定義\n"
        code += "def process_image(image):\n"
        
        # 履歴があれば複数の処理を適用
        if history:
            for i, entry in enumerate(history):
                entry_process_type = entry.get("process_type", "grayscale")
                entry_params = entry.get("params", {})
                
                indented_code = "    " + self._generate_process_code(entry_process_type, entry_params, "image" if i == 0 else "result").replace("\n", "\n    ")
                
                if i == 0:
                    code += f"    result = {indented_code}\n"
                else:
                    code += f"    result = {indented_code}\n"
        else:
            # 単一の処理を適用
            indented_code = "    " + self._generate_process_code(process_type, params, "image").replace("\n", "\n    ")
            code += f"    result = {indented_code}\n"
        
        code += "    return result\n\n"
        
        # 画像処理ループ
        code += "# フォルダ内の画像を一括処理\n"
        code += "print(f\"処理対象画像: {len(image_files)}個\")\n"
        code += "start_time = time.time()\n\n"
        code += "for i, image_file in enumerate(image_files):\n"
        code += "    # 画像を読み込み\n"
        code += "    image = cv2.imread(image_file)\n"
        code += "    if image is None:\n"
        code += "        print(f\"画像の読み込みに失敗しました: {image_file}\")\n"
        code += "        continue\n\n"
        code += "    # 画像を処理\n"
        code += "    result = process_image(image)\n\n"
        code += "    # 出力ファイル名を設定\n"
        code += "    filename = os.path.basename(image_file)\n"
        code += "    output_path = os.path.join(output_folder, filename)\n\n"
        code += "    # 結果を保存\n"
        code += "    cv2.imwrite(output_path, result)\n"
        code += "    print(f\"処理完了 ({i+1}/{len(image_files)}): {filename}\")\n\n"
        
        # 処理時間の表示
        code += "# 処理完了\n"
        code += "elapsed_time = time.time() - start_time\n"
        code += "print(f\"全ての処理が完了しました。処理時間: {elapsed_time:.2f}秒\")\n\n"
        
        # 結果のプレビュー
        code += "# 結果のプレビュー（最初の4枚）\n"
        code += "preview_files = image_files[:4]\n"
        code += "if preview_files:\n"
        code += "    plt.figure(figsize=(15, 10))\n"
        code += "    for i, image_file in enumerate(preview_files):\n"
        code += "        # 元画像\n"
        code += "        plt.subplot(2, len(preview_files), i + 1)\n"
        code += "        input_img = cv2.imread(image_file)\n"
        code += "        plt.imshow(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))\n"
        code += "        plt.title(f\"元画像 {i+1}\")\n"
        code += "        plt.axis('off')\n\n"
        code += "        # 処理後画像\n"
        code += "        plt.subplot(2, len(preview_files), i + 1 + len(preview_files))\n"
        code += "        filename = os.path.basename(image_file)\n"
        code += "        output_path = os.path.join(output_folder, filename)\n"
        code += "        output_img = cv2.imread(output_path)\n"
        code += "        plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))\n"
        code += "        plt.title(f\"処理後 {i+1}\")\n"
        code += "        plt.axis('off')\n\n"
        code += "    plt.tight_layout()\n"
        code += "    plt.show()\n"
        
        return code
        
    def generate_template_matching_code(self, image_path, template_path, method="cv.TM_CCOEFF_NORMED"):
        """テンプレートマッチングのコードを生成する。

        Args:
            image_path: 検索対象の画像のパス
            template_path: テンプレート画像のパス
            method: マッチング手法

        Returns:
            生成されたテンプレートマッチングコード
        """
        # ライブラリに応じたインポート文を取得
        imports = self.get_library_imports()
        
        # 画像読み込みコード
        load_code = self._generate_load_code(image_path, "image")
        load_code += "\n" + self._generate_load_code(template_path, "template")
        
        # テンプレートマッチングコード
        matching_code = "\n# テンプレートマッチングを実行\n"
        
        if self.library == "opencv":
            matching_code += f"result = cv2.matchTemplate(image, template, {method})\n"
            matching_code += "min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)\n\n"
            matching_code += "# マッチング結果を可視化\n"
            matching_code += "img_display = image.copy()\n"
            matching_code += "h, w = template.shape[:2]\n\n"
            matching_code += f"# マッチング手法によって最適な位置が異なる\n"
            matching_code += f"if {method} in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:\n"
            matching_code += "    top_left = min_loc\n"
            matching_code += "else:\n"
            matching_code += "    top_left = max_loc\n\n"
            matching_code += "bottom_right = (top_left[0] + w, top_left[1] + h)\n"
            matching_code += "cv2.rectangle(img_display, top_left, bottom_right, (0, 255, 0), 2)\n\n"
            matching_code += "# 結果を表示\n"
            matching_code += "plt.figure(figsize=(12, 4))\n\n"
            matching_code += "plt.subplot(1, 3, 1)\n"
            matching_code += "plt.title('Image')\n"
            matching_code += "plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))\n"
            matching_code += "plt.axis('off')\n\n"
            matching_code += "plt.subplot(1, 3, 2)\n"
            matching_code += "plt.title('Template')\n"
            matching_code += "plt.imshow(cv2.cvtColor(template, cv2.COLOR_BGR2RGB))\n"
            matching_code += "plt.axis('off')\n\n"
            matching_code += "plt.subplot(1, 3, 3)\n"
            matching_code += "plt.title('Matching Result')\n"
            matching_code += "plt.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))\n"
            matching_code += "plt.axis('off')\n\n"
            matching_code += "plt.tight_layout()\n"
            matching_code += "plt.show()\n"
        else:
            # デフォルトはOpenCV
            matching_code += f"result = cv2.matchTemplate(image, template, {method})\n"
            matching_code += "min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)\n\n"
            matching_code += "# マッチング結果を可視化\n"
            matching_code += "img_display = image.copy()\n"
            matching_code += "h, w = template.shape[:2]\n\n"
            matching_code += f"# マッチング手法によって最適な位置が異なる\n"
            matching_code += f"if {method} in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:\n"
            matching_code += "    top_left = min_loc\n"
            matching_code += "else:\n"
            matching_code += "    top_left = max_loc\n\n"
            matching_code += "bottom_right = (top_left[0] + w, top_left[1] + h)\n"
            matching_code += "cv2.rectangle(img_display, top_left, bottom_right, (0, 255, 0), 2)\n\n"
            matching_code += "# 結果を表示\n"
            matching_code += "plt.figure(figsize=(12, 4))\n\n"
            matching_code += "plt.subplot(1, 3, 1)\n"
            matching_code += "plt.title('Image')\n"
            matching_code += "plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))\n"
            matching_code += "plt.axis('off')\n\n"
            matching_code += "plt.subplot(1, 3, 2)\n"
            matching_code += "plt.title('Template')\n"
            matching_code += "plt.imshow(cv2.cvtColor(template, cv2.COLOR_BGR2RGB))\n"
            matching_code += "plt.axis('off')\n\n"
            matching_code += "plt.subplot(1, 3, 3)\n"
            matching_code += "plt.title('Matching Result')\n"
            matching_code += "plt.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))\n"
            matching_code += "plt.axis('off')\n\n"
            matching_code += "plt.tight_layout()\n"
            matching_code += "plt.show()\n"
            
        # 最終的なコードを組み立て
        code = f"{imports}\nimport matplotlib.pyplot as plt\n\n{load_code}\n{matching_code}"
        
        return self.format_code(code)
        
    def format_code(self, code):
        """コードを整形する。

        Args:
            code: 整形する生のコード

        Returns:
            整形されたコード
        """
        # 空行の連続を1つにする
        lines = code.split('\n')
        formatted_lines = []
        prev_empty = False
        
        for line in lines:
            if not line.strip():
                if not prev_empty:
                    formatted_lines.append(line)
                prev_empty = True
            else:
                formatted_lines.append(line)
                prev_empty = False
                
        return '\n'.join(formatted_lines)
        
    def save_to_file(self, code, path):
        """コードをファイルに保存する。

        Args:
            code: 保存するコード
            path: 保存先のパス
        """
        # ディレクトリが存在しない場合は作成する
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        with open(path, 'w', encoding='utf-8') as f:
            f.write(code)
            
    def get_library_imports(self):
        """選択されたライブラリに応じたインポート文を取得する。

        Returns:
            ライブラリのインポート文
        """
        imports = ""
        
        if self.library == "opencv":
            imports = "import cv2\nimport numpy as np"
        elif self.library == "pillow":
            imports = "from PIL import Image, ImageEnhance, ImageFilter\nimport numpy as np"
        elif self.library == "scipy":
            imports = "import numpy as np\nimport scipy.ndimage as ndi\nfrom scipy import misc"
        elif self.library == "scikit-image":
            imports = "import numpy as np\nfrom skimage import io, color, filters, feature, segmentation, morphology"
        elif self.library == "numpy":
            imports = "import numpy as np\nimport matplotlib.pyplot as plt"
        else:
            # デフォルトはOpenCV
            imports = "import cv2\nimport numpy as np"
            
        return imports
        
    def _generate_code_from_history(self, history):
        """履歴から複数の処理を組み合わせたコードを生成する。

        Args:
            history: 処理履歴

        Returns:
            生成されたコード
        """
        code = "# 画像を読み込む\n"
        code += "image_path = 'path/to/your/image.jpg'  # 画像ファイルのパスを指定\n"
        code += self._generate_load_code("image_path", "image")
        code += "\n# 処理を適用\n"
        
        for i, entry in enumerate(history):
            process_type = entry.get("process_type", "grayscale")
            params = entry.get("params", {})
            
            # 処理の前提条件をチェック
            prereq = self.get_process_prerequisites(process_type)
            if prereq:
                code += f"# 注意: {prereq['warning']}\n"
            
            if i == 0:
                var_name = "result"
                code += f"{var_name} = "
            else:
                var_name = f"result{i+1}"
                code += f"{var_name} = "
                
            code += self._generate_process_code(process_type, params, f"result{i}" if i > 0 else "image")
            code += "\n"
            
        code += "\n# 結果を表示\n"
        code += "import matplotlib.pyplot as plt\n\n"
        code += "plt.figure(figsize=(15, 5))\n\n"
        code += "plt.subplot(1, 3, 1)\n"
        code += "plt.title('元画像')\n"
        code += "plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))\n"
        code += "plt.axis('off')\n\n"
        
        # 中間結果を表示（履歴が3つ以上ある場合）
        if len(history) >= 3:
            middle_idx = len(history) // 2
            middle_var = f"result{middle_idx}" if middle_idx > 0 else "result"
            code += f"plt.subplot(1, 3, 2)\n"
            code += f"plt.title('中間処理（{history[middle_idx].get('process_type', 'grayscale')}）')\n"
            code += f"plt.imshow(cv2.cvtColor({middle_var}, cv2.COLOR_BGR2RGB))\n"
            code += f"plt.axis('off')\n\n"
            
            code += f"plt.subplot(1, 3, 3)\n"
            code += f"plt.title('最終結果')\n"
            code += f"plt.imshow(cv2.cvtColor({var_name}, cv2.COLOR_BGR2RGB))\n"
            code += f"plt.axis('off')\n\n"
        else:
            code += f"plt.subplot(1, 3, 2)\n"
            code += f"plt.title('処理後画像')\n"
            code += f"plt.imshow(cv2.cvtColor({var_name}, cv2.COLOR_BGR2RGB))\n"
            code += f"plt.axis('off')\n\n"
            
            # 差分を表示
            code += f"plt.subplot(1, 3, 3)\n"
            code += f"plt.title('差分（元画像 - 処理後）')\n"
            code += f"diff = cv2.absdiff(image, {var_name})\n"
            code += f"plt.imshow(cv2.applyColorMap(cv2.convertScaleAbs(diff, alpha=5), cv2.COLORMAP_JET))\n"
            code += f"plt.axis('off')\n\n"
            
        code += "plt.tight_layout()\n"
        code += "plt.show()\n"
        
        # 結果の保存
        code += "\n# 結果を保存\n"
        code += "output_path = 'path/to/output/image.jpg'  # 出力パスを指定\n"
        code += f"cv2.imwrite(output_path, {var_name})\n"
        code += "print(f\"処理結果を保存しました: {output_path}\")\n"
        
        return code
        
    def _generate_code_for_process(self, process_type, params):
        """単一の処理のコードを生成する。

        Args:
            process_type: 処理の種類
            params: 処理のパラメータ

        Returns:
            生成されたコード
        """
        code = "# 画像を読み込む\n"
        code += "image_path = 'path/to/your/image.jpg'  # 画像ファイルのパスを指定\n"
        code += self._generate_load_code("image_path", "image")
        
        # 処理の前提条件をチェック
        prereq = self.get_process_prerequisites(process_type)
        if prereq:
            code += f"\n# 注意: {prereq['warning']}\n"
            
            # 前提条件に応じた前処理コードを追加
            if prereq["prerequisite"] == "grayscale":
                code += "\n# グレースケール変換（推奨前処理）\n"
                code += "if len(image.shape) == 3:\n"
                code += "    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)\n"
                code += "    # 3チャンネルに戻す（表示用）\n"
                code += "    image_to_process = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)\n"
                code += "else:\n"
                code += "    image_to_process = image.copy()\n"
            elif prereq["prerequisite"] == "binary":
                code += "\n# 2値化処理（推奨前処理）\n"
                code += "if len(image.shape) == 3:\n"
                code += "    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)\n"
                code += "else:\n"
                code += "    gray = image.copy()\n"
                code += "_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)\n"
                code += "# 3チャンネルに戻す（表示用）\n"
                code += "image_to_process = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)\n"
            elif prereq["prerequisite"] == "edges":
                code += "\n# エッジ検出（推奨前処理）\n"
                code += "if len(image.shape) == 3:\n"
                code += "    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)\n"
                code += "else:\n"
                code += "    gray = image.copy()\n"
                code += "edges = cv2.Canny(gray, 100, 200)\n"
                code += "# 3チャンネルに戻す（表示用）\n"
                code += "image_to_process = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)\n"
            else:
                code += "\n# 処理用画像を準備\n"
                code += "image_to_process = image.copy()\n"
        else:
            code += "\n# 処理用画像を準備\n"
            code += "image_to_process = image.copy()\n"
        
        code += "\n# 処理を適用\n"
        if prereq and prereq["prerequisite"] in ["grayscale", "binary", "edges"]:
            code += "result = " + self._generate_process_code(process_type, params, "image_to_process")
        else:
            code += "result = " + self._generate_process_code(process_type, params, "image")
        
        code += "\n\n# 結果を表示\n"
        code += "import matplotlib.pyplot as plt\n\n"
        code += "plt.figure(figsize=(15, 5))\n\n"
        code += "plt.subplot(1, 3, 1)\n"
        code += "plt.title('元画像')\n"
        code += "plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))\n"
        code += "plt.axis('off')\n\n"
        
        if prereq and prereq["prerequisite"] in ["grayscale", "binary", "edges"]:
            code += "plt.subplot(1, 3, 2)\n"
            code += f"plt.title('前処理画像 ({prereq['prerequisite']})')\n"
            code += "plt.imshow(cv2.cvtColor(image_to_process, cv2.COLOR_BGR2RGB))\n"
            code += "plt.axis('off')\n\n"
            
            code += "plt.subplot(1, 3, 3)\n"
            code += "plt.title('処理後画像')\n"
            code += "plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))\n"
            code += "plt.axis('off')\n\n"
        else:
            code += "plt.subplot(1, 3, 2)\n"
            code += "plt.title('処理後画像')\n"
            code += "plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))\n"
            code += "plt.axis('off')\n\n"
            
            # 差分を表示
            code += "plt.subplot(1, 3, 3)\n"
            code += "plt.title('差分（元画像 - 処理後）')\n"
            code += "diff = cv2.absdiff(image, result)\n"
            code += "plt.imshow(cv2.applyColorMap(cv2.convertScaleAbs(diff, alpha=5), cv2.COLORMAP_JET))\n"
            code += "plt.axis('off')\n\n"
            
        code += "plt.tight_layout()\n"
        code += "plt.show()\n"
        
        # 結果の保存
        code += "\n# 結果を保存\n"
        code += "output_path = 'path/to/output/image.jpg'  # 出力パスを指定\n"
        code += "cv2.imwrite(output_path, result)\n"
        code += "print(f\"処理結果を保存しました: {output_path}\")\n"
        
        return code
        
    def _generate_load_code(self, path, var_name):
        """画像読み込みコードを生成する。

        Args:
            path: 画像ファイルのパス
            var_name: 変数名

        Returns:
            生成されたコード
        """
        if self.library == "opencv":
            return f"{var_name} = cv2.imread({path})"
        elif self.library == "pillow":
            return f"{var_name} = Image.open({path})"
        elif self.library == "scipy":
            return f"{var_name} = misc.imread({path})"
        elif self.library == "scikit-image":
            return f"{var_name} = io.imread({path})"
        else:
            # デフォルトはOpenCV
            return f"{var_name} = cv2.imread({path})"
            
    def _generate_process_code(self, process_type, params, input_var):
        """処理コードを生成する。

        Args:
            process_type: 処理の種類
            params: 処理のパラメータ
            input_var: 入力変数名

        Returns:
            生成されたコード
        """
        if process_type == "grayscale":
            if self.library == "opencv":
                return f"cv2.cvtColor({input_var}, cv2.COLOR_BGR2GRAY)"
            elif self.library == "pillow":
                return f"{input_var}.convert('L')"
            else:
                return f"cv2.cvtColor({input_var}, cv2.COLOR_BGR2GRAY)"
                
        elif process_type == "invert":
            if self.library == "opencv":
                return f"cv2.bitwise_not({input_var})"
            elif self.library == "pillow":
                return f"ImageOps.invert({input_var})"
            else:
                return f"cv2.bitwise_not({input_var})"
                
        elif process_type == "rotate":
            angle = params.get("angle", 90)
            if self.library == "opencv":
                return f"""(lambda img, angle: 
                    cv2.warpAffine(
                        img, 
                        cv2.getRotationMatrix2D(
                            (img.shape[1] // 2, img.shape[0] // 2), 
                            {angle}, 
                            1.0
                        ), 
                        (img.shape[1], img.shape[0])
                    )
                )({input_var}, {angle})"""
            elif self.library == "pillow":
                return f"{input_var}.rotate({angle})"
            else:
                return f"""(lambda img, angle: 
                    cv2.warpAffine(
                        img, 
                        cv2.getRotationMatrix2D(
                            (img.shape[1] // 2, img.shape[0] // 2), 
                            {angle}, 
                            1.0
                        ), 
                        (img.shape[1], img.shape[0])
                    )
                )({input_var}, {angle})"""
                
        elif process_type == "resize":
            width = params.get("width", 300)
            height = params.get("height", 300)
            if self.library == "opencv":
                return f"cv2.resize({input_var}, ({width}, {height}))"
            elif self.library == "pillow":
                return f"{input_var}.resize(({width}, {height}))"
            else:
                return f"cv2.resize({input_var}, ({width}, {height}))"
                
        elif process_type == "blur":
            kernel_size = params.get("kernel_size", 5)
            blur_type = params.get("blur_type", "gaussian")
            
            if blur_type == "gaussian":
                if self.library == "opencv":
                    return f"cv2.GaussianBlur({input_var}, ({kernel_size}, {kernel_size}), 0)"
                elif self.library == "pillow":
                    return f"{input_var}.filter(ImageFilter.GaussianBlur({kernel_size}))"
                else:
                    return f"cv2.GaussianBlur({input_var}, ({kernel_size}, {kernel_size}), 0)"
            elif blur_type == "median":
                if self.library == "opencv":
                    return f"cv2.medianBlur({input_var}, {kernel_size})"
                else:
                    return f"cv2.medianBlur({input_var}, {kernel_size})"
            elif blur_type == "box":
                if self.library == "opencv":
                    return f"cv2.blur({input_var}, ({kernel_size}, {kernel_size}))"
                else:
                    return f"cv2.blur({input_var}, ({kernel_size}, {kernel_size}))"
            elif blur_type == "bilateral":
                sigma_color = params.get("sigma_color", 75)
                sigma_space = params.get("sigma_space", 75)
                if self.library == "opencv":
                    return f"cv2.bilateralFilter({input_var}, {kernel_size}, {sigma_color}, {sigma_space})"
                else:
                    return f"cv2.bilateralFilter({input_var}, {kernel_size}, {sigma_color}, {sigma_space})"
            else:
                return f"cv2.GaussianBlur({input_var}, ({kernel_size}, {kernel_size}), 0)"
                
        elif process_type == "sharpen":
            if self.library == "opencv":
                return f"""(lambda img: 
                    cv2.filter2D(
                        img, 
                        -1, 
                        np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                    )
                )({input_var})"""
            elif self.library == "pillow":
                return f"ImageEnhance.Sharpness({input_var}).enhance(2.0)"
            else:
                return f"""(lambda img: 
                    cv2.filter2D(
                        img, 
                        -1, 
                        np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                    )
                )({input_var})"""
                
        elif process_type == "edge_detection":
            method = params.get("method", "canny")
            if self.library == "opencv":
                if method == "canny":
                    return f"cv2.Canny(cv2.cvtColor({input_var}, cv2.COLOR_BGR2GRAY) if len({input_var}.shape) == 3 else {input_var}, 100, 200)"
                elif method == "sobel":
                    return f"""(lambda img: 
                        cv2.magnitude(
                            cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5),
                            cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
                        )
                    )(cv2.cvtColor({input_var}, cv2.COLOR_BGR2GRAY) if len({input_var}.shape) == 3 else {input_var})"""
                else:
                    return f"cv2.Laplacian(cv2.cvtColor({input_var}, cv2.COLOR_BGR2GRAY) if len({input_var}.shape) == 3 else {input_var}, cv2.CV_64F)"
            else:
                if method == "canny":
                    return f"cv2.Canny(cv2.cvtColor({input_var}, cv2.COLOR_BGR2GRAY) if len({input_var}.shape) == 3 else {input_var}, 100, 200)"
                elif method == "sobel":
                    return f"""(lambda img: 
                        cv2.magnitude(
                            cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5),
                            cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
                        )
                    )(cv2.cvtColor({input_var}, cv2.COLOR_BGR2GRAY) if len({input_var}.shape) == 3 else {input_var})"""
                else:
                    return f"cv2.Laplacian(cv2.cvtColor({input_var}, cv2.COLOR_BGR2GRAY) if len({input_var}.shape) == 3 else {input_var}, cv2.CV_64F)"
        
        # その他の処理タイプに対するコード生成...
        # 実際のアプリケーションでは、すべての処理タイプに対応するコードを生成する必要があります
        
        # デフォルトの場合は元の画像を返す
        return input_var


class CodeGeneratorUI:
    """コード生成機能を提供するUIクラス。"""
    
    def __init__(self):
        """初期化メソッド。"""
        self.code_generator = CodeGenerator()
        
    def render_ui(self, history, pyperclip_available=True) -> Tuple[bool, str]:
        """コード生成UIを描画する。
        
        Args:
            history: 処理履歴
            pyperclip_available: PyperClipが利用可能かどうか
            
        Returns:
            (generated_code_ready, generated_code): コード生成完了フラグとコード
        """
        st.sidebar.header("コード生成")
        
        # コード生成の出力形式
        if history:
            code_output_options = ["単一画像処理", "フォルダ一括処理"]
            code_output_option = st.sidebar.radio(
                "出力形式",
                code_output_options,
                index=0
            )
            folder_processing = (code_output_option == "フォルダ一括処理")
            
            # 処理範囲の選択
            if len(history) > 1:
                code_generation_options = ["選択中の処理のみ", "すべての処理履歴"]
                code_generation_option = st.sidebar.radio(
                    "処理範囲",
                    code_generation_options,
                    index=0
                )
                use_selected_only = (code_generation_option == "選択中の処理のみ")
            else:
                use_selected_only = True
            
            # コード生成ボタン
            generate_code_button = st.sidebar.button("コードを生成")
            
            if generate_code_button:
                try:
                    # コード生成器を初期化（常にOpenCVを使用）
                    self.code_generator = CodeGenerator(library="opencv")
                    
                    # コードを生成
                    if use_selected_only and "selected_history_index" in st.session_state:
                        if st.session_state.selected_history_index is not None:
                            entry = history[st.session_state.selected_history_index]
                            process_type = entry["process_type"]
                            params = entry["params"]
                            
                            # 処理の前提条件をチェック
                            prereq = self.code_generator.get_process_prerequisites(process_type)
                            if prereq:
                                st.sidebar.warning(f"注意: {prereq['warning']}")
                                
                            generated_code = self.code_generator.generate_code(
                                process_type, 
                                params, 
                                None, 
                                folder_processing
                            )
                            st.session_state.generated_code = generated_code
                            st.sidebar.success("選択中の処理のコードを生成しました。")
                            return True, generated_code
                    else:
                        # すべての履歴からコードを生成
                        generated_code = self.code_generator.generate_code(
                            None, 
                            None, 
                            history, 
                            folder_processing
                        )
                        st.session_state.generated_code = generated_code
                        st.sidebar.success("すべての処理履歴からコードを生成しました。")
                        return True, generated_code
                        
                except Exception as e:
                    st.sidebar.error(f"コード生成中にエラーが発生しました: {str(e)}")
            
            # コードをコピーするボタン
            if "generated_code" in st.session_state and pyperclip_available:
                if st.sidebar.button("コードをクリップボードにコピー"):
                    try:
                        import pyperclip
                        pyperclip.copy(st.session_state.generated_code)
                        st.sidebar.success("コードをクリップボードにコピーしました。")
                    except Exception as e:
                        st.sidebar.error(f"コピー中にエラーが発生しました: {str(e)}")
                        
            # コードを保存するボタン
            if "generated_code" in st.session_state:
                save_path = st.sidebar.text_input("保存先ファイルパス", "generated_code.py")
                if st.sidebar.button("コードをファイルに保存"):
                    try:
                        self.code_generator.save_to_file(st.session_state.generated_code, save_path)
                        st.sidebar.success(f"コードを {save_path} に保存しました。")
                    except Exception as e:
                        st.sidebar.error(f"保存中にエラーが発生しました: {str(e)}")
        else:
            st.sidebar.info("処理履歴がありません。処理を適用するとコードが生成できます。")
            
        # 生成されたコードを返す
        if "generated_code" in st.session_state:
            return True, st.session_state.generated_code
        else:
            return False, ""