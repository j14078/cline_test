#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
コード生成モジュール。

このモジュールは画像処理確認アプリで使用される処理のPythonコード生成機能を提供します。
"""

import os
import textwrap
from typing import Dict, List, Any, Optional


class CodeGenerator:
    """Pythonコードを生成するクラス。"""
    
    def __init__(self, library="opencv"):
        """初期化メソッド。

        Args:
            library: コード生成に使用するライブラリ（"opencv", "pillow", "scipy", "scikit-image", "numpy"）
        """
        self.library = library
        self.supported_libraries = ["opencv", "pillow", "scipy", "scikit-image", "numpy"]
        
        if library not in self.supported_libraries:
            raise ValueError(f"サポートされていないライブラリです: {library}")
        
    def generate_code(self, process_type, params=None, history=None):
        """処理に対応するPythonコードを生成する。

        Args:
            process_type: 処理の種類
            params: 処理のパラメータ
            history: 処理履歴（複数の処理を組み合わせる場合）

        Returns:
            生成されたPythonコード
        """
        if params is None:
            params = {}
            
        # ライブラリに応じたインポート文を取得
        imports = self.get_library_imports()
        
        # 処理タイプに応じたコードを生成
        if history:
            # 履歴から複数の処理を組み合わせたコードを生成
            process_code = self._generate_code_from_history(history)
        else:
            # 単一の処理のコードを生成
            process_code = self._generate_code_for_process(process_type, params)
            
        # 最終的なコードを組み立て
        code = f"{imports}\n\n{process_code}"
        
        return self.format_code(code)
        
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
        code += self._generate_load_code(history[0].image_path, "image")
        code += "\n# 処理を適用\n"
        
        for i, entry in enumerate(history):
            if i == 0:
                var_name = "result"
                code += f"{var_name} = "
            else:
                var_name = f"result{i+1}"
                code += f"{var_name} = "
                
            code += self._generate_process_code(entry.process_type, entry.params, f"result{i}" if i > 0 else "image")
            code += "\n"
            
        code += "\n# 結果を表示\n"
        code += "plt.figure(figsize=(12, 6))\n\n"
        code += "plt.subplot(1, 2, 1)\n"
        code += "plt.title('Original')\n"
        code += "plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))\n"
        code += "plt.axis('off')\n\n"
        code += "plt.subplot(1, 2, 2)\n"
        code += "plt.title('Processed')\n"
        code += f"plt.imshow(cv2.cvtColor({var_name}, cv2.COLOR_BGR2RGB))\n"
        code += "plt.axis('off')\n\n"
        code += "plt.tight_layout()\n"
        code += "plt.show()\n"
        
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
        
        code += "\n# 処理を適用\n"
        code += "result = " + self._generate_process_code(process_type, params, "image")
        
        code += "\n# 結果を表示\n"
        code += "import matplotlib.pyplot as plt\n\n"
        code += "plt.figure(figsize=(12, 6))\n\n"
        code += "plt.subplot(1, 2, 1)\n"
        code += "plt.title('Original')\n"
        code += "plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))\n"
        code += "plt.axis('off')\n\n"
        code += "plt.subplot(1, 2, 2)\n"
        code += "plt.title('Processed')\n"
        code += "plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))\n"
        code += "plt.axis('off')\n\n"
        code += "plt.tight_layout()\n"
        code += "plt.show()\n"
        
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
            width = params.get("width", "image.shape[1] // 2")
            height = params.get("height", "image.shape[0] // 2")
            if self.library == "opencv":
                return f"cv2.resize({input_var}, ({width}, {height}))"
            elif self.library == "pillow":
                return f"{input_var}.resize(({width}, {height}))"
            else:
                return f"cv2.resize({input_var}, ({width}, {height}))"
                
        elif process_type == "blur":
            kernel_size = params.get("kernel_size", 5)
            if self.library == "opencv":
                return f"cv2.GaussianBlur({input_var}, ({kernel_size}, {kernel_size}), 0)"
            elif self.library == "pillow":
                return f"{input_var}.filter(ImageFilter.GaussianBlur({kernel_size}))"
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