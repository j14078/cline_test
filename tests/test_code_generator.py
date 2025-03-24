#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
コード生成モジュールのテストコード。
"""

import pytest
import sys
import os

# プロジェクトのsrcディレクトリをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from code_generator import CodeGenerator


def test_code_generator_init():
    """CodeGeneratorの初期化をテストする。"""
    # デフォルト設定でインスタンス化
    generator = CodeGenerator()
    assert generator.library == "opencv"
    assert "opencv" in generator.supported_libraries
    
    # 異なるライブラリを指定
    generator = CodeGenerator(library="pillow")
    assert generator.library == "pillow"
    
    # サポートされていないライブラリを指定すると例外が発生
    with pytest.raises(ValueError):
        CodeGenerator(library="unsupported_library")


def test_get_library_imports():
    """ライブラリのインポート文生成をテストする。"""
    # OpenCV
    generator = CodeGenerator(library="opencv")
    imports = generator.get_library_imports()
    assert "import cv2" in imports
    assert "import numpy as np" in imports
    
    # Pillow
    generator = CodeGenerator(library="pillow")
    imports = generator.get_library_imports()
    assert "from PIL import Image" in imports
    
    # scikit-image
    generator = CodeGenerator(library="scikit-image")
    imports = generator.get_library_imports()
    assert "from skimage import io" in imports


def test_generate_code_for_process():
    """単一処理のコード生成をテストする。"""
    generator = CodeGenerator()
    
    # グレースケール処理
    code = generator._generate_code_for_process("grayscale", {})
    assert "cv2.COLOR_BGR2GRAY" in code
    
    # リサイズ処理
    code = generator._generate_code_for_process("resize", {"width": 300, "height": 200})
    assert "cv2.resize" in code
    assert "300" in code
    assert "200" in code


def test_format_code():
    """コードの整形をテストする。"""
    generator = CodeGenerator()
    
    # 空行が複数続くコードを整形
    unformatted_code = "import cv2\n\n\nimport numpy as np\n\n\n# コード"
    formatted_code = generator.format_code(unformatted_code)
    
    # 連続する空行が1つにまとめられている
    assert "import cv2\n\nimport numpy as np\n\n# コード" == formatted_code


def test_generate_process_code():
    """各処理タイプのコード生成をテストする。"""
    generator = CodeGenerator()
    
    # グレースケール変換
    code = generator._generate_process_code("grayscale", {}, "image")
    assert "cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)" == code
    
    # ぼかし処理
    code = generator._generate_process_code("blur", {"kernel_size": 7}, "image")
    assert "cv2.GaussianBlur(image, (7, 7), 0)" == code
    
    # 回転処理
    code = generator._generate_process_code("rotate", {"angle": 45}, "image")
    assert "cv2.warpAffine" in code
    assert "45" in code
    assert "image" in code


if __name__ == "__main__":
    pytest.main()