#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CodeGeneratorクラスのテスト。
"""

import os
import sys
import tempfile
import pytest

# srcディレクトリをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.code_generator import CodeGenerator


class TestCodeGenerator:
    """CodeGeneratorクラスのテスト。"""
    
    def setup_method(self):
        """各テストメソッドの前に実行される。"""
        self.generator = CodeGenerator()
        
    def test_init(self):
        """初期化のテスト。"""
        assert self.generator.library == "opencv"
        assert "opencv" in self.generator.supported_libraries
        
        # サポートされていないライブラリでの初期化
        with pytest.raises(ValueError):
            CodeGenerator(library="unsupported_library")
            
    def test_get_library_imports(self):
        """ライブラリインポート文取得のテスト。"""
        # OpenCV
        generator = CodeGenerator(library="opencv")
        imports = generator.get_library_imports()
        assert "import cv2" in imports
        assert "import numpy as np" in imports
        
        # Pillow
        generator = CodeGenerator(library="pillow")
        imports = generator.get_library_imports()
        assert "from PIL import Image" in imports
        assert "import numpy as np" in imports
        
        # SciPy
        generator = CodeGenerator(library="scipy")
        imports = generator.get_library_imports()
        assert "import numpy as np" in imports
        assert "import scipy.ndimage as ndi" in imports
        
        # scikit-image
        generator = CodeGenerator(library="scikit-image")
        imports = generator.get_library_imports()
        assert "import numpy as np" in imports
        assert "from skimage import" in imports
        
        # NumPy
        generator = CodeGenerator(library="numpy")
        imports = generator.get_library_imports()
        assert "import numpy as np" in imports
        assert "import matplotlib.pyplot as plt" in imports
        
    def test_generate_code(self):
        """コード生成のテスト。"""
        # グレースケール変換
        code = self.generator.generate_code("grayscale")
        assert "cv2.cvtColor" in code
        assert "cv2.COLOR_BGR2GRAY" in code
        
        # リサイズ
        code = self.generator.generate_code("resize", {"width": 100, "height": 100})
        assert "cv2.resize" in code
        
        # ぼかし処理
        code = self.generator.generate_code("blur", {"kernel_size": 5})
        assert "cv2.GaussianBlur" in code
        
    def test_generate_comparison_code(self):
        """画像比較コード生成のテスト。"""
        code = self.generator.generate_comparison_code("image1.jpg", "image2.jpg", ["psnr", "ssim"])
        assert "cv2.imread" in code
        assert "PSNR" in code
        assert "SSIM" in code
        assert "plt.figure" in code
        
    def test_generate_template_matching_code(self):
        """テンプレートマッチングコード生成のテスト。"""
        code = self.generator.generate_template_matching_code("image.jpg", "template.jpg", "cv2.TM_CCOEFF_NORMED")
        assert "cv2.imread" in code
        assert "cv2.matchTemplate" in code
        assert "cv2.minMaxLoc" in code
        assert "cv2.rectangle" in code
        assert "plt.figure" in code
        
    def test_format_code(self):
        """コード整形のテスト。"""
        # 空行の連続を1つにする
        code = "line1\n\n\nline2"
        formatted_code = self.generator.format_code(code)
        assert formatted_code == "line1\n\nline2"
        
        # 先頭と末尾の空行は残す
        code = "\nline1\nline2\n"
        formatted_code = self.generator.format_code(code)
        assert formatted_code == "\nline1\nline2\n"
        
    def test_save_to_file(self):
        """ファイル保存のテスト。"""
        code = "print('Hello, World!')"
        
        # 一時ファイルを作成
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            temp_path = temp.name
            
        try:
            # コードを保存
            self.generator.save_to_file(code, temp_path)
            
            # ファイルが存在するか確認
            assert os.path.exists(temp_path)
            
            # 内容が正しいか確認
            with open(temp_path, 'r', encoding='utf-8') as f:
                content = f.read()
                assert content == code
        finally:
            # 一時ファイルを削除
            os.unlink(temp_path)
            
    def test_generate_code_with_different_libraries(self):
        """異なるライブラリでのコード生成のテスト。"""
        # OpenCV
        opencv_generator = CodeGenerator(library="opencv")
        opencv_code = opencv_generator.generate_code("grayscale")
        assert "cv2.cvtColor" in opencv_code
        assert "cv2.COLOR_BGR2GRAY" in opencv_code
        
        # Pillow
        pillow_generator = CodeGenerator(library="pillow")
        pillow_code = pillow_generator.generate_code("grayscale")
        assert "convert('L')" in pillow_code
        
    def test_generate_code_for_process(self):
        """_generate_code_for_process メソッドのテスト。"""
        # このメソッドはプライベートですが、generate_code メソッドを通じてテストできます
        code = self.generator.generate_code("grayscale")
        assert "# 画像を読み込む" in code
        assert "# 処理を適用" in code
        assert "# 結果を表示" in code
        
    def test_generate_load_code(self):
        """_generate_load_code メソッドのテスト。"""
        # このメソッドはプライベートですが、generate_code メソッドを通じてテストできます
        code = self.generator.generate_code("grayscale")
        assert "cv2.imread" in code