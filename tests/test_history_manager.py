#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HistoryManagerクラスのテスト。
"""

import os
import sys
import json
import tempfile
import datetime
import pytest
from unittest.mock import patch

# srcディレクトリをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.history_manager import HistoryManager, HistoryEntry


class TestHistoryManager:
    """HistoryManagerクラスのテスト。"""
    
    def setup_method(self):
        """各テストメソッドの前に実行される。"""
        self.manager = HistoryManager()
        
        # テスト用の履歴エントリを作成
        self.test_entry = HistoryEntry.create(
            process_type="grayscale",
            params={},
            image_path="test_image.jpg",
            result_image_path="result_image.jpg"
        )
        
    def test_init(self):
        """初期化のテスト。"""
        assert self.manager.history == []
        assert self.manager.history_file is None
        
    def test_add_entry(self):
        """履歴エントリ追加のテスト。"""
        entry_id = self.manager.add_entry(
            process_type="grayscale",
            params={},
            image_path="test_image.jpg",
            result_image_path="result_image.jpg"
        )
        
        assert len(self.manager.history) == 1
        assert self.manager.history[0].id == entry_id
        assert self.manager.history[0].process_type == "grayscale"
        
    def test_get_entry(self):
        """履歴エントリ取得のテスト。"""
        entry_id = self.manager.add_entry(
            process_type="grayscale",
            params={},
            image_path="test_image.jpg",
            result_image_path="result_image.jpg"
        )
        
        entry = self.manager.get_entry(entry_id)
        assert entry is not None
        assert entry.id == entry_id
        
        # 存在しないIDの場合
        assert self.manager.get_entry("non_existent_id") is None
        
    def test_get_entry_by_index(self):
        """インデックスによる履歴エントリ取得のテスト。"""
        self.manager.add_entry(
            process_type="grayscale",
            params={},
            image_path="test_image.jpg",
            result_image_path="result_image.jpg"
        )
        
        entry = self.manager.get_entry_by_index(0)
        assert entry is not None
        assert entry.process_type == "grayscale"
        
        # 範囲外のインデックスの場合
        assert self.manager.get_entry_by_index(1) is None
        assert self.manager.get_entry_by_index(-1) is None
        
    def test_get_all_entries(self):
        """すべての履歴エントリ取得のテスト。"""
        self.manager.add_entry(
            process_type="grayscale",
            params={},
            image_path="test_image.jpg",
            result_image_path="result_image.jpg"
        )
        self.manager.add_entry(
            process_type="blur",
            params={"kernel_size": 5},
            image_path="test_image.jpg",
            result_image_path="result_image2.jpg"
        )
        
        entries = self.manager.get_all_entries()
        assert len(entries) == 2
        assert entries[0].process_type == "grayscale"
        assert entries[1].process_type == "blur"
        
    def test_clear_history(self):
        """履歴クリアのテスト。"""
        self.manager.add_entry(
            process_type="grayscale",
            params={},
            image_path="test_image.jpg",
            result_image_path="result_image.jpg"
        )
        
        assert len(self.manager.history) == 1
        
        self.manager.clear_history()
        assert len(self.manager.history) == 0
        
    def test_remove_entry(self):
        """履歴エントリ削除のテスト。"""
        entry_id = self.manager.add_entry(
            process_type="grayscale",
            params={},
            image_path="test_image.jpg",
            result_image_path="result_image.jpg"
        )
        
        assert len(self.manager.history) == 1
        
        # 存在するIDの削除
        result = self.manager.remove_entry(entry_id)
        assert result is True
        assert len(self.manager.history) == 0
        
        # 存在しないIDの削除
        result = self.manager.remove_entry("non_existent_id")
        assert result is False
        
    def test_save_and_load_history(self):
        """履歴の保存と読み込みのテスト。"""
        # 一時ファイルを作成
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            temp_path = temp.name
            
        try:
            # 履歴ファイルを設定
            manager = HistoryManager(history_file=temp_path)
            
            # エントリを追加
            manager.add_entry(
                process_type="grayscale",
                params={},
                image_path="test_image.jpg",
                result_image_path="result_image.jpg"
            )
            manager.add_entry(
                process_type="blur",
                params={"kernel_size": 5},
                image_path="test_image.jpg",
                result_image_path="result_image2.jpg"
            )
            
            # 新しいマネージャーで読み込み
            new_manager = HistoryManager(history_file=temp_path)
            
            # 履歴が正しく読み込まれたか確認
            assert len(new_manager.history) == 2
            assert new_manager.history[0].process_type == "grayscale"
            assert new_manager.history[1].process_type == "blur"
            assert new_manager.history[1].params == {"kernel_size": 5}
        finally:
            # 一時ファイルを削除
            os.unlink(temp_path)
            
    def test_get_latest_entry(self):
        """最新の履歴エントリ取得のテスト。"""
        # 空の場合
        assert self.manager.get_latest_entry() is None
        
        # エントリを追加
        self.manager.add_entry(
            process_type="grayscale",
            params={},
            image_path="test_image.jpg",
            result_image_path="result_image.jpg"
        )
        self.manager.add_entry(
            process_type="blur",
            params={"kernel_size": 5},
            image_path="test_image.jpg",
            result_image_path="result_image2.jpg"
        )
        
        latest = self.manager.get_latest_entry()
        assert latest is not None
        assert latest.process_type == "blur"
        
    def test_get_entry_count(self):
        """履歴エントリ数取得のテスト。"""
        assert self.manager.get_entry_count() == 0
        
        self.manager.add_entry(
            process_type="grayscale",
            params={},
            image_path="test_image.jpg",
            result_image_path="result_image.jpg"
        )
        
        assert self.manager.get_entry_count() == 1
        
        self.manager.add_entry(
            process_type="blur",
            params={"kernel_size": 5},
            image_path="test_image.jpg",
            result_image_path="result_image2.jpg"
        )
        
        assert self.manager.get_entry_count() == 2
        
    def test_get_entries_by_process_type(self):
        """処理タイプによる履歴エントリ取得のテスト。"""
        self.manager.add_entry(
            process_type="grayscale",
            params={},
            image_path="test_image.jpg",
            result_image_path="result_image.jpg"
        )
        self.manager.add_entry(
            process_type="blur",
            params={"kernel_size": 5},
            image_path="test_image.jpg",
            result_image_path="result_image2.jpg"
        )
        self.manager.add_entry(
            process_type="grayscale",
            params={},
            image_path="test_image2.jpg",
            result_image_path="result_image3.jpg"
        )
        
        grayscale_entries = self.manager.get_entries_by_process_type("grayscale")
        assert len(grayscale_entries) == 2
        
        blur_entries = self.manager.get_entries_by_process_type("blur")
        assert len(blur_entries) == 1
        
        # 存在しない処理タイプ
        non_existent_entries = self.manager.get_entries_by_process_type("non_existent")
        assert len(non_existent_entries) == 0
        
    def test_get_entries_by_date(self):
        """日付による履歴エントリ取得のテスト。"""
        today = datetime.date.today()
        
        # 今日の日付のエントリを追加
        self.manager.add_entry(
            process_type="grayscale",
            params={},
            image_path="test_image.jpg",
            result_image_path="result_image.jpg"
        )
        
        # 今日の日付のエントリを取得
        today_entries = self.manager.get_entries_by_date(today)
        assert len(today_entries) == 1
        
        # 昨日の日付のエントリを取得
        yesterday = today - datetime.timedelta(days=1)
        yesterday_entries = self.manager.get_entries_by_date(yesterday)
        assert len(yesterday_entries) == 0