#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
履歴管理モジュール。

このモジュールは画像処理確認アプリで使用される処理履歴の管理機能を提供します。
"""

import os
import json
import datetime
import uuid
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any


@dataclass
class HistoryEntry:
    """履歴エントリを表すデータクラス。"""
    
    id: str
    process_type: str
    params: Dict[str, Any]
    timestamp: str
    image_path: str
    result_image_path: str
    
    @classmethod
    def create(cls, process_type, params, image_path, result_image_path):
        """新しい履歴エントリを作成する。

        Args:
            process_type: 処理の種類
            params: 処理のパラメータ
            image_path: 元画像のパス
            result_image_path: 処理結果画像のパス

        Returns:
            新しい履歴エントリ
        """
        return cls(
            id=str(uuid.uuid4()),
            process_type=process_type,
            params=params,
            timestamp=datetime.datetime.now().isoformat(),
            image_path=image_path,
            result_image_path=result_image_path
        )


class HistoryManager:
    """処理履歴を管理するクラス。"""
    
    def __init__(self, history_file=None):
        """初期化メソッド。

        Args:
            history_file: 履歴を保存するファイルのパス（Noneの場合は保存しない）
        """
        self.history: List[HistoryEntry] = []
        self.history_file = history_file
        
        # 履歴ファイルが指定されていて、存在する場合は読み込む
        if history_file and os.path.exists(history_file):
            self.load_history()
        
    def add_entry(self, process_type, params, image_path, result_image_path):
        """履歴エントリを追加する。

        Args:
            process_type: 処理の種類
            params: 処理のパラメータ
            image_path: 元画像のパス
            result_image_path: 処理結果画像のパス

        Returns:
            追加された履歴エントリのID
        """
        entry = HistoryEntry.create(process_type, params, image_path, result_image_path)
        self.history.append(entry)
        
        # 履歴ファイルが指定されている場合は保存する
        if self.history_file:
            self.save_history()
            
        return entry.id
        
    def get_entry(self, entry_id):
        """指定されたIDの履歴エントリを取得する。

        Args:
            entry_id: 履歴エントリのID

        Returns:
            履歴エントリ、見つからない場合はNone
        """
        for entry in self.history:
            if entry.id == entry_id:
                return entry
        return None
        
    def get_entry_by_index(self, index):
        """指定されたインデックスの履歴エントリを取得する。

        Args:
            index: 履歴エントリのインデックス

        Returns:
            履歴エントリ、インデックスが範囲外の場合はNone
        """
        if 0 <= index < len(self.history):
            return self.history[index]
        return None
        
    def get_all_entries(self):
        """すべての履歴エントリを取得する。

        Returns:
            履歴エントリのリスト
        """
        return self.history
        
    def clear_history(self):
        """履歴をクリアする。"""
        self.history = []
        
        # 履歴ファイルが指定されている場合は保存する
        if self.history_file:
            self.save_history()
            
    def remove_entry(self, entry_id):
        """指定されたIDの履歴エントリを削除する。

        Args:
            entry_id: 削除する履歴エントリのID

        Returns:
            削除に成功した場合はTrue、見つからない場合はFalse
        """
        for i, entry in enumerate(self.history):
            if entry.id == entry_id:
                del self.history[i]
                
                # 履歴ファイルが指定されている場合は保存する
                if self.history_file:
                    self.save_history()
                    
                return True
        return False
        
    def save_history(self):
        """履歴をファイルに保存する。"""
        if not self.history_file:
            return
            
        # ディレクトリが存在しない場合は作成する
        directory = os.path.dirname(self.history_file)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        # 履歴をJSON形式で保存する
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump([asdict(entry) for entry in self.history], f, ensure_ascii=False, indent=2)
            
    def load_history(self):
        """履歴をファイルから読み込む。"""
        if not self.history_file or not os.path.exists(self.history_file):
            return
            
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                history_data = json.load(f)
                
            self.history = [HistoryEntry(**entry) for entry in history_data]
        except (json.JSONDecodeError, KeyError) as e:
            print(f"履歴ファイルの読み込みに失敗しました: {e}")
            self.history = []
            
    def get_latest_entry(self):
        """最新の履歴エントリを取得する。

        Returns:
            最新の履歴エントリ、履歴が空の場合はNone
        """
        if not self.history:
            return None
        return self.history[-1]
        
    def get_entry_count(self):
        """履歴エントリの数を取得する。

        Returns:
            履歴エントリの数
        """
        return len(self.history)
        
    def get_entries_by_process_type(self, process_type):
        """指定された処理タイプの履歴エントリを取得する。

        Args:
            process_type: 処理タイプ

        Returns:
            指定された処理タイプの履歴エントリのリスト
        """
        return [entry for entry in self.history if entry.process_type == process_type]
        
    def get_entries_by_date(self, date):
        """指定された日付の履歴エントリを取得する。

        Args:
            date: 日付（datetime.date）

        Returns:
            指定された日付の履歴エントリのリスト
        """
        return [
            entry for entry in self.history
            if datetime.datetime.fromisoformat(entry.timestamp).date() == date
        ]