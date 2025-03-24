#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
画像処理確認アプリのUIコンポーネントモジュール。

このモジュールはStreamlitを使用したUIコンポーネントを提供します。
"""

import streamlit as st
from PIL import Image
import cv2
import numpy as np
import datetime


def show_header():
    """アプリケーションのヘッダーを表示する。"""
    st.title("画像処理確認アプリ")


def convert_to_pil(image):
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


def show_image_display(images, current_image_index, history, selected_history_index):
    """処理前と処理後の画像を表示する。

    Args:
        images: 画像のリスト
        current_image_index: 現在表示中の画像インデックス
        history: 処理履歴
        selected_history_index: 選択中の履歴インデックス
        
    Returns:
        (prev_button, next_button, use_processed_button, restore_original_button): 各種ボタンがクリックされたかどうか
    """
    prev_button = False
    next_button = False
    use_processed_button = False
    restore_original_button = False
    
    # 画像が読み込まれていない場合
    if len(images) == 0:
        st.info("フォルダを選択して画像を読み込んでください。")
        return prev_button, next_button, use_processed_button, restore_original_button
    
    # 元画像（オリジナル）
    original_image = None
    if 'original_images' in st.session_state and len(st.session_state.original_images) > current_image_index:
        original_image = st.session_state.original_images[current_image_index]
    else:
        original_image = images[current_image_index]
    
    # 現在の画像（処理前）
    current_image = images[current_image_index]
    
    # 処理後画像
    processed_image = None
    if len(history) > 0:
        # 選択された履歴エントリがある場合はそれを表示、なければ最新の履歴エントリを表示
        history_index = selected_history_index if selected_history_index is not None else len(history) - 1
        history_entry = history[history_index]
        
        # 一時的な処理結果がある場合はそれを使用（画像ナビゲーション時）
        if 'temp_processed_image' in st.session_state and st.session_state.temp_processed_image is not None:
            processed_image = st.session_state.temp_processed_image
        else:
            processed_image = history_entry["processed_image"]
        
        # 元画像と処理後画像のインデックスが一致しているかチェック
        if "image_index" in history_entry and history_entry["image_index"] != current_image_index:
            st.warning("⚠️ 注意: 現在表示されている元画像と処理後画像は異なる画像です")
    else:
        # 履歴がない場合は処理前画像をそのまま表示
        processed_image = current_image
    
    # 3カラムレイアウトで表示
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("元画像（オリジナル）")
        st.image(convert_to_pil(original_image), use_container_width=True)
    
    with col2:
        st.subheader("処理前画像")
        st.image(convert_to_pil(current_image), use_container_width=True)
    
    with col3:
        st.subheader("処理後画像")
        st.image(convert_to_pil(processed_image), use_container_width=True)
        
        # 処理の情報表示
        if len(history) > 0 and selected_history_index is not None:
            history_entry = history[selected_history_index]
            # 処理タイプとパラメータを表示
            st.info(f"処理: {history_entry['process_type']} - パラメータ: {history_entry['params']}")
            
            # 処理の連鎖情報を表示（あれば）
            if "chain_info" in history_entry and history_entry["chain_info"]:
                st.success(f"処理連鎖: {history_entry['chain_info']}")
    
    # 画像ナビゲーションボタン
    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)
    with nav_col1:
        prev_button = st.button("前の画像")
    with nav_col2:
        next_button = st.button("次の画像")
    with nav_col3:
        # 処理後画像を元画像として使用するボタン
        if len(history) > 0:
            use_processed_button = st.button("処理後画像を元画像として使用")
    with nav_col4:
        # オリジナルに戻るボタン
        if 'original_images' in st.session_state and len(st.session_state.original_images) > 0:
            restore_original_button = st.button("オリジナル画像に戻す")
    
    # ビューモードスイッチ
    st.radio("表示モード", ["3画像表示", "差分表示"], key="view_mode", horizontal=True)
    
    # 差分表示モード
    if st.session_state.get("view_mode") == "差分表示" and processed_image is not None:
        st.subheader("処理前後の差分")
        try:
            # 画像サイズが同じ場合のみ差分を計算
            if current_image.shape == processed_image.shape:
                # 差分の計算
                diff = cv2.absdiff(current_image, processed_image)
                
                # 差分を増幅してわかりやすく
                diff_amplified = cv2.convertScaleAbs(diff, alpha=5.0)
                
                # ヒートマップカラーマップを適用
                diff_color = cv2.applyColorMap(diff_amplified, cv2.COLORMAP_JET)
                
                # 差分画像を表示
                st.image(convert_to_pil(diff_color), use_container_width=True, caption="差分（強調表示）")
            else:
                st.warning("処理前後の画像サイズが異なるため、差分表示できません")
        except Exception as e:
            st.error(f"差分表示でエラーが発生しました: {str(e)}")
    
    return prev_button, next_button, use_processed_button, restore_original_button


def show_history_navigation(history, selected_history_index):
    """履歴ナビゲーションボタンを表示する。

    Args:
        history: 処理履歴
        selected_history_index: 選択中の履歴インデックス
    
    Returns:
        (prev_clicked, next_clicked): 前/次ボタンがクリックされたかどうか
    """
    col1, col2 = st.columns(2)
    prev_clicked = False
    next_clicked = False
    
    with col1:
        if st.button("処理の1つ前に戻る") and selected_history_index is not None:
            if selected_history_index > 0:
                prev_clicked = True
                st.success("1つ前の処理に戻りました。")
            else:
                st.warning("これ以上前の処理はありません。")
    with col2:
        if st.button("処理の1つ次に進む") and selected_history_index is not None:
            if selected_history_index < len(history) - 1:
                next_clicked = True
                st.success("1つ次の処理に進みました。")
            else:
                st.warning("これ以上次の処理はありません。")
    
    return prev_clicked, next_clicked


def show_history_list(history, selected_history_index, process_types_reverse=None):
    """履歴リストを表示する。

    Args:
        history: 処理履歴
        selected_history_index: 選択中の履歴インデックス
        process_types_reverse: 処理タイプの英語名から日本語名への辞書（オプション）
    
    Returns:
        選択された履歴のインデックス、未選択の場合はNone
    """
    if len(history) == 0:
        st.info("履歴がありません。")
        return None
        
    # 履歴選択
    history_options = []
    for i, entry in enumerate(history):
        # 処理タイプを日本語表示
        process_type = entry['process_type']
        if process_types_reverse and process_type in process_types_reverse:
            process_type = process_types_reverse[process_type]
            
        base_text = f"{i+1}. {entry['timestamp']} - {process_type}"
        # 処理連鎖情報を追加（あれば）
        if "chain_info" in entry and entry["chain_info"]:
            base_text += f" [{entry['chain_info']}]"
        history_options.append(base_text)
    
    selected_history = st.selectbox(
        "履歴を選択",
        history_options,
        index=selected_history_index if selected_history_index is not None else 0
    )
    
    if selected_history:
        selected_index = history_options.index(selected_history)
        
        # 選択された履歴の詳細を表示
        history_entry = history[selected_index]
        
        # 処理タイプを日本語表示
        process_type = history_entry['process_type']
        if process_types_reverse and process_type in process_types_reverse:
            process_type = process_types_reverse[process_type]
            
        st.write(f"処理タイプ: {process_type}")
        st.write(f"パラメータ: {history_entry['params']}")
        
        # 処理連鎖情報を表示（あれば）
        if "chain_info" in history_entry and history_entry["chain_info"]:
            st.write(f"処理連鎖: {history_entry['chain_info']}")
        
        # プログレスバー
        st.progress((selected_index + 1) / len(history))
        
        return selected_index
        
    return None


def show_code_display(generated_code):
    """生成されたコードを表示する。

    Args:
        generated_code: 表示するコード
    """
    if generated_code is not None:
        st.header("生成されたコード")
        st.code(generated_code, language="python")


def show_control_panel():
    """操作パネルを表示する。
    
    Returns:
        アクション名とその実行フラグ
    """
    st.subheader("操作パネル")
    
    # UIの分割
    col1, col2, col3 = st.columns(3)
    
    with col1:
        save_button = st.button("現在の画像を保存")
    
    with col2:
        reset_button = st.button("処理をリセット")
    
    with col3:
        undo_button = st.button("1つ前に戻る")
        
    return {
        "save": save_button,
        "reset": reset_button,
        "undo": undo_button
    }