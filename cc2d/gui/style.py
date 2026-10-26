"""Оформление интерфейса: цвета и таблица стилей Qt."""

from __future__ import annotations

from pathlib import Path

#: Значок стрелки выпадающего списка (путь с прямыми косыми чертами для Qt).
ARROW = (Path(__file__).parent / "assets" / "arrow_down.svg").as_posix()

#: Основные цвета интерфейса.
COLORS = {
    "bg": "#f3f5f8",
    "card": "#ffffff",
    "border": "#dbe1e8",
    "text": "#1f2a37",
    "muted": "#6b7785",
    "accent": "#2f6fde",
    "accent_hover": "#285fc0",
    "accent_pressed": "#214fa3",
    "error": "#c62828",
    "xy": "#d8e5fb",
    "labels": "#fbedd2",
    "best": "#d9f0de",
}

STYLESHEET = f"""
QWidget {{
    color: {COLORS['text']};
    font-family: "Segoe UI";
    font-size: 10pt;
}}
QMainWindow, QDialog, #CompareWindow, #HelpWindow {{
    background: {COLORS['bg']};
}}
QGroupBox {{
    background: {COLORS['card']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    margin-top: 14px;
    padding: 12px 10px 8px 10px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 4px;
    color: {COLORS['text']};
}}
QGroupBox QWidget {{
    font-weight: normal;
}}
QLineEdit, QComboBox {{
    background: white;
    border: 1px solid {COLORS['border']};
    border-radius: 5px;
    padding: 4px 6px;
    min-height: 20px;
}}
QComboBox::drop-down {{
    border: none;
    width: 22px;
}}
QComboBox::down-arrow {{
    image: url("{ARROW}");
    width: 10px;
    height: 6px;
    margin-right: 6px;
}}
QComboBox QAbstractItemView {{
    border: 1px solid {COLORS['border']};
    selection-background-color: #e4ecf9;
    selection-color: {COLORS['text']};
}}
QLineEdit:focus, QComboBox:focus {{
    border: 1px solid {COLORS['accent']};
}}
QLineEdit:disabled, QComboBox:disabled {{
    background: #f0f2f5;
    color: #9aa4af;
}}
QLineEdit[invalid="true"] {{
    border: 1px solid {COLORS['error']};
    background: #fff5f5;
}}
QLabel#FieldError, QLabel#GroupError {{
    color: {COLORS['error']};
    font-size: 9pt;
}}
QLabel#Muted {{
    color: {COLORS['muted']};
    font-size: 9pt;
}}
QLabel#Placeholder {{
    color: {COLORS['muted']};
    font-size: 11pt;
}}
QPushButton {{
    background: white;
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    padding: 6px 14px;
}}
QPushButton:hover {{
    background: #eef3fb;
    border-color: #b9c8dd;
}}
QPushButton:pressed {{
    background: #dfe8f6;
}}
QPushButton:disabled {{
    color: #a6afb9;
    background: #f5f6f8;
    border-color: #e4e8ed;
}}
QPushButton#Primary {{
    background: {COLORS['accent']};
    color: white;
    border: 1px solid {COLORS['accent']};
    font-weight: 600;
}}
QPushButton#Primary:hover {{
    background: {COLORS['accent_hover']};
}}
QPushButton#Primary:pressed {{
    background: {COLORS['accent_pressed']};
}}
QPushButton#Primary:disabled {{
    background: #a9c2ee;
    border-color: #a9c2ee;
    color: #eef3fc;
}}
QPushButton#Link {{
    border: none;
    background: transparent;
    color: {COLORS['accent']};
    padding: 2px 4px;
}}
QPushButton#Link:hover {{
    text-decoration: underline;
}}
QPushButton#Link:disabled {{
    color: #a6afb9;
}}
QFrame#AlgorithmCard {{
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    background: #fbfcfe;
}}
QFrame#AlgorithmCard[active="true"] {{
    border: 1px solid #9dbbef;
    background: #f5f9ff;
}}
QCheckBox#AlgorithmTitle {{
    font-weight: 600;
}}
QTableView, QTableWidget {{
    background: white;
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    gridline-color: #e8ecf1;
    selection-background-color: #cfdcf3;
    selection-color: {COLORS['text']};
}}
QHeaderView::section {{
    background: #eef1f5;
    border: none;
    border-right: 1px solid #e0e5eb;
    border-bottom: 1px solid #e0e5eb;
    padding: 4px 6px;
    font-weight: 600;
}}
QProgressBar {{
    border: 1px solid {COLORS['border']};
    border-radius: 5px;
    background: white;
    text-align: center;
    min-height: 16px;
    max-width: 220px;
}}
QProgressBar::chunk {{
    background: {COLORS['accent']};
    border-radius: 4px;
}}
QStatusBar {{
    background: {COLORS['card']};
    border-top: 1px solid {COLORS['border']};
}}
QMenuBar {{
    background: {COLORS['card']};
    border-bottom: 1px solid {COLORS['border']};
}}
QMenuBar::item:selected, QMenu::item:selected {{
    background: #e4ecf9;
}}
QTextBrowser {{
    background: white;
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    padding: 8px;
}}
QSplitter::handle {{
    background: transparent;
}}
"""
