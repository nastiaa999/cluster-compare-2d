# Сборка дистрибутива ClusterCompare 2D (п. 4.6.3 ТЗ).
# Запуск: .venv\Scripts\pyinstaller cc2d.spec --noconfirm
# Результат: dist\ClusterCompare2D\ClusterCompare2D.exe (папка целиком переносится на другой компьютер).
# Сборка в папку, а не в один файл: так программа запускается быстрее (п. 4.1.4.1 ТЗ).

# Модули, которые программе не нужны: исключаются, чтобы уменьшить размер (п. 4.1.4.9 ТЗ).
EXCLUDES = [
    "tkinter", "pytest", "sphinx", "IPython", "PyInstaller",
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick",
    "PySide6.QtQml", "PySide6.QtQuick", "PySide6.QtQuickWidgets", "PySide6.Qt3DCore",
    "PySide6.Qt3DRender", "PySide6.QtMultimedia", "PySide6.QtMultimediaWidgets",
    "PySide6.QtCharts", "PySide6.QtDataVisualization", "PySide6.QtGraphs",
    "PySide6.QtPdf", "PySide6.QtPdfWidgets", "PySide6.QtBluetooth", "PySide6.QtNfc",
    "PySide6.QtPositioning", "PySide6.QtLocation", "PySide6.QtSensors",
    "PySide6.QtSerialPort", "PySide6.QtSql", "PySide6.QtTest", "PySide6.QtDesigner",
    "PySide6.QtHelp", "PySide6.QtOpenGL", "PySide6.QtOpenGLWidgets", "PySide6.QtRemoteObjects",
    "PySide6.QtScxml", "PySide6.QtSpatialAudio", "PySide6.QtTextToSpeech", "PySide6.QtWebChannel",
    "PySide6.QtWebSockets", "PySide6.QtHttpServer", "PySide6.QtNetworkAuth",
    "matplotlib.backends.backend_tkagg", "matplotlib.backends.backend_wx",
]

a = Analysis(
    ["cc2d/__main__.py"],
    pathex=["."],
    datas=[("cc2d/gui/assets", "cc2d/gui/assets")],
    hiddenimports=["matplotlib.backends.backend_qtagg"],
    excludes=EXCLUDES,
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ClusterCompare2D",
    console=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="ClusterCompare2D")
