
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow

from GUI import DisplayUI
from GUI.VideoDisplay import Display
from yolov3_deepsort.yolo_video import detectAPI

if __name__ == '__main__':
    # input_path = 'yolov3_deepsort/img/video-02.avi'
    # output_path = 'yolov3_deepsort/output'
    # detectAPI(input_path, output_path)
    app = QApplication(sys.argv)
    mainWnd = QMainWindow()
    ui = DisplayUI.Ui_MainWindow()

    # 可以理解成将创建的 ui 绑定到新建的 mainWnd 上
    ui.setupUi(mainWnd)
    display = Display(ui, mainWnd)
    mainWnd.show()
    sys.exit(app.exec_())