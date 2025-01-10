import sys
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QTextEdit,
    QCheckBox
)
from PyQt5.QtGui import QIcon, QColor
import torch
from run import generate_comment
from model import RNN, vocab_to_int, int_to_vocab, device
import darkdetect


import ctypes
myappid = u'mycompany.myproduct.subproduct.version'
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("RNN YouTube Comments")
        self.setGeometry(100, 100, 800, 600)
        self.setWindowIcon(QIcon('./ui/nugget.png'))

        self.text_edit = QTextEdit(self)
        self.text_edit.setPlaceholderText(
            "Generated comments will appear here...")

        self.generate_button = QPushButton("Generate Comment", self)
        self.generate_button.clicked.connect(self.generate_comment)

        self.hide_eos_checkbox = QCheckBox("Hide <EOS>", self)
        self.hide_eos_checkbox.setChecked(True)

        layout = QVBoxLayout()
        layout.addWidget(self.hide_eos_checkbox)
        layout.addWidget(self.text_edit)
        layout.addWidget(self.generate_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        with open('vocab_size.txt', 'r') as f:
            vocab_size = int(f.read())

        embedding_dim = 64
        hidden_dim = 128
        output_dim = vocab_size

        self.model = RNN(
            vocab_size, embedding_dim, hidden_dim, output_dim).to(device)
        self.model.load_state_dict(torch.load('rnn_model.pth'))
        self.model.eval()

    def generate_comment(self):
        generated_comment = generate_comment(
            self.model,
            vocab_to_int,
            int_to_vocab)
        if self.hide_eos_checkbox.isChecked():
            generated_comment = generated_comment.replace('<EOS>', '')
        self.text_edit.setText(generated_comment)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Dark mode if the system is set to dark mode.
    if darkdetect.isDark():
        app.setStyle("Fusion")
        dark_palette = app.palette()
        dark_palette.setColor(app.palette().Window, QColor(53, 53, 53))
        dark_palette.setColor(app.palette().WindowText, QColor(255, 255, 255))
        dark_palette.setColor(app.palette().Base, QColor(25, 25, 25))
        dark_palette.setColor(app.palette().AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(app.palette().ToolTipBase, QColor(255, 255, 255))
        dark_palette.setColor(app.palette().ToolTipText, QColor(255, 255, 255))
        dark_palette.setColor(app.palette().Text, QColor(255, 255, 255))
        dark_palette.setColor(app.palette().Button, QColor(53, 53, 53))
        dark_palette.setColor(app.palette().ButtonText, QColor(255, 255, 255))
        dark_palette.setColor(app.palette().BrightText, QColor(255, 0, 0))
        dark_palette.setColor(app.palette().Link, QColor(42, 130, 218))
        dark_palette.setColor(app.palette().Highlight, QColor(42, 130, 218))
        dark_palette.setColor(app.palette().HighlightedText, QColor(0, 0, 0))
        app.setPalette(dark_palette)
    else:
        app.setStyle("Fusion")

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
