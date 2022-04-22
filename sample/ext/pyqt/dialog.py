"""Dialog-Style application."""

import sys

from PyQt6.QtWidgets import QApplication
from PyQt6.QtWidgets import QDialog
from PyQt6.QtWidgets import QDialogButtonBox
from PyQt6.QtWidgets import QFormLayout
from PyQt6.QtWidgets import QLineEdit
from PyQt6.QtWidgets import QVBoxLayout


class Dialog(QDialog):
    """Dialog."""

    def __init__(self, parent=None):
        """Initializer."""
        super().__init__(parent)
        self.setWindowTitle("QDialog")
        dlgLayout = QVBoxLayout()
        formLayout = QFormLayout()
        formLayout.addRow("Name:", QLineEdit())
        formLayout.addRow("Age:", QLineEdit())
        formLayout.addRow("Job:", QLineEdit())
        formLayout.addRow("Hobbies:", QLineEdit())
        dlgLayout.addLayout(formLayout)
        btns = QDialogButtonBox()
        btns.setStandardButtons(
            QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok
        )
        dlgLayout.addWidget(btns)
        self.setLayout(dlgLayout)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    dlg = Dialog()
    dlg.show()
    sys.exit(app.exec())
