Hello,

Compiling PyQt5 on the Jetson Nano involves building the PyQt5 library from source. PyQt5 is a set of Python bindings for the Qt application framework, and building it from source ensures that it is compatible with the Jetson Nano’s architecture and environment.

Here are the steps to compile PyQt5 on the Jetson Nano:

Install Prerequisites:
sudo apt-get update
sudo apt-get install build-essential python3-dev python3-pip python3-pyqt5.qtsvg python3-pyqt5.qtwebkit

Install SIP:

download for : Riverbank Computing | Download
tar -xvzf sip-4.19.25.tar.gz
cd sip-4.19.25.tar.gz
make
sudo make install

install PyQt5:
download for :PyQt5 · PyPI
tar -xvzf PyQt5-5.15.0.tar.gz
cd PyQt5-5.15.0.tar.gz
python3 configure.py --qmake /usr/lib/aarch64-linux-gnu/qt5/bin/qmake
make
sudo make install

Test PyQt5:
import sys
from PyQt5.QtWidgets import QApplication, QLabel

app = QApplication(sys.argv)
label = QLabel(“Hello, PyQt5!”)
label.show()
sys.exit(app.exec_())
