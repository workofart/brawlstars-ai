import win32gui, win32com.client
import re


class WindowMgr:
    """Encapsulates some calls to the winapi for window management"""

    def __init__ (self):
        """Constructor"""
        self._handle = None

    def find_window(self, class_name, window_name=None):
        """find a window by its class_name"""
        self._handle = win32gui.FindWindow(class_name, window_name)

    def _window_enum_callback(self, hwnd, wildcard):
        """Pass to win32gui.EnumWindows() to check all the opened windows"""
        # if re.match(wildcard, str(win32gui.GetWindowText(hwnd))) is not None:
        if wildcard == str(win32gui.GetWindowText(hwnd)):
            self._handle = hwnd

    def find_window_wildcard(self, wildcard):
        """find a window whose title matches the wildcard regex"""
        self._handle = None
        win32gui.EnumWindows(self._window_enum_callback, wildcard)

    def set_foreground(self):
        shell = win32com.client.Dispatch("WScript.Shell")
        shell.SendKeys('%')
        """put the window in the foreground"""
        win32gui.SetForegroundWindow(self._handle)


w = WindowMgr()
w.find_window_wildcard("MEmu")
w.set_foreground()
w.set_foreground()
print(win32gui.GetWindowText(w._handle))