// 课程书架的启动程序（LectureAnalyzer.exe）：用随附的 Python 运行时启动桌面程序，然后自己退出。
// 由 build.ps1 用系统自带的 csc.exe 编译：csc /target:winexe /codepage:65001 /win32icon:app.ico launcher.cs
using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;

static class Launcher
{
    [STAThread]
    static int Main()
    {
        string home = AppDomain.CurrentDomain.BaseDirectory;
        string python = Path.Combine(home, "runtime", "pythonw.exe");
        if (!File.Exists(python))
        {
            MessageBox.Show("找不到随附的 Python 运行时：\n" + python + "\n\n请重新安装课程书架。", "课程书架",
                            MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 1;
        }
        ProcessStartInfo start = new ProcessStartInfo(python, "-m lecture_analyzer.desktop");
        start.WorkingDirectory = home;
        start.UseShellExecute = false;
        start.EnvironmentVariables["PYTHONNOUSERSITE"] = "1";  // 不读用户目录里另外装的 Python 包
        Process.Start(start);
        return 0;
    }
}
