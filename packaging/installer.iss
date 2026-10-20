; 课程书架的安装包（Inno Setup 6.5 及以上）。由 build.ps1 调用：
;   ISCC.exe /DAppVersion=0.1.0 /DBuildDir=<build\LectureAnalyzer> /DOutputDir=<dist> /DMaxDirLength=<N> installer.iss
; 按用户安装，不需要管理员权限。卸载时保留书库文件夹和 %APPDATA%\LectureAnalyzer 里记下的书库位置。

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef BuildDir
  #define BuildDir "..\build\LectureAnalyzer"
#endif
#ifndef OutputDir
  #define OutputDir "..\dist"
#endif
; 安装目录（带末尾的反斜杠）最长多少个字符：Windows 路径默认最长 259 个字符，要给程序里最深的文件留出位置。
; build.ps1 按实际的程序目录算出来传进来（M4 验收：装到较深的目录时安装中途失败，只报「MoveFile 失败；错误代码 3」）
#ifndef MaxDirLength
  #define MaxDirLength 70
#endif

[Setup]
AppId={{8C3A1F5E-6B2D-4E7A-9F1C-2D4B6A8E0C13}
AppName=课程书架
AppVersion={#AppVersion}
AppVerName=课程书架 {#AppVersion}
AppPublisher=lecture-analyzer
DefaultDirName={localappdata}\Programs\LectureAnalyzer
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OutputDir}
OutputBaseFilename=LectureAnalyzer-Setup-{#AppVersion}
SetupIconFile={#BuildDir}\app.ico
UninstallDisplayIcon={app}\LectureAnalyzer.exe
UninstallDisplayName=课程书架
; 程序目录约 4GB，大部分是 PyTorch 的显卡运行库；分文件压缩，安装时不用整包解压
Compression=lzma2/normal
SolidCompression=no
WizardStyle=modern
ShowLanguageDialog=no

[Languages]
; 简体中文翻译从 Inno Setup 7 起随附；6.x 要把官方仓库里的 Files\Languages\Unofficial\ChineseSimplified.isl
; 放进 Inno Setup 的 Languages 文件夹。没有这个文件时安装向导用英文
#if FileExists(AddBackslash(CompilerPath) + "Languages\ChineseSimplified.isl")
Name: "chs"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"
#else
Name: "en"; MessagesFile: "compiler:Default.isl"
#endif

[Tasks]
Name: "desktopicon"; Description: "在桌面创建快捷方式"; Flags: unchecked

[Files]
Source: "{#BuildDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\课程书架"; Filename: "{app}\LectureAnalyzer.exe"; AppUserModelID: "LectureAnalyzer.Desktop"
Name: "{autoprograms}\课程书架命令行"; Filename: "{cmd}"; Parameters: "/k lec --help"; WorkingDir: "{app}"
Name: "{autodesktop}\课程书架"; Filename: "{app}\LectureAnalyzer.exe"; Tasks: desktopicon; AppUserModelID: "LectureAnalyzer.Desktop"

[Run]
Filename: "{app}\LectureAnalyzer.exe"; Description: "打开课程书架"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; 运行时生成的缓存（__pycache__ 等）不在安装记录里，卸载时把这两个文件夹整个删掉
Type: filesandordirs; Name: "{app}\runtime"
Type: filesandordirs; Name: "{app}\tools"

[Code]
function DirTooLong(): Boolean;
begin
  Result := Length(AddBackslash(WizardDirValue)) > {#MaxDirLength};
end;

function TooLongMessage(): String;
begin
  Result := Format('安装路径太长（%d 个字符）。程序里有很深的文件，Windows 默认只支持 259 个字符的完整路径，' +
    '请选一个不超过 %d 个字符的文件夹，例如默认位置。', [Length(AddBackslash(WizardDirValue)), {#MaxDirLength}]);
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  // 静默安装时不弹框，交给 PrepareToInstall 检查
  if (CurPageID = wpSelectDir) and not WizardSilent() and DirTooLong() then
  begin
    SuppressibleMsgBox(TooLongMessage(), mbError, MB_OK, IDOK);
    Result := False;
  end;
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  // 静默安装（/DIR=）不经过选择文件夹的页面，开始安装前再检查一次
  Result := '';
  if DirTooLong() then
    Result := TooLongMessage();
end;
