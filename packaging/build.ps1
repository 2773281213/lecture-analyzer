<#
课程书架的打包脚本：生成 dist\LectureAnalyzer-Setup-<版本>.exe（按用户安装的 setup.exe）。

步骤：
  1. 构建本项目的 wheel；
  2. 复制 uv 管理的独立 Python 3.12 作为程序自带的运行时（runtime\）；
  3. 按 uv.lock 导出的固定版本装好全部依赖（含 PyTorch CUDA 13.0 和本地语音识别），再装本项目；
  4. 随附 7-Zip：用系统自带的 msiexec /a 从官方 MSI 解出 7z.exe、7z.dll 和许可证（不安装）；
  5. 用系统自带的 csc.exe 编译启动程序 LectureAnalyzer.exe（图标 app.ico）；
  6. 写入 lec.cmd 和第三方许可证清单，并检查运行时里的关键模块都能导入；
  7. 用 Inno Setup 编译安装包。

用法（在任意目录）：
  powershell -ExecutionPolicy Bypass -File packaging\build.ps1 -InnoSetup <ISCC.exe> -SevenZipMsi <7z*-x64.msi>
  加 -SkipInstaller 只生成 build\LectureAnalyzer（可以直接双击里面的 LectureAnalyzer.exe 试用），不编译安装包。
#>
param(
    [string]$InnoSetup = "",
    [string]$SevenZipMsi = "",
    [switch]$SkipInstaller
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$work = Join-Path $root "build"
$app = Join-Path $work "LectureAnalyzer"
$runtime = Join-Path $app "runtime"
$python = Join-Path $runtime "python.exe"
$dist = Join-Path $root "dist"
$icon = Join-Path $root "src\lecture_analyzer\web\static\app.ico"
$csc = Join-Path $env:WINDIR "Microsoft.NET\Framework64\v4.0.30319\csc.exe"
$torchIndex = "https://download.pytorch.org/whl/cu130"

function Step([string]$text) { Write-Host "== $text" -ForegroundColor Cyan }
function Check([string]$what) { if ($LASTEXITCODE -ne 0) { throw "$what 失败（退出码 $LASTEXITCODE）" } }

if (-not $SkipInstaller -and -not ($InnoSetup -and (Test-Path $InnoSetup))) {
    throw "请用 -InnoSetup 指定 ISCC.exe（Inno Setup 6.5 及以上），或加 -SkipInstaller 只生成程序目录"
}
if (-not ($SevenZipMsi -and (Test-Path $SevenZipMsi))) {
    throw "请用 -SevenZipMsi 指定 7-Zip 的 x64 MSI（https://www.7-zip.org/download.html）"
}
$version = (Select-String -Path (Join-Path $root "pyproject.toml") -Pattern '^version = "(.+)"').Matches[0].Groups[1].Value

Step "1/7 构建 wheel（版本 $version）"
Remove-Item -Recurse -Force (Join-Path $work "wheel") -ErrorAction SilentlyContinue
uv build --wheel --out-dir (Join-Path $work "wheel") $root
Check "构建 wheel"

Step "2/7 复制独立的 Python 运行时"
$basePython = uv python find --system --managed-python 3.12
Check "查找 uv 管理的 Python 3.12"
Remove-Item -Recurse -Force $app -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $app | Out-Null
Copy-Item -Recurse (Split-Path -Parent $basePython) $runtime
Remove-Item -Force (Join-Path $runtime "Lib\EXTERNALLY-MANAGED") -ErrorAction SilentlyContinue  # 程序私有的运行时，允许装包
foreach ($unused in @("Lib\test", "Lib\idlelib", "Lib\tkinter", "tcl", "include", "libs")) {
    Remove-Item -Recurse -Force (Join-Path $runtime $unused) -ErrorAction SilentlyContinue
}

Step "3/7 按 uv.lock 安装依赖（含 PyTorch CUDA 13.0）"
$requirements = Join-Path $work "requirements.txt"
uv export --project $root --frozen --no-dev --extra asr --no-hashes --no-emit-project --format requirements-txt -o $requirements
Check "导出依赖清单"
uv pip install --python $python --no-deps --compile-bytecode --index-strategy unsafe-best-match `
    --extra-index-url $torchIndex -r $requirements
Check "安装依赖"
$wheel = Get-ChildItem (Join-Path $work "wheel\*.whl") | Select-Object -First 1
uv pip install --python $python --no-deps --compile-bytecode $wheel.FullName
Check "安装本项目"
# PyTorch 的 C++ 头文件只在编译扩展时用到（约 9300 个文件），去掉可以加快压缩和安装
Remove-Item -Recurse -Force (Join-Path $runtime "Lib\site-packages\torch\include") -ErrorAction SilentlyContinue

Step "4/7 随附 7-Zip（从 MSI 解出，不安装）"
$extract = Join-Path $env:TEMP "lecture-analyzer-7zip-msi"  # msiexec 的目标路径用临时目录（纯英文路径更稳妥）
Remove-Item -Recurse -Force $extract -ErrorAction SilentlyContinue
$msi = (Resolve-Path $SevenZipMsi).Path
$process = Start-Process msiexec.exe -ArgumentList @("/a", "`"$msi`"", "/qn", "TARGETDIR=`"$extract`"") -Wait -PassThru
if ($process.ExitCode -ne 0) { throw "解开 7-Zip 的 MSI 失败（退出码 $($process.ExitCode)）" }
$sevenZip = Get-ChildItem -Recurse -Path $extract -Filter "7z.exe" | Select-Object -First 1
$tools = Join-Path $app "tools\7zip"
New-Item -ItemType Directory -Force $tools | Out-Null
foreach ($name in @("7z.exe", "7z.dll", "License.txt")) { Copy-Item (Join-Path $sevenZip.DirectoryName $name) $tools }
Remove-Item -Recurse -Force $extract

Step "5/7 编译启动程序"
& $csc /nologo /codepage:65001 /target:winexe "/win32icon:$icon" "/out:$(Join-Path $app 'LectureAnalyzer.exe')" `
    (Join-Path $PSScriptRoot "launcher.cs")
Check "编译启动程序"
Copy-Item $icon (Join-Path $app "app.ico")

Step "6/7 命令行入口、第三方许可证清单、检查运行时"
Copy-Item (Join-Path $PSScriptRoot "lec.cmd") $app
& $python -I -X utf8 (Join-Path $PSScriptRoot "notices.py") (Join-Path $app "THIRD-PARTY-NOTICES.txt")
Check "生成第三方许可证清单"
& $python -I -X utf8 -c "import lecture_analyzer.desktop, lecture_analyzer.asr.funasr_backend, funasr, modelscope, torch, webview, keyring; print('运行时检查通过：torch', torch.__version__)"
Check "检查运行时"
$files = Get-ChildItem -Recurse -File $app
$size = ($files | Measure-Object Length -Sum).Sum / 1GB
# Windows 路径默认最长 259 个字符：安装目录要给最深的文件留出位置，安装包按这个长度检查
$deepest = ($files | ForEach-Object { $_.FullName.Length - $app.Length - 1 } | Measure-Object -Maximum).Maximum
$maxDir = 259 - $deepest
Write-Host ("程序目录：{0}（{1:N1} GB）；最深的文件相对路径 {2} 个字符，安装目录最长 {3} 个字符" -f $app, $size, $deepest, $maxDir)

if ($SkipInstaller) {
    Write-Host "已跳过安装包，可以直接运行 $(Join-Path $app 'LectureAnalyzer.exe') 试用"
    return
}

Step "7/7 编译安装包（压缩约 4GB 的程序目录，需要一些时间）"
& $InnoSetup "/DAppVersion=$version" "/DBuildDir=$app" "/DOutputDir=$dist" "/DMaxDirLength=$maxDir" (Join-Path $PSScriptRoot "installer.iss")
Check "编译安装包"
Get-Item (Join-Path $dist "LectureAnalyzer-Setup-$version.exe") | Format-List FullName, Length
