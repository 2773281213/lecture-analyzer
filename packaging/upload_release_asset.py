"""将本地编译的安装包上传到 GitHub Release v0.1.0。"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import httpx

REPO = "2773281213/lecture-analyzer"
TAG = "v0.1.0"
ASSET_NAME = "LectureAnalyzer-Setup-0.1.0.exe"
ASSET_PATH = Path(r"F:\claude工作区\lecture-analyzer\dist\LectureAnalyzer-Setup-0.1.0.exe")


def get_github_token() -> str:
    cred_input = "protocol=https\nhost=github.com\n\n"
    res = subprocess.run(["git", "credential", "fill"], input=cred_input, capture_output=True, text=True, check=True)
    for line in res.stdout.splitlines():
        if line.startswith("password="):
            return line.split("=", 1)[1]
    raise RuntimeError("未在 git credential 中找到 github.com 的令牌")


def chunk_generator(filepath: Path, chunk_size: int = 1024 * 1024):
    total = filepath.stat().st_size
    sent = 0
    t0 = time.time()
    last_log = t0
    last_sent = 0
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            sent += len(chunk)
            now = time.time()
            if now - last_log >= 3.0 or sent == total:
                speed = (sent - last_sent) / max(0.001, now - last_log) / (1024 * 1024)
                pct = sent / total * 100
                mb = sent / (1024 * 1024)
                total_mb = total / (1024 * 1024)
                elapsed = now - t0
                eta = (total - sent) / max(1, (sent / max(0.001, elapsed)))
                print(
                    f"[上传进度] {mb:.1f} MB / {total_mb:.1f} MB ({pct:.1f}%) "
                    f"- 速度: {speed:.2f} MB/s - 已用时: {elapsed:.0f}s - 预计剩余: {eta:.0f}s",
                    flush=True,
                )
                last_log = now
                last_sent = sent
            yield chunk


def main() -> None:
    if not ASSET_PATH.exists():
        print(f"错误: 找不到文件 {ASSET_PATH}", file=sys.stderr)
        sys.exit(1)

    file_size = ASSET_PATH.stat().st_size
    print(f"准备上传: {ASSET_NAME} ({file_size:,} 字节，约 {file_size / (1024**3):.2f} GB)")

    token = get_github_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "lecture-analyzer-publisher",
    }

    timeout = httpx.Timeout(connect=60.0, read=1800.0, write=1800.0, pool=60.0)
    with httpx.Client(timeout=timeout) as client:
        # 1. 获取 Release 信息
        r = client.get(f"https://api.github.com/repos/{REPO}/releases/tags/{TAG}", headers=headers)
        if r.status_code != 200:
            print(f"获取 Release 失败: HTTP {r.status_code} - {r.text}", file=sys.stderr)
            sys.exit(1)
        release = r.json()
        release_id = release["id"]
        print(f"找到 Release: id={release_id}, tag={release['tag_name']}, url={release['html_url']}")

        # 2. 检查是否已有同名 asset
        for asset in release.get("assets", []):
            if asset["name"] == ASSET_NAME:
                print(f"发现已有资产 (id={asset['id']}, size={asset['size']})，正在删除旧资产...")
                del_r = client.delete(
                    f"https://api.github.com/repos/{REPO}/releases/assets/{asset['id']}", headers=headers
                )
                if del_r.status_code == 204:
                    print("旧资产已删除。")
                else:
                    print(f"删除旧资产返回: {del_r.status_code} - {del_r.text}")

        # 3. 上传新资产
        upload_url = f"https://uploads.github.com/repos/{REPO}/releases/{release_id}/assets?name={ASSET_NAME}"
        upload_headers = {
            **headers,
            "Content-Type": "application/octet-stream",
            "Content-Length": str(file_size),
        }

        print(f"开始向 {upload_url} 上传...")
        t_start = time.time()
        stream = chunk_generator(ASSET_PATH)
        resp = client.post(upload_url, headers=upload_headers, content=stream)

        if resp.status_code == 201:
            data = resp.json()
            total_time = time.time() - t_start
            print(f"\n上传成功！用时 {total_time:.1f} 秒")
            print(f"资产下载地址: {data.get('browser_download_url')}")
            print(f"Release 页面: {release['html_url']}")
        else:
            print(f"\n上传失败: HTTP {resp.status_code} - {resp.text}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
