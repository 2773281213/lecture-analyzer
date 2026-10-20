"""书库的表结构与逐版本迁移：每个版本一段固定 SQL（不拼接），在一个事务里执行。

v1（M1）：课程、讲、转写、课件页、笔记、考点信号、模型调用记录。
v2（M2）：一讲可有多个文件（节）、讲的标签、课程提示词与书架顺序、课程记忆（L2）、总复习。
v3（M3）：授课语言通用化（现场翻译单独一列）、课件选择、课程资料库（课件、教材、试题……）、
          每讲对应的课件页（取代 v1 每讲一份课件的 slides_path）、笔记所用课件页的快照、考前报告、后台任务。
v4（M4）：问答（按课程的连续对话，可以限定某一讲）；考点信号的原话附笔记语言的意思（外语授课的课程）。
"""

from __future__ import annotations

import sqlite3

SCHEMA_VERSION = 4

_V1 = """
BEGIN;
CREATE TABLE IF NOT EXISTS course (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    subject TEXT NOT NULL,
    teaching TEXT NOT NULL,
    note_language TEXT NOT NULL,
    asr TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS lecture (
    id INTEGER PRIMARY KEY,
    course_id INTEGER NOT NULL REFERENCES course(id) ON DELETE CASCADE,
    seq INTEGER NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    date TEXT NOT NULL DEFAULT '',
    kind TEXT NOT NULL DEFAULT 'normal',
    media_path TEXT NOT NULL,
    slides_path TEXT,
    duration_ms INTEGER,
    status TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    UNIQUE (course_id, seq)
);
CREATE TABLE IF NOT EXISTS segment (
    lecture_id INTEGER NOT NULL REFERENCES lecture(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    start_ms INTEGER NOT NULL,
    end_ms INTEGER NOT NULL,
    text TEXT NOT NULL,
    lang TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (lecture_id, idx)
);
CREATE TABLE IF NOT EXISTS doc_page (
    owner_type TEXT NOT NULL,
    owner_id INTEGER NOT NULL,
    page_no INTEGER NOT NULL,
    title TEXT NOT NULL,
    content_md TEXT NOT NULL,
    terms TEXT NOT NULL,
    PRIMARY KEY (owner_type, owner_id, page_no)
);
CREATE TABLE IF NOT EXISTS lecture_note (
    lecture_id INTEGER PRIMARY KEY REFERENCES lecture(id) ON DELETE CASCADE,
    notes_json TEXT NOT NULL,
    model_profile TEXT NOT NULL,
    model TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS signal (
    id INTEGER PRIMARY KEY,
    lecture_id INTEGER NOT NULL REFERENCES lecture(id) ON DELETE CASCADE,
    type TEXT NOT NULL,
    knowledge_point TEXT NOT NULL,
    strength INTEGER NOT NULL,
    quote TEXT NOT NULL,
    segment_ids TEXT NOT NULL,
    slide_pages TEXT NOT NULL,
    valid INTEGER NOT NULL,
    match_method TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS llm_call (
    id INTEGER PRIMARY KEY,
    lecture_id INTEGER,
    step TEXT NOT NULL,
    model_profile TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cache_read_tokens INTEGER NOT NULL,
    cost_usd REAL,
    created_at TEXT NOT NULL
);
PRAGMA user_version = 1;
COMMIT;
"""

_V2 = """
BEGIN;
CREATE TABLE lecture_part (
    lecture_id INTEGER NOT NULL REFERENCES lecture(id) ON DELETE CASCADE,
    part INTEGER NOT NULL,
    media_path TEXT NOT NULL,
    duration_ms INTEGER,
    PRIMARY KEY (lecture_id, part)
);
INSERT INTO lecture_part (lecture_id, part, media_path, duration_ms)
    SELECT id, 1, media_path, duration_ms FROM lecture;
ALTER TABLE lecture DROP COLUMN media_path;
ALTER TABLE llm_call ADD COLUMN course_id INTEGER;
ALTER TABLE course ADD COLUMN instructions TEXT NOT NULL DEFAULT '';
ALTER TABLE course ADD COLUMN position INTEGER NOT NULL DEFAULT 0;
UPDATE course SET position = id;
CREATE TABLE lecture_tag (
    lecture_id INTEGER NOT NULL REFERENCES lecture(id) ON DELETE CASCADE,
    tag TEXT NOT NULL,
    PRIMARY KEY (lecture_id, tag)
);
CREATE TABLE knowledge_point (
    id INTEGER PRIMARY KEY,
    course_id INTEGER NOT NULL REFERENCES course(id) ON DELETE CASCADE,
    key INTEGER NOT NULL,
    name TEXT NOT NULL,
    chapter TEXT NOT NULL DEFAULT '',
    aliases TEXT NOT NULL DEFAULT '[]',
    UNIQUE (course_id, key)
);
CREATE TABLE kp_lecture (
    kp_id INTEGER NOT NULL REFERENCES knowledge_point(id) ON DELETE CASCADE,
    lecture_id INTEGER NOT NULL REFERENCES lecture(id) ON DELETE CASCADE,
    note_index INTEGER NOT NULL,
    role TEXT NOT NULL,
    minutes REAL NOT NULL,
    PRIMARY KEY (lecture_id, note_index)
);
CREATE INDEX kp_lecture_kp ON kp_lecture (kp_id);
CREATE TABLE kp_signal (
    signal_id INTEGER PRIMARY KEY REFERENCES signal(id) ON DELETE CASCADE,
    kp_id INTEGER NOT NULL REFERENCES knowledge_point(id) ON DELETE CASCADE
);
CREATE TABLE kp_reference (
    id INTEGER PRIMARY KEY,
    lecture_id INTEGER NOT NULL REFERENCES lecture(id) ON DELETE CASCADE,
    note_index INTEGER NOT NULL,
    kp_id INTEGER NOT NULL REFERENCES knowledge_point(id) ON DELETE CASCADE,
    target_lecture_id INTEGER NOT NULL REFERENCES lecture(id) ON DELETE CASCADE,
    relation TEXT NOT NULL,
    explanation TEXT NOT NULL,
    segment_ids TEXT NOT NULL
);
CREATE INDEX kp_reference_lecture ON kp_reference (lecture_id);
CREATE TABLE integration_log (
    lecture_id INTEGER PRIMARY KEY REFERENCES lecture(id) ON DELETE CASCADE,
    result_json TEXT NOT NULL,
    model_profile TEXT NOT NULL,
    model TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE review_chapter (
    course_id INTEGER NOT NULL REFERENCES course(id) ON DELETE CASCADE,
    chapter TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    result_json TEXT NOT NULL,
    model_profile TEXT NOT NULL,
    model TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (course_id, chapter)
);
CREATE TABLE kp_review (
    course_id INTEGER NOT NULL REFERENCES course(id) ON DELETE CASCADE,
    name_norm TEXT NOT NULL,
    reviewed_at TEXT NOT NULL,
    PRIMARY KEY (course_id, name_norm)
);
PRAGMA user_version = 2;
COMMIT;
"""

_V3 = """
BEGIN;
ALTER TABLE course ADD COLUMN interpreted_to TEXT NOT NULL DEFAULT '';
UPDATE course SET interpreted_to = substr(teaching, instr(teaching, '+') + 1),
                  teaching = substr(teaching, 1, instr(teaching, '+') - 1)
    WHERE instr(teaching, '+') > 0;
ALTER TABLE course ADD COLUMN slides TEXT NOT NULL DEFAULT 'unset';
ALTER TABLE lecture DROP COLUMN slides_path;
ALTER TABLE lecture_note ADD COLUMN pages TEXT NOT NULL DEFAULT '[]';
CREATE TABLE material (
    id INTEGER PRIMARY KEY,
    course_id INTEGER NOT NULL REFERENCES course(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    title TEXT NOT NULL,
    path TEXT NOT NULL,
    source TEXT NOT NULL,
    position INTEGER NOT NULL,
    page_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE INDEX material_course ON material (course_id);
CREATE TABLE material_item (
    id INTEGER PRIMARY KEY,
    material_id INTEGER NOT NULL REFERENCES material(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    label TEXT NOT NULL DEFAULT '',
    item_type TEXT NOT NULL,
    content_md TEXT NOT NULL,
    score REAL,
    answer_md TEXT NOT NULL DEFAULT '',
    pages TEXT NOT NULL DEFAULT '[]',
    UNIQUE (material_id, idx)
);
CREATE TABLE item_kp (
    item_id INTEGER NOT NULL REFERENCES material_item(id) ON DELETE CASCADE,
    kp_key INTEGER NOT NULL,
    PRIMARY KEY (item_id, kp_key)
);
CREATE TABLE lecture_page (
    lecture_id INTEGER NOT NULL REFERENCES lecture(id) ON DELETE CASCADE,
    material_id INTEGER NOT NULL REFERENCES material(id) ON DELETE CASCADE,
    page_no INTEGER NOT NULL,
    source TEXT NOT NULL,
    PRIMARY KEY (lecture_id, material_id, page_no)
);
CREATE TABLE report (
    id INTEGER PRIMARY KEY,
    course_id INTEGER NOT NULL REFERENCES course(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    params TEXT NOT NULL,
    result_json TEXT NOT NULL,
    models TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE job (
    id INTEGER PRIMARY KEY,
    course_id INTEGER REFERENCES course(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    params TEXT NOT NULL,
    state TEXT NOT NULL,
    progress TEXT NOT NULL DEFAULT '',
    result TEXT NOT NULL DEFAULT '',
    error TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
PRAGMA user_version = 3;
COMMIT;
"""

_V4 = """
BEGIN;
CREATE TABLE qa (
    id INTEGER PRIMARY KEY,
    course_id INTEGER NOT NULL REFERENCES course(id) ON DELETE CASCADE,
    lecture_id INTEGER REFERENCES lecture(id) ON DELETE SET NULL,
    thread INTEGER NOT NULL,
    question TEXT NOT NULL,
    result_json TEXT NOT NULL,
    models TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX qa_course_thread ON qa (course_id, thread);
ALTER TABLE signal ADD COLUMN meaning TEXT NOT NULL DEFAULT '';
PRAGMA user_version = 4;
COMMIT;
"""

MIGRATIONS = {1: _V1, 2: _V2, 3: _V3, 4: _V4}


def migrate(conn: sqlite3.Connection, backup_path_for: dict[int, str] | None = None) -> int:
    """把数据库升级到最新版本，返回升级前的版本号。

    backup_path_for：{旧版本: 备份文件路径}，从这些版本升级前先做一致性备份。
    每个版本的 SQL 自带 BEGIN/COMMIT；出错时回滚并抛出，数据库停在上一个完整版本。
    """
    start = conn.execute("PRAGMA user_version").fetchone()[0]
    backups = backup_path_for or {}
    if start in backups and start < SCHEMA_VERSION:
        target = sqlite3.connect(backups[start])
        try:
            conn.backup(target)
        finally:
            target.close()  # sqlite3 的 with 只提交不关闭，需显式关闭
    for version in range(start + 1, SCHEMA_VERSION + 1):
        try:
            conn.executescript(MIGRATIONS[version])
        except sqlite3.Error:
            if conn.in_transaction:
                conn.rollback()
            raise
    return start
