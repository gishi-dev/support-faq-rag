"""質問と回答結果の記録。資料で答えられなかった質問を、FAQ追加の候補として管理する。"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

STATUSES = {"open": "未対応", "added": "FAQに追加済み", "ignored": "対応しない"}


class Logbook:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with closing(self._connect()) as db, db:
            db.execute(
                """create table if not exists questions (
                    id integer primary key,
                    asked_at text not null,
                    question text not null,
                    answered integer not null,
                    sources text not null,
                    model text not null,
                    status text not null default 'open',
                    note text not null default ''
                )"""
            )

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        return db

    def record(self, question: str, answered: bool, sources: list[str], model: str) -> None:
        with closing(self._connect()) as db, db:
            db.execute(
                "insert into questions (asked_at, question, answered, sources, model) values (?, ?, ?, ?, ?)",
                (
                    datetime.now().isoformat(timespec="seconds"),
                    question,
                    int(answered),
                    json.dumps(sources, ensure_ascii=False),
                    model,
                ),
            )

    def stats(self) -> dict[str, int]:
        with closing(self._connect()) as db:
            row = db.execute(
                "select count(*) as total, coalesce(sum(answered), 0) as answered, "
                "coalesce(sum(case when answered = 0 and status = 'open' then 1 else 0 end), 0) as open "
                "from questions"
            ).fetchone()
        return dict(row)

    def unanswered(self) -> list[sqlite3.Row]:
        with closing(self._connect()) as db:
            return db.execute(
                "select * from questions where answered = 0 order by status = 'open' desc, id desc"
            ).fetchall()

    def set_status(self, question_id: int, status: str, note: str) -> None:
        if status not in STATUSES:
            raise ValueError(status)
        with closing(self._connect()) as db, db:
            db.execute("update questions set status = ?, note = ? where id = ?", (status, note, question_id))
