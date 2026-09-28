"""検索結果を根拠に、Claude APIで回答を作る。

検索結果は search_result ブロックとして渡し、APIの引用機能で根拠を返させる。
根拠（引用）が1つも付かなかった回答は「資料では答えられなかった」ものとして扱う。
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import anthropic

from .search import Hit

SYSTEM_PROMPT = """\
あなたは、家電メーカー「ソヨカ電機」のカスタマーサポート担当です。
お客様からの質問に、検索結果として渡される自社の製品資料だけを根拠に、日本語で答えてください。

- 資料に書かれていることだけを答える。資料にない仕様・料金・日数・対応可否を推測で補わない。
- 資料で答えられない質問には「その点についてはご案内できる情報がありません」と短く伝え、サポート窓口（電話：平日10時〜18時、0120-000-000）を案内する。このときは資料を引用せず、推測による注意や理由も付け足さない。
- 「資料」「検索結果」など、回答の裏側の仕組みには触れない。
- 発熱・異臭・煙・繰り返すエラーなど安全に関わる内容では、まず使用を中止して電源プラグを抜くよう伝える。
- 結論を先に書き、手順があるときだけ番号付きの手順にする。3〜6文程度にまとめる。太字などの強調記号は使わない。
- 聞かれたことに答える。関連する別の注意事項は、質問に直接関わる場合だけ添える。
- 製品が特定できず答えが製品ごとに異なる場合は、両方の製品について答えるか、どちらの製品かを尋ねる。
"""

FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass
class Source:
    number: int
    title: str
    source: str
    cited_texts: list[str] = field(default_factory=list)


@dataclass
class Answer:
    markdown: str
    sources: list[Source]
    answered: bool
    model: str
    stop_reason: str | None


def _search_result_blocks(hits: list[Hit]) -> list[dict]:
    return [
        {
            "type": "search_result",
            "source": f"docs/{hit.chunk.path}#{hit.chunk.heading}",
            "title": hit.chunk.title,
            "content": [{"type": "text", "text": hit.chunk.text}],
            "citations": {"enabled": True},
        }
        for hit in hits
    ]


def build_messages(history: list[tuple[str, str]], question: str, hits: list[Hit]) -> list[dict]:
    """過去のやり取りは本文だけを渡し、検索結果は今回の質問にだけ付ける。"""
    messages: list[dict] = []
    for user_text, assistant_text in history:
        messages.append({"role": "user", "content": user_text})
        messages.append({"role": "assistant", "content": assistant_text})
    messages.append(
        {"role": "user", "content": [*_search_result_blocks(hits), {"type": "text", "text": question}]}
    )
    return messages


def _render(message) -> tuple[str, list[Source]]:
    """本文に [1] のような番号を付け、番号ごとの出典一覧を作る。"""
    sources: dict[str, Source] = {}
    parts: list[str] = []
    for block in message.content:
        if block.type != "text":
            continue
        parts.append(block.text)
        numbers: list[int] = []
        for citation in block.citations or []:
            if citation.type != "search_result_location":
                continue
            source = sources.setdefault(
                citation.source, Source(len(sources) + 1, citation.title, citation.source)
            )
            if citation.cited_text not in source.cited_texts:
                source.cited_texts.append(citation.cited_text)
            if source.number not in numbers:
                numbers.append(source.number)
        if numbers:
            parts.append("".join(f"[{n}]" for n in numbers))
    # 「。**内部」のように句読点の直後で閉じた ** は、Markdownで太字にならず記号が残る。
    return "".join(parts).replace("**", "").strip(), list(sources.values())


def stream_answer(
    client: anthropic.Anthropic,
    *,
    model: str,
    effort: str,
    history: list[tuple[str, str]],
    question: str,
    hits: list[Hit],
) -> Iterator[str | Answer]:
    """回答の文字列を少しずつ返し、最後に出典付きの Answer を1つ返す。"""
    with client.beta.messages.stream(
        model=model,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=build_messages(history, question, hits),
        output_config={"effort": effort},
        # 安全上の理由で断られた場合に、別モデルで答え直す（サーバー側フォールバック）。
        betas=[FALLBACK_BETA],
        fallbacks="default",
    ) as stream:
        yield from stream.text_stream
        message = stream.get_final_message()

    if message.stop_reason == "refusal":
        text = "この質問にはお答えできません。サポート窓口（平日10時〜18時、0120-000-000）へお問い合わせください。"
        yield Answer(text, [], False, message.model, message.stop_reason)
        return
    markdown, sources = _render(message)
    yield Answer(markdown, sources, bool(sources), message.model, message.stop_reason)
