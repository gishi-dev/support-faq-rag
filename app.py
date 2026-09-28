"""製品サポートAIチャット（デモ）の画面。

起動: streamlit run app.py
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import anthropic
import streamlit as st
from dotenv import load_dotenv

from supportrag.answer import Answer, stream_answer
from supportrag.loader import fingerprint, load_chunks
from supportrag.logbook import STATUSES, Logbook
from supportrag.search import HybridSearcher

load_dotenv()
ROOT = Path(__file__).parent
DOCS_DIR = ROOT / os.getenv("SUPPORT_RAG_DOCS_DIR", "docs")
CACHE_DIR = ROOT / ".cache"
MODEL = os.getenv("SUPPORT_RAG_MODEL", "claude-opus-5")
EFFORT = os.getenv("SUPPORT_RAG_EFFORT", "medium")
EMBED_MODEL = os.getenv("SUPPORT_RAG_EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
HISTORY_TURNS = 3
EXAMPLES = [
    "加湿器にE3と出て止まりました",
    "アロマオイルを入れても大丈夫？",
    "Wi-Fiにつながりません",
    "保証は何年ですか",
    "加湿器を食洗機で洗えますか",
]


@st.cache_resource(show_spinner="資料を読み込んでいます…")
def get_searcher(docs_fingerprint: str) -> HybridSearcher:
    return HybridSearcher(load_chunks(DOCS_DIR), EMBED_MODEL, CACHE_DIR)


@st.cache_resource
def get_logbook() -> Logbook:
    return Logbook(ROOT / "data" / "questions.sqlite3")


def show_answer(turn: dict) -> None:
    st.markdown(turn["markdown"])
    if not turn["answered"]:
        st.warning("資料では答えられなかったため、「答えられなかった質問」に記録しました。", icon="📝")
    if turn["sources"]:
        with st.expander(f"出典（{len(turn['sources'])}件）"):
            for source in turn["sources"]:
                st.markdown(f"**[{source.number}] {source.title}**")
                for cited in source.cited_texts:
                    st.caption(cited)
    with st.expander("検索結果（上位5件）"):
        st.dataframe(
            [
                {
                    "順位": rank,
                    "資料": hit.chunk.title,
                    "意味検索の順位": hit.vector_rank,
                    "キーワード検索の順位": hit.keyword_rank or "該当なし",
                }
                for rank, hit in enumerate(turn["hits"], 1)
            ],
            hide_index=True,
            width="stretch",
        )


def ask(question: str, searcher: HybridSearcher, logbook: Logbook) -> None:
    history = [(t["question"], re.sub(r"\[\d+\]", "", t["markdown"])) for t in st.session_state.turns]
    hits = searcher.search(question)
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        streamed = ""
        answer: Answer | None = None
        try:
            for piece in stream_answer(
                anthropic.Anthropic(),
                model=MODEL,
                effort=EFFORT,
                history=history[-HISTORY_TURNS:],
                question=question,
                hits=hits,
            ):
                if isinstance(piece, Answer):
                    answer = piece
                else:
                    streamed += piece
                    placeholder.markdown(streamed + "▌")
        except anthropic.AuthenticationError:
            placeholder.error("APIキーが正しくありません。.env の ANTHROPIC_API_KEY を確認してください。")
            return
        except anthropic.RateLimitError:
            placeholder.error("混み合っています。少し時間をおいてから、もう一度お試しください。")
            return
        except anthropic.APIConnectionError:
            placeholder.error("Claude APIに接続できませんでした。ネットワークを確認してください。")
            return
        except anthropic.APIStatusError as error:
            placeholder.error(f"Claude APIでエラーが発生しました（{error.status_code}）。")
            return
        placeholder.empty()
        turn = {
            "question": question,
            "markdown": answer.markdown,
            "sources": answer.sources,
            "answered": answer.answered,
            "hits": hits,
        }
        show_answer(turn)
    st.session_state.turns.append(turn)
    logbook.record(question, answer.answered, [s.title for s in answer.sources], answer.model)


def chat_tab(searcher: HybridSearcher, logbook: Logbook) -> None:
    # タブ内の入力欄はその場に描画されるため、会話は入力欄より上の枠へ書く。
    conversation = st.container()
    with conversation:
        for turn in st.session_state.turns:
            with st.chat_message("user"):
                st.markdown(turn["question"])
            with st.chat_message("assistant"):
                show_answer(turn)

    has_key = bool(os.getenv("ANTHROPIC_API_KEY"))
    if not has_key:
        st.info("回答を作るには、.env に ANTHROPIC_API_KEY を設定してください（.env.example を参照）。")
    question = st.chat_input("製品について質問してください", disabled=not has_key)
    question = question or st.session_state.pop("example", None)
    if question and has_key:
        with conversation:
            ask(question, searcher, logbook)


def unanswered_tab(logbook: Logbook) -> None:
    stats = logbook.stats()
    rate = f"{stats['answered'] / stats['total']:.0%}" if stats["total"] else "―"
    left, middle, right = st.columns(3)
    left.metric("質問の数", stats["total"])
    middle.metric("資料で答えられた割合", rate)
    right.metric("未対応の質問", stats["open"])
    st.caption("資料で答えられなかった質問です。FAQや資料へ追加する候補として確認します。")

    rows = logbook.unanswered()
    if not rows:
        st.write("まだありません。")
    for row in rows:
        with st.container(border=True):
            st.markdown(f"**{row['question']}**")
            st.caption(f"{row['asked_at'].replace('T', ' ')}　状態：{STATUSES[row['status']]}")
            with st.form(f"status-{row['id']}"):
                status = st.selectbox(
                    "状態",
                    list(STATUSES),
                    index=list(STATUSES).index(row["status"]),
                    format_func=STATUSES.get,
                )
                note = st.text_input("メモ", value=row["note"], placeholder="例：保証規定に追記した")
                if st.form_submit_button("保存"):
                    logbook.set_status(row["id"], status, note)
                    st.rerun()


def docs_tab(searcher: HybridSearcher) -> None:
    st.caption(f"{DOCS_DIR.name}/ にMarkdownを置くと、検索の対象になります。見出し（##）ごとに区切って検索します。")
    by_doc: dict[str, list] = {}
    for chunk in searcher.chunks:
        by_doc.setdefault(chunk.doc_title, []).append(chunk)
    for doc_title, chunks in by_doc.items():
        with st.expander(f"{doc_title}（{len(chunks)}項目）"):
            for chunk in chunks:
                st.markdown(f"**{chunk.heading}**")
                st.markdown(chunk.text)


def main() -> None:
    st.set_page_config(page_title="ギシ電機 サポートAI（デモ）", page_icon="💬", layout="wide")
    st.session_state.setdefault("turns", [])
    searcher = get_searcher(fingerprint(load_chunks(DOCS_DIR)))
    logbook = get_logbook()

    with st.sidebar:
        st.title("ギシ電機 サポートAI")
        st.caption("架空の家電メーカーの製品資料で動くデモです。実在の会社・製品とは関係ありません。")
        st.subheader("質問の例")
        for example in EXAMPLES:
            if st.button(example, width="stretch"):
                st.session_state.example = example
        st.divider()
        if st.button("会話をリセット", width="stretch"):
            st.session_state.turns = []
            st.rerun()
        st.caption(f"回答モデル：{MODEL}（effort: {EFFORT}）")
        st.caption(f"埋め込みモデル：{EMBED_MODEL.split('/')[-1]}")

    chat, unanswered, docs = st.tabs(["チャット", "答えられなかった質問", "資料一覧"])
    with chat:
        chat_tab(searcher, logbook)
    with unanswered:
        unanswered_tab(logbook)
    with docs:
        docs_tab(searcher)


main()
