"""Prompt-content regression tests for the P1-P5 synthesis improvements.

These lock in the text of specific instructions in the α / β / Leader
system prompts so future edits don't silently drop them. Real LLM
behavior is out of scope — we only verify the instructions are *present*
and unambiguous.

P1: 核心要素 に出典必須化 (Leader)
P2: 矢印圧縮禁止 (Leader)
P3: 章あたり最低 5 トピック (β)
P4: MOC に概念別索引テーブル (Leader)
P5: 学習順序は時間別コース (Leader)

Plus residual-miss policy (Leader) and the legacy "γ" label cleanup.
"""

from __future__ import annotations

from pipeline_youtube.synthesis.agents import (
    BETA_SYSTEM_PROMPT,
    LEADER_SYSTEM_PROMPT,
)


class TestP1InlineCitations:
    def test_core_elements_require_inline_citations(self):
        """Leader prompt must force `(出典: [[...]])` on every 核心要素 item."""
        assert "核心要素" in LEADER_SYSTEM_PROMPT
        assert "各項目末尾" in LEADER_SYSTEM_PROMPT
        assert "出典: [[<動画 note 名>#^MM-SS]]" in LEADER_SYSTEM_PROMPT


class TestP2ArrowExpansion:
    def test_arrow_chains_must_be_expanded(self):
        """Leader prompt must forbid `A→B→C→D` style step compression."""
        assert "工程列挙の展開" in LEADER_SYSTEM_PROMPT
        assert "矢印" in LEADER_SYSTEM_PROMPT
        assert "3 ステップ以上" in LEADER_SYSTEM_PROMPT
        assert "独立した箇条書き" in LEADER_SYSTEM_PROMPT


class TestP3MinTopicsPerChapter:
    def test_beta_requires_five_topics_per_chapter(self):
        """β must refuse to emit chapters with < 5 topics, even for `unique`."""
        assert "5 トピック" in BETA_SYSTEM_PROMPT
        assert "unique 章でも 5 以上" in BETA_SYSTEM_PROMPT


class TestP4ConceptIndexInMoc:
    def test_moc_must_include_concept_index_table(self):
        """MOC gets a `## 概念別索引` table so readers can cross-look topics."""
        assert "## 概念別索引" in LEADER_SYSTEM_PROMPT
        assert "| 概念 | 章 |" in LEADER_SYSTEM_PROMPT


class TestP5LearningPaths:
    def test_learning_section_lists_three_courses(self):
        """`## 学習順序の推奨` must split into 3 reader-intent courses."""
        assert "全章通読コース" in LEADER_SYSTEM_PROMPT
        assert "30 分で要点把握コース" in LEADER_SYSTEM_PROMPT
        assert "深掘りコース" in LEADER_SYSTEM_PROMPT


class TestResidualMissPolicy:
    """Leader must not silently drop missing_topic_ids after β retry exhaustion."""

    def test_residual_miss_policy_section_exists(self):
        assert "残存ミス補完ポリシー" in LEADER_SYSTEM_PROMPT

    def test_policy_gated_on_missing_topic_ids_nonempty(self):
        """Explicitly: no-op when missing_topic_ids is empty."""
        assert "missing_topic_ids` が空でない場合のみ適用" in LEADER_SYSTEM_PROMPT
        assert "missing_topic_ids` が空の場合、このポリシーは一切適用しない" in LEADER_SYSTEM_PROMPT

    def test_policy_does_not_mutate_chapter_structure(self):
        """Leader must not rearrange β's chapters when filling in residual misses."""
        assert "章構成" in LEADER_SYSTEM_PROMPT
        assert "変更しない" in LEADER_SYSTEM_PROMPT

    def test_policy_appends_as_trailing_section(self):
        """Residual misses land in `### 補足` at chapter tail."""
        assert "### 補足" in LEADER_SYSTEM_PROMPT


class TestLegacyGammaLabelRemoved:
    """After γ removal (PR #10), no prompt should still call coverage 'γ'."""

    def test_no_gamma_coverage_report_label(self):
        import inspect

        from pipeline_youtube.synthesis import agents as agents_mod

        source = inspect.getsource(agents_mod.call_leader)
        assert "## γ coverage report" not in source


class TestExistingConstraintsIntact:
    """Guard against regression of earlier prompt guarantees."""

    def test_leader_still_forbids_hallucinated_images(self):
        assert "新規ファイル名創作禁止" in LEADER_SYSTEM_PROMPT

    def test_leader_still_requires_json_only(self):
        assert "JSON 単体" in LEADER_SYSTEM_PROMPT

    def test_beta_still_limits_title_filename_chars(self):
        for ch in ("\\", "/", ":", "*", "?", '"', "<", ">", "|"):
            assert ch in BETA_SYSTEM_PROMPT
