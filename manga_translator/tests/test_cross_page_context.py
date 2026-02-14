"""Tests for CrossPageContext — cross-page translation consistency."""


from manga_translator.cross_page_context import CrossPageContext


class TestCrossPageContextDefaults:
    def test_init_defaults(self):
        ctx = CrossPageContext()
        assert ctx.dialogue_history == []
        assert ctx.character_names == {}
        assert ctx.glossary == {}
        assert ctx.pages_processed == 0
        assert ctx.page_window == 2
        assert ctx.max_dialogue_lines == 10


class TestDialogueHistory:
    def test_update_from_page(self):
        ctx = CrossPageContext()
        ctx.update_from_page(0, ["こんにちは"], ["Hello"])
        assert ctx.pages_processed == 1
        assert len(ctx.dialogue_history) == 1
        assert ctx.dialogue_history[0].page_num == 0
        assert ctx.dialogue_history[0].source_texts == ["こんにちは"]
        assert ctx.dialogue_history[0].translated_texts == ["Hello"]

    def test_page_window_limit(self):
        ctx = CrossPageContext(page_window=2)
        ctx.update_from_page(0, ["text0"], ["trans0"])
        ctx.update_from_page(1, ["text1"], ["trans1"])
        ctx.update_from_page(2, ["text2"], ["trans2"])
        # Only last 2 pages kept
        assert len(ctx.dialogue_history) == 2
        assert ctx.dialogue_history[0].page_num == 1
        assert ctx.dialogue_history[1].page_num == 2
        # But pages_processed tracks all
        assert ctx.pages_processed == 3

    def test_get_dialogue_summary_empty(self):
        ctx = CrossPageContext()
        assert ctx.get_dialogue_summary() == ""

    def test_get_dialogue_summary(self):
        ctx = CrossPageContext()
        ctx.update_from_page(0, ["こんにちは", "元気？"], ["Hello", "How are you?"])
        summary = ctx.get_dialogue_summary()
        assert "Previous dialogue:" in summary
        assert "こんにちは -> Hello" in summary
        assert "元気？ -> How are you?" in summary

    def test_get_dialogue_summary_max_lines(self):
        ctx = CrossPageContext()
        texts_src = [f"text_{i}" for i in range(20)]
        texts_tgt = [f"trans_{i}" for i in range(20)]
        ctx.update_from_page(0, texts_src[:10], texts_tgt[:10])
        ctx.update_from_page(1, texts_src[10:], texts_tgt[10:])
        summary = ctx.get_dialogue_summary(max_lines=5)
        # Only last 5 lines
        lines = [line for line in summary.split("\n") if "->" in line]
        assert len(lines) == 5


class TestCharacterNames:
    def test_add_and_get(self):
        ctx = CrossPageContext()
        ctx.add_character_name("太郎", "Taro")
        ctx.add_character_name("花子", "Hanako")
        names = ctx.get_character_map()
        assert names == {"太郎": "Taro", "花子": "Hanako"}

    def test_detect_names_from_translations(self):
        ctx = CrossPageContext()
        translations = [
            "Hey Sakura Chan, let's go!",
            "Naruto Uzumaki is here.",
            "The weather is nice today.",
        ]
        names = ctx.detect_names_from_translations(translations)
        assert "Sakura Chan" in names
        assert "Naruto Uzumaki" in names

    def test_detect_names_ignores_common_words(self):
        ctx = CrossPageContext()
        translations = ["The cat sat on the mat."]
        names = ctx.detect_names_from_translations(translations)
        assert names == []

    def test_name_consistency_no_issues(self):
        ctx = CrossPageContext()
        ctx.add_character_name("太郎", "Taro Yamada")
        warnings = ctx.check_name_consistency(["Taro Yamada said hello."])
        assert warnings == []

    def test_name_consistency_flags_mismatch(self):
        ctx = CrossPageContext()
        ctx.character_names["Sakura Haruno"] = "Sakura Haruno"
        warnings = ctx.check_name_consistency(["Sakura Chan is here."])
        assert len(warnings) == 1
        assert "Sakura" in warnings[0]


class TestGlossary:
    def test_add_and_get(self):
        ctx = CrossPageContext()
        ctx.add_glossary_term("忍術", "Ninjutsu")
        ctx.add_glossary_term("写輪眼", "Sharingan")
        glossary = ctx.get_glossary()
        assert glossary == {"忍術": "Ninjutsu", "写輪眼": "Sharingan"}

    def test_glossary_persists_across_pages(self):
        ctx = CrossPageContext()
        ctx.add_glossary_term("忍術", "Ninjutsu")
        ctx.update_from_page(0, ["text"], ["trans"])
        ctx.update_from_page(1, ["text2"], ["trans2"])
        assert ctx.get_glossary() == {"忍術": "Ninjutsu"}
