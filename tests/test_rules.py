from hybridmod.config import RulesCfg
from hybridmod.rules import RuleEngine, hash_token, normalise_token


def test_normalise_folds_leetspeak_and_repeats():
    assert normalise_token("B4DW0RD") == "badword"
    assert normalise_token("baaaadword") == "baadword"
    assert normalise_token("b@dw*rd") == "badwrd"


def test_lexicon_block(rules):
    r = rules.evaluate("you are a badword")
    assert r.action == "block" and r.rule == "lexicon"


def test_lexicon_catches_obfuscation_and_plural(rules):
    assert rules.evaluate("total B4DW0RD").action == "block"
    assert rules.evaluate("all of you badwords").action == "block"


def test_lexicon_no_substring_false_positive(rules):
    # a token that merely contains the lexicon word is not blocked
    assert rules.evaluate("notabadwordatall here").action == "pass"


def test_emoji_only_allowed(rules):
    assert rules.evaluate("😂😂👍").action == "allow"
    assert rules.evaluate("   ").rule == "empty"


def test_short_appreciation_allowed(rules):
    r = rules.evaluate("Thank you!")
    assert r.action == "allow" and r.rule == "short_benign"
    assert rules.evaluate("Great article, thanks").action == "allow"


def test_long_text_with_thanks_not_auto_allowed(rules):
    assert rules.evaluate("thanks for nothing you clueless people, honestly").action == "pass"


def test_spam_links_promo_and_repetition(rules):
    assert rules.evaluate("see http://a.com http://b.com www.c.com").action == "spam"
    assert rules.evaluate("Earn $500 per day, click here").rule == "promo_phrase"
    assert rules.evaluate(" ".join(["buy"] * 12)).rule == "repetition"


def test_single_link_is_fine(rules):
    assert rules.evaluate("source: https://example.org/report").action == "pass"


def test_rules_disabled_pass_everything():
    eng = RuleEngine(RulesCfg(enabled=False), {hash_token("badword", "hybridmod-v1")})
    assert eng.evaluate("badword").action == "pass"


def test_lexicon_file_loading(tmp_path):
    cfg = RulesCfg()
    p = tmp_path / "lex.txt"
    p.write_text("# comment\n" + hash_token("badword", cfg.lexicon_salt) + "\n")
    eng = RuleEngine.from_file(cfg, p)
    assert eng.evaluate("badword").action == "block"
    assert "badword" not in p.read_text()  # only hashes on disk
