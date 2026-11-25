import pytest

from pubmedqa_abst.parse import parse_output


@pytest.mark.parametrize(
    "text,exp",
    [
        ('{"answer":"yes","confidence":85}', ("yes", 85, "ok")),
        ('{"answer":"MAYBE","confidence":0}', ("maybe", 0, "ok")),
        ('{"answer":"no","confidence":85.0}', ("no", 85, "ok")),
        ('  {"answer":"no","confidence":100}\n', ("no", 100, "ok")),
        ('{"answer":"no","confidence":"85"}', ("no", None, "invalid_confidence")),
        ('{"answer":"no","confidence":true}', ("no", None, "invalid_confidence")),
        ('{"answer":"no","confidence":101}', ("no", None, "invalid_confidence")),
        ('{"answer":"no","confidence":-1}', ("no", None, "invalid_confidence")),
        ('{"answer":"no","confidence":85.5}', ("no", None, "invalid_confidence")),
        ('{"answer":"no","confidence":null}', ("no", None, "invalid_confidence")),
        ('{"answer":"no"}', ("no", None, "invalid_confidence")),
        ('{"answer":"probably","confidence":80}', (None, None, "invalid_format")),
        ('{"answer":1,"confidence":80}', (None, None, "invalid_format")),
        ('{"confidence":80}', (None, None, "invalid_format")),
        ('The answer is yes {"answer":"yes","confidence":80}', (None, None, "invalid_format")),
        ("", (None, None, "invalid_format")),
        ("[1,2]", (None, None, "invalid_format")),
    ],
)
def test_parse(text, exp):
    p = parse_output(text)
    assert (p.answer, p.confidence, p.status) == exp


def test_parse_none_text_is_invalid_format():
    assert parse_output(None).status == "invalid_format"
