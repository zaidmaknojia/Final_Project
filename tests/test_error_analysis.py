from pubmedqa_abst.data import Item
from pubmedqa_abst.error_analysis import categorize, confident_errors, features, sample_examples
from pubmedqa_abst.scores import ItemScore


def it(iid, q="Is X helpful?"):
    return Item(iid, q, ("c",), "yes")


def sc(iid, gold, a0, sv, scn):
    return ItemScore(iid, "E+", gold, a0, a0 == gold, sv, scn, a0 is not None, 0)


def test_features_regexes():
    assert features(it("1", "Is X not associated with Y?"))["negation"]
    assert features(it("1", "Is A better than B?"))["comparative"]
    assert features(it("1", "Does 5 mg help?"))["numeric"]
    f = features(it("1", "Is surgery helpful?"))
    assert not (f["negation"] or f["comparative"] or f["numeric"])


def test_confident_errors_only_wrong_and_accepted():
    scores = [sc("a", "yes", "no", .9, .2), sc("b", "yes", "yes", .9, 1.0),
              sc("c", "no", "yes", .5, 1.0), sc("d", "maybe", "yes", .95, 1.0)]
    items = {k: it(k) for k in "abcd"}
    rows = confident_errors(scores, items, {"s_verb": .8, "s_cons": .9}, "s_verb")
    assert [r["item_id"] for r in rows] == ["a", "d"]
    by_id = {r["item_id"]: r for r in rows}
    assert by_id["a"]["confusion"] == "yes->no" and by_id["a"]["scores_agree"] is False
    assert by_id["d"]["scores_agree"] is True


def test_confident_errors_none_threshold_accepts_nothing():
    rows = confident_errors([sc("a", "yes", "no", .9, .9)], {"a": it("a")},
                            {"s_verb": None, "s_cons": .5}, "s_verb")
    assert rows == []


def test_categorize_fields():
    row = categorize(sc("a", "maybe", "yes", .9, .9), it("a", "Is A less than B?"), True)
    assert row["confusion"] == "maybe->yes" and row["condition"] == "E+" and row["comparative"]


def test_sample_examples_seeded_and_capped():
    rows = [{"confusion": "yes->no", "item_id": str(i)} for i in range(10)] + \
           [{"confusion": "maybe->yes", "item_id": "m"}]
    a = sample_examples(rows, 3, seed=5)
    assert a == sample_examples(rows, 3, seed=5)
    assert sum(r["confusion"] == "yes->no" for r in a) == 3 and any(r["item_id"] == "m" for r in a)
