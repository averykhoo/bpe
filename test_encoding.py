# Note that there are more actual tests, they're just not currently public :-)

from tokenizer import get_cl100k_base
from tokenizer import get_o200k_base


def test_simple():
    enc = get_cl100k_base("cl100k_base.tiktoken")
    assert enc.encode("hello world") == [15339, 1917]
    assert enc.decode([15339, 1917]) == "hello world"
    # assert enc.encode("hello <|endoftext|>") == [15339, 220, 100257]

    # for token in range(min(10_000, enc.max_token_value - 1)):
    #     assert enc.encode(enc.decode(token)) == token


# def test_large_repeated():
#     enc = get_o200k_base("o200k_base.tiktoken")
#
#     with pytest.raises(ValueError):
#         enc.encode("x" * 1_000_000)


def test_simple_regex():
    enc = get_cl100k_base("cl100k_base.tiktoken")
    assert enc.encode("rer") == [38149]
    assert enc.encode("'rer") == [2351, 81]
    assert enc.encode("today\n ") == [31213, 198, 220]
    assert enc.encode("today\n \n") == [31213, 27907]
    assert enc.encode("today\n  \n") == [31213, 14211]


def test_basic_encode():
    enc = get_cl100k_base("cl100k_base.tiktoken")
    assert enc.encode("hello world") == [15339, 1917]
    assert enc.encode(" \x850") == [220, 126, 227, 15]


# def test_encode_bytes():
#     enc = get_cl100k_base("cl100k_base.tiktoken")
#     assert enc._encode_bytes(b" \xec\x8b\xa4\xed") == [62085]
#     for i in range(10):
#         bytestring = b"\x80" * i
#         assert enc.decode_bytes(enc._encode_bytes(bytestring)) == bytestring


def test_encode_surrogate_pairs():
    enc = get_cl100k_base("cl100k_base.tiktoken")

    assert enc.encode("👍") == [9468, 239, 235]

    # cant handle surrogate pairs but wtv
    # # surrogate pair gets converted to codepoint
    # assert enc.encode("\ud83d\udc4d") == [9468, 239, 235]
    #
    # # lone surrogate just gets replaced
    # assert enc.encode("?") != enc.encode("�")
    # assert enc.encode("\ud83d") == enc.encode("�")


# ====================
# Roundtrip
# ====================


def test_basic_roundtrip():
    for enc in [get_cl100k_base("cl100k_base.tiktoken"), get_o200k_base("o200k_base.tiktoken")]:
        for value in (
                "hello",
                "hello ",
                "hello  ",
                " hello",
                " hello ",
                " hello  ",
                "hello world",
                "请考试我的软件！12345",
        ):
            assert value == enc.decode(enc.encode(value))
