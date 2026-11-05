"""PDQ test vectors.

``REFERENCE_VECTORS``: hashes of the generated images in ``pdq_corpus.py``,
computed once with Meta's unmodified pure-Python PDQ reference implementation
(ThreatExchange ``pdq/python``, commit :data:`THREATEXCHANGE_COMMIT`), called
as ``PDQHasher().fromImage(...)`` on the RGB array (no resizing). The NumPy
port must reproduce them bit for bit.

``THREATEXCHANGE_VECTORS``: the expected hashes published in ThreatExchange's
``pdq/python/pdqhashing/tests/pdq_test.py`` for images in its ``pdq/data``
directory (``misc-images/b.jpg`` is listed there but missing from the
repository, so it is left out). The images are not committed: the
integration test downloads them at the pinned commit and checks each
file's SHA-256 and size first. Only the hash values (facts, BSD-licensed
test data) are kept here.
"""

from __future__ import annotations

THREATEXCHANGE_COMMIT = "a9a36d4552875c385c4b40f2be847db2b613ed44"
THREATEXCHANGE_DATA_URL = (
    f"https://raw.githubusercontent.com/facebook/ThreatExchange/{THREATEXCHANGE_COMMIT}/pdq/data/"
)

# name -> (hex hash, quality)
REFERENCE_VECTORS: dict[str, tuple[str, int]] = {
    "blocks-64x64": (
        "8726f80d5aaff07d07c039fb39d239d207a5c644e43dce0ea510877266d213df",
        100,
    ),
    "smooth-64x64": (
        "850c684368e3971cb2e2783eeefc353f0c43bd874617434ee7a8c77d10ef4294",
        100,
    ),
    "noise-48x40": (
        "6fc6934712eef016b6650d65a9d48b603919cee9fafcf1fc017da84398890e85",
        100,
    ),
    "shapes-96x80": (
        "28aaaae9d936544d2c9a93594db6364deaabaaadaaac36494d329764a6c969b2",
        57,
    ),
    "smooth-129x100": (
        "04c7f9d039d006affb38d263e220a34bd23477856562bef31f99a2bb24a4b8cd",
        100,
    ),
    "blocks-200x150": (
        "0f138f0cf0174cca07b752f052e4ad4bad0bad4df06933954fea70f3f04cce0b",
        100,
    ),
    "shapes-257x190": (
        "63089cb763089ce7621d8d68761e08e1f73a0ce3631db4a64b51f6b4ad6af39c",
        64,
    ),
    "smooth-300x300": (
        "4f49ab03abc35cfcb0b77c3e5795078414da220d8fc7c55000468fec77b4eae9",
        100,
    ),
    "noise-333x257": (
        "f5755ce423b01de05075d3d1976618dba2013f1b44d6159e465182edf6e256b7",
        100,
    ),
    "smooth-400x240": (
        "686f907744806e889794d187a79fd3d00cac3b61d7bd381ae37136f770cba162",
        100,
    ),
    "shapes-512x384": (
        "96667c8996263d9ebe273ccec3f1c8d9c1f141d9be273e0e161e3c058b6241d8",
        56,
    ),
    "blocks-640x360": (
        "f0119b7af84a93702deead4d78994eb6876652b2d2116c8f0fad64850fea994c",
        100,
    ),
    "smooth-150x520": (
        "6cf00fdd5fd5f022930f0a79ba08f70e5e008ead8a4e0157c0ddfe7bce164c58",
        100,
    ),
    "shapes-31x200": (
        "5556cda95544cd54cccc7564aa9932ab272caa995572393babb37376a8a95544",
        77,
    ),
}

# (path under pdq/data, SHA-256 of the file, size in bytes, published hash)
THREATEXCHANGE_VECTORS: tuple[tuple[str, str, int, str], ...] = (
    (
        "misc-images/c.png",
        "93aa0ba454d3805e02be04a883b839a85bee83d0cbac3d306331584c1ebec0c6",
        1178841,
        "e64cc9d91e623842f8d1f1d9a398e78c9f199a3bd87924f2b7e11e0bf061b064",
    ),
    (
        "misc-images/small.jpg",
        "329479790610cd0a2668eea439374cf279ef58d7330400dbb4e3569429ef5e3d",
        1756,
        "0007001f003f003f007f00ff00ff00ff01ff01ff01ff03ff03ff03ff03ff03ff",
    ),
    (
        "misc-images/wee.jpg",
        "05636c50d0c2ffe3a820f964eaba35b404deadd10538fe452ad9fb16fdd823d7",
        13664,
        "6227401f601ff4ccafcc9fad4b0d95d371a2eb7265a3285234d228ca94deeb2d",
    ),
    (
        "reg-test-input/labelme-subset/q0003.jpg",
        "9f93a23582535bc62088240e38ee050c9c0e2eed5c4741339b8415bf04a0604e",
        3187,
        "54a977c221d14c1c43ba5e6e21d4a13989a3553f1462611cbb85fda7be83b677",
    ),
    (
        "reg-test-input/labelme-subset/q0004.jpg",
        "d517f25c0142f9d897a00959acc16ac2a67cf20ade0997bb5ab378f607ab7f3b",
        3345,
        "992d44af36d69e6ca6b812585928bac11def254ef5398c6d07466c9abcc65b92",
    ),
    (
        "reg-test-input/labelme-subset/q0122.jpg",
        "3e7b6b95d72f6ea61c0e6c3f2a4afc7c78ee2ab08b9ac51c58388e50deb139e9",
        6928,
        "cfb2009ddd21c6dab0046a7745b5984757a8a4535b3377aea2591d32b33ff940",
    ),
    (
        "reg-test-input/labelme-subset/q0291.jpg",
        "25db315820a1faee9bc4a8086914f4c6fcb818448a0e1a707fe160bb67818846",
        9559,
        "a0fe94f1e5cc1cc8dd855948498dc9243f7ca27336f036d7f212b74bc103c9a7",
    ),
    (
        "reg-test-input/labelme-subset/q0746.jpg",
        "acf23ed39052296ed032981dcdc063a1e442691be54e2927043a08f97675a1f1",
        11530,
        "1049d96239e24d4dca2c55512b8bdb77425f4dbcf575a0a95555aaab5554aaaa",
    ),
    (
        "reg-test-input/labelme-subset/q1050.jpg",
        "e570fd8266489c94f1da079da69e62c24e90851c424e5a26f63ebdd13f0e51a4",
        17984,
        "489db672e9190276d452aeab41eba20f02375fe4092d88defdf491a5c55c5f70",
    ),
    (
        "reg-test-input/labelme-subset/q2821.jpg",
        "418b85299aa697927c29ff139280a988ec3c44df540a35c221dcb0fcf8051fcd",
        19660,
        "b150231ffae4710ffcf4f18bb574b109a576f14bb8543189f8743289f174b109",
    ),
    (
        "reg-test-input/dih/bridge-1-original.jpg",
        "b5b0799616df52d475a3968dc7e54f1d0724c912244ffa6175bc786375dd7298",
        361182,
        "d8f8f0cce0f4a84f0e370a22028f67f0b36e2ed596623e1d33e6b39c4e9c9b22",
    ),
    (
        "reg-test-input/dih/bridge-2-rotate-90.jpg",
        "a4ec0b39cf38469e3c5e16869a5ea1f23995b42a4b5de15d223e905b87788317",
        373855,
        "38a50efd71c83f429013d68d0ffffc52e34e0e15ada952a9d29684214aa9e5af",
    ),
    (
        "reg-test-input/dih/bridge-3-rotate-180.jpg",
        "fc9073d9b08f7b4d90e0fbdb4693fa7c58a9746dfc4c339fce5b3320905a3b3b",
        356813,
        "2dadda64b5a142e5d362209057da895ae63b8c7fc277b4b766b319361f893188",
    ),
    (
        "reg-test-input/dih/bridge-4-rotate-270.jpg",
        "ed7769b97c8eab168eefe5cc5a49a1372d25cfecb354e457153a8e45f208daa5",
        357530,
        "a5f0a457248995e8c9065c275aaa54d8b61ba4bdf8fcfc0387c32f8b0bfc4f05",
    ),
    (
        "reg-test-input/dih/bridge-5-flipx.jpg",
        "3069d9eb8d9de760229cf99ddcc5a7834f08e3924b37c080c2df8bc5836737fb",
        356799,
        "d8f80f31e0f417b00e37f5dd028f980fb36ed12a9662c1e233e64c634e9c64dd",
    ),
    (
        "reg-test-input/dih/bridge-6-flipy.jpg",
        "c2225164ee53418ec001fa33bf9a4cf95a73514f493c87f676a1af9f9624d19b",
        372851,
        "0dad259bb1a1bd18d362576556da32a1e63b7380c2374b4866b3c6c91b89ce77",
    ),
    (
        "reg-test-input/dih/bridge-7-flip-plus-1.jpg",
        "355dd5a4ccef6e205ed705c73bce8676981f9f943a06453a2f68d9d8bba3f7f1",
        373862,
        "f0a5e10271dcc0bd9c5309720fff018de34ef1e8ada9a956d2967ade1ea91a50",
    ),
    (
        "reg-test-input/dih/bridge-8-flip-minus-1.jpg",
        "092bfe6026946f3b9322662d010fb16dabf3da7ece9f0cbaad99772fdf16a30e",
        357443,
        "69f05aa8a4996a17c146a2da5aaaab07b61b5b60f8fc07fc83c3d0740bfcb0fa",
    ),
)
