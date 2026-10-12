import os
import tempfile
import unittest

from picurate.common.images import ImageRecord
from picurate.dedupe.keeping import builtin_policies, load_policies


def rec(path, mtime=0.0, size=0, width=1, height=1, fmt="JPEG"):
    return ImageRecord(path, mtime, size, width, height, fmt)


class BuiltinPolicyTests(unittest.TestCase):
    def setUp(self):
        self.policies = builtin_policies()

    def test_highest_quality_order(self):
        p = self.policies["highest-quality"]
        a = rec("a.jpg", width=100, height=100)
        b = rec("b.png", width=100, height=100, fmt="PNG")
        c = rec("c.gif", width=200, height=200, fmt="GIF")
        self.assertEqual([r.path for r in p.sort([a, b, c])], ["c.gif", "b.png", "a.jpg"])
        # same pixels and format: larger file wins before newer file
        d1 = rec("d1.jpg", mtime=9, size=1)
        d2 = rec("d2.jpg", mtime=1, size=2)
        self.assertEqual(p.select([d1, d2]).path, "d2.jpg")

    def test_pic_dir_prefers_wallpaper_dir_then_source(self):
        p = self.policies["pic-dir"]
        pixiv = rec("misc/12345678_p0.jpg", width=10, height=10)
        wall = rec("Wallpaper/random.jpg", width=1, height=1)
        self.assertEqual(p.select([pixiv, wall]).path, "Wallpaper/random.jpg")
        yande = rec("x/yande.re 123 tag.jpg", width=999, height=999)
        self.assertEqual(p.select([yande, pixiv]).path, "misc/12345678_p0.jpg")
        danbooru = rec("x/__hatsune_miku_vocaloid__0123456789abcdef0123456789abcdef.png", fmt="PNG")
        konachan = rec("x/Konachan.com - 123 tag.jpg")
        other = rec("x/other.jpg", width=5000, height=5000)
        self.assertEqual(
            [r.path for r in p.sort([other, konachan, danbooru, yande, pixiv])],
            [pixiv.path, yande.path, danbooru.path, konachan.path, other.path],
        )
        # no source matched at all: quality decides
        self.assertEqual(p.select([rec("x/a.jpg", width=1, height=1), rec("x/b.jpg", width=2, height=2)]).path, "x/b.jpg")


class UserConfigTests(unittest.TestCase):
    def _load(self, text):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "p.toml")
            with open(path, "w") as f:
                f.write(text)
            return load_policies(path)

    def test_user_file_adds_and_overrides(self):
        policies = self._load("""
[policy.newest]
criteria = [{ attr = "mtime", prefer = "min" }]
[policy.shortest-path]
description = "shortest"
criteria = [{ attr = "path", match = '^keep/' }, { attr = "ext", match = ['\\.png$', '\\.jpg$'] }]
""")
        self.assertEqual(policies["newest"].select([rec("a", mtime=1), rec("b", mtime=2)]).path, "a")  # overridden to oldest
        self.assertIn("largest", policies)  # builtins still there
        sp = policies["shortest-path"]
        self.assertEqual(sp.description, "shortest")
        self.assertEqual(sp.select([rec("x/a.jpg"), rec("keep/b.jpg"), rec("keep/c.png")]).path, "keep/c.png")


if __name__ == "__main__":
    unittest.main()
