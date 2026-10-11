import json
import os
import tempfile
import sqlite3
import unittest
from unittest import mock

import numpy as np
import PIL.Image
import torch
from click.testing import CliRunner
from safetensors.torch import save_file

from picurate.tagger.captions import (
    CaptionOptions,
    Captioner,
    Vocabulary,
    expand_character,
    parse_replacements,
    prompt_form,
    resolution_tags,
)
from picurate.tagger import cli as cli_mod
from picurate.tagger.cli import cli
from picurate.tagger.model import TaggerConfig, TaggerPreprocess, ViTDetTagger, load_model
from picurate.tagger.tag_db import Binding, TagDB, TagDBMismatchError

# A vocabulary in the checkpoint's layout (category blocks in this order, "style" = artists), general tags in
# popularity order like the real one.
SPLIT = [
    ["general", ["1girl", "solo", "long_hair", "smile", "2boys", "simple_background", "^_^"]],
    ["character", ["artoria_pendragon_(fate)", "hatsune_miku"]],
    ["copyright", ["fate_(series)", "original"]],
    ["style", ["some_artist"]],
    ["meta", ["highres", "md5_mismatch", "official_art", "watercolor_(medium)"]],
    ["rating", ["rating:s", "rating:g", "rating:q", "rating:e"]],
]
TAGS = [t for _, tags in SPLIT for t in tags]
CONFIG = {
    "tags": TAGS,
    "tags_split": [[c, len(t)] for c, t in SPLIT],
    "category_best_threshold": {"general": 0.3, "character": 0.3, "copyright": 0.3, "style": 0.3, "meta": 0.3, "rating": 0.4},
}


def _probs(**scores: float) -> np.ndarray:
    p = np.zeros(len(TAGS), np.float32)
    for tag, v in scores.items():
        p[TAGS.index(tag.replace("__", ":"))] = v
    return p


# general sorted by confidence: smile > solo > 1girl; 2boys counts as a count tag
SCENE = _probs(
    smile=0.95, solo=0.9, **{"1girl": 0.8, "2boys": 0.4, "^_^": 0.5}, long_hair=0.1,
    **{"artoria_pendragon_(fate)": 0.7, "fate_(series)": 0.6, "some_artist": 0.7},
    md5_mismatch=0.9, official_art=0.5, **{"watercolor_(medium)": 0.4}, highres=0.9,
    rating__s=0.3, rating__g=0.6,
)


class CaptionTests(unittest.TestCase):
    def setUp(self):
        self.vocab = Vocabulary.from_config(CONFIG)

    def caption(self, probs=SCENE, size=(1000, 1300), **kw) -> str:
        return Captioner(self.vocab, CaptionOptions(**kw)).caption(probs, *size).text

    def test_helpers(self):
        self.assertEqual(prompt_form("long_hair"), "long hair")
        self.assertEqual(prompt_form("^_^"), "^_^")
        self.assertEqual(expand_character("artoria_pendragon_(fate)"), ["artoria_pendragon", "fate"])
        self.assertEqual(expand_character("name_(costume)_(series)"), ["name_(costume)", "series"])
        self.assertEqual(expand_character("hatsune_miku"), ["hatsune_miku"])
        self.assertEqual(resolution_tags(500, 500), ["lowres"])
        self.assertEqual(resolution_tags(827, 1208), ["highres"])
        self.assertEqual(resolution_tags(3200, 100), ["highres", "absurdres"])
        self.assertEqual(resolution_tags(1000, 10000), ["highres", "absurdres", "incredibly_absurdres"])
        self.assertEqual(parse_replacements(r"a b,c;x\,y,z"), {"a_b": "c", "x,y": "z"})
        with self.assertRaises(ValueError):
            parse_replacements("a,b,c")

    def test_vocabulary_maps_style_to_artist(self):
        self.assertEqual(self.vocab.slices["artist"], slice(11, 12))
        self.assertEqual(self.vocab.thresholds["artist"], 0.3)

    def test_anima_sections(self):
        text = self.caption(style="anima", prefix=["masterpiece", "best quality"], thresholds={"artist": 0.5})
        self.assertEqual(
            text,
            "masterpiece, best quality, highres, official art, watercolor (medium), safe, 1girl, 2boys, "
            "artoria pendragon (fate), fate (series), @some artist, smile, solo, ^_^",
        )

    def test_sdxl_matches_wd14_layout(self):
        text = self.caption(style="sdxl", always_first=["1girl", "1boy"], thresholds={"artist": 0.5})
        # characters expanded first, then series and artist, general in vocabulary (popularity) order, meta, rating last
        self.assertEqual(
            text,
            "1girl, artoria pendragon, fate, fate (series), some artist, solo, smile, 2boys, ^_^, "
            "highres, official art, watercolor (medium), general",
        )

    def test_resolution_and_meta_modes(self):
        # size: highres from the pixels, the predicted one ignored; 400x400 is lowres
        self.assertIn("lowres", self.caption(size=(400, 400)))
        self.assertNotIn("highres", self.caption(size=(400, 400)))
        self.assertIn("highres", self.caption(size=(400, 400), resolution="predicted"))
        self.assertNotIn("res", self.caption(resolution="none"))
        self.assertIn("md5 mismatch", self.caption(meta="all"))
        self.assertNotIn("md5 mismatch", self.caption())
        none = self.caption(meta="none")
        self.assertNotIn("official art", none)
        self.assertNotIn("highres", none)

    def test_thresholds_filters_and_replacements(self):
        self.assertNotIn("some artist", self.caption(thresholds={"artist": 2}))
        self.assertIn("long hair", self.caption(thresholds={"general": 0.05}))
        text = self.caption(
            undesired=frozenset({"smile", "safe"}), replacements={"solo": "alone", "^_^": ""}, thresholds={"artist": 2}
        )
        self.assertNotIn("smile", text)
        self.assertNotIn("safe", text)
        self.assertNotIn("^_^", text)
        self.assertIn("alone", text)
        self.assertNotIn("safe", self.caption(rating=False))

    def test_sdxl_series_once_and_software_medium_dropped(self):
        probs = _probs(**{"artoria_pendragon_(fate)": 0.9, "fate_(series)": 0.9}, rating__g=0.6)
        vocab = Vocabulary.from_config({
            **CONFIG,
            "tags": [t if t != "fate_(series)" else "fate" for t in TAGS],
            "tags_split": CONFIG["tags_split"],
        })  # fmt: skip
        text = Captioner(vocab, CaptionOptions(style="sdxl")).caption(probs, 600, 600).text
        self.assertEqual(text, "artoria pendragon, fate, general")
        meta = Vocabulary.from_config({**CONFIG, "tags": [t if t != "md5_mismatch" else "photoshop_(medium)" for t in TAGS]})
        filtered = Captioner(meta, CaptionOptions()).caption(SCENE, 600, 600).text
        self.assertNotIn("photoshop", filtered)
        self.assertIn("watercolor (medium)", filtered)
        self.assertIn("photoshop (medium)", Captioner(meta, CaptionOptions(meta="all")).caption(SCENE, 600, 600).text)

    def test_always_first_follows_sd_scripts(self):
        both = _probs(solo=0.9, **{"1girl": 0.8, "2boys": 0.8}, rating__g=0.6)
        self.assertEqual(self.caption(both, style="sdxl", always_first=["1girl", "2boys"]), "2boys, 1girl, solo, highres, general")
        self.assertEqual(self.caption(both, style="sdxl", always_first=["1girl", "1boy"]), "1girl, solo, 2boys, highres, general")

    def test_override_artist(self):
        options = dict(override_artists=["nnn_yryr", "other_one"], thresholds={"artist": 0.1})
        anima = self.caption(style="anima", **options)
        self.assertIn("fate (series), @nnn yryr, @other one, smile", anima)
        self.assertNotIn("some artist", anima)
        self.assertIn("fate (series), nnn yryr, other one, 1girl", self.caption(style="sdxl", **options))
        # even with artist predictions disabled
        self.assertIn("@nnn yryr", self.caption(override_artists=["nnn_yryr"], thresholds={"artist": 2}))

    def test_bad_options(self):
        for kw in ({"style": "pony"}, {"meta": "some"}, {"resolution": "big"}):
            with self.assertRaises(ValueError):
                CaptionOptions(**kw)


# -- a tiny checkpoint with the real architecture, for the model and the CLI ------------------------------------------

TINY = {
    "img_size": 56, "patch_size": 14, "embed_dim": 32, "depth": 2, "num_heads": 2, "mlp_ratio": 2.0, "window_size": 2,
    "global_att_blocks": [1], "pretrain_img_size": 28, "rope_pt_size": None, "num_classes": len(TAGS),
    "use_rope": True, "use_interp_rope": True, "use_abs_pos": True, "tile_abs_pos": True, "pretrain_use_cls_token": True,
    "retain_cls_token": False, "rel_pos_blocks": [], "init_values": None, "ln_pre": True, "ln_post": False,
    "bias_patch_embed": False, "qkv_bias": True, "act_layer": "GELU", "norm_layer": "LayerNorm", "in_chans": 3,
}  # fmt: skip


def _write_tiny_checkpoint(directory: str) -> None:
    torch.manual_seed(0)
    net = ViTDetTagger(TaggerConfig.from_json(TINY))
    for p in net.parameters():
        torch.nn.init.normal_(p, std=0.2)
    save_file({k: v.contiguous() for k, v in net.state_dict().items()}, os.path.join(directory, "model.safetensors"))
    with open(os.path.join(directory, "config.json"), "w") as f:
        json.dump({**TINY, **CONFIG}, f)


class ModelTests(unittest.TestCase):
    def test_preprocess_flattens_alpha_on_white_and_pads_black(self):
        img = PIL.Image.new("RGBA", (40, 20), (255, 0, 0, 0))  # fully transparent red
        x = TaggerPreprocess(56, np.dtype(np.float32))(img)
        self.assertEqual(x.shape, (3, 56, 56))
        self.assertTrue(np.allclose(x[:, 28, 28], 1.0))  # white content
        self.assertTrue(np.allclose(x[:, 0, 28], -1.0))  # black padding above
        self.assertEqual(TaggerPreprocess(56, np.dtype(np.float16))(PIL.Image.new("L", (56, 56))).dtype, np.float16)

    def test_rejects_unsupported_config(self):
        with self.assertRaises(ValueError):
            TaggerConfig.from_json({**TINY, "rel_pos_blocks": [0]})

    def test_load_casts_weights_but_keeps_rope_fp32(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_tiny_checkpoint(tmp)
            x = torch.rand(2, 3, 56, 56) * 2 - 1
            ref = load_model(tmp).model(x)
            half = load_model(tmp, dtype=torch.bfloat16).model
            self.assertEqual(half.head.weight.dtype, torch.bfloat16)
            self.assertEqual(half.blocks[0].attn.rope_cos.dtype, torch.float32)
            self.assertEqual(ref.shape, (2, len(TAGS)))
            self.assertLess((half(x.bfloat16()).float().sigmoid() - ref.sigmoid()).abs().max().item(), 0.05)


class CliTests(unittest.TestCase):
    def test_tags_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir, image_dir = os.path.join(tmp, "model"), os.path.join(tmp, "img")
            os.makedirs(model_dir)
            os.makedirs(os.path.join(image_dir, "sub"))
            _write_tiny_checkpoint(model_dir)
            PIL.Image.new("RGB", (64, 32), (200, 10, 10)).save(os.path.join(image_dir, "a.png"))
            PIL.Image.new("RGBA", (30, 30)).save(os.path.join(image_dir, "b.webp"))
            PIL.Image.new("RGB", (30, 30)).save(os.path.join(image_dir, "sub", "c.jpg"))
            with open(os.path.join(image_dir, "b.txt"), "w") as f:
                f.write("keep me\n")
            base = ["run", image_dir, "--model", model_dir, "--device", "cpu", "-b", "2", "-j", "1", "--threshold", "0"]
            runner = CliRunner()

            r = runner.invoke(cli, base + ["--existing", "skip", "--prefix", "masterpiece"], catch_exceptions=False)
            self.assertEqual(r.exit_code, 0, r.output)
            with open(os.path.join(image_dir, "a.txt")) as f:
                a = f.read()
            self.assertTrue(a.startswith("masterpiece, lowres, "), a)  # 64x32 is lowres by size
            self.assertTrue(a.endswith("\n"))
            with open(os.path.join(image_dir, "b.txt")) as f:
                self.assertEqual(f.read(), "keep me\n")
            self.assertFalse(os.path.exists(os.path.join(image_dir, "sub", "c.txt")))  # not recursive

            r = runner.invoke(cli, base + ["--existing", "append", "-r", "--style", "sdxl"], catch_exceptions=False)
            self.assertEqual(r.exit_code, 0, r.output)
            with open(os.path.join(image_dir, "b.txt")) as f:
                b = f.read().strip()
            self.assertTrue(b.startswith("keep me, 1girl, "), b)
            self.assertTrue(os.path.exists(os.path.join(image_dir, "sub", "c.txt")))

            out = os.path.join(tmp, "captions.jsonl")
            r = runner.invoke(cli, base + ["-o", out, "--frequency"], catch_exceptions=False)
            self.assertEqual(r.exit_code, 0, r.output)
            self.assertIn("Tag frequencies:", r.output)
            with open(out) as f:
                rows = [json.loads(line) for line in f]
            self.assertEqual(sorted(os.path.basename(row["image_path"]) for row in rows), ["a.png", "b.webp"])
            self.assertEqual(next(row for row in rows if row["image_path"].endswith("a.png"))["image_size"], [64, 32])

            r = runner.invoke(cli, base + ["-o", out, "--override-artist-tag", "@nnn yryr"], catch_exceptions=False)
            with open(out) as f:
                self.assertTrue(all("@nnn yryr" in json.loads(line)["caption"] for line in f))

            r = runner.invoke(cli, base + ["-o", os.path.join(tmp, "x.csv")])
            self.assertNotEqual(r.exit_code, 0)



class TagDBTests(unittest.TestCase):
    def test_store_keeps_floor_and_ratings_and_round_trips_vocabulary(self):
        vocab = Vocabulary.from_config(CONFIG)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "t.sqlite")
            rec = cli_mod.ImageRecord("x/a.png", 1.5, 10, 64, 32, "PNG")
            with TagDB(path) as db:
                db.bind(Binding("m@1", "fp16", 0.3), vocab)
                db.store([(rec, SCENE)], vocab)
                db.store([(rec, SCENE)], vocab)  # replacing an image does not duplicate it
                self.assertEqual(db.load_index(), {"x/a.png": 1.5})
                stored = {TAGS[i]: p for i, p in db.conn.execute("SELECT tag_id, prob FROM scores")}
                # everything >= 0.3 plus all four ratings, even the zero ones
                expected = {t for t in TAGS if SCENE[TAGS.index(t)] >= 0.3} | {"rating:s", "rating:g", "rating:q", "rating:e"}
                self.assertEqual(set(stored), expected)
                again = db.vocabulary()
                self.assertEqual(list(again.tags), TAGS)
                self.assertEqual(again.slices, vocab.slices)
                self.assertEqual(again.thresholds, vocab.thresholds)
                (got_rec, probs), = db.iter_scores(len(TAGS))
                self.assertEqual(got_rec, rec)
                self.assertTrue(np.allclose(probs, np.where((SCENE >= 0.3) | (np.arange(len(TAGS)) >= 15), SCENE, 0)))
                # same binding: fine; any difference: refused, or reset when asked
                db.bind(Binding("m@1", "fp16", 0.3), vocab)
                for other in (Binding("m@2", "fp16", 0.3), Binding("m@1", "fp32", 0.3), Binding("m@1", "fp16", 0.1)):
                    with self.assertRaises(TagDBMismatchError):
                        db.bind(other, vocab)
                db.bind(Binding("m@2", "fp16", 0.3), vocab, reset_on_mismatch=True)
                self.assertEqual(len(db), 0)
                self.assertEqual(db.conn.execute("SELECT COUNT(*) FROM scores").fetchone()[0], 0)
                db.store([(rec, SCENE)], vocab)
                db.delete(["x/a.png"])
                self.assertEqual(db.conn.execute("SELECT COUNT(*) FROM scores").fetchone()[0], 0)  # cascades


class UpdateExportTests(unittest.TestCase):
    def test_interrupted_update_resumes_and_export_matches_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir, image_dir = os.path.join(tmp, "model"), os.path.join(tmp, "img")
            os.makedirs(model_dir)
            os.makedirs(image_dir)
            _write_tiny_checkpoint(model_dir)
            for i in range(5):
                PIL.Image.new("RGB", (40 + i, 30), (40 * i, 0, 0)).save(os.path.join(image_dir, f"{i}.png"))
            db = os.path.join(tmp, "tags.sqlite")
            model = ["--model", model_dir, "--device", "cpu", "-b", "2", "-j", "1"]
            update = ["update", image_dir, "-d", db, "--score-floor", "0"] + model
            runner = CliRunner()

            real_store = TagDB.store
            calls = []

            def store_then_interrupt(self, rows, vocab):
                if calls:
                    raise KeyboardInterrupt
                calls.append(1)
                real_store(self, rows, vocab)

            with mock.patch.object(TagDB, "store", store_then_interrupt):
                r = runner.invoke(cli, update)
            self.assertEqual(r.exit_code, 130, r.output)
            self.assertEqual(sqlite3.connect(db).execute("SELECT COUNT(*) FROM images").fetchone()[0], 2)  # first batch only

            with mock.patch.object(cli_mod, "run_pipeline", wraps=cli_mod.run_pipeline) as pipe:
                r = runner.invoke(cli, update, catch_exceptions=False)
                self.assertEqual(r.exit_code, 0, r.output)
                self.assertEqual(len(pipe.call_args.args[1]), 3)  # resumed: only the three missing images
            self.assertEqual(sqlite3.connect(db).execute("SELECT COUNT(*) FROM images").fetchone()[0], 5)

            # nothing changed: no model load at all
            with mock.patch.object(cli_mod, "Tagger", side_effect=AssertionError("loaded")):
                r = runner.invoke(cli, update, catch_exceptions=False)
            self.assertEqual(r.exit_code, 0, r.output)

            # a different floor is a different binding
            r = runner.invoke(cli, ["update", image_dir, "-d", db, "--score-floor", "0.5", "-f"] + model, catch_exceptions=False)
            self.assertEqual(r.exit_code, 0, r.output)
            r = runner.invoke(cli, ["export", image_dir, "-d", db, "-t", "0.1"])
            self.assertNotEqual(r.exit_code, 0)
            self.assertIn("0.5", r.output)
            r = runner.invoke(cli, ["update", image_dir, "-d", db, "--score-floor", "0", "-f"] + model, catch_exceptions=False)
            self.assertEqual(r.exit_code, 0, r.output)

            # export reproduces run's captions (fp32 on CPU, same batches)
            for style in ("anima", "sdxl"):
                run_out, exp_out = os.path.join(tmp, f"run_{style}.json"), os.path.join(tmp, f"exp_{style}.json")
                caption = ["--style", style, "-t", "0.3", "--artist-threshold", "0.3"]
                runner.invoke(cli, ["run", image_dir, "-o", run_out] + model + caption, catch_exceptions=False)
                r = runner.invoke(cli, ["export", image_dir, "-d", db, "-o", exp_out] + caption, catch_exceptions=False)
                self.assertEqual(r.exit_code, 0, r.output)
                with open(run_out) as a, open(exp_out) as b:
                    self.assertEqual(json.load(a), json.load(b))

            # orphans and stale files
            os.remove(os.path.join(image_dir, "0.png"))
            os.utime(os.path.join(image_dir, "1.png"), (1_000, 1_000))
            with self.assertLogs("picurate.tagger.cli", level="WARNING") as logs:
                r = runner.invoke(cli, ["export", image_dir, "-d", db], catch_exceptions=False)
            self.assertTrue(any("1.png" in m for m in logs.output))
            self.assertFalse(os.path.exists(os.path.join(image_dir, "1.txt")))
            self.assertTrue(os.path.exists(os.path.join(image_dir, "2.txt")))
            r = runner.invoke(cli, update, catch_exceptions=False)
            self.assertEqual(r.exit_code, 0, r.output)
            self.assertEqual(sorted(TagDB(db).load_index()), ["1.png", "2.png", "3.png", "4.png"])


if __name__ == "__main__":
    unittest.main()
