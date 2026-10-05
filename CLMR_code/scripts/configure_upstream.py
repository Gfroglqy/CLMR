from __future__ import annotations
import argparse
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace(text, old, new):
    if old not in text:
        if new in text:
            return text
        raise ValueError("Pinned upstream code does not match compatibility rule: " + old[:100])
    if text.count(old) != 1:
        raise ValueError("Ambiguous compatibility rule")
    return text.replace(old, new)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-root", type=Path, default=ROOT / "external")
    cli = parser.parse_args()
    base = cli.external_root.resolve()
    for name in ("melbert", "dags", "roppt", "contrastwsd"):
        repo = base / name
        if not repo.exists():
            continue
        for filename in ("main.py", "trainer.py"):
            path = repo / filename
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8")
            if "from tensorboardX import SummaryWriter\n" in text and "except ImportError:" not in text:
                text = text.replace("from tensorboardX import SummaryWriter\n", "try:\n    from tensorboardX import SummaryWriter\nexcept ImportError:\n    class SummaryWriter:\n        def add_scalar(self, *args, **kwargs):\n            return None\n        def close(self):\n            return None\n")
            lines = []
            for line in text.splitlines():
                if line.startswith("from transformers import ") and re.search(r"\bAdamW\b", line):
                    line = line.replace("AdamW, ", "").replace(", AdamW", "")
                    lines.extend([line, "from torch.optim import AdamW"])
                else:
                    lines.append(line)
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        for filename in ("datasets.py", "run_classifier_dataset_utils.py"):
            path = repo / filename
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8")
            text = text.replace("import simplejson as json", "import json")
            text = text.replace("from lxml import etree\n", "").replace("from allennlp.modules.elmo import batch_to_ids\n", "")
            text = text.replace("from appdirs import unicode\n", "")
            path.write_text(text, encoding="utf-8")
    dags = base / "dags"
    if dags.exists():
        if not (dags / "utils.py").exists() and not (dags / "utils").exists():
            utilities = base / "roppt/utils"
            if not utilities.exists():
                raise FileNotFoundError("DAGS omits utils; download RoPPT's published utilities too")
            shutil.copytree(utilities, dags / "utils")
        path = dags / "datasets.py"
        text = path.read_text(encoding="utf-8")
        for old, new in (
            ("get_dataset(args.dataset_name)", "get_dataset(args.dataset_name, args.data_dir)"),
            ("def get_dataset(dataset_name):", "def get_dataset(dataset_name, data_dir=None):"),
            ("aspect = word_tokenize(e['aspect_sentiment'][i][0].lower())", "aspect = TreebankWordTokenizer().tokenize(e['aspect_sentiment'][i][0].lower())"),
            ("'ori_index': e['ori_index'], 'ori_pos': e['ori_pos'], 'ori_aspect': e['ori_aspect']", "'ori_index': frm, 'ori_pos': e.get('target_pos', [e['tags'][item[0]] for item in e['from_to']])[i], 'ori_aspect': e['aspect_sentiment'][i][0]"),
        ):
            text = replace(text, old, new)
        anchor = "    train = list(read_sentence_depparsed(ds_train[dataset_name]))"
        if "if data_dir:" not in text:
            injection = "    if data_dir:\n        ds_train[dataset_name] = os.path.join(data_dir, 'train_rospacy.json')\n        ds_test[dataset_name] = os.path.join(data_dir, 'test_rospacy.json')\n        ds_val[dataset_name] = os.path.join(data_dir, 'val_rospacy.json')\n\n"
            text = replace(text, anchor, injection + anchor)
        path.write_text(text, encoding="utf-8")
        path = dags / "modeling.py"
        text = replace(path.read_text(encoding="utf-8"), "torch.cat([attn_output,attn_output_2.mean(1)], dim=1)", "torch.cat([attn_output, attn_output_2], dim=1)")
        path.write_text(text, encoding="utf-8")
    contrast = base / "contrastwsd/run_classifier_dataset_utils.py"
    if contrast.exists():
        text = contrast.read_text(encoding="utf-8")
        for old, new in (
            ("self.input_ids_3 = input_ids_3,", "self.input_ids_3 = input_ids_3"),
            ("self.input_mask_3 = input_mask_3,", "self.input_mask_3 = input_mask_3"),
            ("def _read_tsv(cls, input_file, quotechar=None):", "def _read_tsv(cls, input_file, quotechar='\"'):"),
        ):
            text = replace(text, old, new)
        if "word_sense = word_sense[:max(" not in text:
            anchor = " # >>    # Second features (Target word)"
            text = replace(text, anchor, "        word_sense = word_sense[:max(0, max_seq_length - len(text_b) - 3)]\n\n" + anchor)
        if "definition = definition[:max(" not in text:
            anchor = "        # preparing the target word with the symbolof feature."
            text = replace(text, anchor, "        definition = definition[:max(0, max_seq_length - len(text_b) - 3)]\n\n" + anchor)
        contrast.write_text(text, encoding="utf-8")
    print("Compatibility edits applied only to downloaded, ignored third-party copies.")


if __name__ == "__main__":
    main()
