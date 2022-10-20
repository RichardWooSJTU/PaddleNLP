# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import shutil
import tempfile
import unittest

from paddlenlp.transformers import SPIECE_UNDERLINE, MBart50Tokenizer
from paddlenlp.transformers.mbart.modeling import shift_tokens_right
from ...testing_utils import get_tests_dir, nested_simplify, slow

from ..test_tokenizer_common import TokenizerTesterMixin

SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece.model")

EN_CODE = 250004
RO_CODE = 250020


class MBart50TokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = MBart50Tokenizer
    test_sentencepiece = True

    test_offsets = False

    def setUp(self):
        super().setUp()

        # We have a SentencePiece fixture for testing
        tokenizer = MBart50Tokenizer(SAMPLE_VOCAB,
                                     src_lang="en_XX",
                                     tgt_lang="ro_RO",
                                     keep_accents=True)
        tokenizer.save_pretrained(self.tmpdirname)

    def test_convert_token_and_id(self):
        """Test ``_convert_token_to_id`` and ``_convert_id_to_token``."""
        token = "<s>"
        token_id = 0

        self.assertEqual(self.get_tokenizer()._convert_token_to_id(token),
                         token_id)
        self.assertEqual(self.get_tokenizer()._convert_id_to_token(token_id),
                         token)

    def test_get_vocab(self):
        vocab_keys = list(self.get_tokenizer().get_vocab().keys())

        self.assertEqual(vocab_keys[0], "<s>")
        self.assertEqual(vocab_keys[1], "<pad>")
        self.assertEqual(vocab_keys[-1], "<mask>")
        self.assertEqual(len(vocab_keys), 1_054)

    def test_vocab_size(self):
        self.assertEqual(self.get_tokenizer().vocab_size, 1_054)

    def test_full_tokenizer(self):
        tokenizer = MBart50Tokenizer(SAMPLE_VOCAB,
                                     src_lang="en_XX",
                                     tgt_lang="ro_RO",
                                     keep_accents=True)

        tokens = tokenizer.tokenize("This is a test")
        self.assertListEqual(tokens, ["▁This", "▁is", "▁a", "▁t", "est"])

        self.assertListEqual(
            tokenizer.convert_tokens_to_ids(tokens),
            [
                value + tokenizer.fairseq_offset
                for value in [285, 46, 10, 170, 382]
            ],
        )

        tokens = tokenizer.tokenize("I was born in 92000, and this is falsé.")
        self.assertListEqual(
            tokens,
            # fmt: off
            [
                SPIECE_UNDERLINE + "I", SPIECE_UNDERLINE + "was",
                SPIECE_UNDERLINE + "b", "or", "n", SPIECE_UNDERLINE + "in",
                SPIECE_UNDERLINE + "", "9", "2", "0", "0", "0", ",",
                SPIECE_UNDERLINE + "and", SPIECE_UNDERLINE + "this",
                SPIECE_UNDERLINE + "is", SPIECE_UNDERLINE + "f", "al", "s", "é",
                "."
            ],
            # fmt: on
        )
        ids = tokenizer.convert_tokens_to_ids(tokens)
        self.assertListEqual(
            ids,
            [
                value + tokenizer.fairseq_offset for value in [
                    8, 21, 84, 55, 24, 19, 7, 2, 602, 347, 347, 347, 3, 12, 66,
                    46, 72, 80, 6, 2, 4
                ]
            ],
        )

        back_tokens = tokenizer.convert_ids_to_tokens(ids)
        self.assertListEqual(
            back_tokens,
            # fmt: off
            [
                SPIECE_UNDERLINE + "I", SPIECE_UNDERLINE + "was",
                SPIECE_UNDERLINE + "b", "or", "n", SPIECE_UNDERLINE + "in",
                SPIECE_UNDERLINE + "", "<unk>", "2", "0", "0", "0", ",",
                SPIECE_UNDERLINE + "and", SPIECE_UNDERLINE + "this",
                SPIECE_UNDERLINE + "is", SPIECE_UNDERLINE + "f", "al", "s",
                "<unk>", "."
            ],
            # fmt: on
        )


class MBart50OneToManyIntegrationTest(unittest.TestCase):
    checkpoint_name = "mbart-large-50-one-to-many-mmt"
    src_text = [
        " UN Chief Says There Is No Military Solution in Syria",
        """ Secretary-General Ban Ki-moon says his response to Russia's stepped up military support for Syria is that "there is no military solution" to the nearly five-year conflict and more weapons will only worsen the violence and misery for millions of people.""",
    ]
    tgt_text = [
        "Şeful ONU declară că nu există o soluţie militară în Siria",
        "Secretarul General Ban Ki-moon declară că răspunsul său la intensificarea sprijinului militar al Rusiei"
        ' pentru Siria este că "nu există o soluţie militară" la conflictul de aproape cinci ani şi că noi arme nu vor'
        " face decât să înrăutăţească violenţele şi mizeria pentru milioane de oameni.",
    ]
    expected_src_tokens = [
        EN_CODE, 8274, 127873, 25916, 7, 8622, 2071, 438, 67485, 53, 187895, 23,
        51712, 2
    ]

    @classmethod
    def setUpClass(cls):
        cls.tokenizer: MBart50Tokenizer = MBart50Tokenizer.from_pretrained(
            cls.checkpoint_name, src_lang="en_XX", tgt_lang="ro_RO")
        cls.pad_token_id = 1
        return cls

    def check_language_codes(self):
        self.assertEqual(self.tokenizer.fairseq_tokens_to_ids["ar_AR"], 250001)
        self.assertEqual(self.tokenizer.fairseq_tokens_to_ids["en_EN"], 250004)
        self.assertEqual(self.tokenizer.fairseq_tokens_to_ids["ro_RO"], 250020)
        self.assertEqual(self.tokenizer.fairseq_tokens_to_ids["mr_IN"], 250038)

    def test_tokenizer_decode_ignores_language_codes(self):
        self.assertIn(RO_CODE, self.tokenizer.all_special_ids)
        generated_ids = [
            RO_CODE, 884, 9019, 96, 9, 916, 86792, 36, 18743, 15596, 5, 2
        ]
        result = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        expected_romanian = self.tokenizer.decode(generated_ids[1:],
                                                  skip_special_tokens=True)
        self.assertEqual(result, expected_romanian)
        self.assertNotIn(self.tokenizer.eos_token, result)

    def test_tokenizer_truncation(self):
        src_text = ["this is gunna be a long sentence " * 20]
        assert isinstance(src_text[0], str)
        desired_max_length = 10
        ids = self.tokenizer(src_text,
                             max_length=desired_max_length,
                             truncation=True).input_ids[0]
        self.assertEqual(ids[0], EN_CODE)
        self.assertEqual(ids[-1], 2)
        self.assertEqual(len(ids), desired_max_length)

    def test_mask_token(self):
        self.assertListEqual(
            self.tokenizer.convert_tokens_to_ids(["<mask>", "ar_AR"]),
            [250053, 250001])

    def test_special_tokens_unaffacted_by_save_load(self):
        tmpdirname = tempfile.mkdtemp()
        original_special_tokens = self.tokenizer.fairseq_tokens_to_ids
        self.tokenizer.save_pretrained(tmpdirname)
        new_tok = MBart50Tokenizer.from_pretrained(tmpdirname)
        self.assertDictEqual(new_tok.fairseq_tokens_to_ids,
                             original_special_tokens)

    def test_seq2seq_max_target_length(self):
        batch = self.tokenizer(self.src_text,
                               padding=True,
                               truncation=True,
                               max_length=3,
                               return_tensors="pd")
        targets = self.tokenizer(self.tgt_text,
                                 padding=True,
                                 truncation=True,
                                 max_length=10,
                                 return_tensors="pd")
        labels = targets["input_ids"]
        batch["decoder_input_ids"] = shift_tokens_right(
            labels, self.tokenizer.pad_token_id)

        self.assertEqual(batch.input_ids.shape[1], 3)
        self.assertEqual(batch.decoder_input_ids.shape[1], 10)

    def test_tokenizer_translation(self):
        inputs = self.tokenizer._build_translation_inputs("A test",
                                                          return_tensors="pd",
                                                          src_lang="en_XX",
                                                          tgt_lang="ar_AR")

        self.assertEqual(
            nested_simplify(inputs),
            {
                # en_XX, A, test, EOS
                "input_ids": [[250004, 62, 3034, 2]],
                "attention_mask": [[1, 1, 1, 1]],
                # ar_AR
                "forced_bos_token_id": 250001,
            },
        )
