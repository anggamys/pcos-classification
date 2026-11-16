"""Test pemilihan device: GPU default, fallback CPU, dan jalur gagal.

Jalankan dari root repo:

    python -m unittest tests.test_device -v
"""

import sys
import unittest
from argparse import Namespace
from unittest import mock

import torch

from src.cli import build_config, parse_args
from src.device import DEVICE_CHOICES, resolve_device


class ResolveDeviceTests(unittest.TestCase):
    def test_auto_prefers_cuda_when_available(self) -> None:
        with mock.patch.object(torch.cuda, "is_available", return_value=True):
            self.assertEqual(resolve_device("auto").type, "cuda")

    def test_auto_falls_back_to_cpu_without_cuda(self) -> None:
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            self.assertEqual(resolve_device("auto").type, "cpu")

    def test_explicit_cpu_ignores_available_cuda(self) -> None:
        with mock.patch.object(torch.cuda, "is_available", return_value=True):
            self.assertEqual(resolve_device("cpu").type, "cpu")

    def test_explicit_cuda_requires_gpu(self) -> None:
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=False),
            self.assertRaises(RuntimeError) as ctx,
        ):
            resolve_device("cuda")

        self.assertIn("--device cpu", str(ctx.exception))

    def test_unknown_preference_is_rejected(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            resolve_device("tpu")

        self.assertIn("auto", str(ctx.exception))
        self.assertEqual(DEVICE_CHOICES, ("auto", "cuda", "cpu"))


class DeviceCliTests(unittest.TestCase):
    def _parse(self, argv: list[str]) -> Namespace:
        with mock.patch.object(sys, "argv", ["main.py", *argv]):
            return parse_args()

    def test_default_device_is_auto(self) -> None:
        args = self._parse([])

        self.assertEqual(args.device, "auto")
        self.assertEqual(build_config(args)["device"], "auto")

    def test_explicit_device_reaches_config(self) -> None:
        args = self._parse(["--device", "cuda"])

        self.assertEqual(build_config(args)["device"], "cuda")

    def test_invalid_device_is_rejected(self) -> None:
        with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
            self._parse(["--device", "tpu"])


if __name__ == "__main__":
    unittest.main()
