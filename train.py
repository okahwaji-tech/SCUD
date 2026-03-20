"""Backward-compatible training entry point. Use `scud-train` CLI instead."""

from scud.cli import train

if __name__ == "__main__":
    train()
