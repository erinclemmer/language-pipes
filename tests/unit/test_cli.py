import argparse
from pathlib import Path

import pytest
import toml

import language_pipes.cli as cli


@pytest.fixture()
def parser():
    return cli.build_parser()


class DummyLanguagePipes:
    def __init__(self, version, config):
        self.version = version
        self.config = config


def test_keygen_creates_file(tmp_path):
    key_path = tmp_path / "network.key"
    cli.main(["keygen", str(key_path)])
    assert key_path.exists()
    assert key_path.stat().st_size > 0


def test_keygen_overwrites_existing(tmp_path):
    key_path = tmp_path / "network.key"
    key_path.write_bytes(b"old content")
    cli.main(["keygen", str(key_path)])
    assert key_path.read_bytes() != b"old content"


def test_version_flag_short():
    with pytest.raises(SystemExit) as exc:
        cli.main(["-V"])
    assert exc.value.code == 0


def test_version_flag_long():
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0


def test_version_constant_format():
    parts = cli.VERSION.split(".")
    assert len(parts) >= 2
    assert all(part.isdigit() for part in parts)


def test_parser_has_subcommands(parser):
    args = parser.parse_args(["keygen", "test.key"])
    assert args.command == "keygen"
    assert args.output == "test.key"

    args = parser.parse_args(["serve", "--node-id", "test"])
    assert args.command == "serve"
    assert args.node_id == "test"

    args = parser.parse_args(["init"])
    assert args.command == "init"


def test_apply_overrides_requires_hosted_models(parser, capsys):
    args = parser.parse_args(["serve", "--node-id", "node-1"])
    with pytest.raises(SystemExit):
        cli.apply_overrides({}, args)
    captured = capsys.readouterr()
    assert "hosted_models" in captured.out


def test_apply_overrides_requires_node_id(parser, capsys):
    args = parser.parse_args([
        "serve",
        "--hosted-models",
        "id=Qwen/Qwen3-1.7B,device=cpu,memory=4",
    ])
    with pytest.raises(SystemExit):
        cli.apply_overrides({}, args)
    captured = capsys.readouterr()
    assert "node_id" in captured.out


def test_apply_overrides_parses_hosted_models(parser):
    args = parser.parse_args([
        "serve",
        "--node-id", "node-1",
        "--hosted-models",
        "id=Qwen/Qwen3-1.7B,device=cpu,memory=4,load_ends=true",
    ])
    config = cli.apply_overrides({}, args)
    assert config["hosted_models"] == [
        {
            "id": "Qwen/Qwen3-1.7B",
            "device": "cpu",
            "max_memory": 4.0,
            "load_ends": True,
        }
    ]


def test_main_serve_loads_config_without_model_init(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "LanguagePipes", DummyLanguagePipes)

    config_path = tmp_path / "config.toml"
    toml.dump(
        {
            "node_id": "node-1",
            "hosted_models": [
                {
                    "id": "Qwen/Qwen3-1.7B",
                    "device": "cpu",
                    "max_memory": 4,
                    "load_ends": False,
                }
            ],
        },
        config_path.open("w", encoding="utf-8"),
    )

    instance = cli.main(["serve", "--config", str(config_path)])
    assert isinstance(instance, DummyLanguagePipes)
    assert instance.config.router.node_id == "node-1"


def test_apply_overrides_errors_on_bad_hosted_models(parser):
    args = parser.parse_args([
        "serve",
        "--node-id", "node-1",
        "--hosted-models",
        "id=Qwen/Qwen3-1.7B,device=cpu",
    ])
    with pytest.raises(ValueError):
        cli.apply_overrides({}, args)
