from pathlib import Path
from typing import List

import pytest

from scaling.core import RunnerConfig, runner_main


@pytest.mark.short
@pytest.mark.parametrize(
    "hosts,expected_world_size",
    [
        (["localhost"], 8),
        (["localhost slots=8"], 8),
        (["localhost slots=0,"], 1),
        (["localhost slots=0,1,2"], 3),
    ],
)
@pytest.mark.parametrize("use_hostsfile", [True, False])
def test_should_run_runner(tmp_path: Path, hosts: List[str], expected_world_size: int, use_hostsfile: bool):
    if use_hostsfile:
        hostsfile = tmp_path / "hostfile"
        with open(hostsfile, "w", encoding="UTF-8") as f:
            for host in hosts:
                f.write(f"{host}\n")
        _hosts = None
    else:
        hostsfile = None
        _hosts = hosts

    config = RunnerConfig.from_dict(
        {
            "runner_type": "pdsh",
            "hostsfile": hostsfile,
            "hosts": _hosts,
            "master_port": 29500,
            "master_addr": None,
            "script": str(Path(__file__).parent.absolute() / "runner_script.py"),
            "docker_config": {
                "docker_container": "container_name",
                "docker_sudo": False,
                "docker_mounts": None,
            },
        }
    )
    returncode = runner_main(
        config=config,
        payload={"testcase": "test_should_run_runner", "cache_dir": str(tmp_path)},
    )

    assert returncode == 0
    process_outputs = list(tmp_path.glob("*.json"))
    assert len(process_outputs) == expected_world_size, "did not write process output"


@pytest.mark.short
def test_create_runner_config_with_legacy_field_names(tmp_path: Path):
    hostsfile = tmp_path / "hostfile"
    with open(hostsfile, "w", encoding="UTF-8") as f:
        f.write("localhost")
    config = RunnerConfig.from_dict(
        {
            "runner_type": "pdsh",
            "hostfile": hostsfile,
            "hosts": ["localhost"],
            "master_port": 29500,
            "master_addr": None,
            "script": str(Path(__file__).parent.absolute() / "runner_script.py"),
            "docker_config": {
                "docker_container": "container_name",
                "docker_sudo": False,
                "docker_mounts": None,
            },
        }
    )
    assert config.hostsfile == hostsfile
