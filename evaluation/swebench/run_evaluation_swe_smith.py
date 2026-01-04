import argparse
import json
import os
import shlex
import shutil
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from docker import from_env, DockerClient
from docker.errors import APIError, ImageNotFound, NotFound
from docker.models.containers import Container, ExecResult
from datasets import load_dataset
from tqdm import tqdm

AGENT_IMAGE = "pywen/agent:0.1"
AGENT_IMAGE_DOCKERFILE = "Dockerfile.pywen-agent"

AGENT_IMAGE_PATH_IN_CONTAINER = "/opt/Pywen"
UV_BIN_IN_CONTAINER = "/root/.local/bin/uv"
UV_SHARE_IN_CONTAINER = "/root/.local/share/uv"

HOST_AGENT_CACHE_DIRNAME = "pywen_agent_cache"
HOST_UV_BIN_DIRNAME = "uv_bin"
HOST_UV_SHARE_DIRNAME = "uv_share"
RESULTS_DIRNAME = "results"


def _add_swesmith_repo_to_path() -> None:
    base = Path(__file__).resolve().parents[3]
    candidate = base / "SWE-smith"
    if candidate.exists():
        sys.path.insert(0, str(candidate))


def _get_swebench_constants() -> tuple[str, str, str, str, str]:
    try:
        from swebench.harness.constants import (
            DOCKER_WORKDIR,
            DOCKER_USER,
            KEY_INSTANCE_ID,
            KEY_MODEL,
            KEY_PREDICTION,
        )
    except Exception:
        return "/testbed", "root", "instance_id", "model_name_or_path", "model_patch"
    return DOCKER_WORKDIR, DOCKER_USER, KEY_INSTANCE_ID, KEY_MODEL, KEY_PREDICTION


def _get_problem_statement(instance: dict) -> str:
    return (
        instance.get("problem_statement")
        or instance.get("issue")
        or instance.get("problem")
        or ""
    )


def _load_dataset(dataset_path: str, instance_key: str) -> list[dict]:
    if dataset_path in ("SWE-smith", "SWE-bench/SWE-smith"):
        dataset = load_dataset("SWE-bench/SWE-smith", split="train")
        return [dict(x) for x in dataset]
    if dataset_path.endswith(".json"):
        with open(dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)
    if dataset_path.endswith(".jsonl"):
        with open(dataset_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    raise ValueError(
        f"Unsupported dataset_path: {dataset_path}. Use HF dataset, .json, or .jsonl."
    )


def _filter_instance_ids(
    ids: Iterable[str],
    instance_ids: list[str] | None,
    pattern: str | None,
    limit: int | None,
) -> list[str]:
    ids_list = list(ids)
    if instance_ids:
        ids_list = [iid for iid in ids_list if iid in set(instance_ids)]
    if pattern:
        import re

        regex = re.compile(pattern)
        ids_list = [iid for iid in ids_list if regex.search(iid)]
    if limit:
        ids_list = ids_list[:limit]
    return ids_list


class DockerOps:
    def __init__(self):
        self.client: DockerClient = from_env()

    def image_exists(self, name: str) -> bool:
        try:
            self.client.images.get(name)
            return True
        except ImageNotFound:
            return False

    def pull_image(self, name: str) -> None:
        last_err = None
        for _ in range(3):
            try:
                self.client.images.pull(name)
                return
            except Exception as e:
                last_err = e
        raise RuntimeError(f"Failed to pull image {name}: {last_err}")

    def build_image(self, tag: str, dockerfile: Path, context: Path) -> None:
        try:
            stream = self.client.api.build(
                path=str(context),
                dockerfile=str(dockerfile.name),
                tag=tag,
                decode=True,
                rm=True,
            )
            for chunk in stream:
                if "error" in chunk:
                    raise RuntimeError(chunk["error"])
        except APIError as e:
            raise RuntimeError(f"Docker build failed: {e}")

    def run_container(
        self,
        image: str,
        *,
        command: str = "/bin/bash",
        detach: bool = True,
        tty: bool = False,
        stdin_open: bool = True,
        environment: dict | None = None,
        volumes: dict | None = None,
        working_dir: str | None = None,
    ) -> Container:
        try:
            container = self.client.containers.run(
                image=image,
                command=command,
                detach=detach,
                tty=tty,
                stdin_open=stdin_open,
                environment=environment or {},
                volumes=volumes or {},
                working_dir=working_dir,
            )
            return container
        except Exception as e:
            raise RuntimeError(f"Failed to run container from {image}: {e}")

    def exec_sh(
        self,
        container: Container,
        shell_cmd: str,
        check: bool = True,
        user: str | None = None,
        workdir: str | None = None,
    ) -> str:
        try:
            res: ExecResult = container.exec_run(
                cmd=["/bin/bash", "--noprofile", "--norc", "-lc", shell_cmd],
                tty=False,
                stdin=False,
                user=user,
                workdir=workdir,
            )
        except Exception as e:
            raise RuntimeError(f"Exec failed: {shell_cmd}\n{e}")

        code = getattr(res, "exit_code", None)
        out = getattr(res, "output", None)
        if code is None:
            code = res[0]
            out = res[1]
        text = out.decode("utf-8", "ignore") if isinstance(out, (bytes, bytearray)) else (out or "")

        if check and code != 0:
            raise RuntimeError(f"Command failed ({code}): {shell_cmd}\n{text}")
        return text

    def stop_and_remove(self, container: Container | None) -> None:
        if container is None:
            return
        try:
            container.stop(timeout=5)
        except Exception:
            pass
        try:
            container.remove()
        except Exception:
            pass

    def cp_from_container(self, container: Container, src_path: str, dst: Path) -> None:
        dst.mkdir(parents=True, exist_ok=True)
        try:
            stream, _ = container.get_archive(src_path)
        except NotFound:
            raise RuntimeError(f"Path not found in container: {src_path}")
        import tarfile
        import io as _io

        bio = _io.BytesIO()
        for chunk in stream:
            bio.write(chunk)
        bio.seek(0)
        with tarfile.open(fileobj=bio, mode="r|*") as tar:
            tar.extractall(path=dst)


@dataclass
class SweSmithConfig:
    dataset_path: str
    run_id: str
    max_workers: int
    instance_ids: list[str] | None
    pattern: str | None
    limit: int | None
    force: bool
    agent_name: str
    config_path: str
    mode: str
    evaluate: bool
    eval_workers: int


class SweSmithRunner:
    def __init__(self, cfg: SweSmithConfig):
        _add_swesmith_repo_to_path()
        try:
            from swesmith.constants import HF_DATASET
        except Exception as e:
            raise RuntimeError(f"Failed to import SWE-smith (swesmith): {e}")

        (
            self.docker_workdir,
            self.docker_user,
            self.key_instance_id,
            self.key_model,
            self.key_prediction,
        ) = _get_swebench_constants()

        self.ops = DockerOps()
        self.cfg = cfg
        self.dataset_path = cfg.dataset_path or HF_DATASET
        self.dataset = _load_dataset(self.dataset_path, self.key_instance_id)

        eval_dir = Path(__file__).parent.resolve()
        self.working_dir = (eval_dir / "pywen_workspace").resolve()
        self.working_dir.mkdir(parents=True, exist_ok=True)

        self.config_src = self._resolve_config_path(Path(cfg.config_path))
        self.config_dest = self.working_dir / f"pywen_config{self.config_src.suffix}"
        shutil.copy(self.config_src, self.config_dest)

        self.results_dir = (eval_dir / RESULTS_DIRNAME).resolve()
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.task_id = f"SWE-smith_{cfg.run_id}".replace("/", "_")
        self.task_results_dir = self.results_dir / self.task_id
        self.task_results_dir.mkdir(parents=True, exist_ok=True)

        all_ids = [item[self.key_instance_id] for item in self.dataset]
        if cfg.instance_ids is not None:
            filtered_ids = [iid for iid in all_ids if iid in set(cfg.instance_ids)]
        else:
            filtered_ids = _filter_instance_ids(all_ids, None, cfg.pattern, cfg.limit)
        self.instance_ids = filtered_ids

        self.host_agent_cache = self.working_dir / HOST_AGENT_CACHE_DIRNAME
        self.host_agent_cache.mkdir(parents=True, exist_ok=True)
        self.skip_completed = not cfg.force

    def _resolve_config_path(self, user_path: Path) -> Path:
        if user_path and user_path.exists():
            return user_path
        home_default = Path.home() / ".pywen" / "pywen_config.yaml"
        if home_default.exists():
            return home_default
        repo_root = Path(__file__).parents[2].resolve()
        example = repo_root / "pywen_config.example.yaml"
        if example.exists():
            return example
        raise FileNotFoundError(
            "No YAML config found. Expected one of: "
            f"{user_path} (from --config) OR {home_default} OR {example}"
        )

    def ensure_agent_image_and_cache(self) -> None:
        repo_root = Path(__file__).parents[2].resolve()
        dockerfile = repo_root / AGENT_IMAGE_DOCKERFILE
        if not dockerfile.exists():
            raise FileNotFoundError(f"Missing {AGENT_IMAGE_DOCKERFILE} at repo root: {dockerfile}")

        if not self.ops.image_exists(AGENT_IMAGE):
            print(f"Building agent image {AGENT_IMAGE} ...")
            self.ops.build_image(tag=AGENT_IMAGE, dockerfile=dockerfile, context=repo_root)
        else:
            print(f"Found agent image {AGENT_IMAGE}")

        target_pywen = self.host_agent_cache / "Pywen"
        target_uv_bin = self.host_agent_cache / HOST_UV_BIN_DIRNAME
        target_uv_share = self.host_agent_cache / HOST_UV_SHARE_DIRNAME

        need_export = not (
            target_pywen.exists()
            and (target_uv_bin / "uv").exists()
            and (target_uv_share / "uv").exists()
        )
        if need_export:
            print("Exporting agent cache from image ...")
            container = None
            try:
                container = self.ops.run_container(AGENT_IMAGE, command="sleep 60")
                shutil.rmtree(target_pywen, ignore_errors=True)
                shutil.rmtree(target_uv_bin, ignore_errors=True)
                shutil.rmtree(target_uv_share, ignore_errors=True)
                self.ops.cp_from_container(container, AGENT_IMAGE_PATH_IN_CONTAINER, self.host_agent_cache)
                target_uv_bin.mkdir(parents=True, exist_ok=True)
                self.ops.cp_from_container(container, UV_BIN_IN_CONTAINER, target_uv_bin)
                target_uv_share.mkdir(parents=True, exist_ok=True)
                self.ops.cp_from_container(container, UV_SHARE_IN_CONTAINER, target_uv_share)
            finally:
                self.ops.stop_and_remove(container)
        else:
            print("Found existing host cache of pywen-agent. Skipping export.")

    def _instance_completed(self, instance_id: str) -> bool:
        instance_res_dir = self.task_results_dir / instance_id
        patch_path = instance_res_dir / f"{instance_id}.patch"
        log_path = instance_res_dir / "run.log"
        if patch_path.exists() and patch_path.stat().st_size > 0:
            return True
        if log_path.exists() and log_path.stat().st_size > 0:
            return True
        return False

    def _get_instance(self, instance_id: str) -> dict | None:
        for inst in self.dataset:
            if inst[self.key_instance_id] == instance_id:
                return inst
        return None

    def run_one_instance(self, instance_id: str) -> None:
        if self.skip_completed and self._instance_completed(instance_id):
            print(f"⏭️  Skipping {instance_id} (already completed)")
            return

        instance = self._get_instance(instance_id)
        if not instance:
            print(f"Instance {instance_id} not found")
            return

        try:
            from swesmith.profiles import registry
        except Exception as e:
            print(f"Failed to import SWE-smith profiles: {e}")
            return

        try:
            profile = registry.get_from_inst(instance)
        except Exception as e:
            print(f"Failed to resolve SWE-smith profile for {instance_id}: {e}")
            return

        image_name = profile.image_name
        if not self.ops.image_exists(image_name):
            print(f"Pulling {image_name} ...")
            self.ops.pull_image(image_name)

        instance_res_dir = self.task_results_dir / instance_id
        instance_res_dir.mkdir(parents=True, exist_ok=True)
        _write_problem_statement(instance_res_dir, _get_problem_statement(instance))

        host_pywen = (self.host_agent_cache / "Pywen").resolve()
        host_uv_bin = (self.host_agent_cache / HOST_UV_BIN_DIRNAME).resolve()
        host_uv_share = (self.host_agent_cache / HOST_UV_SHARE_DIRNAME).resolve()

        volumes = {
            str(instance_res_dir): {"bind": "/results", "mode": "rw"},
            str(host_pywen): {"bind": AGENT_IMAGE_PATH_IN_CONTAINER, "mode": "ro"},
            str(host_uv_bin): {"bind": "/root/.local/bin", "mode": "ro"},
            str(host_uv_share): {"bind": "/root/.local/share", "mode": "ro"},
        }
        environment = {"PATH": "/root/.local/bin:" + os.environ.get("PATH", "")}

        container = None
        try:
            container = self.ops.run_container(
                image=image_name,
                command="/bin/bash",
                environment=environment,
                volumes=volumes,
                tty=True,
                stdin_open=True,
            )

            self.ops.exec_sh(
                container,
                f"git checkout {shlex.quote(instance_id)}",
                user=self.docker_user,
                workdir=self.docker_workdir,
            )

            problem_stmt_path = instance_res_dir / "problem_statement.txt"
            problem_stmt = problem_stmt_path.read_text(encoding="utf-8") if problem_stmt_path.exists() else ""
            prompt = (
                "You are tasked with resolving a GitHub issue in this repository.\n"
                f"The repository is checked out at {self.docker_workdir}.\n\n"
                "IMPORTANT: You must analyze the issue, locate the relevant code files, make the necessary changes to fix the bug, "
                "and verify your solution. Do not just acknowledge the task - you must actually implement the fix.\n\n"
                f"Issue Description:\n{problem_stmt}\n\n"
                "Please start by exploring the codebase to understand the issue, then implement the fix.\n"
            )
            (instance_res_dir / "problem_statement_for_pywen.txt").write_text(prompt, encoding="utf-8")

            instance_cfg = instance_res_dir / self.config_dest.name
            shutil.copy(self.config_dest, instance_cfg)

            quoted_prompt = shlex.quote(prompt)
            run_cmd = (
                f"cd {shlex.quote(self.docker_workdir)} && "
                "if [ -f /opt/miniconda3/etc/profile.d/conda.sh ]; then "
                "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed; "
                "fi && "
                f"{AGENT_IMAGE_PATH_IN_CONTAINER}/.venv/bin/pywen "
                f"--config /results/{self.config_dest.name} "
                "--permission-mode yolo "
                f"--agent {shlex.quote(self.cfg.agent_name)} "
                f"-p {quoted_prompt}"
            )
            output = self.ops.exec_sh(
                container,
                run_cmd,
                check=False,
                user=self.docker_user,
                workdir=self.docker_workdir,
            )
            (instance_res_dir / "run.log").write_text(output, encoding="utf-8")

            try:
                home_dir = self.ops.exec_sh(container, "echo $HOME", user=self.docker_user).strip()
                if not home_dir:
                    home_dir = "/root" if self.docker_user == "root" else f"/home/{self.docker_user}"
                traj_src = f"{home_dir}/.pywen/trajectories"
                trajectories_dir = instance_res_dir / "trajectories"
                self.ops.cp_from_container(container, traj_src, trajectories_dir)
            except RuntimeError:
                pass

            patch_out = self.ops.exec_sh(
                container,
                f"cd {shlex.quote(self.docker_workdir)} && git diff",
                check=False,
                user=self.docker_user,
                workdir=self.docker_workdir,
            )
            if patch_out.strip():
                (instance_res_dir / f"{instance_id}.patch").write_text(patch_out, encoding="utf-8")
                print(f"✅ Patch saved: {instance_id}")
            else:
                print(f"⚠️  No patch generated for {instance_id}")
        except Exception as e:
            print(f"Error running {instance_id}: {e}")
            traceback.print_exc()
        finally:
            self.ops.stop_and_remove(container)

    def run_all(self) -> None:
        self.ensure_agent_image_and_cache()
        if self.skip_completed:
            completed = [iid for iid in self.instance_ids if self._instance_completed(iid)]
            remaining = [iid for iid in self.instance_ids if not self._instance_completed(iid)]
            print(f"📊 Status: {len(completed)} completed, {len(remaining)} remaining (total: {len(self.instance_ids)})")
            if len(remaining) == 0:
                print("✅ All instances already completed!")
                return

        with ThreadPoolExecutor(max_workers=self.cfg.max_workers) as executor:
            futures = {executor.submit(self.run_one_instance, iid): iid for iid in self.instance_ids}
            for fut in tqdm(as_completed(futures), total=len(futures)):
                iid = futures[fut]
                try:
                    fut.result()
                except Exception as e:
                    print(f"Instance {iid} crashed: {e}")

    def collect_predictions(self, instance_ids: list[str] | None = None) -> Path:
        if instance_ids is None:
            instance_ids = self.instance_ids

        predictions = []
        missing_count = 0
        for instance_id in instance_ids:
            patch_path = self.task_results_dir / instance_id / f"{instance_id}.patch"
            if not patch_path.exists():
                missing_count += 1
                continue
            patch_content = patch_path.read_text(encoding="utf-8")
            if not patch_content.strip():
                missing_count += 1
                continue
            predictions.append(
                {
                    self.key_instance_id: instance_id,
                    self.key_model: self.cfg.run_id,
                    self.key_prediction: patch_content,
                    "patch": patch_content,
                }
            )

        # Convert list to dict for SWE-smith eval compatibility
        # The eval harness expects .json files to be dict format
        predictions_dict = {pred[self.key_instance_id]: pred for pred in predictions}
        
        predictions_path = self.task_results_dir / "predictions.json"
        with open(predictions_path, "w", encoding="utf-8") as f:
            json.dump(predictions_dict, f, indent=2, ensure_ascii=False)
        print(f"✅ Collected {len(predictions)} patches into {predictions_path}")
        if missing_count > 0:
            print(f"⚠️  {missing_count} instances have no patch")
        return predictions_path

    def evaluate(self, predictions_path: Path, instance_ids: list[str] | None = None) -> None:
        try:
            from swesmith.harness import eval as swe_eval
            from swesmith.constants import HF_DATASET
        except Exception as e:
            raise RuntimeError(f"Failed to import SWE-smith eval harness: {e}")

        dataset_path = self.dataset_path
        if dataset_path in ("SWE-smith", "SWE-bench/SWE-smith"):
            dataset_path = HF_DATASET

        swe_eval.main(
            run_id=self.cfg.run_id,
            workers=self.cfg.eval_workers,
            predictions_path=str(predictions_path),
            dataset_path=dataset_path,
            instance_ids=instance_ids,
            redo_existing=False,
            report_only=False,
            f2p_only=False,
        )


def _write_problem_statement(instance_dir: Path, content: str) -> int:
    with open(instance_dir / "problem_statement.txt", "w", encoding="utf-8") as f:
        return f.write(content)


def parse_args() -> SweSmithConfig:
    parser = argparse.ArgumentParser(
        description="Run Pywen on SWE-smith instances using official SWE-smith Docker images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-path",
        default="SWE-bench/SWE-smith",
        help="Dataset path: HF dataset name, .json, or .jsonl",
    )
    parser.add_argument("--instance-ids", nargs="+", help="Specific instance IDs to run")
    parser.add_argument("--pattern", type=str, default=None, help="Regex to filter instance_id")
    parser.add_argument("--limit", type=int, default=None, help="Only run first N instances")
    parser.add_argument("--max-workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--config", type=str, default=str(Path.home() / ".pywen/pywen_config.yaml"))
    parser.add_argument("--agent", default="pywenswe", help="Agent to use")
    parser.add_argument("--run-id", type=str, default="pywen-agent", help="Run identifier")
    parser.add_argument(
        "--mode",
        type=str,
        default="expr",
        choices=["expr", "collect", "eval", "e2e"],
        help="Mode: expr=generate patches, collect=only predictions.json, eval=only evaluate, e2e=expr+collect+eval",
    )
    parser.add_argument("--force", action="store_true", help="Force re-run all instances")
    parser.add_argument("--eval-workers", type=int, default=4, help="Workers for SWE-smith eval")
    args = parser.parse_args()

    return SweSmithConfig(
        dataset_path=args.dataset_path,
        run_id=args.run_id,
        max_workers=args.max_workers,
        instance_ids=args.instance_ids,
        pattern=args.pattern,
        limit=args.limit,
        force=args.force,
        agent_name=args.agent,
        config_path=args.config,
        mode=args.mode,
        evaluate=args.mode in ("eval", "e2e"),
        eval_workers=args.eval_workers,
    )


def main() -> None:
    cfg = parse_args()
    runner = SweSmithRunner(cfg)

    if cfg.instance_ids is None and cfg.mode != "collect":
        print("⚠️  Warning: No --instance-ids specified. Will run ALL instances in dataset.")
        print(f"   Dataset: {cfg.dataset_path}")
        resp = input("   Continue? [y/N]: ")
        if resp.lower() != "y":
            print("Aborted.")
            return

    if cfg.mode in ("expr", "e2e"):
        runner.run_all()

    predictions_path = runner.task_results_dir / "predictions.json"
    if cfg.mode in ("collect", "e2e"):
        predictions_path = runner.collect_predictions(cfg.instance_ids)

    if cfg.mode == "eval" and not predictions_path.exists():
        predictions_path = runner.collect_predictions(cfg.instance_ids)

    if cfg.evaluate:
        runner.evaluate(predictions_path, cfg.instance_ids)


if __name__ == "__main__":
    main()
