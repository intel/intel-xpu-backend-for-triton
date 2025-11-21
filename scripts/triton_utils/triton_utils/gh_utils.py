from __future__ import annotations

from abc import abstractmethod
from typing import Any, Iterator, Optional, TypeVar, Type
from dataclasses import dataclass, asdict, fields

import os
import subprocess
from pathlib import Path
import shutil
import time

import json

from pprint import pprint


class CLIUtils:

    @classmethod
    def run(  #pylint: disable=R0913, R0917
        cls,
        cmd: list[str],
        check: bool = True,
        retries: int = 0,
        retry_backoff_seconds: float = 1.0,
        capture_output: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        attempt = 0
        run_res: subprocess.CompletedProcess[str] | None = None
        while attempt <= retries:
            run_res = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if run_res.returncode == 0:
                break
            if attempt < retries:
                sleep_time = retry_backoff_seconds * (2**attempt)
                print(f"[retry] Command failed (attempt {attempt + 1}/{retries + 1}) exit={run_res.returncode}."
                      f" Retrying in {sleep_time:.1f}s...")
                time.sleep(sleep_time)
                attempt += 1
                continue
            if run_res.returncode != 0:
                raise ValueError(
                    f"Command '{' '.join(cmd)}' failed with exit code {run_res.returncode}.\nstdout: {run_res.stdout}\nstderr: {run_res.stderr}"
                )
        assert run_res is not None
        if check and run_res.returncode != 0:
            raise ValueError(
                f"Command '{' '.join(cmd)}' failed with exit code {run_res.returncode}.\nstdout: {run_res.stdout}\nstderr: {run_res.stderr}"
            )
        if capture_output:
            if len(cmd) >= 3 and cmd[1] == "api":
                with open("".join(cmd[2]).replace("/", "\\"), "w", encoding="utf-8") as f:
                    f.write(run_res.stdout)
        return run_res

    @classmethod
    def ensure_gh_available(cls) -> None:
        cls.run(["gh", "auth", "status"], check=True)

    @classmethod
    def gh_json(cls, args: list[str], concatenated_jsons: bool = False) -> Any:

        def _iter_concatenated_jsons(s: str) -> Iterator[Any]:
            decoder = json.JSONDecoder()
            i = 0
            n = len(s)
            while i < n:
                # skip whitespace between JSON documents
                while i < n and s[i].isspace():
                    i += 1
                if i >= n:
                    break
                obj, end = decoder.raw_decode(s, i)
                yield obj
                i = end

        if concatenated_jsons:
            raw_output = cls.run(["gh", *args]).stdout
            outputs = list(_iter_concatenated_jsons(raw_output))
            return outputs
        return json.loads(cls.run(["gh", *args]).stdout)


T = TypeVar("T", bound="FromDictMixin")


@dataclass
class FromDictMixin:

    @classmethod
    def from_dict(cls: Type[T], data: dict[str, Any]) -> T:
        field_names = {f.name for f in fields(cls)}
        kwargs: dict[str, Any] = {k: v for k, v in data.items() if k in field_names}
        return cls(**kwargs)  # type: ignore[arg-type]


@dataclass
class GHAObj(FromDictMixin):
    id: str
    name: str
    path: str
    repo: str
    branch: str


@dataclass
class GHArtifact(FromDictMixin):
    name: str
    size_in_bytes: int
    expired: bool
    created_at: str
    workflow_run_id: str
    repo: str

    def download_artifact(self, dest_dir: Path) -> None:
        artifact_dir = dest_dir / self.name
        artifact_dir.mkdir(parents=True, exist_ok=True)
        CLIUtils.run(
            [
                "gh",
                "run",
                "download",
                str(self.workflow_run_id),
                "-R",
                self.repo,
                "-D",
                str(artifact_dir),
                "--pattern",
                self.name,
            ],
            check=True,
            retries=3,
            retry_backoff_seconds=1.5,
        )
        src_dir = artifact_dir / self.name
        for file_item in src_dir.iterdir():
            shutil.move(str(file_item), str(artifact_dir / file_item.name))
        src_dir.rmdir()


@dataclass
class GHAWorkflow(GHAObj):

    def __str__(self):
        return f"workflow: '{self.name}'('{self.path}')\nid: {self.id}\nrepo: '{self.repo}'\nbranch: '{self.branch}'\n"

    def get_latest_success_run(self) -> GHAWorkflowRun:
        data = CLIUtils.gh_json([
            "api",
            f"repos/{self.repo}/actions/workflows/{self.id}/runs",
            "-H",
            "Accept: application/vnd.github+json",
            "--method",
            "GET",
            "--field",
            f"branch={self.branch}",
            "--field",
            "status=completed",
            "--field",
            "per_page=50",
        ])
        runs = data["workflow_runs"]
        for run_obj in runs:
            if run_obj["conclusion"] == "success":
                return GHAWorkflowRun.from_dict({"repo": self.repo, "branch": self.branch} | run_obj)
        raise RuntimeError(f"No successful completed runs found for workflow_id={self.id} on branch='{self.branch}'.")

    @classmethod
    def from_workflow_path(cls, repo: str, branch: str, wf_path: str) -> GHAWorkflow:
        data = CLIUtils.gh_json([
            "api",
            f"repos/{repo}/actions/workflows",
            "-H",
            "Accept: application/vnd.github+json",
        ])
        workflows = data["workflows"]
        matches = [wf for wf in workflows if wf["path"] == wf_path]
        if not matches:
            raise RuntimeError(f"Workflow '{wf_path}' not found in repo {repo}.")
        return GHAWorkflow(
            id=matches[0]["id"],
            name=matches[0]["name"],
            repo=repo,
            path=wf_path,
            branch=branch,
        )


@dataclass
class GHAWorkflowRun(GHAObj):
    html_url: str
    head_sha: str
    run_number: str
    created_at: str
    updated_at: str

    def get_runs_for_workflow(self, wf: GHAWorkflow) -> list[GHAWorkflowRun]:
        data = CLIUtils.gh_json([
            "api",
            f"repos/{self.repo}/actions/workflows/{wf.id}/runs",
            "-H",
            "Accept: application/vnd.github+json",
            "--method",
            "GET",
            "--field",
            f"head_sha={self.head_sha}",
            "--field",
            "status=completed",
            "--field",
            "per_page=250",
        ])
        runs_data: list[dict[str, Any]] = data["workflow_runs"]
        runs: list[GHAWorkflowRun] = []
        for run_obj in runs_data:
            if run_obj.get("conclusion") == "success":
                run_obj_dict: dict[str, Any] = run_obj  # hint for type checker
                runs.append(GHAWorkflowRun.from_dict({"repo": self.repo, "branch": self.branch} | run_obj_dict))
        if len(runs) == 0:
            raise RuntimeError(
                f"No successful completed runs found for workflow_id={self.id} on branch='{self.branch}'.")
        return runs

    def get_latest_success_run_for_workflow(self, wf: GHAWorkflow) -> GHAWorkflowRun:
        return self.get_runs_for_workflow(wf)[0]

    @classmethod
    def from_run_id(cls, repo: str, run_id: str | None) -> GHAWorkflowRun:
        if run_id is None:
            raise ValueError("run_id must be provided.")
        data: dict[str, Any] = CLIUtils.gh_json(
            ["api", f"repos/{repo}/actions/runs/{run_id}", "-H", "Accept: application/vnd.github+json"])
        # Extract branch from the head_branch field
        branch = data.get("head_branch", "")
        data_dict: dict[str, Any] = data  # explicit type
        return GHAWorkflowRun.from_dict({"repo": repo, "branch": branch} | data_dict)

    def gh_af_from_dict(self, af_dict: dict[str, Any]) -> GHArtifact:
        return GHArtifact.from_dict(af_dict | {"workflow_run_id": self.id, "repo": self.repo})

    @property
    def artifacts(self) -> list[GHArtifact]:
        data = CLIUtils.gh_json([
            "api",
            f"repos/{self.repo}/actions/runs/{self.id}/artifacts",
            "-H",
            "Accept: application/vnd.github+json",
            "--paginate",
        ], concatenated_jsons=True)
        artifacts_data: list[dict[str, Any]] = []
        for page_data in data:  # each page_data is a dict with an "artifacts" list
            artifacts_data.extend(page_data["artifacts"])
        if not artifacts_data:
            raise ValueError(f"[info] No artifacts found for run {self.id}")
        return [self.gh_af_from_dict(af_dict) for af_dict in artifacts_data]

    def download_artifacts(self, dest_dir: Path) -> None:
        dest_dir.mkdir(parents=True, exist_ok=True)
        # gh run download downloads into CWD or given dir; we specify -D
        CLIUtils.run(
            [
                "gh",
                "run",
                "download",
                str(self.id),
                "-R",
                self.repo,
                "-D",
                str(dest_dir),
            ],
            check=True,
        )


@dataclass
class GHTestReportProcessor:
    download_dir: Path
    repo: str
    branch: str

    @abstractmethod
    def get_artifacts(self) -> list[GHArtifact]:
        pass

    @classmethod
    def is_test_report(cls, artifact: GHArtifact) -> bool:
        return (artifact.name.startswith("test-reports") and not artifact.name.endswith("lts"))

    @classmethod
    def print_artifacts(cls, header: str, artifacts: list[GHArtifact]) -> None:
        print(header)
        for af in artifacts:
            print(f"Name: {af.name}, size: {af.size_in_bytes} bytes")

    def get_test_report_artifacts(self) -> list[GHArtifact]:
        artifacts = self.get_artifacts()
        self.print_artifacts("\nArtifacts:\n", artifacts)
        test_artifacts = [af for af in self.get_artifacts() if self.is_test_report(af)]
        self.print_artifacts("\nTest report artifacts:\n", test_artifacts)
        print(f"\nTotal artifacts: {len(artifacts)}, test report artifacts: {len(test_artifacts)}\n")
        if len(test_artifacts) == 0:
            raise ValueError("[info] No test report artifacts found for run")
        return test_artifacts

    def download_test_reports(self, ) -> None:
        os.makedirs(self.download_dir, exist_ok=True)
        if any(self.download_dir.iterdir()):
            raise ValueError(f"Report download directory {self.download_dir} is not empty.")
        test_report_artifacts = self.get_test_report_artifacts()
        print("Donloading test report artifacts:\n")
        for test_af in test_report_artifacts:
            print(f"Name: {test_af.name}, size: {test_af.size_in_bytes} bytes")
            test_af.download_artifact(self.download_dir)


@dataclass
class GHABuildTestReportProcessor(GHTestReportProcessor):
    gh_run_id: str | None = "18661995769"

    def get_artifacts(self) -> list[GHArtifact]:
        return GHAWorkflowRun.from_run_id(repo=self.repo, run_id=self.gh_run_id).artifacts


@dataclass
class GHANightlyTestReportProcessor(GHTestReportProcessor):
    nightly_wheels_wf_path: str = ".github/workflows/nightly-wheels.yml"
    build_and_test_wf_path = ".github/workflows/build-test.yml"
    gh_run_id: Optional[str] = "18568696494"

    def get_artifacts(self) -> list[GHArtifact]:
        nightly_wheels_wf = GHAWorkflow.from_workflow_path(repo=self.repo, wf_path=self.nightly_wheels_wf_path,
                                                           branch=self.branch)
        print(nightly_wheels_wf)

        if self.gh_run_id:
            nightly_wheels_run = GHAWorkflowRun.from_run_id(repo=self.repo, run_id=self.gh_run_id)
        else:
            nightly_wheels_run = nightly_wheels_wf.get_latest_success_run()
            print(f"Latest successful run for {nightly_wheels_wf.name}:")

        pprint(asdict(nightly_wheels_run), sort_dicts=False, width=80)
        print("\nArtifacts:\n")
        for nw_af in nightly_wheels_run.artifacts:
            print(f"Name: {nw_af.name}, size: {nw_af.size_in_bytes} bytes")

        build_and_test_wf = GHAWorkflow.from_workflow_path(repo=self.repo, branch=self.branch,
                                                           wf_path=self.build_and_test_wf_path)
        print(build_and_test_wf)

        build_and_test_wf_run = nightly_wheels_run.get_latest_success_run_for_workflow(build_and_test_wf)
        print(f"Latest successful run for {build_and_test_wf.name} and commit id {nightly_wheels_run.head_sha}:")
        pprint(asdict(build_and_test_wf_run), sort_dicts=False, width=80)
        return build_and_test_wf_run.artifacts

        # TODO:   # pylint: disable=fixme
        # Show runner labels which produced artifact
        # Print when commit id was merged into the branch
        # def get_commit_merge_info_via_gh(repo: str, commit_sha: str) -> None:
        #     """
        #     Try to identify when a commit was merged by looking up PR info via gh CLI.
        #     """
        #     data = gh_json([
        #         "pr", "list",
        #         "-R", repo,
        #         "--state", "merged",
        #         "--search", commit_sha,
        #         "--json", "number,mergedAt,mergeCommit"
        #     ])
        #     if not data:
        #         print(f"[warn] No merged PR found containing commit {commit_sha}")
        #         return
        #     pr = data[0]
        #     print(f"[ok] Commit {commit_sha} merged via PR #{pr['number']} at {pr['mergedAt']} (merge commit {pr['mergeCommit']['oid']})")


def main() -> None:
    pass


if __name__ == "__main__":
    main()
