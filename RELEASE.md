# Releasing Triton

Triton releases provide a stable snapshot of the code base encapsulated into a binary that can easily be consumed through PyPI. Additionally, releases represent points in time when we, as the development team, can signal to the community that certain new features are available, what improvements have been made, and any changes that are coming that may impact them (i.e. breaking changes).

## Release Compatibility Matrix

Following is the Release Compatibility Matrix for Triton releases:

| Triton version | Python version | Manylinux version |
| --- | --- | --- |
| 3.2.0 | >=3.9, <=3.13 | glibc 2.17+ x86-64 |
| 3.1.0 | >=3.8, <=3.12 | glibc 2.17+ x86-64 |
| 3.0.0 | >=3.8, <=3.12 | glibc 2.17+ x86-64 |
| 2.3.1 | >=3.7, <=3.12 | glibc 2.17+ x86-64 |
| 2.3.0 | >=3.7, <=3.12 | glibc 2.17+ x86-64 |
| 2.2.0 | >=3.7, <=3.12 | glibc 2.17+ x86-64 |
| 2.1.0 | >=3.7, <=3.11 | glibc 2.17+ x86-64 |
| 2.0.0 | >=3.6, <=3.11 | glibc 2.17+ x86-64 |
| 1.1.1 | >=3.6, <=3.9 | glibc 2.17+ x86-64 |
| 1.1.0 | >=3.6, <=3.9 | glibc 2.17+ x86-64 |
| 1.0.0 | >=3.6, <=3.9 | glibc 2.17+ x86-64 |

## Release Cadence

Following is the release cadence for year 2024/2025. All future release dates below are tentative. Please note: Patch Releases are optional.

| Minor Version | Release branch cut | Release date | Patch Release date |
| --- | --- | --- | --- |
| 3.5.0 | Sep 2025 | Oct 2025 | --- |
| 3.4.0 | Jun 2025 | Jul 2025 | --- |
| 3.3.0 | Feb/Mar 2025 | Apr 2025 | --- |
| 3.2.0 | Dec 2024 | Jan 2025 | --- |
| 3.1.0 | Jun 2024 | Oct 2024 | --- |
| 3.0.0 | Jun 2024 | Jul 2024 | --- |
| 2.3.0 | Dec 2023 | Apr 2024 | May 2024 |
| 2.2.0 | Dec 2023 | Jan 2024 | --- |

## Release Cherry-Pick Criteria

After branch cut, we approach finalizing the release branch with clear criteria on what cherry picks are allowed in. Note: a cherry pick is a process to land a PR in the release branch after branch cut. These are typically limited to ensure that the team has sufficient time to complete a thorough round of testing on a stable code base.

* Regression fixes - that address functional/performance regression against the most recent release (e.g. 3.2 for 3.3 release)
* Critical fixes - critical fixes for severe issue such as silent incorrectness, backwards compatibility, crashes, deadlocks, (large) memory leaks
* Fixes to new features introduced in the most recent release (e.g. 3.2 for 3.3 release)
* Documentation improvements
* Release branch specific changes (e.g. change version identifiers or CI fixes)

Please note: **No feature work allowed for cherry picks**. All PRs that are considered for cherry-picks need to be merged on trunk, the only exception are Release branch specific changes. An issue is for tracking cherry-picks to the release branch is created after the branch cut. **Only issues that have ‘cherry-picks’ in the issue tracker will be considered for the release.**

# Intel Release Process

Intel XPU Backend for Triton releases are aligned to the upstream `triton-lang/triton` project and to `PyTorch`. To make a release:

1. Select a commit common to upstream [Triton](https://github.com/triton-lang/triton). Often this commit will be selected by PyTorch at [`pytorch/.ci/docker/ci_commit_pins/triton.txt`](https://github.com/pytorch/pytorch/blob/main/.ci/docker/ci_commit_pins/triton.txt). For example:
```
[FRONTEND] Fix wrong livein set in loop codegen (#4018)
```
2. Using `git log --graph --oneline` find the selected commit in the default branch. Because upstream commits are merged using merge commits, and GitHub displays commits in chronological order, the ordering of commits on GitHub's website cannot be trusted.
```
* | 23c4cdf1 [intel] Fix bf16 representation
* | e4ac248d Merge commit '445d5edb8a3796b0e6e589682231b2c50fe14871'
|\|
| * 445d5edb [BACKEND] fix bf16 representation in TritonNvidiaGPU and bf16 tl.sort bug (#3975)
* | 0cdcffa2 Merge commit '100e2aaca903ed99564242f933198a6c221d3b50'
|\|
| * 100e2aac [AMD][WMMA] Support dot3d (#3674)
| * 706174da [CI] Add macos build test (#3994)
| * 47f7d45c [AMD] Replace wave with warp where possible. (#3978)
| * d3fb1dc1 [AMD] Move MFMA shortcut check to not compute scratch buffer shape if it is not needed (#3790)
| * b847042d Remove redundant options from passes (#4015)
| * 513f38c4 [FRONTEND] Fix wrong livein set in loop codegen (#4018)
| * d4b16818 [DOCS] improve Triton Linear layout doc (#4005)
* | c37ca9c6 [GEN] Update libGenISAIntrinsics
* | 4d936645 Merge commit '18d691e491b9f6184c505f9b553a49075e67d4bd'
|\|
| * 18d691e4 [BACKEND] Update LLVM version to https://github.com/llvm/llvm-project/commit/10dc3a8e916d73291269e5e2b82dd22681489aa1 (#4010)
| * 1d571390 Remove odd character at beginning of file (#4011)
| * 9d9ec144 [DOCS] fixes to tl.dot (#4006)
* | 8ab1e456 [TritonGEN] Add `triton_gen.cache_controls` operation (#1087)
* | 8b91a5b9 Added new workflow to run pytorch inductor test (#1219)
```

3. Checkout a commit from the default branch that is close to the desired upstream commit. In this case, the commit we want is in the middle of a merge. But, this particular commit does not effect any of our code and we don't want other commits from the same merge. So in this case, we will select the last commit before the merge and then cherry-pick our way forward.
```
git checkout c37ca9c6 # [GEN] Update libGenISAIntrinsics
```

4. Next, create a release branch.
```
git checkout -b release/X.Y.Z
```

5. Finally, cherry-pick (or revert) commits to get to the desired state. When cherry-picking, use `-x` to preserve references:
```
git cherry-pick -x d4b16818
git cherry-pick -x 513f38c4
```

6. Push the release branch to GitHub and start CI pipelines.
