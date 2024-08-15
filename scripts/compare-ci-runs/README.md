# Comparing CI Runs

This script can be used to compare the results of two runs of the
"Performance E2E" CI workflow.

## Prerequisites

To evaluate the data from CI runs, the Python packages `pandas` and its
corresponding dependencies are required.

If you also plan to plot the performance comparison, the packages `matplotlib`
and `seaborn` and their dependencies are also required.

All those packages can simply be installed through `pip`

```sh
pip install pandas seaborn matplotlib
```

The installation and use inside a virtual environment is recommended, but not
strictly required.

### Github CLI

To download artifacts from CI runs, the script uses the Github CLI `gh`.

The script assumes that the `gh` executable can be found on the `$PATH` and
that authentication has happened.

Authentication for the Github CLI is only required once, see the
[manual](https://cli.github.com/manual/gh_auth_login).

## Usage

The script supports three modes of execution, explained below. Some aspects of
the script are independent from the mode of execution and will be explained
first.

### Output

In general, the script compares the performance results of two different
configurations from two different CI runs.

One configuration is referred to as the "numerator", while the other is called
the "denominator".

The raw performance of the numerator and denominator in the CI runs is captured
as "speedup" over Pytorch native (i.e., not Triton-compiled) execution.

The metric used to compare the those two configurations is the
"relative difference" between the speedups.

It is calculated as:
```
(speedup(numerator) - speedup(denominator)) / speedup(denominator)
```

As also explained in the script's output, a relative difference of (close to) 0
means that numerator and denominator configuration perform identically.
A relative difference < 0 means that the denominator outperformed the numerator,
while a relative difference of > 0 means that the numerator was faster than the
denominator.

If not deactivated via an option (see below), the script will also visualize the
data in a box plot, with each combination of `[training|inference] x data-type`
being one entry on the x-axis of the plot.

### Common Options

While some options are used to control the execution mode of the script as
described below, some options are common to all modes of execution:

* `-p`/`--path`: The output path, `DIR`. Per default, the script would store
all intermediate data and output in the current directory. With this option,
users can select a different location to store data and output. For the
remainder of this README, the path where data is stored will be referred to
as `DIR`.
* `--no-plot`: Disable plotting. Per default, the script will also visualize
the output in a plot. With this option, you can deactive plotting, which also
means the corresponding prerequisite Python packages (see above) do not need to
be installed.

### Full Mode

In this mode, the full set of capabilities of the script is used. This means,
the script will perform the following three steps:

1. Download the artifacts from both *Performance E2E* CI runs containing the
performance data using the Github CLI. The raw data is stored in subdirectories
of `DIR`, with the given names for numerator and denominator used as names for
the subdirectories.
2. Process the raw data, i.e., the performance data stored for each benchmark
configuration in the artifact and store the result in a file called
`preprocessed-data-[numerator]-[denominator].csv` in `DIR`.
3. Evaluate the pre-processed data and print a summary of the evaluation to
`stdout`. If not deactivated via the option, also visualize the data in a plot
and store it to `performance-plot-[numerator]-[denominator].pdf` in `DIR`.

To allow the script to download the CI artifact, the user must not only specify
a name for numerator and denominator, but must also provide the ID of the CI
run. This ID can be extracted from the URL of a CI run, e.g.:
`https://github.com/intel/intel-xpu-backend-for-triton/actions/runs/10180538030`

In this case, the ID is `10180538030`.

To specify a name and the ID for numerator and denominator, name and ID are
separated by a `:`, for example:

```sh
compare-runs.py -n "32:10197715410" -d "16:10197720144" -p data/run5
```

In this case "32" is used as the name for the numerator and "16" is used as the
name for the denominator.

### Local Mode

The local mode can be used by adding the `-l`/`--local` option.

In this mode, the script skips the first step from above and assumes that the
CI artifacts have already been downloaded and placed in `DIR/[numerator]` and
`DIR/[denominator]`.

While it is possible to manually download the data and place them in the right
directories, it is recommended to use the full mode once to download the data
and store it in the expected layout.

As no download is required in local mode, only the name has to be specified
for the numerator and denominator options, e.g.:

```sh
compare-runs.py -n "32" -d "16" -p data/run5 -l
```

The local mode can also be useful to avoid repeated downloads from Github that
might trigger rate limiting.

### Evaluation Mode

The evaluation mode can be used by adding the `-e`/`--eval-only` option.

In this mode, the script skips the first two steps from above and assumes that
the pre-processed data is stored in
`DIR/preprocessed-data-[numerator]-[denominator].csv`.

As no download is required in evaluation mode, only the name has to be specified
for the numerator and denominator options, e.g.:

```sh
compare-runs.py -n "32" -d "16" -p data/run5 -e
```

The evaluation mode can be useful to avoid repeated pre-processing of data.
