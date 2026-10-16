# Releasing

This page is for the maintainer who cuts a release. Users start at
[README.md](README.md), contributors at [CONTRIBUTING.md](CONTRIBUTING.md).

Development happens on a private development repository. The public repository
receives **one squashed commit per release**, whose only parent is the public
`main`, and an annotated tag `v<version>` on it. The commit's tree is the release
commit's tree with the files that
[`tools/release/export_manifest.json`](tools/release/export_manifest.json) excludes
removed, and **nothing else changed**: no file is edited for the public tree. A
sentence that would be false or dangling without the excluded files is fixed in the
development repository, so the two trees never say different things.

The export is one command, [`tools/release/export_public.py`](tools/release/export_public.py).
It builds the commit and the tag in a scratch clone, refuses on any defect it can
check, and prints the push commands. It never pushes, and it never writes to either
repository. Never build a public tree by hand.

## Before the export: the release commit

The release commit is on the development repository's `main`, reviewed, with its
continuous-integration run read. At that commit:

- `pyproject.toml` and `CITATION.cff` carry the release version, and
  `CITATION.cff`'s `date-released` is the release date;
- `CHANGELOG.md` has the heading `## <version> — <title> (<date>)`; the title becomes
  the tag's message, `meep-gpu <version> — <title>`;
- the hygiene guard, `tools/ci/check_hygiene.py`, is green: no host name, account
  name, local path or personal address in any tracked file;
- every file the manifest requires on the public repository is present, among them
  `.github/workflows/publish.yml`, the workflow the package index's trusted publisher
  is registered to (it runs on the public repository only);
- no file names an excluded file or quotes its digest.

The message is written from the release's CHANGELOG section: a `Summary:` line, a
blank line, `Description:` with numbered items, a blank line, then one
`Co-authored-by: <name> <address>` trailer per co-author named in `CITATION.cff`,
and nothing after them. It names no development commit and no local path.

## The export

    python tools/release/export_public.py \
        --dev-repo <development checkout> --dev-ref <release commit> \
        --public-repo <public checkout or URL> \
        --message-file <message file> --scratch <empty directory> \
        --report <scratch report>.json

`--version` defaults to `pyproject.toml`'s and `--date` to the CHANGELOG heading's.
The manifest is read from the release commit, never from a working copy, so the
exclusions that ship are the ones that were reviewed. `--guard-names-file` hands the
hygiene guard an untracked local file of names to look for besides the ones its
rules hold; it can only add names, and the guard's own arguments come from the
release commit.

It prints one line per step, then the verdict of each check, and refuses (exit 1,
nothing committed) if any fails:

| Check | Refuses when |
|---|---|
| `required` | a file the manifest requires on the public repository is absent: land it in the development repository; it is never copied in |
| `removals` | the public `main` carries a file the export drops that is not excluded, was never deleted in the release commit's history and is not listed under `dropped_from_public` (with its reason): a public-only file would be lost |
| `references` | an exported file names an excluded file, by path or by file name |
| `digests` | an exported file carries an excluded file's sha256 (or a Python file's code digest): a record that binds it would break |
| `shape` | the exported tree's `ls-tree -r` differs from the release commit's by anything other than the excluded files |
| `version` | the version or the date disagree across `pyproject.toml`, `CITATION.cff` and the CHANGELOG heading, or the public repository already has the tag or the version |
| `message` | the message is out of the form above |
| `guard` | the release commit's hygiene guard is absent, or exits non-zero on the exported tree |

On success it writes one commit (author, committer and tagger are `CITATION.cff`'s
contact, dated the release date at the manifest's `release_time`), moves `main` of
the scratch clone to it, and makes the annotated tag. Exit 2 means the command
could not run at all: bad arguments, an unreadable repository or manifest, or a
scratch directory that is not empty.

## After the export

1. Run the suites on the exported tree it leaves under `<scratch>/tree`: the
   commands of [CONTRIBUTING.md](CONTRIBUTING.md), `tools/ci/check_failures.py` on
   their reports, and `tools/ci/check_wheel.py`.
2. Run the push commands it printed, in order. Pushing `main` starts the tests
   workflow; read that run. Pushing the tag starts the publish workflow, which
   uploads the version to the package index. An index takes a version once, so push
   the tag only after the `main` run is green, and bump the version before the next
   release.

## Changing what the public tree leaves out

Add an entry to `excluded` in the manifest, with its reason, in a reviewed commit on
the development repository: a hand-off runbook, a driver written for one machine, or
anything the hygiene guard cannot accept in public. Then run the export: the
`references` and `digests` checks name every file that still points at the excluded
one, and those are fixed in the development repository. A file the public repository
should stop carrying that the development history never had goes under
`dropped_from_public`, with its reason.
