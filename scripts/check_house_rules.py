#!/usr/bin/env python3
"""Mechanical enforcement of this repository's house rules, over the tree and the history.

Every rule here exists because its absence has already cost something. The
absence of exactly this check is what forced a retroactive history rewrite
across 23 of 25 repositories on 2026-07-28, and a rewrite is not a cleanup: it
is a dated, public event that invalidates every clone and every commit hash
anyone else recorded, and it has to be disclosed as such. This file runs in
milliseconds. Omitting it cost a rewrite. That asymmetry is the entire argument.

Four rules, all refusals, all default-on:

1. ASCII only, over every tracked text file, not only ``.py``. Any byte above
   127 fails, reported with the file and the line. Markdown, YAML and TOML are
   where curly quotes actually arrive, because those are the files written in
   editors that substitute punctuation silently.
2. No em dash, no curly quotes, no unicode arrows. These are a subset of rule 1
   and are still reported as their own category, because "byte 0xe2 at column
   34" does not tell an author to type ``--``.
3. No co-author trailer and no assistant-attribution string, over the working
   tree AND over ``git log --all``. A tree-only grep is the version of this
   check that passes while the trailer sits in a commit message that has
   already been pushed, which is the failure that produced the rewrite.
4. Author and committer on every commit is ``di-omics``, exactly one identity.

This is a sibling of the checker in the plr-lab-robot repository and the two are
kept deliberately identical in behaviour, reindented to this repository's
four-space style. Two copies can drift, and nothing here can detect that: there
is no cross-repository test and there is not going to be one, because a test
that clones a second repository to compare a file is a test that fails for
network reasons and then gets deleted. What is here instead is ``--self-test``,
which proves this copy can still catch every rule it claims to enforce, run in
CI beside the real scan. A drifted copy that has stopped catching something is
the failure that matters; a drifted comment is not.

This repository has no ``pyproject.toml`` and no dependency manifest for tooling,
so the stdlib-only constraint below is not a preference. There is nowhere to
declare a dependency, and a checker that needed one would simply stop running.

What this deliberately does not have, and what each omission prevents:

* No allowlist. An allowlist is where the first exception lives and where the
  second one hides. A file that genuinely needs a byte above 127 is a
  conversation, not a config entry.
* No self-exemption. The banned strings are assembled from fragments below so
  this file does not match its own patterns. Exempting the checker from the
  check is how a checker starts lying, and skipping a path by name would also
  skip a real violation that happened to live there.
* No pass on an empty result. Zero commits visible is reported as a failure,
  not as a clean history, because the usual cause is a shallow clone, and
  "nothing was checked" has to stay a different answer from "the check passed".
* No offending character in the output. Findings name codepoints as ``U+XXXX``
  and never echo the character, so a CI log under a byte-oriented locale cannot
  fail to encode the report, and so pasting the report into a file does not
  create the violation it describes.

Usage::

    python scripts/check_house_rules.py               # tracked files + full history
    python scripts/check_house_rules.py --walk        # working tree, untracked included
    python scripts/check_house_rules.py --paths a b   # named files only, no git needed
    python scripts/check_house_rules.py --self-test   # prove the checker can still fail
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

#: The one identity permitted as author and as committer. Sole authorship is a
#: property of this repository, so a second name is a defect and not a merge.
EXPECTED_IDENTITY = "di-omics"

#: Record and field separators for ``git log --format``. Both are ASCII control
#: characters that cannot appear in a commit message, so a body containing
#: newlines, pipes or NUL bytes still parses back into whole records.
_RECORD_SEP = "\x1e"
_FIELD_SEP = "\x1f"

#: Codepoints that have an ASCII spelling, mapped to that spelling. Written as
#: escapes so this file is itself ASCII and can be checked by its own rules.
TYPOGRAPHY: Dict[str, Tuple[str, str]] = {
    "\u2014": ("em dash", "--"),
    "\u2013": ("en dash", "-"),
    "\u2012": ("figure dash", "-"),
    "\u2015": ("horizontal bar", "--"),
    "\u2212": ("minus sign", "-"),
    "\u2018": ("left single quote", "'"),
    "\u2019": ("right single quote", "'"),
    "\u201a": ("low single quote", "'"),
    "\u201b": ("reversed single quote", "'"),
    "\u201c": ("left double quote", '"'),
    "\u201d": ("right double quote", '"'),
    "\u201e": ("low double quote", '"'),
    "\u2032": ("prime", "'"),
    "\u2033": ("double prime", '"'),
    "\u2190": ("leftwards arrow", "<-"),
    "\u2192": ("rightwards arrow", "->"),
    "\u2194": ("left right arrow", "<->"),
    "\u21d0": ("leftwards double arrow", "<="),
    "\u21d2": ("rightwards double arrow", "=>"),
    "\u21d4": ("left right double arrow", "<=>"),
    "\u2026": ("horizontal ellipsis", "..."),
    "\u2022": ("bullet", "*"),
    "\u00a0": ("no-break space", " "),
    "\u202f": ("narrow no-break space", " "),
    "\ufeff": ("byte order mark", "nothing at all"),
}

# Assembled from fragments on purpose: this file has to be scannable by the
# scan it defines, and a checker that has to skip itself is a checker with a
# hole in it exactly where someone would put the thing being hidden.
_TRAILER = "co-authored" + "-by"
_GENERATED = "generated" + " with"
_TOOL_A = "clau" + "de code"
_TOOL_B = "chat" + "gpt"

#: Substrings banned in tracked text and in commit messages, case-insensitive.
#: The phrasing check is deliberately blunt. A grep that tries to tell an
#: innocent "produced with" from an attribution trailer is a grep that can be
#: argued with in review, and the argument is more expensive than rewording.
ATTRIBUTION: Tuple[Tuple[str, str], ...] = (
    (_TRAILER, "a co-author trailer; every commit here has one author and one committer"),
    (_GENERATED, "an attribution phrase; reword it, the check does not read intent"),
    (_TOOL_A, "an assistant tool name"),
    (_TOOL_B, "an assistant tool name"),
)

#: Extensions read as bytes and never as text. The default rule is inverted on
#: purpose: a file is text unless it proves otherwise, so a new extension is
#: checked rather than silently exempt. The font entries earn their place here:
#: ``.fonts/`` holds four tracked ``.ttf`` files, and a checker that read them as
#: text would report thousands of findings and be turned off within the hour.
#: They are counted in the "skipped as binary" total rather than passed over in
#: silence, so a font that somehow became a text file shows up as a changed count.
BINARY_SUFFIXES = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".ico", ".webp",
    ".pdf", ".zip", ".gz", ".bz2", ".xz", ".tar", ".whl", ".pyc", ".pyo",
    ".so", ".dylib", ".dll", ".o", ".a", ".npy", ".npz", ".pkl", ".bin",
    ".mp4", ".mov", ".avi", ".wav", ".mp3", ".woff", ".woff2", ".ttf", ".otf",
})

#: Directories skipped by ``--walk``. These hold generated or vendored files,
#: none of which are ever committed, so scanning them reports other people's
#: bytes. ``git ls-files`` needs no such list, which is why it is the default.
#:
#: ``frames``, ``videos`` and ``output`` are this repository's own three: the
#: demos and the legacy ROI pipeline write extracted frames, rendered videos and
#: contact sheets into them, all three are in ``.gitignore``, and a ``--walk`` that
#: scanned them would spend its time on decoded video and report on nothing that
#: anybody is going to publish.
WALK_SKIP_DIRS = frozenset({
    ".git", ".venv", "venv", "build", "dist", "node_modules",
    "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".tox",
    "frames", "videos", "output",
})


class GitUnavailable(RuntimeError):
    """Raised when a git query cannot run, so a missing git never reads as clean."""


@dataclass(frozen=True)
class Violation:
    """One finding. Carries enough to fix it without re-running anything."""

    category: str
    path: str
    detail: str
    line: Optional[int] = None
    column: Optional[int] = None

    def as_line(self) -> str:
        where = self.path
        if self.line is not None:
            where += ":" + str(self.line)
            if self.column is not None:
                where += ":" + str(self.column)
        return "%s: [%s] %s" % (where, self.category, self.detail)


def _codepoint_note(ch: str) -> str:
    """Describe a character without ever printing it. See the module docstring."""
    return "U+%04X %s" % (ord(ch), unicodedata.name(ch, "unnamed codepoint"))


def scan_text(
        name: str, lineno: int, text: str, replacement_reported: bool = False) -> List[Violation]:
    """Scan one already-decoded line for every content rule.

    Kept pure and line-scoped so the same code path checks a tracked file and a
    commit message body. Two scanners would drift, and the one that drifted would
    be the history scanner, which is the one nobody looks at until a rewrite.

    ``replacement_reported`` is set only by :func:`scan_raw`, and only when strict
    decoding already failed on this line, so that one bad byte is not reported
    twice. It defaults to False because U+FFFD is a real codepoint that a valid
    UTF-8 file can contain, and it is not ASCII: suppressing it unconditionally
    would be an exemption granted to the one character that looks like an error.
    """
    found: List[Violation] = []
    for column, ch in enumerate(text, start=1):
        if ord(ch) < 128:
            continue
        if replacement_reported and ch == "\ufffd":
            continue
        known = TYPOGRAPHY.get(ch)
        if known is None:
            found.append(Violation(
                "non-ascii", name, "%s is not ASCII" % _codepoint_note(ch), lineno, column))
        else:
            label, replacement = known
            found.append(Violation(
                "typography", name,
                "%s (%s); write \"%s\" instead" % (label, _codepoint_note(ch), replacement),
                lineno, column))
    lowered = text.lower()
    for needle, why in ATTRIBUTION:
        index = lowered.find(needle)
        if index >= 0:
            found.append(Violation(
                "attribution", name, "%s (%d characters starting here)" % (why, len(needle)),
                lineno, index + 1))
    return found


def scan_raw(name: str, lineno: int, raw: bytes) -> List[Violation]:
    """Decode one raw line, reporting undecodable bytes rather than dropping them."""
    found: List[Violation] = []
    replaced = False
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        found.append(Violation(
            "non-ascii", name,
            "byte 0x%02x is not decodable as UTF-8, so the file is not ASCII either"
            % raw[exc.start],
            lineno, exc.start + 1))
        text = raw.decode("utf-8", "replace")
        replaced = True
    found.extend(scan_text(name, lineno, text, replacement_reported=replaced))
    return found


def looks_binary(path: str, data: bytes) -> bool:
    """True for files read as bytes. Reported in the summary, never silently."""
    if os.path.splitext(path)[1].lower() in BINARY_SUFFIXES:
        return True
    return b"\x00" in data[:8192]


def scan_file(path: str, name: Optional[str] = None) -> Tuple[List[Violation], bool]:
    """Scan one file on disk. Returns (violations, skipped_as_binary)."""
    with open(path, "rb") as handle:
        data = handle.read()
    label = path if name is None else name
    if looks_binary(path, data):
        return [], True
    found: List[Violation] = []
    for lineno, raw in enumerate(data.split(b"\n"), start=1):
        found.extend(scan_raw(label, lineno, raw))
    return found, False


def scan_paths(root: str, names: Sequence[str]) -> Tuple[List[Violation], List[str]]:
    """Scan a list of repo-relative (or absolute) paths. Returns (violations, skipped)."""
    found: List[Violation] = []
    skipped: List[str] = []
    for name in names:
        path = name if os.path.isabs(name) else os.path.join(root, name)
        if not os.path.isfile(path) or os.path.islink(path):
            continue
        hits, binary = scan_file(path, name)
        if binary:
            skipped.append(name)
        else:
            found.extend(hits)
    return found, skipped


def _git(root: str, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", root] + list(args), capture_output=True)
    if proc.returncode != 0:
        raise GitUnavailable(
            "git %s failed in %s: %s"
            % (" ".join(args), root, proc.stderr.decode("utf-8", "replace")))
    return proc.stdout.decode("utf-8", "replace")


def tracked_files(root: str) -> List[str]:
    """The file list the rules apply to, via ``git ls-files -z``.

    NUL-delimited because a path containing a space or a newline is still a path,
    and a checker that mangles it is a checker that skips it.
    """
    return [name for name in _git(root, "ls-files", "-z").split("\x00") if name]


def walked_files(root: str) -> List[str]:
    """Every file in the working tree, generated directories excluded.

    For local use before anything is staged. ``git ls-files`` cannot see a file
    that has not been added yet, and the rules apply to it the moment it is.
    """
    out: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames if d not in WALK_SKIP_DIRS and not d.endswith(".egg-info"))
        for filename in sorted(filenames):
            full = os.path.join(dirpath, filename)
            out.append(os.path.relpath(full, root))
    return out


def commit_records(root: str) -> List[Tuple[str, str]]:
    """Every commit reachable from every ref, as ``(sha, body)``.

    ``--all`` and not ``HEAD``: the 2026-07-28 event was about history that had
    already been pushed, and a trailer on an unmerged branch is just as published.
    """
    raw = _git(root, "log", "--all", "--format=format:" + _RECORD_SEP + "%H" + _FIELD_SEP + "%B")
    records: List[Tuple[str, str]] = []
    for chunk in raw.split(_RECORD_SEP):
        if not chunk.strip():
            continue
        sha, _, body = chunk.partition(_FIELD_SEP)
        records.append((sha.strip(), body))
    return records


def identity_lines(root: str) -> List[str]:
    """``git log --all --format=%an|%cn`` output, one line per commit."""
    return _git(root, "log", "--all", "--format=%an|%cn").splitlines()


def history_violations(records: Sequence[Tuple[str, str]]) -> List[Violation]:
    """Apply the content rules to commit message bodies.

    A commit message is published text. An em dash in one survives every later
    fix to the working tree, and removing it later means rewriting history.
    """
    found: List[Violation] = []
    for sha, body in records:
        name = "commit " + sha[:12]
        for lineno, line in enumerate(body.splitlines(), start=1):
            found.extend(scan_text(name, lineno, line))
    return found


def identity_violations(lines: Sequence[str]) -> List[Violation]:
    """Require exactly one ``author|committer`` pair, and require it to be ours.

    An empty list is a failure and not a pass. The usual cause is a shallow
    clone, where every commit but one is invisible and the check would otherwise
    report a clean history it never read.
    """
    entries = sorted({line.strip() for line in lines if line.strip()})
    expected = EXPECTED_IDENTITY + "|" + EXPECTED_IDENTITY
    if not entries:
        return [Violation(
            "identity", "git log --all",
            "no commits visible, so nothing was checked; a shallow clone (fetch-depth 1) "
            "is the usual cause and it must not read as a clean history")]
    found: List[Violation] = []
    for entry in entries:
        if entry != expected:
            found.append(Violation(
                "identity", "git log --all",
                "author|committer \"%s\" is not \"%s\"; %d distinct identities in history, "
                "expected exactly 1" % (entry, expected, len(entries))))
    return found


def self_test() -> List[str]:
    """Seed one violation of every rule and require the checker to catch each one.

    A checker nobody has seen fail is a checker that does not work, and the way
    this one would fail is silently: a pattern that no longer matches reports zero
    findings, which reads exactly like a clean repository. This runs in CI beside
    the real scan, using stdlib only so it needs no install step to stay honest.

    The seeds are built with ``chr()`` and by concatenation so that this file does
    not contain the strings it bans. Returns a list of failure descriptions, empty
    when every seeded violation was caught.
    """
    import tempfile

    failures: List[str] = []
    seeds = [
        ("non-ascii", "caf" + chr(0x00E9) + " is not ascii"),
        ("typography", "a clause " + chr(0x2014) + " a dash"),
        ("typography", "capture " + chr(0x2192) + " screen"),
        ("attribution", _TRAILER.title() + ": Someone <someone@example.com>"),
        ("attribution", _GENERATED.capitalize() + " a tool"),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        for index, (category, text) in enumerate(seeds):
            path = os.path.join(tmp, "seed_%d.md" % index)
            with open(path, "wb") as handle:
                handle.write(("clean first line\n" + text + "\n").encode("utf-8"))
            found, _ = scan_paths(tmp, [path])
            got = sorted({v.category for v in found})
            if got != [category]:
                failures.append("seed %d: expected [%s], got %s" % (index, category, got))
            elif [v.line for v in found] != [2] * len(found):
                failures.append("seed %d: wrong line number reported" % index)

        clean = os.path.join(tmp, "clean.md")
        with open(clean, "wb") as handle:
            handle.write(b"plain ascii, arrows as ->, dashes as --\n")
        found, _ = scan_paths(tmp, [clean])
        if found:
            failures.append("clean fixture produced %d finding(s)" % len(found))

    if not identity_violations(["Someone Else|" + EXPECTED_IDENTITY]):
        failures.append("a foreign author was not caught")
    if not identity_violations([]):
        failures.append("an empty history was accepted as clean")
    if identity_violations([EXPECTED_IDENTITY + "|" + EXPECTED_IDENTITY]):
        failures.append("the expected identity was rejected")
    if not history_violations([("0" * 40, "subject " + chr(0x2014) + " dash")]):
        failures.append("a commit message violation was not caught")
    return failures


def _report(found: Sequence[Violation], max_report: int, stream) -> None:
    for violation in found[:max_report]:
        stream.write(violation.as_line() + "\n")
    if len(found) > max_report:
        stream.write("... and %d more\n" % (len(found) - max_report))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check ASCII, typography, attribution strings and commit identity.")
    parser.add_argument("--root", default=".", help="repository root (default: .)")
    parser.add_argument(
        "--paths", nargs="+", default=None,
        help="check only these files; skips git entirely, for fixtures and pre-commit use")
    parser.add_argument(
        "--walk", action="store_true",
        help="check the working tree instead of the tracked file list, untracked included")
    parser.add_argument(
        "--no-history", action="store_true",
        help="skip the commit message and identity rules (they need a full clone)")
    parser.add_argument(
        "--self-test", action="store_true",
        help="seed one violation of every rule and require each to be caught")
    parser.add_argument("--max-report", type=int, default=200, help="findings printed")
    args = parser.parse_args(argv)

    if args.self_test:
        failures = self_test()
        for line in failures:
            sys.stdout.write("house rules self-test: %s\n" % line)
        if failures:
            sys.stdout.write(
                "house rules self-test: %d seeded violation(s) missed\n" % len(failures))
            return 1
        sys.stdout.write("house rules self-test: every seeded violation was caught\n")
        return 0

    root = os.path.abspath(args.root)
    found: List[Violation] = []
    checked = 0
    skipped: List[str] = []
    commits = 0

    try:
        if args.paths is not None:
            names = list(args.paths)
        elif args.walk:
            names = walked_files(root)
        else:
            names = tracked_files(root)
        hits, skipped = scan_paths(root, names)
        checked = len(names) - len(skipped)
        found.extend(hits)

        if args.paths is None and not args.no_history:
            records = commit_records(root)
            commits = len(records)
            found.extend(history_violations(records))
            found.extend(identity_violations(identity_lines(root)))
    except GitUnavailable as exc:
        sys.stderr.write("house rules: %s\n" % exc)
        return 2

    sys.stdout.write(
        "house rules: %d text files checked, %d skipped as binary, %d commits read\n"
        % (checked, len(skipped), commits))
    if not found:
        sys.stdout.write("house rules: clean\n")
        return 0
    _report(found, args.max_report, sys.stdout)
    sys.stdout.write("house rules: %d violation(s)\n" % len(found))
    return 1


if __name__ == "__main__":
    sys.exit(main())
