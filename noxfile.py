import nox
import pathlib
import shutil
import nox

DIR = pathlib.Path(__file__).parent.resolve()
VENV_DIR = pathlib.Path("./.venv").resolve()
nox.options.sessions = ["test", "coverage"]


@nox.session
def build(session: nox.Session) -> None:
    """
    Build an SDist and wheel with ``flit``.
    """

    dist_dir = DIR.joinpath("dist")
    if dist_dir.exists():
        shutil.rmtree(dist_dir)

    session.install(".[dev]")
    session.run("flit", "build")


@nox.session(python=["3.8"], venv_backend="conda")
def test(session):
    session.install(".[test]")
    session.run("pytest")


@nox.session
def coverage(session) -> None:
    """
    Run the unit and regular tests, and save coverage report
    """
    session.install(".[test]", "pytest-cov")
    session.run("pytest", "--cov=./", "--cov-report=xml", *session.posargs)
