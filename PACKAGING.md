# Packaging Instructions
## Publish with a GitHub Release
An action is set up to automatically publish a release on PyPI each type a version tag is created.

### Version Tag Structure
A version tag should have the form `vMAJOR.MINOR.PATCH[EXTRAS]`
where `[EXTRAS]` are optional extras as allowed by [PEP 440](https://www.python.org/dev/peps/pep-0440/).

Versioning numbers should follow [Semantic Versioning](https://semver.org/).
To summarize, while in development mode: `v0.MINOR.PATCH`, `PATCH` should be incremented for backwards-compatible bug
fixes and `MINOR` for everything else (breaking changes and new features).

### Instructions
[Create a new release on GitHub](https://github.com/causalincentives/pycid/releases/new) with an appropriate version
tag and title.
For example: tag `v0.2.1` and title `Version 0.2.1`.

Publish the release. Check the actions tab to see whether the publish to PyPI succeeded.


## Publish Manually
### Validate
Make sure that:
* All changes are commited to `master` and the working directory has no changes or new files.
* The code passes all tests by running [tests/check-code.sh](tests/check-code.sh).

### Tag a Version
The code version is created based on the lasted git tag on each run of `pip install/build`.
The version is stored in `pycid/version.py` or can be seen by running `pip show pycid` on the local install.
Note that the package version does not automatically update in an editable install so `pip install` must be re-run to
update the version.

Versioning numbers should follow [Semantic Versioning](https://semver.org/).
To summarize, while in development mode: `0.MINOR.PATCH`, `PATCH` should be incremented for backwards-compatible bug
fixes and `MINOR` for everything else (breaking changes and new features).

Once the API has been stabilized, the version can be updated to `1.0.0`.

**Instructions for tagging version 0.2.1**:
```shell
git checkout master
git fetch --all --tags                  # Fetch any recent tags from origin
git tag                                 # List the existing tags to see what the next should be
git tag -a "v0.2.1" -m "Version 0.2.1"  # Create an annotated tag.
                                        # Optionally omit -m "..." to write a more detailed message
git push --tags                         # Pust tags to origin
```

At this point the GitHub action will take over and publish to PyPI if all tests pass.
The following steps are not necessary and are included for reference only.

### Test Version
Make sure the versioning worked by re-installing the package (`pip install --editable .`)
and make sure that `pip show pycid` shows the new version number without any extra suffixes.

### Build and Upload to PyPI
*This has been replaced by a GitHub action and should not be done manually. It is for reference only.*

Detailed instructions are
[here](https://packaging.python.org/tutorials/packaging-projects/#generating-distribution-archives).

Get the `build` and `twine` packages:
```shell
python3 -m pip install --upgrade build twine
```

Build the project:
```shell
python3 -m build
```

Create a [PyPI account](https://pypi.org/manage/projects/) and contact the pycid owner to get added as a maintainer.
Create an API token ([instructions](https://packaging.python.org/tutorials/packaging-projects/#uploading-the-distribution-archives)).

Upload the project:
```shell
twine upload dist/*
```
For the username use `__token__` and enter the API token as the password.

Afterwards, delete the dist directory because it can interfere with editable installs:
```shell
rm -r dist/
```
