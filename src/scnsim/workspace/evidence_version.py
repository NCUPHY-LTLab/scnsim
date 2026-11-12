"""The one version number of stored Workspace evidence.

The Workspace envelope, the operation database, every numeric block (in its
magic bytes and its header) and every result manifest declare this number.
A store that declares another number is rejected before anything is changed.
There is no migration and no reader for another version.
"""

EVIDENCE_VERSION = 4
