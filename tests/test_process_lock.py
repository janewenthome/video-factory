from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from video_editor.process_lock import (
    HEAVY_LOCK_FD_ENV,
    release_heavy_job,
    try_acquire_heavy_job,
)


class ProcessLockTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="vf-heavy-lock-")
        self.queue_file = Path(self.temp_dir.name) / "state" / "queue.json"
        self.environment = os.environ.copy()
        self.environment["VIDEO_FACTORY_QUEUE_FILE"] = str(self.queue_file)
        self.environment.pop(HEAVY_LOCK_FD_ENV, None)
        self.environment_patch = patch.dict(os.environ, {
            "VIDEO_FACTORY_QUEUE_FILE": str(self.queue_file),
            HEAVY_LOCK_FD_ENV: "",
        })
        self.environment_patch.start()

    def tearDown(self) -> None:
        self.environment_patch.stop()
        self.temp_dir.cleanup()

    def test_other_process_cannot_start_and_queue_child_can_inherit_lock(self) -> None:
        descriptor = try_acquire_heavy_job()
        self.assertIsNotNone(descriptor)
        assert descriptor is not None
        try:
            script = "\n".join(
                [
                    "from video_editor.process_lock import HeavyJobBusy, exclusive_heavy_job",
                    "import sys",
                    "try:",
                    "    with exclusive_heavy_job():",
                    "        pass",
                    "except HeavyJobBusy:",
                    "    sys.exit(75)",
                ]
            )
            blocked = subprocess.run(
                [sys.executable, "-c", script],
                env=self.environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(blocked.returncode, 75, blocked.stderr)

            inherited_env = self.environment.copy()
            inherited_env[HEAVY_LOCK_FD_ENV] = str(descriptor)
            inherited = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "\n".join(
                        [
                            "from video_editor.process_lock import exclusive_heavy_job",
                            "with exclusive_heavy_job():",
                            "    print('inherited')",
                        ]
                    ),
                ],
                env=inherited_env,
                pass_fds=(descriptor,),
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("inherited", inherited.stdout)
            self.assertIsNone(try_acquire_heavy_job())
        finally:
            release_heavy_job(descriptor)

        released = try_acquire_heavy_job()
        self.assertIsNotNone(released)
        if released is not None:
            release_heavy_job(released)


if __name__ == "__main__":
    unittest.main()
