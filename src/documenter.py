import atexit
import os
import shutil
import sys
import glob
from datetime import datetime, timedelta


class Documenter:
    """Class that makes network runs self-documenting. All output data including the saved
    model, log file, parameter file and plots are saved into an output folder."""

    def __init__(self, run_name, existing_run=None, read_only=False):
        """If existing_run is None, a new output folder named as run_name prefixed by date
        and time is created. stdout and stderr are redirected into a log file. The method
        close is registered to be automatically called when the program exits."""
        self.run_name = run_name
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if existing_run is None:
            now = datetime.now()
            while True:
                full_run_name = now.strftime("%m%d_%H%M%S") + "_" + run_name
                self.basedir = os.path.join(script_dir, "../results", full_run_name)
                try:
                    os.mkdir(self.basedir)
                    break
                except FileExistsError:
                    now += timedelta(seconds=1)
        else:
            self.basedir = existing_run

        if not read_only:
            self.tee = Tee(self.add_file(self._next_log_name(), False))
            atexit.register(self.close)

    def _next_log_name(self) -> str:
        plain = os.path.join(self.basedir, "log.txt")
        if os.path.exists(plain):
            os.rename(plain, os.path.join(self.basedir, "log_0.txt"))

        nums = [
            int(os.path.basename(f)[4:-4])
            for f in glob.glob(os.path.join(self.basedir, "log_*.txt"))
            if os.path.basename(f)[4:-4].isdigit()
        ]
        n = (max(nums) + 1) if nums else 0
        return f"log_{n}.txt"

    def add_file(self, name, add_run_name=True):
        """Returns the path in the output folder for a file with the given name. If
        add_run_name is True, the run name is appended to the file name. If a file with
        the same name already exists in the output folder, it is moved to a subfolder 'old'.
        """
        new_file = self.get_file(name, add_run_name)
        old_dir = os.path.join(self.basedir, "old")
        if os.path.exists(new_file):
            os.makedirs(old_dir, exist_ok=True)
            shutil.move(new_file, os.path.join(old_dir, os.path.basename(new_file)))
        return new_file

    def get_file(self, name, add_run_name=False):
        """Returns the path in the output folder for a file with the given name. If
        add_run_name is True, the run name is appended to the file name."""
        if add_run_name:
            name_base, name_ext = os.path.splitext(name)
            name = f"{name_base}_{self.run_name}{name_ext}"
        return os.path.join(self.basedir, name)

    def close(self):
        """Ends redirection of stdout and changes the file permissions of the output folder
        such that other people on the cluster can access the files."""
        self.tee.close()
        os.system("chmod -R 755 " + self.basedir)


class Tee:
    """Class to replace stdout and stderr. It redirects all printed data to std_out as well
    as a log file."""

    def __init__(self, log_file):
        """Creates log file and redirects stdout and stderr."""
        self.log_file = open(log_file, "w")
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self

    def close(self):
        """Closes log file and restores stdout and stderr."""
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.log_file.close()

    def write(self, data):
        if not hasattr(self, "_buf"):
            self._buf = ""

        self._buf += data

        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            ts = datetime.now().strftime("%m-%d %H:%M:%S")
            out = f"[{ts}] {line}\n"
            self.log_file.write(out)
            self.stdout.write(out)

        self.log_file.flush()
        self.stdout.flush()

    def flush(self):
        """Flushes buffered data to the file."""
        self.log_file.flush()

    def isatty(self):
        """Returns False to indicate that the class is not a tty."""
        return False
