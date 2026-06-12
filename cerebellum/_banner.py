"""Terminal banner for the cerebellum CLI, after cerebellum_banner_500kb.png:
ASCII brain (cyan upper hemisphere, magenta underside) over the blocky wordmark.
Falls back to plain text when stdout is not a tty or NO_COLOR is set."""
import os
import sys

_CYN = "\033[96m"
_MAG = "\033[95m"
_DIM = "\033[2m"
_RST = "\033[0m"

# brain: top rows cyan, bottom rows magenta, like the logo's split
_BRAIN = [
    (_CYN, r"                 _,.--~~~~--.,_"),
    (_CYN, r"              ,-' (@#%&)(o#@#) '-,"),
    (_CYN, r"            ,'  #%@( )@#%  &*%@#  ',"),
    (_CYN, r"           |  (@#&%)  (%&#@)  (@#%) |"),
    (_CYN, r"           |  #%@(  )#&%  (@##)  ,--'"),
    (_MAG, r"            ',  (%@#)  (&%#)  ,-' ,--,"),
    (_MAG, r"              '-.,_      _,-'  ,'(@#)',"),
    (_MAG, r"                   '~~~~'      '-,__,-'"),
]

_WORDMARK = [
    r" ██████╗███████╗██████╗ ███████╗██████╗ ███████╗██╗     ██╗     ██╗   ██╗███╗   ███╗",
    r"██╔════╝██╔════╝██╔══██╗██╔════╝██╔══██╗██╔════╝██║     ██║     ██║   ██║████╗ ████║",
    r"██║     █████╗  ██████╔╝█████╗  ██████╔╝█████╗  ██║     ██║     ██║   ██║██╔████╔██║",
    r"██║     ██╔══╝  ██╔══██╗██╔══╝  ██╔══██╗██╔══╝  ██║     ██║     ██║   ██║██║╚██╔╝██║",
    r"╚██████╗███████╗██║  ██║███████╗██████╔╝███████╗███████╗███████╗╚██████╔╝██║ ╚═╝ ██║",
    r" ╚═════╝╚══════╝╚═╝  ╚═╝╚══════╝╚═════╝ ╚══════╝╚══════╝╚══════╝ ╚═════╝ ╚═╝     ╚═╝",
]

_TAGLINE = "dissect the brain · cut one sense · the other four come back sharper"


def _want_color() -> bool:
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _width() -> int:
    import shutil
    return shutil.get_terminal_size((80, 24)).columns


def banner() -> str:
    color = _want_color()
    wide = _width() >= 86
    out = []
    if wide:
        for tint, line in _BRAIN:
            out.append(f"{tint}{line}{_RST}" if color else line)
        for i, line in enumerate(_WORDMARK):
            tint = _CYN if i < 3 else _MAG
            out.append(f"{tint}{line}{_RST}" if color else line)
        bar = "▔" * 84
        out.append(f"{_MAG}{bar}{_RST}" if color else bar)
    else:
        word = "C E R E B E L L U M"
        out.append(f"{_CYN}{word}{_RST}" if color else word)
    tag = f"  {_TAGLINE}"
    out.append(f"{_DIM}{tag}{_RST}" if color else tag)
    return "\n".join(out)


if __name__ == "__main__":
    print(banner())
