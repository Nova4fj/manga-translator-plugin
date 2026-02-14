"""UI dialog stubs for GIMP GTK integration and standalone CLI."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def show_progress_dialog(title: str = "Translating...") -> Optional[object]:
    """Show a progress dialog. Returns a dialog object if GTK is available."""
    try:
        import gtk

        dialog = gtk.Window(gtk.WINDOW_TOPLEVEL)
        dialog.set_title(title)
        dialog.set_size_request(400, 100)
        dialog.set_position(gtk.WIN_POS_CENTER)

        vbox = gtk.VBox(False, 8)
        vbox.set_border_width(16)

        label = gtk.Label(title)
        vbox.pack_start(label, False, False, 0)

        progress_bar = gtk.ProgressBar()
        vbox.pack_start(progress_bar, False, False, 0)

        dialog.add(vbox)
        dialog.show_all()

        dialog._label = label
        dialog._progress_bar = progress_bar
        return dialog

    except ImportError:
        logger.debug("GTK not available, using text progress")
        return None


def update_progress(dialog: Optional[object], fraction: float, message: str = "") -> None:
    """Update a progress dialog or print to console."""
    if dialog is not None:
        try:
            dialog._progress_bar.set_fraction(fraction)
            if message:
                dialog._label.set_text(message)
            import gtk
            while gtk.events_pending():
                gtk.main_iteration()
            return
        except Exception:
            pass

    # Console fallback
    pct = int(fraction * 100)
    bar = "=" * (pct // 2) + "-" * (50 - pct // 2)
    msg = f" {message}" if message else ""
    print(f"\r[{bar}] {pct}%{msg}", end="", flush=True)
    if fraction >= 1.0:
        print()


def close_dialog(dialog: Optional[object]) -> None:
    """Close a dialog window."""
    if dialog is not None:
        try:
            dialog.destroy()
        except Exception:
            pass
