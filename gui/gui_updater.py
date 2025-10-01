"""
GUI update helper for thread-safe GUI operations.

This module provides the GUIUpdater class to handle batched, thread-safe
updates to Tkinter GUI components from background threads.
"""
import tkinter as tk
from typing import Optional, Callable, Any, Dict
from collections import deque
import threading


class GUIUpdater:
    """
    Helper class for thread-safe GUI updates.

    Provides batched update mechanism to minimize GUI thread overhead when
    receiving frequent updates from background threads.

    Attributes:
        root (tk.Tk): Root Tkinter window.
        update_interval_ms (int): Milliseconds between batch updates.
        _update_queue (deque): Queue of pending GUI updates.
        _running (bool): Whether updater is actively processing.
        _lock (threading.Lock): Thread synchronization lock.
    """

    def __init__(self, root: tk.Tk, update_interval_ms: int = 50):
        """
        Initialize GUI updater.

        Args:
            root: Root Tkinter window.
            update_interval_ms: Milliseconds between batch updates (default: 50ms = 20 FPS).
        """
        self.root = root
        self.update_interval_ms = update_interval_ms
        self._update_queue: deque = deque()
        self._running = False
        self._lock = threading.Lock()

    def queue_update(self, update_type: str, **kwargs):
        """
        Queue a GUI update for batch processing.

        Args:
            update_type: Type of update ('status', 'progress', 'log', etc.).
            **kwargs: Update-specific parameters.
        """
        with self._lock:
            self._update_queue.append({
                'type': update_type,
                'data': kwargs
            })

    def queue_status(self, message: str):
        """
        Queue a status message update.

        Args:
            message: Status message to display.
        """
        self.queue_update('status', message=message)

    def queue_progress(self, current: int, total: int, details: Optional[Dict[str, Any]] = None):
        """
        Queue a progress update.

        Args:
            current: Current progress value.
            total: Total progress value.
            details: Optional additional progress details.
        """
        self.queue_update('progress', current=current, total=total, details=details)

    def queue_log(self, message: str):
        """
        Queue a log message update.

        Args:
            message: Log message to append.
        """
        self.queue_update('log', message=message)

    def start_updates(self, handlers: Dict[str, Callable]):
        """
        Start processing queued updates.

        Args:
            handlers: Dictionary mapping update_type -> handler function.
                Each handler receives the update data dict as **kwargs.

        Example:
            >>> updater.start_updates({
            ...     'status': lambda message: status_label.config(text=message),
            ...     'progress': lambda current, total, **kw: progress_bar.set(current/total)
            ... })
        """
        self._running = True
        self._handlers = handlers
        self._process_updates()

    def stop_updates(self):
        """Stop processing updates."""
        self._running = False

    def _process_updates(self):
        """Process all queued updates (internal method)."""
        if not self._running:
            return

        # Process all pending updates in batch
        updates_to_process = []
        with self._lock:
            while self._update_queue:
                updates_to_process.append(self._update_queue.popleft())

        # Execute handlers for each update
        for update in updates_to_process:
            update_type = update['type']
            update_data = update['data']

            if update_type in self._handlers:
                try:
                    self._handlers[update_type](**update_data)
                except Exception as e:
                    # Silently ignore handler errors to prevent GUI freezing
                    print(f"Warning: GUI update handler error for '{update_type}': {e}")

        # Schedule next batch update
        if self._running:
            self.root.after(self.update_interval_ms, self._process_updates)

    def clear_queue(self):
        """Clear all pending updates."""
        with self._lock:
            self._update_queue.clear()

    def has_pending_updates(self) -> bool:
        """
        Check if there are pending updates.

        Returns:
            True if updates are queued.
        """
        with self._lock:
            return len(self._update_queue) > 0


class ProgressTracker:
    """
    Helper for tracking and formatting progress information.

    Works with GUIUpdater to provide consistent progress display.
    """

    def __init__(self, gui_updater: GUIUpdater):
        """
        Initialize progress tracker.

        Args:
            gui_updater: GUIUpdater instance to send updates to.
        """
        self.updater = gui_updater
        self.current = 0
        self.total = 0
        self.details = {}

    def update(self, current: int, total: int, **details):
        """
        Update progress and send to GUI.

        Args:
            current: Current progress value.
            total: Total progress value.
            **details: Additional progress details (batch, simulation, etc.).
        """
        self.current = current
        self.total = total
        self.details = details
        self.updater.queue_progress(current, total, details)

    def increment(self, amount: int = 1, **details):
        """
        Increment progress by amount.

        Args:
            amount: Amount to increment by.
            **details: Additional progress details.
        """
        self.update(self.current + amount, self.total, **details)

    def reset(self, total: int = 0):
        """
        Reset progress tracker.

        Args:
            total: New total value.
        """
        self.current = 0
        self.total = total
        self.details = {}

    def get_percentage(self) -> float:
        """
        Get current progress as percentage.

        Returns:
            Progress percentage (0-100).
        """
        if self.total == 0:
            return 0.0
        return (self.current / self.total) * 100.0

    def format_status(self, template: str = "{current}/{total} ({percent:.1f}%)") -> str:
        """
        Format progress as string.

        Args:
            template: Format string with {current}, {total}, {percent} placeholders.

        Returns:
            Formatted progress string.
        """
        return template.format(
            current=self.current,
            total=self.total,
            percent=self.get_percentage()
        )
