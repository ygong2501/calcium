"""
Cache cleaning utilities for calcium simulation system.
"""
import os
import shutil
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time

class CacheCleanerDialog:
    """Dialog for cleaning Python cache files."""
    
    def __init__(self, parent, root_dir=None):
        """
        Initialize the cache cleaner dialog.
        
        Args:
            parent: Parent window
            root_dir (str, optional): Root directory to scan. Defaults to current directory.
        """
        self.parent = parent
        
        # Default to project root if not specified
        if root_dir is None:
            # Get the parent directory of the script location
            self.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        else:
            self.root_dir = root_dir
        
        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Cache Cleaner")
        self.dialog.geometry("600x400")
        self.dialog.resizable(True, True)
        self.dialog.transient(parent)  # Set to be a transient window of the parent
        self.dialog.grab_set()  # Make dialog modal
        
        # Center the dialog on the parent window
        self.center_window()
        
        # Directory selection
        dir_frame = ttk.Frame(self.dialog)
        dir_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(dir_frame, text="Directory:").pack(side=tk.LEFT, padx=5)
        
        self.dir_var = tk.StringVar(value=self.root_dir)
        dir_entry = ttk.Entry(dir_frame, textvariable=self.dir_var, width=50)
        dir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        browse_btn = ttk.Button(dir_frame, text="Browse...", command=self.browse_directory)
        browse_btn.pack(side=tk.LEFT, padx=5)
        
        # Cache file types
        types_frame = ttk.LabelFrame(self.dialog, text="Cache Types to Clean")
        types_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Variable for each cache type
        self.pycache_var = tk.BooleanVar(value=True)
        self.pyc_var = tk.BooleanVar(value=True)
        self.pyo_var = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(types_frame, text="__pycache__ directories", variable=self.pycache_var).pack(anchor=tk.W, padx=5, pady=3)
        ttk.Checkbutton(types_frame, text=".pyc files", variable=self.pyc_var).pack(anchor=tk.W, padx=5, pady=3)
        ttk.Checkbutton(types_frame, text=".pyo files", variable=self.pyo_var).pack(anchor=tk.W, padx=5, pady=3)
        
        # Results display
        results_frame = ttk.LabelFrame(self.dialog, text="Cache Files Found")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create scrollable text widget for results
        self.results_text = tk.Text(results_frame, wrap=tk.WORD, height=10)
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Add scrollbar to the text widget
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=scrollbar.set)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.dialog, textvariable=self.status_var, anchor=tk.W, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(self.dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.scan_btn = ttk.Button(button_frame, text="Scan", command=self.scan_for_cache)
        self.scan_btn.pack(side=tk.LEFT, padx=5)
        
        self.clean_btn = ttk.Button(button_frame, text="Clean", command=self.clean_cache, state=tk.DISABLED)
        self.clean_btn.pack(side=tk.LEFT, padx=5)
        
        close_btn = ttk.Button(button_frame, text="Close", command=self.dialog.destroy)
        close_btn.pack(side=tk.RIGHT, padx=5)
        
        # Initialize variables to store scan results
        self.cache_files = []
    
    def center_window(self):
        """Center the dialog window on the parent window."""
        self.dialog.update_idletasks()
        
        # Get parent window position and size
        parent_x = self.parent.winfo_rootx()
        parent_y = self.parent.winfo_rooty()
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()
        
        # Get dialog size
        dialog_width = self.dialog.winfo_width()
        dialog_height = self.dialog.winfo_height()
        
        # Calculate position
        x = parent_x + (parent_width - dialog_width) // 2
        y = parent_y + (parent_height - dialog_height) // 2
        
        # Set position
        self.dialog.geometry(f"+{x}+{y}")
    
    def browse_directory(self):
        """Open directory browser dialog."""
        directory = filedialog.askdirectory(
            initialdir=self.dir_var.get(),
            title="Select Directory"
        )
        
        if directory:
            self.dir_var.set(directory)
    
    def scan_for_cache(self):
        """Scan for cache files."""
        directory = self.dir_var.get()
        
        if not os.path.isdir(directory):
            messagebox.showerror("Error", f"Directory not found: {directory}")
            return
        
        # Disable buttons during scan
        self.scan_btn.config(state=tk.DISABLED)
        self.clean_btn.config(state=tk.DISABLED)
        
        # Clear results
        self.results_text.delete(1.0, tk.END)
        self.status_var.set("Scanning...")
        
        # Start scan in a separate thread
        threading.Thread(target=self._scan_thread, args=(directory,), daemon=True).start()
    
    def _scan_thread(self, directory):
        """Background thread for scanning."""
        try:
            self.cache_files = []
            
            # Check which cache types to scan for
            scan_pycache = self.pycache_var.get()
            scan_pyc = self.pyc_var.get()
            scan_pyo = self.pyo_var.get()
            
            total_size = 0
            
            # Walk through the directory tree
            for root, dirs, files in os.walk(directory):
                # Check for __pycache__ directories
                if scan_pycache:
                    for dir_name in dirs:
                        if dir_name == "__pycache__":
                            cache_path = os.path.join(root, dir_name)
                            size = self._get_dir_size(cache_path)
                            total_size += size
                            self.cache_files.append((cache_path, size, 'dir'))
                            
                            # Update results in the GUI thread
                            self.dialog.after(0, lambda p=cache_path, s=size: self._add_result_line(p, s, 'dir'))
                
                # Check for .pyc and .pyo files
                for file_name in files:
                    if (scan_pyc and file_name.endswith('.pyc')) or (scan_pyo and file_name.endswith('.pyo')):
                        cache_path = os.path.join(root, file_name)
                        size = os.path.getsize(cache_path)
                        total_size += size
                        self.cache_files.append((cache_path, size, 'file'))
                        
                        # Update results in the GUI thread
                        self.dialog.after(0, lambda p=cache_path, s=size: self._add_result_line(p, s, 'file'))
            
            # Update status with total count and size
            count = len(self.cache_files)
            self.dialog.after(0, lambda: self.status_var.set(
                f"Found {count} cache items ({self._format_size(total_size)})"
            ))
            
            # Enable buttons
            self.dialog.after(0, lambda: self.scan_btn.config(state=tk.NORMAL))
            if count > 0:
                self.dialog.after(0, lambda: self.clean_btn.config(state=tk.NORMAL))
        
        except Exception as e:
            self.dialog.after(0, lambda: self.status_var.set(f"Error: {str(e)}"))
            self.dialog.after(0, lambda: self.scan_btn.config(state=tk.NORMAL))
    
    def _add_result_line(self, path, size, item_type):
        """Add a line to the results text widget."""
        item_icon = '📁' if item_type == 'dir' else '📄'
        rel_path = os.path.relpath(path, self.dir_var.get())
        size_str = self._format_size(size)
        line = f"{item_icon} {rel_path} ({size_str})\n"
        
        self.results_text.insert(tk.END, line)
        self.results_text.see(tk.END)  # Scroll to show the new line
    
    def _get_dir_size(self, path):
        """Calculate the size of a directory in bytes."""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                file_path = os.path.join(dirpath, f)
                if not os.path.islink(file_path):  # Skip symbolic links
                    total_size += os.path.getsize(file_path)
        return total_size
    
    def _format_size(self, size_bytes):
        """Format bytes into a human-readable size string."""
        if size_bytes < 1024:
            return f"{size_bytes} bytes"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
    
    def clean_cache(self):
        """Clean found cache files."""
        if not self.cache_files:
            messagebox.showinfo("Info", "No cache files to clean.")
            return
        
        # Confirm before cleaning
        count = len(self.cache_files)
        total_size = sum(item[1] for item in self.cache_files)
        
        confirm = messagebox.askyesno(
            "Confirm Cache Cleanup",
            f"Are you sure you want to delete {count} cache items ({self._format_size(total_size)})?\n\n"
            "This action cannot be undone."
        )
        
        if not confirm:
            return
        
        # Disable buttons during cleaning
        self.scan_btn.config(state=tk.DISABLED)
        self.clean_btn.config(state=tk.DISABLED)
        
        # Start cleaning in a separate thread
        threading.Thread(target=self._clean_thread, daemon=True).start()
    
    def _clean_thread(self):
        """Background thread for cleaning."""
        try:
            self.status_var.set("Cleaning cache files...")
            
            # Track progress
            total = len(self.cache_files)
            deleted = 0
            errors = 0
            
            for path, _, item_type in self.cache_files:
                try:
                    # Delete file or directory
                    if item_type == 'dir':
                        if os.path.exists(path):
                            shutil.rmtree(path)
                    else:
                        if os.path.exists(path):
                            os.remove(path)
                    
                    deleted += 1
                    
                    # Update status occasionally (not for every file to reduce GUI updates)
                    if deleted % 10 == 0 or deleted == total:
                        self.dialog.after(0, lambda d=deleted, t=total: 
                                        self.status_var.set(f"Deleted {d}/{t} cache items..."))
                
                except Exception as e:
                    errors += 1
                    # Log errors in results
                    self.dialog.after(0, lambda p=path, e=str(e): 
                                    self.results_text.insert(tk.END, f"Error deleting {p}: {e}\n"))
            
            # Final status update
            self.dialog.after(0, lambda d=deleted, t=total, e=errors: 
                            self.status_var.set(f"Completed: Deleted {d}/{t} cache items. Errors: {e}"))
            
            # Clear cache files list
            self.cache_files = []
            
            # Re-enable scan button, but keep clean button disabled (nothing to clean now)
            self.dialog.after(0, lambda: self.scan_btn.config(state=tk.NORMAL))
            self.dialog.after(0, lambda: self.clean_btn.config(state=tk.DISABLED))
            
            # Show completion message
            if errors == 0:
                self.dialog.after(0, lambda: messagebox.showinfo(
                    "Cleanup Complete", 
                    f"Successfully deleted {deleted} cache items."
                ))
            else:
                self.dialog.after(0, lambda: messagebox.showwarning(
                    "Cleanup Completed with Errors", 
                    f"Deleted {deleted} cache items. Failed to delete {errors} items."
                ))
        
        except Exception as e:
            self.dialog.after(0, lambda: self.status_var.set(f"Error: {str(e)}"))
            self.dialog.after(0, lambda: self.scan_btn.config(state=tk.NORMAL))


def show_cache_cleaner(parent, root_dir=None):
    """
    Show the cache cleaner dialog.
    
    Args:
        parent: Parent window
        root_dir (str, optional): Root directory to scan. Defaults to current directory.
    """
    CacheCleanerDialog(parent, root_dir)


if __name__ == "__main__":
    # For testing the dialog independently
    root = tk.Tk()
    root.title("Cache Cleaner Test")
    root.geometry("200x100")
    
    ttk.Button(root, text="Open Cache Cleaner", 
               command=lambda: show_cache_cleaner(root)).pack(padx=20, pady=30)
    
    root.mainloop()