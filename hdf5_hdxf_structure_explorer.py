
from __future__ import annotations

import os
import sys
import json
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import h5py
import numpy as np

try:
    import hdf5plugin  # noqa: F401
except Exception:
    hdf5plugin = None

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QBrush, QColor, QFont, QPalette
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QFileDialog, QHeaderView, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QMessageBox, QPushButton, QSplitter, QStatusBar,
    QTabWidget, QTableWidget, QTableWidgetItem, QTextEdit, QToolBar,
    QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget
)

ROLE_MODE = Qt.UserRole + 1
ROLE_FILE = Qt.UserRole + 2
ROLE_PATH = Qt.UserRole + 3
ROLE_KIND = Qt.UserRole + 4
ROLE_EXTRA = Qt.UserRole + 5


def human_bytes(n: Optional[int]) -> str:
    if n is None:
        return "-"
    n = int(n)
    x = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if x < 1024.0 or unit == "TiB":
            return f"{int(x)} B" if unit == "B" else f"{x:.2f} {unit}"
        x /= 1024.0
    return str(n)


def short_value(v: Any, limit: int = 2500) -> str:
    try:
        if isinstance(v, bytes):
            try:
                s = v.decode("utf-8")
            except Exception:
                s = repr(v)
        elif isinstance(v, np.ndarray):
            if v.size <= 64:
                s = np.array2string(v, threshold=64)
            else:
                s = f"<array shape={v.shape} dtype={v.dtype} elements={v.size}>"
        else:
            s = str(v)
    except Exception:
        s = repr(v)
    return s if len(s) <= limit else s[:limit] + " ..."


def filter_pipeline(ds: h5py.Dataset) -> list[dict[str, Any]]:
    out = []
    try:
        plist = ds.id.get_create_plist()
        for i in range(plist.get_nfilters()):
            info = plist.get_filter(i)
            name = info[3] if len(info) > 3 else None
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            out.append(
                {
                    "id": info[0],
                    "flags": info[1] if len(info) > 1 else None,
                    "cd_values": info[2] if len(info) > 2 else None,
                    "name": name,
                }
            )
    except Exception:
        pass
    return out


@dataclass
class Occurrence:
    file_path: str
    kind: str
    shape: Optional[tuple] = None
    dtype: Optional[str] = None
    logical_bytes: Optional[int] = None
    storage_bytes: Optional[int] = None
    chunks: Any = None
    compression: Any = None
    filters: Optional[list] = None
    target: Any = None
    attr_names: Optional[list[str]] = None


class HDF5FolderExplorer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HDF5 / HDXF Structure Explorer")
        self.resize(1650, 950)

        self.folder_path: Optional[str] = None
        self.files: Dict[str, h5py.File] = {}
        self.master_file: Optional[str] = None

        self.path_occurrences: Dict[str, list[Occurrence]] = defaultdict(list)
        self.path_kinds: Dict[str, str] = {}

        # Optional HDXF archive opened alongside the HDF5 experiment.
        self.hdxf_path: Optional[str] = None
        self.hdxf_zip: Optional[zipfile.ZipFile] = None
        self.hdxf_manifest: Optional[dict[str, Any]] = None
        self.hdxf_payload_encodings: Dict[str, set[str]] = defaultdict(set)

        # Cross-pane path highlighting. The exact counterpart is painted more
        # strongly; every ancestor is painted more softly so a collapsed branch
        # still shows the route to the hidden matching field.
        self._cross_highlighted_items: list[QTreeWidgetItem] = []

        self._build_ui()
        self._build_toolbar()
        self.setStatusBar(QStatusBar(self))

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_toolbar(self):
        tb = QToolBar("Main", self)
        self.addToolBar(tb)

        a = QAction("Open HDF5 Folder", self)
        a.triggered.connect(self.open_folder_dialog)
        tb.addAction(a)

        a = QAction("Open HDXF", self)
        a.triggered.connect(self.open_hdxf_dialog)
        tb.addAction(a)

        a = QAction("Refresh", self)
        a.triggered.connect(self.refresh_all)
        tb.addAction(a)

        tb.addSeparator()

        a = QAction("Expand 2 levels", self)
        a.triggered.connect(lambda: self.expand_depth(2))
        tb.addAction(a)

        a = QAction("Collapse all", self)
        a.triggered.connect(self.collapse_all_views)
        tb.addAction(a)

        tb.addSeparator()

        a = QAction("Field Summary", self)
        a.triggered.connect(self.build_summary)
        tb.addAction(a)

    def _configure_interactive_header(self, view, widths=None):
        """Make every column manually resizable and keep horizontal scrolling available."""
        header = view.header() if isinstance(view, QTreeWidget) else view.horizontalHeader()
        header.setStretchLastSection(False)
        for i in range(header.count()):
            header.setSectionResizeMode(i, QHeaderView.Interactive)
        if widths:
            for i, width in enumerate(widths):
                if i < header.count():
                    view.setColumnWidth(i, width)
        if isinstance(view, QTreeWidget):
            view.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        else:
            view.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            view.setWordWrap(False)

    def _build_ui(self):
        root = QWidget()
        main = QVBoxLayout(root)
        main.setContentsMargins(6, 6, 6, 6)
        main.setSpacing(6)

        # The top comparison area contains two completely independent views.
        # The splitter handle can be dragged to give either HDF5 or HDXF more width.
        self.compare_splitter = QSplitter(Qt.Horizontal)
        self.compare_splitter.setChildrenCollapsible(False)
        self.compare_splitter.setHandleWidth(7)

        # ------------------------------ HDF5 pane ------------------------------
        hdf5_panel = QWidget()
        hdf5_layout = QVBoxLayout(hdf5_panel)
        hdf5_layout.setContentsMargins(0, 0, 0, 0)
        hdf5_layout.setSpacing(4)

        hdf5_top = QHBoxLayout()
        hdf5_title = QLabel("HDF5 experiment")
        f = hdf5_title.font()
        f.setBold(True)
        hdf5_title.setFont(f)
        hdf5_top.addWidget(hdf5_title)
        hdf5_top.addStretch(1)
        hdf5_top.addWidget(QLabel("Search:"))
        self.hdf5_search = QLineEdit()
        self.hdf5_search.setPlaceholderText("detector, beam_center, pedestal ...")
        self.hdf5_search.setMinimumWidth(240)
        self.hdf5_search.textChanged.connect(self.apply_hdf5_search)
        hdf5_top.addWidget(self.hdf5_search)
        hdf5_layout.addLayout(hdf5_top)

        self.hdf5_tree = QTreeWidget()
        self.hdf5_tree.setHeaderLabels(["Object", "Type", "Present", "Shape / Target"])
        self._configure_interactive_header(self.hdf5_tree, [360, 140, 95, 280])
        self.hdf5_tree.itemSelectionChanged.connect(
            lambda: self.selection_changed(self.hdf5_tree)
        )
        hdf5_layout.addWidget(self.hdf5_tree, 1)
        self.compare_splitter.addWidget(hdf5_panel)

        # ------------------------------- HDXF pane -----------------------------
        hdxf_panel = QWidget()
        hdxf_layout = QVBoxLayout(hdxf_panel)
        hdxf_layout.setContentsMargins(0, 0, 0, 0)
        hdxf_layout.setSpacing(4)

        hdxf_top = QHBoxLayout()
        hdxf_title = QLabel("HDXF archive")
        f = hdxf_title.font()
        f.setBold(True)
        hdxf_title.setFont(f)
        hdxf_top.addWidget(hdxf_title)
        hdxf_top.addStretch(1)
        hdxf_top.addWidget(QLabel("Search:"))
        self.hdxf_search = QLineEdit()
        self.hdxf_search.setPlaceholderText("payload, manifest, detector_archive ...")
        self.hdxf_search.setMinimumWidth(240)
        self.hdxf_search.textChanged.connect(self.apply_hdxf_search)
        hdxf_top.addWidget(self.hdxf_search)
        hdxf_layout.addLayout(hdxf_top)

        self.hdxf_tree = QTreeWidget()
        self.hdxf_tree.setHeaderLabels(["Object", "Type", "Source / Count", "Shape / Encoding"])
        self._configure_interactive_header(self.hdxf_tree, [360, 160, 180, 300])
        self.hdxf_tree.itemSelectionChanged.connect(
            lambda: self.selection_changed(self.hdxf_tree)
        )
        hdxf_layout.addWidget(self.hdxf_tree, 1)
        self.compare_splitter.addWidget(hdxf_panel)
        self.compare_splitter.setSizes([820, 820])

        # Detail tabs are shared by the currently selected item, but are placed
        # below the two trees so the HDF5 and HDXF structures remain visible side-by-side.
        self.tabs = QTabWidget()

        # Overview
        page = QWidget()
        lay = QVBoxLayout(page)

        self.info = QTableWidget(0, 2)
        self.info.setHorizontalHeaderLabels(["Property", "Value"])
        self._configure_interactive_header(self.info, [240, 780])
        self.info.verticalHeader().setVisible(False)
        self.info.setEditTriggers(QTableWidget.NoEditTriggers)
        lay.addWidget(self.info)

        preview_row = QHBoxLayout()
        self.preview_btn = QPushButton("Preview dataset value")
        self.preview_btn.setEnabled(False)
        self.preview_btn.clicked.connect(self.preview_dataset)
        preview_row.addWidget(self.preview_btn)
        self.allow_filtered_preview = QCheckBox("Allow filtered value preview")
        self.allow_filtered_preview.setChecked(False)
        self.allow_filtered_preview.setToolTip(
            "Disabled by default. Structure inspection never decodes dataset data."
        )
        preview_row.addWidget(self.allow_filtered_preview)
        preview_row.addStretch(1)
        lay.addLayout(preview_row)

        self.preview = QTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setPlaceholderText("Dataset values are never read automatically.")
        lay.addWidget(self.preview, 1)
        self.tabs.addTab(page, "Overview")

        # Occurrences / Files
        page = QWidget()
        lay = QVBoxLayout(page)
        self.occ_table = QTableWidget(0, 8)
        self.occ_table.setHorizontalHeaderLabels(
            ["File", "Kind", "Shape", "dtype", "Logical", "Storage", "Chunks", "Compression"]
        )
        self._configure_interactive_header(
            self.occ_table, [330, 100, 160, 120, 105, 105, 180, 150]
        )
        self.occ_table.verticalHeader().setVisible(False)
        self.occ_table.setEditTriggers(QTableWidget.NoEditTriggers)
        lay.addWidget(self.occ_table)
        self.tabs.addTab(page, "Occurrences")

        # Attributes
        page = QWidget()
        lay = QVBoxLayout(page)
        self.attrs = QTableWidget(0, 5)
        self.attrs.setHorizontalHeaderLabels(["File", "Attribute", "Value", "dtype", "shape"])
        self._configure_interactive_header(self.attrs, [320, 190, 500, 130, 130])
        self.attrs.verticalHeader().setVisible(False)
        self.attrs.setEditTriggers(QTableWidget.NoEditTriggers)
        lay.addWidget(self.attrs)
        self.tabs.addTab(page, "Attributes")

        # Storage / raw HDXF JSON
        page = QWidget()
        lay = QVBoxLayout(page)
        self.storage = QTextEdit()
        self.storage.setReadOnly(True)
        lay.addWidget(self.storage)
        self.tabs.addTab(page, "Storage / Filters / JSON")

        # Field summary
        page = QWidget()
        lay = QVBoxLayout(page)
        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        lay.addWidget(self.summary)
        self.tabs.addTab(page, "HDF5 Field Summary")

        mono = QFont("Consolas")
        mono.setStyleHint(QFont.Monospace)
        self.preview.setFont(mono)
        self.storage.setFont(mono)
        self.summary.setFont(mono)

        # Vertical splitter: users can freely resize the comparison trees versus details.
        self.main_splitter = QSplitter(Qt.Vertical)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setHandleWidth(7)
        self.main_splitter.addWidget(self.compare_splitter)
        self.main_splitter.addWidget(self.tabs)
        self.main_splitter.setSizes([610, 330])
        main.addWidget(self.main_splitter, 1)

        self.active_tree = self.hdf5_tree
        self.setCentralWidget(root)

    # ------------------------------------------------------------------
    # Folder lifecycle
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        self.close_all_files()
        self.close_hdxf()
        super().closeEvent(event)

    def close_all_files(self):
        for f in list(self.files.values()):
            try:
                f.close()
            except Exception:
                pass
        self.files.clear()

    def close_hdxf(self):
        if self.hdxf_zip is not None:
            try:
                self.hdxf_zip.close()
            except Exception:
                pass
        self.hdxf_zip = None
        self.hdxf_path = None
        self.hdxf_manifest = None
        self.hdxf_payload_encodings.clear()

    def open_hdxf_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open HDXF archive",
            self.folder_path or "",
            "HDXF archives (*.hdxf);;ZIP-compatible archives (*.zip);;All files (*)",
        )
        if path:
            self.load_hdxf(path)

    def load_hdxf(self, path: str):
        path = os.path.abspath(path)
        old_zip = self.hdxf_zip
        try:
            zf = zipfile.ZipFile(path, "r")
            if "manifest.json" not in zf.namelist():
                zf.close()
                raise ValueError("manifest.json is missing; this is not a supported HDXF archive")
            manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
            if not isinstance(manifest, dict):
                zf.close()
                raise ValueError("manifest.json root is not a JSON object")
        except Exception as exc:
            QMessageBox.critical(self, "Open HDXF failed", f"{path}\n\n{type(exc).__name__}: {exc}")
            return

        if old_zip is not None:
            try:
                old_zip.close()
            except Exception:
                pass

        self.hdxf_zip = zf
        self.hdxf_path = path
        self.hdxf_manifest = manifest
        self.hdxf_payload_encodings.clear()
        self._index_hdxf_payload_encodings(manifest)
        self.populate_tree()

        hdxf = manifest.get("hdxf", {}) if isinstance(manifest, dict) else {}
        version = hdxf.get("version", "?") if isinstance(hdxf, dict) else "?"
        profile = hdxf.get("profile", "?") if isinstance(hdxf, dict) else "?"
        self.statusBar().showMessage(
            f"HDXF: {Path(path).name}   version={version}   profile={profile}   "
            f"size={human_bytes(os.path.getsize(path))}"
        )

    def _index_hdxf_payload_encodings(self, value: Any):
        """Build member -> encoding hints by recursively scanning the manifest."""
        if isinstance(value, dict):
            p = value.get("path")
            enc = value.get("encoding")
            if isinstance(p, str) and p.startswith("payloads/") and isinstance(enc, str):
                self.hdxf_payload_encodings[p].add(enc)
            for child in value.values():
                self._index_hdxf_payload_encodings(child)
        elif isinstance(value, list):
            for child in value:
                self._index_hdxf_payload_encodings(child)

    def open_folder_dialog(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Open folder containing one HDF5 experiment",
            "",
        )
        if folder:
            self.load_folder(folder)

    def refresh_folder(self):
        if self.folder_path:
            self.load_folder(self.folder_path)

    def refresh_all(self):
        folder = self.folder_path
        hdxf = self.hdxf_path
        if folder:
            self.load_folder(folder)
        if hdxf and os.path.isfile(hdxf):
            self.load_hdxf(hdxf)

    def load_folder(self, folder: str):
        self.close_all_files()
        self.path_occurrences.clear()
        self.path_kinds.clear()

        folder = os.path.abspath(folder)
        candidates = []
        for p in sorted(Path(folder).iterdir()):
            if not p.is_file():
                continue
            if p.suffix.lower() in {".h5", ".hdf5", ".nxs"}:
                candidates.append(str(p))

        if not candidates:
            QMessageBox.warning(
                self,
                "No HDF5 files",
                "No .h5 / .hdf5 / .nxs files were found in this folder.",
            )
            return

        failed = []
        for path in candidates:
            try:
                self.files[path] = h5py.File(path, "r")
            except Exception as exc:
                failed.append((path, str(exc)))

        if not self.files:
            QMessageBox.critical(
                self,
                "Open failed",
                "None of the HDF5 files in this folder could be opened.",
            )
            return

        self.folder_path = folder
        self.master_file = self.detect_master_file()
        self.scan_all_files()
        self.populate_tree()
        self.build_summary()

        title = f"HDF5 / HDXF Structure Explorer — {Path(folder).name}"
        self.setWindowTitle(title)

        total_size = sum(
            os.path.getsize(p) for p in self.files if os.path.exists(p)
        )
        msg = (
            f"Folder: {folder}   Files: {len(self.files)}   "
            f"Total size: {human_bytes(total_size)}"
        )
        if self.master_file:
            msg += f"   Master: {Path(self.master_file).name}"
        self.statusBar().showMessage(msg)

        if failed:
            QMessageBox.warning(
                self,
                "Some files could not be opened",
                "\n".join(f"{p}: {e}" for p, e in failed[:10]),
            )

    def detect_master_file(self) -> Optional[str]:
        # Strong filename signal first.
        named = [
            p for p in self.files
            if "master" in Path(p).stem.lower()
        ]
        if len(named) == 1:
            return named[0]
        if named:
            named.sort(key=lambda p: len(Path(p).name))
            return named[0]

        # Fallback: file with the largest number of external links.
        best = None
        best_count = -1

        for path, f in self.files.items():
            count = 0

            def visit(name):
                nonlocal count
                parent_path, _, leaf = ("/" + name).rpartition("/")
                if not parent_path:
                    parent_path = "/"
                try:
                    parent = f[parent_path]
                    link = parent.get(leaf, getlink=True)
                    if isinstance(link, h5py.ExternalLink):
                        count += 1
                except Exception:
                    pass

            try:
                f.visit(visit)
            except Exception:
                pass

            if count > best_count:
                best_count = count
                best = path

        return best

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------

    def scan_all_files(self):
        self.path_occurrences.clear()

        for file_path, f in self.files.items():
            # Root group
            self.path_occurrences["/"].append(
                Occurrence(
                    file_path=file_path,
                    kind="group",
                    attr_names=sorted(list(f["/"].attrs.keys())),
                )
            )

            self._scan_group(file_path, f["/"], "/")

    def _scan_group(self, file_path: str, group: h5py.Group, group_path: str):
        try:
            names = sorted(group.keys())
        except Exception:
            return

        for name in names:
            path = "/" + name if group_path == "/" else group_path.rstrip("/") + "/" + name

            try:
                link = group.get(name, getlink=True)
            except Exception:
                continue

            if isinstance(link, h5py.ExternalLink):
                self.path_occurrences[path].append(
                    Occurrence(
                        file_path=file_path,
                        kind="external_link",
                        target=(link.filename, link.path),
                    )
                )
                continue

            if isinstance(link, h5py.SoftLink):
                self.path_occurrences[path].append(
                    Occurrence(
                        file_path=file_path,
                        kind="soft_link",
                        target=link.path,
                    )
                )
                continue

            try:
                obj = group[name]
            except Exception:
                self.path_occurrences[path].append(
                    Occurrence(file_path=file_path, kind="unreadable")
                )
                continue

            if isinstance(obj, h5py.Group):
                self.path_occurrences[path].append(
                    Occurrence(
                        file_path=file_path,
                        kind="group",
                        attr_names=sorted(list(obj.attrs.keys())),
                    )
                )
                self._scan_group(file_path, obj, path)

            elif isinstance(obj, h5py.Dataset):
                logical = None
                storage = None
                try:
                    logical = int(obj.size) * int(obj.dtype.itemsize)
                except Exception:
                    pass
                try:
                    storage = int(obj.id.get_storage_size())
                except Exception:
                    pass

                self.path_occurrences[path].append(
                    Occurrence(
                        file_path=file_path,
                        kind="dataset",
                        shape=tuple(obj.shape),
                        dtype=str(obj.dtype),
                        logical_bytes=logical,
                        storage_bytes=storage,
                        chunks=obj.chunks,
                        compression=obj.compression,
                        filters=filter_pipeline(obj),
                        attr_names=sorted(list(obj.attrs.keys())),
                    )
                )

    # ------------------------------------------------------------------
    # Tree
    # ------------------------------------------------------------------

    def populate_tree(self):
        self._clear_cross_highlights()
        self.hdf5_tree.clear()
        self.hdxf_tree.clear()

        if self.folder_path and self.files:
            root = QTreeWidgetItem(
                [
                    Path(self.folder_path).name,
                    "HDF5 experiment",
                    f"{len(self.files)} files",
                    human_bytes(sum(os.path.getsize(p) for p in self.files)),
                ]
            )
            root.setData(0, ROLE_MODE, "experiment_root")
            self.hdf5_tree.addTopLevelItem(root)

            logical = QTreeWidgetItem(
                ["Merged Logical View", "Merged HDF5", f"{len(self.files)} files", ""]
            )
            logical.setData(0, ROLE_MODE, "merged_root")
            root.addChild(logical)

            physical = QTreeWidgetItem(
                ["Physical Files", "HDF5 files", f"{len(self.files)} files", ""]
            )
            physical.setData(0, ROLE_MODE, "physical_root")
            root.addChild(physical)

            self._populate_merged_tree(logical)
            self._populate_physical_tree(physical)

            root.setExpanded(True)
            logical.setExpanded(True)
            physical.setExpanded(False)

        if self.hdxf_path and self.hdxf_zip is not None and self.hdxf_manifest is not None:
            self._populate_hdxf_tree()

        # Preserve current search filters after a reload.
        self.apply_hdf5_search(self.hdf5_search.text())
        self.apply_hdxf_search(self.hdxf_search.text())

    def _populate_merged_tree(self, parent: QTreeWidgetItem):
        nodes = {"/": parent}

        for path in sorted(self.path_occurrences.keys(), key=lambda p: (p.count("/"), p)):
            if path == "/":
                continue

            occs = self.path_occurrences[path]
            kinds = sorted(set(o.kind for o in occs))
            kind = kinds[0] if len(kinds) == 1 else "mixed"

            parent_path = path.rsplit("/", 1)[0] or "/"
            leaf = path.rsplit("/", 1)[-1]

            pitem = nodes.get(parent_path)
            if pitem is None:
                continue

            present = f"{len(occs)}/{len(self.files)}"

            shape_target = ""
            if kind == "dataset":
                shapes = sorted(set(str(o.shape) for o in occs))
                shape_target = shapes[0] if len(shapes) == 1 else f"{len(shapes)} shapes"
            elif kind in {"external_link", "soft_link"}:
                targets = sorted(set(str(o.target) for o in occs))
                shape_target = targets[0] if len(targets) == 1 else f"{len(targets)} targets"

            item = QTreeWidgetItem([leaf, kind, present, shape_target])
            item.setData(0, ROLE_MODE, "merged")
            item.setData(0, ROLE_PATH, path)
            item.setData(0, ROLE_KIND, kind)
            pitem.addChild(item)
            nodes[path] = item

    def _populate_physical_tree(self, parent: QTreeWidgetItem):
        ordered = sorted(
            self.files.keys(),
            key=lambda p: (
                0 if p == self.master_file else 1,
                Path(p).name.lower()
            )
        )

        for file_path in ordered:
            label = Path(file_path).name
            ftype = "Master HDF5" if file_path == self.master_file else "Data HDF5"
            item = QTreeWidgetItem(
                [
                    label,
                    ftype,
                    "",
                    human_bytes(os.path.getsize(file_path)),
                ]
            )
            item.setData(0, ROLE_MODE, "physical_file")
            item.setData(0, ROLE_FILE, file_path)
            item.setData(0, ROLE_PATH, "/")
            parent.addChild(item)

            self._populate_one_physical_file(file_path, item)

    def _populate_one_physical_file(self, file_path: str, root_item: QTreeWidgetItem):
        f = self.files[file_path]

        def rec(group: h5py.Group, parent: QTreeWidgetItem, group_path: str):
            try:
                names = sorted(group.keys())
            except Exception:
                return

            for name in names:
                path = "/" + name if group_path == "/" else group_path.rstrip("/") + "/" + name

                try:
                    link = group.get(name, getlink=True)
                except Exception as exc:
                    parent.addChild(QTreeWidgetItem([name, "Error", "", str(exc)]))
                    continue

                if isinstance(link, h5py.ExternalLink):
                    target = f"{link.filename} :: {link.path}"
                    item = QTreeWidgetItem([name, "External Link", "", target])
                    item.setData(0, ROLE_MODE, "physical")
                    item.setData(0, ROLE_FILE, file_path)
                    item.setData(0, ROLE_PATH, path)
                    item.setData(0, ROLE_KIND, "external_link")
                    item.setData(0, ROLE_EXTRA, (link.filename, link.path))
                    parent.addChild(item)
                    continue

                if isinstance(link, h5py.SoftLink):
                    item = QTreeWidgetItem([name, "Soft Link", "", link.path])
                    item.setData(0, ROLE_MODE, "physical")
                    item.setData(0, ROLE_FILE, file_path)
                    item.setData(0, ROLE_PATH, path)
                    item.setData(0, ROLE_KIND, "soft_link")
                    item.setData(0, ROLE_EXTRA, link.path)
                    parent.addChild(item)
                    continue

                try:
                    obj = group[name]
                except Exception as exc:
                    parent.addChild(QTreeWidgetItem([name, "Unreadable", "", str(exc)]))
                    continue

                if isinstance(obj, h5py.Group):
                    item = QTreeWidgetItem([name, "Group", "", ""])
                    item.setData(0, ROLE_MODE, "physical")
                    item.setData(0, ROLE_FILE, file_path)
                    item.setData(0, ROLE_PATH, path)
                    item.setData(0, ROLE_KIND, "group")
                    parent.addChild(item)
                    rec(obj, item, path)

                elif isinstance(obj, h5py.Dataset):
                    shape = "scalar" if obj.shape == () else str(tuple(obj.shape))
                    item = QTreeWidgetItem([name, "Dataset", "", shape])
                    item.setData(0, ROLE_MODE, "physical")
                    item.setData(0, ROLE_FILE, file_path)
                    item.setData(0, ROLE_PATH, path)
                    item.setData(0, ROLE_KIND, "dataset")
                    parent.addChild(item)

        rec(f["/"], root_item, "/")

    # ------------------------------------------------------------------
    # HDXF tree / details
    # ------------------------------------------------------------------

    def _hdxf_source_map(self) -> Dict[str, dict[str, Any]]:
        manifest = self.hdxf_manifest or {}
        source = manifest.get("source", {}) if isinstance(manifest, dict) else {}
        files = source.get("files", []) if isinstance(source, dict) else []
        out: Dict[str, dict[str, Any]] = {}
        if isinstance(files, list):
            for record in files:
                if isinstance(record, dict) and isinstance(record.get("id"), str):
                    out[record["id"]] = record
        return out

    def _hdxf_object_shape(self, record: dict[str, Any]) -> str:
        storage = record.get("hdf5_storage")
        if not isinstance(storage, dict):
            return ""
        shape = storage.get("shape")
        if shape is None:
            return "scalar" if record.get("type") == "dataset" else ""
        return str(tuple(shape)) if isinstance(shape, list) else str(shape)

    def _hdxf_payload_encoding(self, record: dict[str, Any]) -> str:
        payload = record.get("payload")
        if isinstance(payload, dict):
            return str(payload.get("encoding", ""))
        return ""

    def _populate_hdxf_tree(self):
        manifest = self.hdxf_manifest or {}
        hdxf_meta = manifest.get("hdxf", {}) if isinstance(manifest, dict) else {}
        version = hdxf_meta.get("version", "?") if isinstance(hdxf_meta, dict) else "?"
        profile = hdxf_meta.get("profile", "?") if isinstance(hdxf_meta, dict) else "?"

        root = QTreeWidgetItem(
            [
                Path(self.hdxf_path).name,
                "HDXF archive",
                f"v{version}",
                human_bytes(os.path.getsize(self.hdxf_path)),
            ]
        )
        root.setData(0, ROLE_MODE, "hdxf_root")
        root.setData(0, ROLE_FILE, self.hdxf_path)
        root.setData(0, ROLE_EXTRA, manifest)
        self.hdxf_tree.addTopLevelItem(root)

        overview = QTreeWidgetItem(["Overview", "HDXF metadata", profile, f"v{version}"])
        overview.setData(0, ROLE_MODE, "hdxf_json")
        overview.setData(0, ROLE_KIND, "overview")
        overview.setData(0, ROLE_EXTRA, hdxf_meta)
        root.addChild(overview)

        objects = manifest.get("objects", {}) if isinstance(manifest, dict) else {}
        logical = QTreeWidgetItem([
            "Logical HDF5 Object Graph", "Object graph",
            f"{len(objects) if isinstance(objects, dict) else 0} objects", ""
        ])
        logical.setData(0, ROLE_MODE, "hdxf_section")
        logical.setData(0, ROLE_KIND, "logical_object_graph")
        root.addChild(logical)

        root_object = manifest.get("root_object") if isinstance(manifest, dict) else None
        if isinstance(root_object, str) and isinstance(objects, dict):
            self._add_hdxf_object_node(logical, root_object, display_name="/", visited=set())

        by_source = QTreeWidgetItem(["Archived Objects by Source File", "Objects", "", ""])
        by_source.setData(0, ROLE_MODE, "hdxf_section")
        by_source.setData(0, ROLE_KIND, "objects_by_source")
        root.addChild(by_source)
        self._populate_hdxf_objects_by_source(by_source)

        legacy = manifest.get("legacy_hdf5") if isinstance(manifest, dict) else None
        if isinstance(legacy, dict):
            legacy_node = QTreeWidgetItem([
                "Legacy HDF5 Reconstruction", "Restore metadata",
                f"{len(legacy.get('external_files', []) or [])} external files", ""
            ])
            legacy_node.setData(0, ROLE_MODE, "hdxf_json")
            legacy_node.setData(0, ROLE_KIND, "legacy_hdf5")
            legacy_node.setData(0, ROLE_EXTRA, legacy)
            root.addChild(legacy_node)
            self._populate_hdxf_legacy_tree(legacy_node, legacy)

        detector = manifest.get("detector_archive") if isinstance(manifest, dict) else None
        if isinstance(detector, dict):
            frames = detector.get("frames", {})
            frame_count = frames.get("frame_count", "?") if isinstance(frames, dict) else "?"
            det_node = QTreeWidgetItem([
                "Detector Archive", "Detector profile", f"{frame_count} frames", ""
            ])
            det_node.setData(0, ROLE_MODE, "hdxf_json")
            det_node.setData(0, ROLE_KIND, "detector_archive")
            det_node.setData(0, ROLE_EXTRA, detector)
            root.addChild(det_node)
            self._populate_hdxf_detector_tree(det_node, detector)

        members = QTreeWidgetItem([
            "Archive Members", "ZIP members", f"{len(self.hdxf_zip.infolist())} members", ""
        ])
        members.setData(0, ROLE_MODE, "hdxf_section")
        members.setData(0, ROLE_KIND, "archive_members")
        root.addChild(members)
        self._populate_hdxf_members(members)

        manifest_node = QTreeWidgetItem([
            "manifest.json", "Manifest", f"{len(manifest)} top-level keys", ""
        ])
        manifest_node.setData(0, ROLE_MODE, "hdxf_json")
        manifest_node.setData(0, ROLE_KIND, "manifest")
        manifest_node.setData(0, ROLE_EXTRA, manifest)
        root.addChild(manifest_node)
        for key in ("hdxf", "source", "links", "legacy_hdf5", "detector_archive"):
            if key in manifest:
                child = QTreeWidgetItem([key, "Manifest section", "", ""])
                child.setData(0, ROLE_MODE, "hdxf_json")
                child.setData(0, ROLE_KIND, f"manifest:{key}")
                child.setData(0, ROLE_EXTRA, manifest[key])
                manifest_node.addChild(child)

        root.setExpanded(True)
        logical.setExpanded(True)
        by_source.setExpanded(False)
        members.setExpanded(False)

    def _add_hdxf_object_node(
        self,
        parent: QTreeWidgetItem,
        object_id: str,
        *,
        display_name: Optional[str] = None,
        visited: set[str],
    ):
        manifest = self.hdxf_manifest or {}
        objects = manifest.get("objects", {})
        if not isinstance(objects, dict):
            return
        record = objects.get(object_id)
        if not isinstance(record, dict):
            item = QTreeWidgetItem([display_name or object_id, "Missing object", "", object_id])
            parent.addChild(item)
            return

        path = str(record.get("source_path") or record.get("source_local_path") or "")
        name = display_name or (path.rsplit("/", 1)[-1] if path not in ("", "/") else "/")
        kind = str(record.get("type", "object"))
        shape = self._hdxf_object_shape(record)
        enc = self._hdxf_payload_encoding(record)
        tail = shape
        if enc:
            tail = f"{shape}  [{enc}]" if shape else enc

        source_id = record.get("source_file")
        source_record = self._hdxf_source_map().get(str(source_id), {})
        source_name = source_record.get("filename", source_id or "")

        if object_id in visited:
            item = QTreeWidgetItem([name, f"{kind} reference", str(source_name), tail])
            item.setData(0, ROLE_MODE, "hdxf_object")
            item.setData(0, ROLE_PATH, path)
            item.setData(0, ROLE_KIND, kind)
            item.setData(0, ROLE_EXTRA, {"object_id": object_id, "record": record, "reference": True})
            parent.addChild(item)
            return

        item = QTreeWidgetItem([name, kind, str(source_name), tail])
        item.setData(0, ROLE_MODE, "hdxf_object")
        item.setData(0, ROLE_PATH, path)
        item.setData(0, ROLE_KIND, kind)
        item.setData(0, ROLE_EXTRA, {"object_id": object_id, "record": record})
        parent.addChild(item)

        if kind != "group":
            return

        next_visited = set(visited)
        next_visited.add(object_id)
        children = record.get("children", [])
        if not isinstance(children, list):
            return

        link_map = {}
        links = manifest.get("links", [])
        if isinstance(links, list):
            for link in links:
                if isinstance(link, dict) and isinstance(link.get("path"), str):
                    link_map[link["path"]] = link

        for child in children:
            if not isinstance(child, dict):
                continue
            child_name = str(child.get("name", "?"))
            child_id = child.get("object")
            if isinstance(child_id, str):
                self._add_hdxf_object_node(
                    item, child_id, display_name=child_name, visited=next_visited
                )
            else:
                link_path = child.get("link_path")
                link = link_map.get(link_path, {}) if isinstance(link_path, str) else {}
                link_type = str(link.get("type", "link"))
                target = (
                    link.get("target_path")
                    or link.get("target_file")
                    or link.get("error")
                    or ""
                )
                li = QTreeWidgetItem([child_name, f"{link_type} link", "", str(target)])
                li.setData(0, ROLE_MODE, "hdxf_json")
                li.setData(0, ROLE_KIND, "link")
                li.setData(0, ROLE_PATH, str(link_path or ""))
                li.setData(0, ROLE_EXTRA, link)
                item.addChild(li)

    def _populate_hdxf_objects_by_source(self, parent: QTreeWidgetItem):
        manifest = self.hdxf_manifest or {}
        objects = manifest.get("objects", {})
        if not isinstance(objects, dict):
            return

        source_map = self._hdxf_source_map()
        grouped: Dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
        for object_id, record in objects.items():
            if isinstance(record, dict):
                grouped[str(record.get("source_file", "unknown"))].append((object_id, record))

        source_ids = list(grouped.keys())
        source_ids.sort(key=lambda sid: str(source_map.get(sid, {}).get("filename", sid)).lower())

        for sid in source_ids:
            source_rec = source_map.get(sid, {})
            filename = str(source_rec.get("filename", sid))
            role = str(source_rec.get("role", ""))
            records = grouped[sid]
            file_node = QTreeWidgetItem([filename, role or "source file", f"{len(records)} objects", ""])
            file_node.setData(0, ROLE_MODE, "hdxf_json")
            file_node.setData(0, ROLE_KIND, "source_file")
            file_node.setData(0, ROLE_EXTRA, source_rec)
            parent.addChild(file_node)

            # Build a path tree from source_local_path.  This exposes reconstruction-only
            # datasets such as /entry/detector/* from each physical data file.
            path_nodes: Dict[str, QTreeWidgetItem] = {"/": file_node}
            for object_id, record in sorted(
                records,
                key=lambda pair: (
                    str(pair[1].get("source_local_path", "/")).count("/"),
                    str(pair[1].get("source_local_path", "/")),
                ),
            ):
                path = str(record.get("source_local_path") or record.get("source_path") or "/")
                if path == "/":
                    continue
                parts = [p for p in path.split("/") if p]
                cur_path = ""
                cur_parent = file_node
                for i, part in enumerate(parts):
                    cur_path += "/" + part
                    existing = path_nodes.get(cur_path)
                    is_leaf = i == len(parts) - 1
                    if existing is not None:
                        cur_parent = existing
                        continue
                    if is_leaf:
                        kind = str(record.get("type", "object"))
                        shape = self._hdxf_object_shape(record)
                        enc = self._hdxf_payload_encoding(record)
                        tail = f"{shape}  [{enc}]" if shape and enc else (shape or enc)
                        node = QTreeWidgetItem([part, kind, "", tail])
                        node.setData(0, ROLE_MODE, "hdxf_object")
                        node.setData(0, ROLE_PATH, path)
                        node.setData(0, ROLE_KIND, kind)
                        node.setData(0, ROLE_EXTRA, {"object_id": object_id, "record": record})
                    else:
                        node = QTreeWidgetItem([part, "Path", "", ""])
                        node.setData(0, ROLE_MODE, "hdxf_virtual_path")
                        node.setData(0, ROLE_PATH, cur_path)
                    cur_parent.addChild(node)
                    path_nodes[cur_path] = node
                    cur_parent = node

    def _populate_hdxf_legacy_tree(self, parent: QTreeWidgetItem, legacy: dict[str, Any]):
        template = legacy.get("master_template") or legacy.get("template")
        if template is not None:
            n = QTreeWidgetItem(["Master template", "Template", "", ""])
            n.setData(0, ROLE_MODE, "hdxf_json")
            n.setData(0, ROLE_KIND, "legacy_master_template")
            n.setData(0, ROLE_EXTRA, template)
            parent.addChild(n)

        coverage = legacy.get("coverage")
        if isinstance(coverage, dict):
            ok = coverage.get("all_external_datasets_archived")
            n = QTreeWidgetItem(["Coverage", "Validation", str(ok), ""])
            n.setData(0, ROLE_MODE, "hdxf_json")
            n.setData(0, ROLE_KIND, "legacy_coverage")
            n.setData(0, ROLE_EXTRA, coverage)
            parent.addChild(n)

        ext_files = legacy.get("external_files", [])
        if not isinstance(ext_files, list):
            return
        files_node = QTreeWidgetItem(["External HDF5 Files", "Restore layouts", f"{len(ext_files)} files", ""])
        files_node.setData(0, ROLE_MODE, "hdxf_section")
        files_node.setData(0, ROLE_KIND, "legacy_external_files")
        parent.addChild(files_node)

        for descriptor in ext_files:
            if not isinstance(descriptor, dict):
                continue
            sf = descriptor.get("source_file", {})
            filename = sf.get("filename", "?") if isinstance(sf, dict) else "?"
            objects = descriptor.get("objects", {})
            count = len(objects) if isinstance(objects, dict) else 0
            fnode = QTreeWidgetItem([str(filename), "Physical HDF5 layout", f"{count} objects", ""])
            fnode.setData(0, ROLE_MODE, "hdxf_json")
            fnode.setData(0, ROLE_KIND, "legacy_external_file")
            fnode.setData(0, ROLE_EXTRA, descriptor)
            files_node.addChild(fnode)
            self._add_hdxf_legacy_file_objects(fnode, descriptor)

    def _add_hdxf_legacy_file_objects(self, parent: QTreeWidgetItem, descriptor: dict[str, Any]):
        objects = descriptor.get("objects", {})
        if not isinstance(objects, dict):
            return
        records = [r for r in objects.values() if isinstance(r, dict)]
        records.sort(key=lambda r: (str(r.get("canonical_path", "/")).count("/"), str(r.get("canonical_path", "/"))))
        nodes: Dict[str, QTreeWidgetItem] = {"/": parent}
        for record in records:
            path = str(record.get("canonical_path", "/"))
            if path == "/":
                continue
            parent_path = path.rsplit("/", 1)[0] or "/"
            pnode = nodes.get(parent_path, parent)
            leaf = path.rsplit("/", 1)[-1]
            kind = str(record.get("type", "object"))
            archive_object = record.get("archive_object")
            storage = record.get("hdf5_storage") if isinstance(record.get("hdf5_storage"), dict) else {}
            shape = storage.get("shape") if isinstance(storage, dict) else None
            shape_text = "scalar" if shape is None and kind == "dataset" else (str(tuple(shape)) if isinstance(shape, list) else "")
            tail = shape_text
            if archive_object:
                tail = f"{shape_text}  archive_object={str(archive_object)[:12]}…" if shape_text else f"archive_object={str(archive_object)[:12]}…"
            node = QTreeWidgetItem([leaf, kind, "", tail])
            node.setData(0, ROLE_MODE, "hdxf_json")
            node.setData(0, ROLE_KIND, "legacy_object")
            node.setData(0, ROLE_PATH, path)
            node.setData(0, ROLE_EXTRA, record)
            pnode.addChild(node)
            nodes[path] = node

    def _populate_hdxf_detector_tree(self, parent: QTreeWidgetItem, detector: dict[str, Any]):
        frames = detector.get("frames")
        if isinstance(frames, dict):
            count = frames.get("frame_count", "?")
            encoding = frames.get("encoding", "")
            n = QTreeWidgetItem(["Frames", "Frame archive", f"{count} frames", str(encoding)])
            n.setData(0, ROLE_MODE, "hdxf_json")
            n.setData(0, ROLE_KIND, "detector_frames")
            n.setData(0, ROLE_EXTRA, frames)
            parent.addChild(n)

            block_index = frames.get("block_index")
            if isinstance(block_index, dict):
                bi = QTreeWidgetItem([
                    "Frame block index", "Binary index", "", str(block_index.get("encoding", ""))
                ])
                bi.setData(0, ROLE_MODE, "hdxf_json")
                bi.setData(0, ROLE_KIND, "frame_block_index")
                bi.setData(0, ROLE_EXTRA, block_index)
                n.addChild(bi)

        calibration = detector.get("calibration")
        if isinstance(calibration, dict):
            state = calibration.get("state", "")
            mode = calibration.get("mode", "")
            n = QTreeWidgetItem(["Calibration", "Calibration", str(state), str(mode)])
            n.setData(0, ROLE_MODE, "hdxf_json")
            n.setData(0, ROLE_KIND, "detector_calibration")
            n.setData(0, ROLE_EXTRA, calibration)
            parent.addChild(n)

    def _populate_hdxf_members(self, parent: QTreeWidgetItem):
        if self.hdxf_zip is None:
            return
        groups: Dict[str, QTreeWidgetItem] = {}
        for info in sorted(self.hdxf_zip.infolist(), key=lambda x: x.filename):
            if info.filename == "manifest.json":
                group_name = "Manifest"
            elif info.filename.startswith("payloads/"):
                suffix = Path(info.filename).suffix.lower() or "(no extension)"
                group_name = f"Payloads {suffix}"
            else:
                group_name = "Other members"

            g = groups.get(group_name)
            if g is None:
                g = QTreeWidgetItem([group_name, "Member group", "", ""])
                g.setData(0, ROLE_MODE, "hdxf_section")
                g.setData(0, ROLE_KIND, "member_group")
                parent.addChild(g)
                groups[group_name] = g

            encs = sorted(self.hdxf_payload_encodings.get(info.filename, set()))
            enc = ", ".join(encs)
            label = Path(info.filename).name if info.filename.startswith("payloads/") else info.filename
            item = QTreeWidgetItem([
                label,
                "ZIP member",
                human_bytes(info.file_size),
                enc or f"compressed={human_bytes(info.compress_size)}",
            ])
            item.setData(0, ROLE_MODE, "hdxf_member")
            item.setData(0, ROLE_PATH, info.filename)
            item.setData(0, ROLE_KIND, "zip_member")
            item.setData(0, ROLE_EXTRA, {
                "filename": info.filename,
                "file_size": info.file_size,
                "compress_size": info.compress_size,
                "compress_type": info.compress_type,
                "crc": info.CRC,
                "encoding_hints": encs,
            })
            g.addChild(item)

    def show_hdxf_details(self, item: QTreeWidgetItem):
        mode = item.data(0, ROLE_MODE)
        kind = item.data(0, ROLE_KIND)
        path = item.data(0, ROLE_PATH)
        extra = item.data(0, ROLE_EXTRA)

        self.preview_btn.setEnabled(False)
        self.occ_table.setRowCount(0)
        self.attrs.setRowCount(0)

        if mode == "hdxf_root":
            manifest = self.hdxf_manifest or {}
            hdxf = manifest.get("hdxf", {}) if isinstance(manifest, dict) else {}
            source = manifest.get("source", {}) if isinstance(manifest, dict) else {}
            objects = manifest.get("objects", {}) if isinstance(manifest, dict) else {}
            links = manifest.get("links", []) if isinstance(manifest, dict) else []
            files = source.get("files", []) if isinstance(source, dict) else []
            self.set_info([
                ("HDXF file", self.hdxf_path),
                ("Archive size", human_bytes(os.path.getsize(self.hdxf_path))),
                ("Version", hdxf.get("version", "?")),
                ("Profile", hdxf.get("profile", "?")),
                ("Source files", len(files) if isinstance(files, list) else 0),
                ("Objects", len(objects) if isinstance(objects, dict) else 0),
                ("Links", len(links) if isinstance(links, list) else 0),
                ("ZIP members", len(self.hdxf_zip.infolist()) if self.hdxf_zip else 0),
            ])
            self.storage.setPlainText(self._json_preview(hdxf))
            return

        if mode == "hdxf_object" and isinstance(extra, dict):
            object_id = extra.get("object_id")
            record = extra.get("record", {})
            if not isinstance(record, dict):
                record = {}
            source_id = record.get("source_file")
            source_rec = self._hdxf_source_map().get(str(source_id), {})
            storage = record.get("hdf5_storage") if isinstance(record.get("hdf5_storage"), dict) else {}
            payload = record.get("payload") if isinstance(record.get("payload"), dict) else {}
            self.set_info([
                ("Object ID", object_id),
                ("Type", record.get("type")),
                ("Source file", source_rec.get("filename", source_id)),
                ("Source path", record.get("source_path")),
                ("Source local path", record.get("source_local_path")),
                ("Reconstruction only", record.get("reconstruction_only", False)),
                ("Shape", storage.get("shape")),
                ("dtype", self._hdxf_dtype_text(storage.get("dtype"))),
                ("Chunks", storage.get("chunks")),
                ("Original HDF5 compression", storage.get("compression")),
                ("Payload encoding", payload.get("encoding")),
                ("Payload members", len(payload.get("members", [])) if isinstance(payload.get("members"), list) else 0),
            ])
            attrs = record.get("attributes")
            if isinstance(attrs, dict):
                rows = []
                for name, val in attrs.items():
                    rows.append((Path(str(source_rec.get("filename", ""))).name, str(name), self._json_preview(val, 1200), "", ""))
                self.attrs.setRowCount(len(rows))
                for r, vals in enumerate(rows):
                    for c, v in enumerate(vals):
                        self.attrs.setItem(r, c, QTableWidgetItem(str(v)))
            self.storage.setPlainText(self._json_preview(record, 150000))
            return

        if mode == "hdxf_member" and isinstance(extra, dict):
            self.set_info([
                ("Member", extra.get("filename")),
                ("Uncompressed size", human_bytes(extra.get("file_size"))),
                ("Compressed size", human_bytes(extra.get("compress_size"))),
                ("ZIP compression type", extra.get("compress_type")),
                ("CRC32", f"0x{int(extra.get('crc', 0)):08x}"),
                ("Encoding hints", ", ".join(extra.get("encoding_hints", []))),
            ])
            self.storage.setPlainText(self._json_preview(extra))
            return

        if mode == "hdxf_json":
            self.set_info([
                ("HDXF section", kind),
                ("Path / label", path or item.text(0)),
                ("Python type", type(extra).__name__),
            ])
            self.storage.setPlainText(self._json_preview(extra, 250000))
            return

        if mode in {"hdxf_section", "hdxf_virtual_path"}:
            self.set_info([
                ("HDXF view", item.text(0)),
                ("Type", kind or item.text(1)),
                ("Path", path or "-"),
            ])
            return

    def _hdxf_dtype_text(self, dtype_info: Any) -> str:
        if isinstance(dtype_info, dict):
            return str(dtype_info.get("numpy") or dtype_info.get("str") or dtype_info)
        return str(dtype_info) if dtype_info is not None else "-"

    def _json_preview(self, value: Any, limit: int = 100000) -> str:
        try:
            text = json.dumps(value, indent=2, ensure_ascii=False)
        except Exception:
            text = repr(value)
        if len(text) > limit:
            return text[:limit] + f"\n\n... <truncated; {len(text) - limit} characters omitted>"
        return text

    def expand_depth(self, depth: int):
        def rec(item, d):
            item.setExpanded(d < depth)
            for i in range(item.childCount()):
                rec(item.child(i), d + 1)

        for tree in (self.hdf5_tree, self.hdxf_tree):
            for i in range(tree.topLevelItemCount()):
                rec(tree.topLevelItem(i), 0)

    def collapse_all_views(self):
        self.hdf5_tree.collapseAll()
        self.hdxf_tree.collapseAll()

    # ------------------------------------------------------------------
    # Cross-pane path highlighting
    # ------------------------------------------------------------------

    def _normalize_hdf5_path(self, value: Any) -> Optional[str]:
        """Return a canonical HDF5 path, or None for non-HDF5 items such as ZIP members."""
        if not isinstance(value, str):
            return None
        value = value.strip()
        if not value.startswith("/"):
            return None
        if value == "/":
            return "/"
        parts = [part for part in value.split("/") if part]
        return "/" + "/".join(parts)

    def _normalize_source_filename(self, value: Any) -> Optional[str]:
        """Normalize a physical source filename for cross-pane identity matching."""
        if not isinstance(value, str):
            return None
        value = value.strip()
        if not value:
            return None
        # HDXF manifests are often produced on Windows, while this UI may be run
        # elsewhere.  Normalizing both slash styles avoids Path() platform quirks.
        value = value.replace("\\", "/")
        return value.rsplit("/", 1)[-1].casefold()

    def _hdxf_source_filename_from_id(self, source_id: Any) -> Optional[str]:
        if source_id is None:
            return None
        record = self._hdxf_source_map().get(str(source_id), {})
        if not isinstance(record, dict):
            return None
        return self._normalize_source_filename(record.get("filename"))

    def _source_filename_from_record(self, record: Any) -> Optional[str]:
        """Extract a physical source filename from an HDXF record/descriptor when possible."""
        if not isinstance(record, dict):
            return None

        # Normal archived-object records use source_file as an ID into source.files.
        source_name = self._hdxf_source_filename_from_id(record.get("source_file"))
        if source_name:
            return source_name

        # Reconstruction/source descriptors may carry a filename directly.
        for key in (
            "filename", "file", "source_filename", "original_filename",
            "relative_path", "path",
        ):
            value = record.get(key)
            if isinstance(value, str) and value.lower().endswith((".h5", ".hdf5", ".nxs")):
                return self._normalize_source_filename(value)

        # Some descriptors nest source information.
        for key in ("source", "source_file_record", "descriptor"):
            nested = record.get(key)
            found = self._source_filename_from_record(nested)
            if found:
                return found

        return None

    def _item_source_filename(
        self,
        tree: QTreeWidget,
        item: QTreeWidgetItem,
    ) -> Optional[str]:
        """
        Return the physical HDF5 filename represented by *item*, if it is source-specific.

        A merged HDF5 node intentionally returns None: selecting it means "this path
        across the whole experiment".  Physical HDF5 nodes and HDXF objects grouped by
        source return a concrete filename, allowing source-aware cross-highlighting.
        """
        if tree is self.hdf5_tree:
            mode = item.data(0, ROLE_MODE)
            if mode in {"physical", "physical_file"}:
                return self._normalize_source_filename(item.data(0, ROLE_FILE))
            return None

        # HDXF leaf objects usually contain their source object record.
        extra = item.data(0, ROLE_EXTRA)
        if isinstance(extra, dict):
            record = extra.get("record")
            found = self._source_filename_from_record(record)
            if found:
                return found

            # Some HDXF JSON nodes store the source descriptor directly in ROLE_EXTRA.
            found = self._source_filename_from_record(extra)
            if found:
                return found

        # Virtual path nodes under "Archived Objects by Source File" do not carry
        # the object record.  Walk upward to the source-file container.
        parent = item.parent()
        while parent is not None:
            kind = parent.data(0, ROLE_KIND)
            pextra = parent.data(0, ROLE_EXTRA)
            if kind in {"source_file", "legacy_external_file"}:
                found = self._source_filename_from_record(pextra)
                if found:
                    return found
                # As a final fallback the displayed node name is the source filename.
                displayed = self._normalize_source_filename(parent.text(0))
                if displayed:
                    return displayed
            parent = parent.parent()

        return None

    def _iter_tree_items(self, tree: QTreeWidget):
        def walk(item: QTreeWidgetItem):
            yield item
            for idx in range(item.childCount()):
                yield from walk(item.child(idx))

        for idx in range(tree.topLevelItemCount()):
            yield from walk(tree.topLevelItem(idx))

    def _clear_cross_highlights(self):
        """Remove only the custom counterpart/path painting, without touching selection."""
        for item in self._cross_highlighted_items:
            try:
                for column in range(item.columnCount()):
                    item.setBackground(column, QBrush())
            except RuntimeError:
                # The underlying Qt item may already have been deleted by tree.clear().
                pass
        self._cross_highlighted_items.clear()

    def _cross_highlight_brushes(self, tree: QTreeWidget) -> tuple[QBrush, QBrush]:
        """Derive theme-aware highlight brushes from the current Qt palette."""
        base = QColor(tree.palette().color(QPalette.Highlight))

        exact = QColor(base)
        exact.setAlpha(115)

        path = QColor(base)
        path.setAlpha(42)

        return QBrush(exact), QBrush(path)

    def _paint_cross_item(self, item: QTreeWidgetItem, brush: QBrush):
        for column in range(item.columnCount()):
            item.setBackground(column, brush)
        self._cross_highlighted_items.append(item)

    def _visible_anchor_for_item(self, item: QTreeWidgetItem) -> QTreeWidgetItem:
        """
        Return the deepest currently visible point on the route to item.

        If a parent is collapsed, that parent becomes the visible anchor.  Because
        every ancestor is already painted, expanding it reveals the next painted
        path segment without changing the user's expansion state automatically.
        """
        chain = []
        cur = item
        while cur is not None:
            chain.append(cur)
            cur = cur.parent()
        chain.reverse()

        if not chain:
            return item

        anchor = chain[0]
        for index, node in enumerate(chain):
            if node.isHidden():
                break
            anchor = node
            if index < len(chain) - 1 and not node.isExpanded():
                break
        return anchor

    def _candidate_matches_source(
        self,
        tree: QTreeWidget,
        item: QTreeWidgetItem,
        source_filename: Optional[str],
    ) -> bool:
        """
        Apply source-file scope to one path match.

        * source_filename is None -> global/merged selection, so every occurrence of
          the path is valid.
        * concrete source filename -> only that physical file is valid.
        * on the HDF5 side the Merged Logical View is additionally valid because it
          is a deliberate aggregate representation of the same physical occurrence.
        """
        if source_filename is None:
            return True

        if tree is self.hdf5_tree and item.data(0, ROLE_MODE) == "merged":
            return True

        candidate_source = self._item_source_filename(tree, item)
        return candidate_source == source_filename

    def _highlight_matching_path(
        self,
        tree: QTreeWidget,
        path: str,
        source_filename: Optional[str] = None,
    ) -> int:
        """
        Highlight matching representations of *path* in *tree*.

        Matching is source-aware when the originating selection represents a concrete
        physical HDF5 file.  This prevents e.g. data_000001:/entry/data/data from
        highlighting the same path in all 60 HDXF source-file branches merely because
        the local HDF5 path is identical.

        A selection from Merged Logical View has no source filename and intentionally
        highlights every physical occurrence.
        """
        normalized = self._normalize_hdf5_path(path)
        if not normalized or normalized == "/":
            return 0

        matches: list[QTreeWidgetItem] = []
        for item in self._iter_tree_items(tree):
            item_path = self._normalize_hdf5_path(item.data(0, ROLE_PATH))
            if item_path != normalized:
                continue
            if not self._candidate_matches_source(tree, item, source_filename):
                continue
            matches.append(item)

        if not matches:
            return 0

        exact_items: Dict[int, QTreeWidgetItem] = {}
        ancestor_items: Dict[int, QTreeWidgetItem] = {}

        for match in matches:
            exact_items[id(match)] = match
            parent = match.parent()
            while parent is not None:
                ancestor_items[id(parent)] = parent
                parent = parent.parent()

        exact_brush, path_brush = self._cross_highlight_brushes(tree)

        # Paint ancestors first so an exact node can never be weakened by a later
        # breadcrumb paint operation.
        for key, ancestor in ancestor_items.items():
            if key not in exact_items:
                self._paint_cross_item(ancestor, path_brush)

        for match in exact_items.values():
            self._paint_cross_item(match, exact_brush)

        # Do not expand/collapse anything automatically.  Scroll to the deepest
        # location that is already visible for the first match.
        anchor = self._visible_anchor_for_item(matches[0])
        try:
            tree.scrollToItem(anchor, QAbstractItemView.PositionAtCenter)
        except Exception:
            tree.scrollToItem(anchor)

        return len(matches)

    def _sync_cross_highlight(self, source_tree: QTreeWidget, item: QTreeWidgetItem):
        self._clear_cross_highlights()

        path = self._normalize_hdf5_path(item.data(0, ROLE_PATH))
        if not path or path == "/":
            return

        # This is the important distinction that the previous implementation missed:
        # path identity alone is not enough.  Every external data file legitimately has
        # local paths such as /entry/data/data and /entry/detector/exptime.  A physical
        # selection therefore carries its source filename into the match operation.
        source_filename = self._item_source_filename(source_tree, item)

        target_tree = self.hdxf_tree if source_tree is self.hdf5_tree else self.hdf5_tree
        count = self._highlight_matching_path(
            target_tree,
            path,
            source_filename=source_filename,
        )

        target_name = "HDXF" if target_tree is self.hdxf_tree else "HDF5"
        source_label = source_filename or "all source files"
        if count:
            self.statusBar().showMessage(
                f"Linked path: {path} — source scope: {source_label} — "
                f"highlighted {count} matching location(s) in {target_name}",
                5000,
            )
        else:
            self.statusBar().showMessage(
                f"Linked path: {path} — source scope: {source_label} — "
                f"no matching location found in {target_name}",
                3500,
            )

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def selection_changed(self, tree: QTreeWidget):
        items = tree.selectedItems()
        if not items:
            return

        self.active_tree = tree
        # Keep one active selection visually unambiguous.
        other = self.hdxf_tree if tree is self.hdf5_tree else self.hdf5_tree
        other.blockSignals(True)
        other.clearSelection()
        other.blockSignals(False)

        item = items[0]
        mode = item.data(0, ROLE_MODE)

        self.preview.clear()
        self.attrs.setRowCount(0)
        self.occ_table.setRowCount(0)
        self.storage.clear()
        self.preview_btn.setEnabled(False)

        if isinstance(mode, str) and mode.startswith("hdxf"):
            self.show_hdxf_details(item)
        elif mode == "merged":
            self.show_merged_details(item)
        elif mode in {"physical", "physical_file"}:
            self.show_physical_details(item)
        elif mode == "experiment_root":
            self.set_info([
                ("Folder", self.folder_path),
                ("Physical HDF5 files", len(self.files)),
                ("Master file", Path(self.master_file).name if self.master_file else "-"),
                ("Unique logical paths", len(self.path_occurrences)),
            ])
        elif mode == "merged_root":
            self.set_info([
                ("View", "Merged Logical View"),
                ("Meaning", "Same HDF5 paths from all physical files are merged into one tree"),
                ("Physical files", len(self.files)),
            ])
        elif mode == "physical_root":
            self.set_info([
                ("View", "Physical Files"),
                ("Physical files", len(self.files)),
            ])

        # Bidirectional counterpart highlighting.  We intentionally do this after
        # rendering details so it never changes the active selection or expansion
        # state in either tree.
        self._sync_cross_highlight(tree, item)

    def set_info(self, rows):
        self.info.setRowCount(len(rows))
        for r, (k, v) in enumerate(rows):
            self.info.setItem(r, 0, QTableWidgetItem(str(k)))
            self.info.setItem(r, 1, QTableWidgetItem(short_value(v)))
        self.info.resizeRowsToContents()

    def show_merged_details(self, item: QTreeWidgetItem):
        path = item.data(0, ROLE_PATH)
        occs = self.path_occurrences.get(path, [])
        if not occs:
            return

        kinds = sorted(set(o.kind for o in occs))
        shapes = sorted(set(str(o.shape) for o in occs if o.shape is not None))
        dtypes = sorted(set(o.dtype for o in occs if o.dtype is not None))
        chunks = sorted(set(str(o.chunks) for o in occs if o.kind == "dataset"))
        compressions = sorted(set(str(o.compression) for o in occs if o.kind == "dataset"))

        total_logical = sum(o.logical_bytes or 0 for o in occs)
        total_storage = sum(o.storage_bytes or 0 for o in occs)

        rows = [
            ("Merged path", path),
            ("Present in files", f"{len(occs)} / {len(self.files)}"),
            ("Kinds", ", ".join(kinds)),
            ("Unique shapes", ", ".join(shapes) if shapes else "-"),
            ("Unique dtypes", ", ".join(dtypes) if dtypes else "-"),
            ("Unique chunk layouts", ", ".join(chunks) if chunks else "-"),
            ("Unique compression values", ", ".join(compressions) if compressions else "-"),
            ("Total logical bytes across files", human_bytes(total_logical)),
            ("Total storage bytes across files", human_bytes(total_storage)),
        ]

        # Master / data split
        master_count = sum(1 for o in occs if o.file_path == self.master_file)
        data_count = len(occs) - master_count
        rows += [
            ("Present in master", "yes" if master_count else "no"),
            ("Present in data files", data_count),
        ]

        self.set_info(rows)
        self.populate_occurrences(occs)
        self.populate_merged_attributes(path, occs)

        if len(kinds) == 1 and kinds[0] == "dataset":
            self.preview_btn.setEnabled(True)

        lines = [f"Merged path: {path}", ""]
        for o in occs[:200]:
            lines.append(f"[{Path(o.file_path).name}]")
            lines.append(f"  kind        : {o.kind}")
            if o.kind == "dataset":
                lines.append(f"  shape       : {o.shape}")
                lines.append(f"  dtype       : {o.dtype}")
                lines.append(f"  chunks      : {o.chunks}")
                lines.append(f"  compression : {o.compression}")
                if o.filters:
                    lines.append("  filters:")
                    for f in o.filters:
                        lines.append(
                            f"    id={f['id']} name={f['name']!r} "
                            f"flags={f['flags']} cd_values={f['cd_values']}"
                        )
            elif o.target is not None:
                lines.append(f"  target      : {o.target}")
            lines.append("")
        self.storage.setPlainText("\n".join(lines))

    def populate_occurrences(self, occs: list[Occurrence]):
        self.occ_table.setRowCount(len(occs))
        for r, o in enumerate(occs):
            vals = [
                Path(o.file_path).name,
                o.kind,
                "-" if o.shape is None else str(o.shape),
                o.dtype or "-",
                human_bytes(o.logical_bytes),
                human_bytes(o.storage_bytes),
                str(o.chunks) if o.chunks is not None else "-",
                str(o.compression) if o.compression is not None else "-",
            ]
            for c, v in enumerate(vals):
                self.occ_table.setItem(r, c, QTableWidgetItem(str(v)))
        self.occ_table.resizeRowsToContents()

    def populate_merged_attributes(self, path: str, occs: list[Occurrence]):
        rows = []
        for o in occs:
            f = self.files.get(o.file_path)
            if f is None:
                continue

            try:
                obj = f[path]
            except Exception:
                continue

            try:
                for name in obj.attrs.keys():
                    try:
                        v = obj.attrs[name]
                        a = np.asarray(v)
                        rows.append(
                            (
                                Path(o.file_path).name,
                                name,
                                short_value(v),
                                str(a.dtype),
                                str(a.shape),
                            )
                        )
                    except Exception as exc:
                        rows.append(
                            (
                                Path(o.file_path).name,
                                name,
                                f"<read error: {exc}>",
                                "?",
                                "?",
                            )
                        )
            except Exception:
                pass

        self.attrs.setRowCount(len(rows))
        for r, vals in enumerate(rows):
            for c, v in enumerate(vals):
                self.attrs.setItem(r, c, QTableWidgetItem(str(v)))
        self.attrs.resizeRowsToContents()

    def show_physical_details(self, item: QTreeWidgetItem):
        file_path = item.data(0, ROLE_FILE)
        path = item.data(0, ROLE_PATH)
        kind = item.data(0, ROLE_KIND)

        if not file_path or file_path not in self.files:
            return

        f = self.files[file_path]

        if item.data(0, ROLE_MODE) == "physical_file":
            self.set_info([
                ("File", file_path),
                ("Role", "Master" if file_path == self.master_file else "Data file"),
                ("File size", human_bytes(os.path.getsize(file_path))),
            ])
            return

        rows = [
            ("File", file_path),
            ("Role", "Master" if file_path == self.master_file else "Data file"),
            ("Path", path),
            ("Type", kind),
        ]

        if kind == "external_link":
            filename, target_path = item.data(0, ROLE_EXTRA)
            target_file = os.path.abspath(os.path.join(os.path.dirname(file_path), filename))
            rows += [
                ("Target file", target_file),
                ("Target path", target_path),
                ("Target exists", os.path.exists(target_file)),
            ]
            self.set_info(rows)
            return

        if kind == "soft_link":
            rows.append(("Target path", item.data(0, ROLE_EXTRA)))
            self.set_info(rows)
            return

        try:
            obj = f[path]
        except Exception as exc:
            rows.append(("Error", str(exc)))
            self.set_info(rows)
            return

        self.populate_single_attributes(file_path, obj)

        if isinstance(obj, h5py.Group):
            rows.append(("Children", len(obj.keys())))

        if isinstance(obj, h5py.Dataset):
            logical = None
            storage = None
            try:
                logical = int(obj.size) * int(obj.dtype.itemsize)
            except Exception:
                pass
            try:
                storage = int(obj.id.get_storage_size())
            except Exception:
                pass

            ratio = "-"
            if logical is not None and storage:
                ratio = f"{logical / storage:.3f}x"

            filters = filter_pipeline(obj)

            rows += [
                ("Shape", obj.shape),
                ("dtype", obj.dtype),
                ("Elements", obj.size),
                ("Logical size", human_bytes(logical)),
                ("Storage size", human_bytes(storage)),
                ("Logical / storage", ratio),
                ("Chunks", obj.chunks),
                ("Compression", obj.compression),
                ("Compression opts", obj.compression_opts),
                ("Shuffle", obj.shuffle),
                ("Fletcher32", obj.fletcher32),
                ("Scaleoffset", obj.scaleoffset),
                ("Max shape", obj.maxshape),
            ]

            lines = [
                f"File            : {file_path}",
                f"Path            : {path}",
                f"Shape           : {obj.shape}",
                f"dtype           : {obj.dtype}",
                f"Chunks          : {obj.chunks}",
                f"Compression     : {obj.compression}",
                f"Compression opts: {obj.compression_opts}",
                f"Logical size    : {human_bytes(logical)}",
                f"Storage size    : {human_bytes(storage)}",
                "",
                "HDF5 filter pipeline:",
            ]

            if filters:
                for flt in filters:
                    lines.append(
                        f"  id={flt['id']}  name={flt['name']!r}  "
                        f"flags={flt['flags']}  cd_values={flt['cd_values']}"
                    )
            else:
                lines.append("  None")

            self.storage.setPlainText("\n".join(lines))
            self.preview_btn.setEnabled(True)

        self.set_info(rows)

    def populate_single_attributes(self, file_path: str, obj):
        rows = []
        try:
            for name in obj.attrs.keys():
                try:
                    v = obj.attrs[name]
                    a = np.asarray(v)
                    rows.append(
                        (
                            Path(file_path).name,
                            name,
                            short_value(v),
                            str(a.dtype),
                            str(a.shape),
                        )
                    )
                except Exception as exc:
                    rows.append(
                        (
                            Path(file_path).name,
                            name,
                            f"<read error: {exc}>",
                            "?",
                            "?",
                        )
                    )
        except Exception:
            pass

        self.attrs.setRowCount(len(rows))
        for r, vals in enumerate(rows):
            for c, v in enumerate(vals):
                self.attrs.setItem(r, c, QTableWidgetItem(str(v)))
        self.attrs.resizeRowsToContents()

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def preview_dataset(self):
        tree = getattr(self, "active_tree", self.hdf5_tree)
        items = tree.selectedItems()
        if not items:
            return

        item = items[0]
        mode = item.data(0, ROLE_MODE)

        file_path = None
        path = item.data(0, ROLE_PATH)

        if mode == "physical":
            file_path = item.data(0, ROLE_FILE)

        elif mode == "merged":
            occs = self.path_occurrences.get(path, [])
            # Prefer master occurrence, otherwise first file.
            if self.master_file:
                for o in occs:
                    if o.file_path == self.master_file and o.kind == "dataset":
                        file_path = o.file_path
                        break
            if file_path is None:
                for o in occs:
                    if o.kind == "dataset":
                        file_path = o.file_path
                        break

        if not file_path or not path:
            return

        f = self.files.get(file_path)
        if f is None:
            return

        try:
            ds = f[path]
        except Exception as exc:
            self.preview.setPlainText(str(exc))
            return

        if not isinstance(ds, h5py.Dataset):
            return

        filters = filter_pipeline(ds)
        if filters and not self.allow_filtered_preview.isChecked():
            self.preview.setPlainText(
                "Preview blocked for safety.\n\n"
                f"File: {Path(file_path).name}\n"
                f"Path: {path}\n\n"
                "This dataset has an HDF5 filter pipeline. "
                "The structure can be inspected without decoding values.\n\n"
                + "\n".join(
                    f"id={flt['id']} name={flt['name']!r}" for flt in filters
                )
                + "\n\nEnable 'Allow filtered value preview' only if needed."
            )
            return

        try:
            if ds.shape == ():
                self.preview.setPlainText(
                    f"File: {Path(file_path).name}\nPath: {path}\n\n"
                    + short_value(ds[()], 20000)
                )
                return

            if int(ds.size) <= 256:
                self.preview.setPlainText(
                    f"File: {Path(file_path).name}\nPath: {path}\n\n"
                    + short_value(ds[...], 20000)
                )
                return

            tail = int(np.prod(ds.shape[1:])) if len(ds.shape) > 1 else 1
            if tail > 500_000:
                self.preview.setPlainText(
                    "Large detector/image dataset: value preview intentionally skipped.\n\n"
                    f"File: {Path(file_path).name}\n"
                    f"Path: {path}\n"
                    f"shape={ds.shape}\n"
                    f"dtype={ds.dtype}\n"
                    f"elements={ds.size}\n\n"
                    "Use your detector Viewer for frame/image data."
                )
                return

            n = min(16, int(ds.shape[0]))
            sl = (slice(0, n),) + tuple(slice(None) for _ in ds.shape[1:])
            v = ds[sl]
            self.preview.setPlainText(
                f"File: {Path(file_path).name}\n"
                f"Path: {path}\n"
                f"First {n} entries along axis 0\n"
                f"Full shape={ds.shape}\n\n"
                + short_value(v, 20000)
            )

        except Exception as exc:
            self.preview.setPlainText(
                f"Preview failed:\n{type(exc).__name__}: {exc}\n\n"
                "This does not affect structure inspection."
            )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _apply_search_to_tree(self, tree: QTreeWidget, text: str):
        needle = text.strip().lower()

        def rec(item: QTreeWidgetItem) -> bool:
            own = (
                not needle
                or needle in item.text(0).lower()
                or needle in item.text(1).lower()
                or needle in item.text(2).lower()
                or needle in item.text(3).lower()
                or needle in str(item.data(0, ROLE_PATH) or "").lower()
            )

            child = False
            for i in range(item.childCount()):
                child = rec(item.child(i)) or child

            visible = own or child
            item.setHidden(not visible)
            if needle and child:
                item.setExpanded(True)
            return visible

        for i in range(tree.topLevelItemCount()):
            rec(tree.topLevelItem(i))

    def apply_hdf5_search(self, text: str):
        self._apply_search_to_tree(self.hdf5_tree, text)

    def apply_hdxf_search(self, text: str):
        self._apply_search_to_tree(self.hdxf_tree, text)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def build_summary(self):
        if not self.path_occurrences:
            return

        dataset_paths = []
        for path, occs in self.path_occurrences.items():
            if any(o.kind == "dataset" for o in occs):
                dataset_paths.append(path)

        groups = {
            "Master detector / geometry": [],
            "Data-file detector auxiliary": [],
            "Calibration / pixel mask": [],
            "Beam": [],
            "Acquisition / timing": [],
            "ROI / image diagnostics": [],
            "XFEL / event": [],
            "Other": [],
        }

        for path in sorted(dataset_paths):
            occs = self.path_occurrences[path]
            p = path.lower()

            present_master = any(o.file_path == self.master_file for o in occs)
            data_count = sum(1 for o in occs if o.file_path != self.master_file)

            if (
                path.startswith("/entry/instrument/detector")
                and not ("calibration" in p or "pixel_mask" in p)
            ):
                cat = "Master detector / geometry"
            elif path.startswith("/entry/detector"):
                cat = "Data-file detector auxiliary"
            elif "calibration" in p or "pedestal" in p or "pixel_mask" in p:
                cat = "Calibration / pixel mask"
            elif "/beam" in p or "wavelength" in p:
                cat = "Beam"
            elif any(
                k in p
                for k in (
                    "exptime", "count_time", "frame_time",
                    "nimages", "ntrigger", "timestamp", "acquisition"
                )
            ):
                cat = "Acquisition / timing"
            elif any(
                k in p
                for k in (
                    "/roi/", "error_pixels", "saturated_pixels",
                    "processing_time", "pixel_sum"
                )
            ):
                cat = "ROI / image diagnostics"
            elif "/xfel/" in p or "pulseid" in p or "eventcode" in p:
                cat = "XFEL / event"
            else:
                cat = "Other"

            shapes = sorted(set(str(o.shape) for o in occs if o.shape is not None))
            dtypes = sorted(set(o.dtype for o in occs if o.dtype is not None))

            groups[cat].append(
                (
                    path,
                    len(occs),
                    len(self.files),
                    present_master,
                    data_count,
                    ", ".join(shapes) if shapes else "-",
                    ", ".join(dtypes) if dtypes else "-",
                )
            )

        lines = [
            f"Folder: {self.folder_path}",
            f"Physical HDF5 files: {len(self.files)}",
            f"Master: {Path(self.master_file).name if self.master_file else '-'}",
            f"Unique merged dataset paths: {len(dataset_paths)}",
            "",
        ]

        for cat, items in groups.items():
            if not items:
                continue
            lines.append(f"[{cat}] ({len(items)})")
            for path, count, total, present_master, data_count, shapes, dtypes in items:
                lines.append(f"  {path}")
                lines.append(
                    f"      present={count}/{total}  "
                    f"master={'yes' if present_master else 'no'}  "
                    f"data_files={data_count}  "
                    f"shape={shapes}  dtype={dtypes}"
                )
            lines.append("")

        self.summary.setPlainText("\n".join(lines))


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("HDF5 / HDXF Structure Explorer")

    win = HDF5FolderExplorer()
    win.show()

    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if os.path.isdir(arg):
            win.load_folder(arg)
        elif os.path.isfile(arg) and arg.lower().endswith(".hdxf"):
            win.load_hdxf(arg)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
