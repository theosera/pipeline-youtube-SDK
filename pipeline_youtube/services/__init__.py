"""Shared service-layer infrastructure (I/O・横断ユーティリティ).

cache / checkpoint / sanitize / path_safety など、ドメイン型 (``domain/``) の
上に乗る共通基盤をここへ集約する。各モジュールは旧パス
(``pipeline_youtube.<name>``) に後方互換 shim を残しているため、既存 import は
無修正で動作する。
"""

from __future__ import annotations
