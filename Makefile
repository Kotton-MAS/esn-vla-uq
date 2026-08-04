.PHONY: sync test cov lint fmt fmt-check type lock-check src-not-ignored audit secrets version-consistency ci pre-commit clean help

help:
	@echo "Available targets:"
	@echo "  sync         - Install dependencies (uv sync --locked)"
	@echo "  test         - Run tests (uv run pytest -q)"
	@echo "  cov          - Run tests with coverage"
	@echo "  lint         - Run ruff check"
	@echo "  fmt          - Run ruff format (modifies files)"
	@echo "  fmt-check    - Check formatting without modifying"
	@echo "  type         - Run mypy"
	@echo "  audit        - Scan dependencies for known vulnerabilities (pip-audit)"
	@echo "  secrets      - Scan the repository for hardcoded secrets (gitleaks)"
	@echo "  ci           - Full CI check (see the 'ci' target for the exact list)"
	@echo "  pre-commit   - Run pre-commit on all files"
	@echo "  clean        - Remove caches and build artifacts"

sync:
	uv sync --locked

# PYTHONPATH を空にしてから実行する。ROS など外部の site-packages が
# PYTHONPATH に載っている環境では、pytest がそこのプラグイン (launch_testing 等) を
# autoload しようとして ModuleNotFoundError で落ちる。本プロジェクトは
# PYTHONPATH に依存しない (src レイアウト + uv の editable install) ため無害。
test:
	PYTHONPATH= uv run pytest -q

cov:
	PYTHONPATH= uv run pytest --cov

lint:
	uv run ruff check .

fmt:
	uv run ruff format .

fmt-check:
	uv run ruff format --check .

type:
	uv run mypy .

lock-check:
	uv lock --check

# src/ 配下のソース・パッケージデータが .gitignore のパターンに巻き込まれて
# 静かに追跡外になる事故を検出する。実例: 未アンカーの `data/` が
# `src/esn_vla_uq/data/` に一致し、git だけでなく ruff / mypy / wheel からも
# 無警告で欠落していた。
src-not-ignored:
	@untracked=$$(git ls-files --others --ignored --exclude-standard -- src/ \
		| grep -E '\.(py|pyi|json|npz|typed)$$' || true); \
	tracked=$$(git ls-files -- src/ | git check-ignore --no-index --stdin || true); \
	if [ -n "$$untracked" ] || [ -n "$$tracked" ]; then \
		echo "ERROR: 以下の src/ 配下のファイルが .gitignore で除外されています:"; \
		[ -n "$$untracked" ] && echo "$$untracked" | sed 's/^/  (untracked) /'; \
		[ -n "$$tracked" ] && echo "$$tracked" | sed 's/^/  (tracked)   /'; \
		echo "アンカー漏れのパターンがないか .gitignore を確認してください。"; \
		exit 1; \
	fi

# 依存の既知脆弱性を検出する。dev グループにしか脆弱性が無くても配布物
# (numpy のみ) と切り分けて判断できるよう、まず機械的に検出する (S9)。
# 推移的依存の修正版は pyproject.toml の `[tool.uv] constraint-dependencies` で
# 引き上げる。
#
# `test` と同じ理由で PYTHONPATH を空にする。ROS など外部の site-packages が
# PYTHONPATH に載っていると、pip-audit がそれらまで監査対象に含めてしまい、
# 本プロジェクトと無関係な数百件の "not found on PyPI" が出て結果が読めなくなる。
audit:
	PYTHONPATH= uv run pip-audit

# ハードコードされたシークレットを検出する (S8)。
# pre-commit の detect-private-key は PEM 形式の秘密鍵しか見ず、`.gitignore` は
# `git add -f` で無効化できる。除外パターンの網羅性だけに依存しないため、
# pre-commit のインストール有無によらず CI から必ず走らせる。
secrets:
	uv run pre-commit run gitleaks --all-files

# CITATION.cff と pyproject.toml のバージョンは手動の二重管理であり、
# リリース時に片方だけ更新される drift が起きる (U3)。
version-consistency:
	@pyproject=$$(grep -m1 '^version = ' pyproject.toml | sed 's/.*"\(.*\)".*/\1/'); \
	citation=$$(grep -m1 '^version: ' CITATION.cff | sed 's/^version: *//;s/^"\(.*\)"$$/\1/'); \
	if [ "$$pyproject" != "$$citation" ]; then \
		echo "ERROR: バージョンが一致しません"; \
		echo "  pyproject.toml: $$pyproject"; \
		echo "  CITATION.cff:   $$citation"; \
		exit 1; \
	fi

# Stop hook (verify-ci.sh) と GitHub Actions が両方これを呼ぶ。
# ここを単一の真実とすることで、ローカルと CI の検証ロジック乖離を防ぐ。
ci: lock-check src-not-ignored version-consistency secrets audit lint fmt-check type test

pre-commit:
	uv run pre-commit run --all-files

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov dist build
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
