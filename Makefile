.PHONY: sync test cov lint fmt fmt-check type lock-check src-not-ignored ci pre-commit clean help

help:
	@echo "Available targets:"
	@echo "  sync         - Install dependencies (uv sync --locked)"
	@echo "  test         - Run tests (uv run pytest -q)"
	@echo "  cov          - Run tests with coverage"
	@echo "  lint         - Run ruff check"
	@echo "  fmt          - Run ruff format (modifies files)"
	@echo "  fmt-check    - Check formatting without modifying"
	@echo "  type         - Run mypy"
	@echo "  ci           - Full CI check: lock + gitignore guard + lint + fmt + type + test"
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

# Stop hook (verify-ci.sh) と GitHub Actions が両方これを呼ぶ。
# ここを単一の真実とすることで、ローカルと CI の検証ロジック乖離を防ぐ。
ci: lock-check src-not-ignored lint fmt-check type test

pre-commit:
	uv run pre-commit run --all-files

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov dist build
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
