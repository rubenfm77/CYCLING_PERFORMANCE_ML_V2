"""Regression test: one broken page module must NOT take down the whole app.

Script-style like its siblings: run it directly, it asserts and prints.
Needs the usual INTERVALS_* credentials in the environment.

Why this test exists. app_modern.py imports every page module while BUILDING THE
PAGE REGISTRY, long before st.navigation(). It used to do that with a single
`from views import (evolution, fitness, ...)` statement, which meant a
module-level ImportError in ANY view was an app-wide crash rather than a
page-local one. A missing constant in one leaf module put all eight pages on a
red screen while the seven healthy ones could not render at all.

The router now imports each view separately and substitutes an honest stand-in
for any that fail. This asserts that behaviour against the REAL app_modern.py
(imported under a stubbed streamlit) rather than a reimplementation of it — the
mechanism is `importlib.import_module(f"views.{name}")`, so patching
`importlib.import_module` before app_modern runs makes it genuinely fail.

Cases:
  A  all pages healthy            -> 8 registered, 8 real renders, no warning
  B  a page raises ImportError    -> 8 registered, 7 real renders, message shown
  C  a page raises RuntimeError   -> same isolation, and its MESSAGE is withheld
"""
import os
import sys
import types
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import logging
logging.getLogger("streamlit").setLevel(logging.ERROR)

SINK = {"error": [], "code": [], "warning": [], "caption": [], "markdown": []}

SLUG_OF = {"overview": "home", "intervals_view": "intervals"}
FAILS = []


class _Col:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Any:
    """Callable, a context manager, and permissive on attributes.

    `with st.sidebar:` and bare widget calls both occur, so the fallback for any
    streamlit attribute not stubbed explicitly has to support both.
    """

    def __call__(self, *a, **k):
        if len(a) == 1 and not k and callable(a[0]):
            return a[0]                       # decorator form
        return None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __getattr__(self, name):
        return _Any()


def _record(kind):
    def f(*a, **k):
        SINK[kind].append(str(a[0]) if a else "")
    return f


class _Nav:
    url_path = ""

    def run(self):
        return None


def build_stub():
    """A streamlit stub permissive enough for app_modern plus the sidebar."""
    st = types.ModuleType("streamlit")

    def _cache(f=None, **k):
        return f if f is not None else (lambda g: g)

    st.__dict__.update({
        "columns": lambda spec=1, **k: [_Col() for _ in range(
            max(1, int(spec) if isinstance(spec, int) else 1))],
        "container": lambda *a, **k: _Col(),
        "expander": lambda label="", *a, **k: _Col(),
        "form": lambda *a, **k: _Col(),
        "tabs": lambda labels=(), *a, **k: [_Col() for _ in labels] or [_Col()],
        "session_state": {},
        "column_config": types.SimpleNamespace(),
        "secrets": types.SimpleNamespace(),
        "cache_data": _cache,
        "cache_resource": _cache,
        "set_page_config": lambda *a, **k: None,
        "switch_page": _record("markdown"),
        "Page": lambda fn, **k: types.SimpleNamespace(fn=fn, **k),
        "navigation": lambda pages, **k: _Nav(),
        "radio": lambda label, options, **k: (list(options)[0] if options else None),
        "error": _record("error"),
        "warning": _record("warning"),
        "caption": _record("caption"),
        "code": _record("code"),
        "stop": lambda *a, **k: (_ for _ in ()).throw(RuntimeError("st.stop")),
    })
    st.__dict__["__getattr__"] = lambda name: _Any()
    return st


def run_case(label, break_module=None, exc=None):
    for k in SINK:
        SINK[k].clear()
    for m in list(sys.modules):
        if m == "app_modern" or m.startswith("views.") or m.startswith("ml."):
            del sys.modules[m]

    sys.modules["streamlit"] = build_stub()
    import importlib
    real = importlib.import_module
    if break_module:
        def patched(name, package=None):
            if name == f"views.{break_module}":
                raise exc
            return real(name, package)
        importlib.import_module = patched

    print("=" * 74)
    print(f"CASE {label}" + (f"   views.{break_module} raises "
                             f"{type(exc).__name__}" if break_module else ""))
    print("=" * 74)
    try:
        import app_modern as AM
    except Exception as e:                                    # noqa: BLE001
        importlib.import_module = real
        print(f"  app_modern RAISED {type(e).__name__}: {e}")
        FAILS.append(f"{label}: app_modern itself failed: {type(e).__name__}: {e}")
        return
    finally:
        importlib.import_module = real

    specs, errs = AM._SPECS, AM._view_errors
    print(f"  app_modern imported                : OK")
    print(f"  pages registered                  : {len(specs)}")
    print(f"  view modules loaded               : {len(AM._views)}")
    print(f"  view import errors                : {sorted(errs)}")

    if len(specs) != 8:
        FAILS.append(f"{label}: {len(specs)} pages registered, expected 8")

    # functools.partial keeps the callable in .func; .args holds (head, ctx).
    if not break_module:
        if errs:
            FAILS.append(f"{label}: unexpected errors {sorted(errs)}")
        if len(AM._views) != 8:
            FAILS.append(f"{label}: {len(AM._views)} modules, expected 8")
        for slug, _i, _t, fn, _d in specs:
            mod = getattr(fn.func, "__module__", None)
            if not (isinstance(mod, str) and mod.startswith("views.")):
                FAILS.append(f"{label}: page {slug} is a stand-in ({mod!r})")
        if SINK["warning"]:
            FAILS.append(f"{label}: warning shown with no broken page")
        print("  every page is a real view render  : OK")
        print("  no spurious warning               : OK")
        return

    broken_slug = SLUG_OF.get(break_module, break_module)
    if sorted(errs) != [break_module]:
        FAILS.append(f"{label}: errors {sorted(errs)}, expected ['{break_module}']")
    if len(AM._views) != 7:
        FAILS.append(f"{label}: {len(AM._views)} healthy modules, expected 7")

    n_real = 0
    for slug, _i, _t, fn, _d in specs:
        inner = fn.func
        if slug == broken_slug:
            if getattr(inner, "__name__", "") != "_render":
                FAILS.append(f"{label}: broken page wired to {inner!r}")
        elif getattr(inner, "__module__", "").startswith("views."):
            n_real += 1
        else:
            FAILS.append(f"{label}: healthy page {slug} is not a real render")
    print(f"  healthy pages still real renders  : {n_real}/7")
    if n_real != 7:
        FAILS.append(f"{label}: {n_real} real renders, expected 7")

    before = len(SINK["caption"])
    entry = [s for s in specs if s[0] == broken_slug][0]
    try:
        entry[3]()                      # head and ctx are already bound
    except Exception as e:                                    # noqa: BLE001
        FAILS.append(f"{label}: stand-in render raised "
                     f"{type(e).__name__}: {e}")
    new_caps = SINK["caption"][before:]
    print(f"  stand-in renders without raising : "
          f"{'OK' if not FAILS or 'stand-in render' not in FAILS[-1] else 'FAILED'}")
    print(f"  warning announced the fault      : "
          f"{SINK['warning'][0][:60] if SINK['warning'] else '(NONE)'}")
    print(f"  error panel                      : "
          f"{SINK['error'][0][:56] if SINK['error'] else '(NONE)'}")
    print(f"  caption                          : "
          f"{new_caps[0][:64] if new_caps else '(NONE)'}")
    if not SINK["warning"]:
        FAILS.append(f"{label}: no warning announced the broken page")
    if not SINK["error"]:
        FAILS.append(f"{label}: stand-in drew no error panel")
    if not new_caps:
        FAILS.append(f"{label}: stand-in printed no caption")

    if isinstance(exc, ImportError):
        if not SINK["code"]:
            FAILS.append(f"{label}: ImportError message not shown")
        else:
            print(f"  ImportError message shown       : "
                  f"{SINK['code'][0][:56]}")
    elif SINK["code"]:
        # A non-ImportError message may contain anything, including data, so it
        # must never be rendered. Type only.
        FAILS.append(f"{label}: non-ImportError message DISPLAYED: "
                     f"{SINK['code'][0][:70]!r}")
    else:
        print("  non-ImportError message withheld : OK")


SECRET = "AKIA1234567890ABCDEF-must-not-appear"
run_case("A")
run_case("B", "forecast", ImportError(
    "cannot import name 'BLANK_TYPE_TOKENS' from 'core.theme'"))
run_case("C", "evolution", RuntimeError(f"boom {SECRET}"))
run_case("D", "trends", ImportError("cannot import name 'x'"))

# Belt and braces: the secret from case C must not be anywhere in the transcript.
if any(SECRET in s for v in SINK.values() for s in v):
    FAILS.append("a withheld exception message leaked into the UI sink")

print()
print("=" * 74)
print("FAILURES:", FAILS if FAILS else "none")
sys.exit(1 if FAILS else 0)