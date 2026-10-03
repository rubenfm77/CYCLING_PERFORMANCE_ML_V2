"""Scan every user-visible string (section/callout/caption/…) for Spanish."""
import ast
import pathlib
import re

SPAN = re.compile(
    r"[\u00c1\u00e1\u00cd\u00ed\u00d3\u00f3\u00da\u00fa\u00d1\u00f1]"
    r"|\b(?:para|del|los|las|unos|unas|sesi\u00f3n|entreno|umbral|duraci\u00f3n"
    r"|fecha|a\u00f1o|potencia|mejora|respecto|tambi\u00e9n|mismo|seg\u00fan"
    r"|cada|entre|hasta|desde|donde|cuando|porque)\b"
    r"|\b\w+ción\b|\b\w+miento\b",
    re.IGNORECASE)

CALLS = {"section", "callout", "metric_card", "page_header", "st.caption",
         "st.markdown", "st.title", "st.subheader", "st.write", "st.error",
         "st.warning", "st.info", "style_figure", "st.selectbox", "st.button",
         "st.checkbox", "st.metric", "st.toast", "st.divider"}


def name_of(node):
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def strings_in(node, out):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        out.append(node.value)
    elif isinstance(node, ast.JoinedStr):
        for v in node.values:
            strings_in(v, out)
    elif isinstance(node, ast.BinOp):
        strings_in(node.left, out)
        strings_in(node.right, out)
    elif isinstance(node, (ast.List, ast.Tuple)):
        for e in node.elts:
            strings_in(e, out)
    elif isinstance(node, ast.Dict):
        for e in node.values:
            strings_in(e, out)


roots = [pathlib.Path("views"), pathlib.Path("core"), pathlib.Path("ml")]
hits = 0
for root in roots:
    for p in sorted(root.rglob("*.py")):
        tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.func:
                continue
            n = name_of(node.func)
            if n not in CALLS:
                continue
            if n == "style_figure":
                args = node.args[1:2]      # skip the figure itself
            else:
                args = list(node.args) + [kw.value for kw in node.keywords]
            for a in args:
                found = []
                strings_in(a, found)
                for s in found:
                    if SPAN.search(s):
                        hits += 1
                        print(f"{p}:{node.lineno} [{n}] {s[:170]!r}")
print(f"\nSpanish user-visible strings: {hits}")
