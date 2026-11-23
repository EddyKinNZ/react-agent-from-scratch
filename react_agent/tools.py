"""Tools the agent may call. Each takes a dict of arguments and returns a string."""
import ast
import operator

import pandas as pd

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
        ast.Mod: operator.mod}


def calculator(args):
    """Evaluate arithmetic safely (no eval): numbers and + - * / ** % only."""
    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("only arithmetic is allowed")
    expr = str(args["expression"])
    if len(expr) > 200:
        raise ValueError("expression too long")
    return str(round(ev(ast.parse(expr, mode="eval").body), 6))


class CsvTool:
    """Read-only questions over one CSV: schema, filter, aggregate. No code execution."""
    AGGS = {"mean", "sum", "min", "max", "count", "median"}

    def __init__(self, path):
        self.df = pd.read_csv(path)

    def describe(self, args):
        return "columns: " + ", ".join(f"{c} ({t})" for c, t in self.df.dtypes.astype(str).items()) \
               + f"\nrows: {len(self.df)}\nfirst rows:\n{self.df.head(3).to_string(index=False)}"

    def aggregate(self, args):
        df = self.df
        for col, val in (args.get("where") or {}).items():
            if col not in df.columns:
                raise ValueError(f"unknown column {col}")
            df = df[df[col].astype(str) == str(val)]
        col, agg = args["column"], args["agg"]
        if col not in df.columns or agg not in self.AGGS:
            raise ValueError(f"column must be one of {list(df.columns)}, agg one of {sorted(self.AGGS)}")
        by = args.get("group_by")
        if by:
            if by not in df.columns:
                raise ValueError(f"unknown column {by}")
            return df.groupby(by)[col].agg(agg).round(3).to_string()
        return str(round(float(df[col].agg(agg)), 3))


def build_tools(csv_path):
    csv = CsvTool(csv_path)
    return {
        "calculator": (calculator, 'Arithmetic. Input: {"expression": "(91.5-87.7)/87.7*100"}'),
        "describe_data": (csv.describe, "Show columns, types and sample rows. Input: {}"),
        "aggregate": (csv.aggregate, 'Aggregate a column. Input: {"column": "punctuality_pct", '
                      '"agg": "mean", "group_by": "area" (optional), "where": {"service": "School"} (optional)}'),
    }
