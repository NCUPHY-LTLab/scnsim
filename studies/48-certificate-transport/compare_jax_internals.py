"""Text comparison of the JAX internals SCNSim's certificate transport relies on.

Reads source files fetched by fetch_jax_sources.sh. Nothing is imported or run
from them. Usage: python -I compare_jax_internals.py <jaxsrc dir>
"""
import re, sys, pathlib, hashlib

root = pathlib.Path(sys.argv[1])
tags = sorted((p.name for p in root.iterdir()), key=lambda t: tuple(int(x) for x in t.split('.')))


def text(tag, rel):
    path = root / tag / rel
    return path.read_text() if path.exists() and path.stat().st_size else ''


def first(pattern, source, flags=re.M | re.S):
    match = re.search(pattern, source, flags)
    return match.group(0) if match else None


def block(source, header, stop=r'\n(?=\S)'):
    """Return the top-level definition starting at `header` (regex)."""
    match = re.search(header, source, re.M)
    if not match:
        return None
    rest = source[match.start():]
    end = re.search(r'\n(?=[^\s#)\]}])', rest[1:])
    return rest[: end.start() + 1] if end else rest


def digest(value):
    return '-' if value is None else hashlib.sha1(re.sub(r'\s+', ' ', value).encode()).hexdigest()[:6]


rows = []


def row(name, fn):
    rows.append((name, [fn(tag) for tag in tags]))


ck = lambda tag: text(tag, 'jax/_src/checkify.py')
row('checkify.py lines', lambda t: str(ck(t).count('\n')))
row('_INACTIVE_CODE defined', lambda t: (first(r'^_INACTIVE_CODE = [^\n]*', ck(t)) or 'absent').replace('_INACTIVE_CODE = ', '='))
row('placeholder code literal', lambda t: (re.search(r'new_code\[effect\] = (\S+)', ck(t)) or [None, 'absent'])[1])
row('placeholder payload', lambda t: 'ones' if 'jnp.ones(a.shape, a.dtype)' in ck(t) else 'other')
row('placeholders sorted(effects)', lambda t: 'yes' if 'for effect in sorted(effects)' in ck(t) else 'no (set order)')
row('CheckifyContext class', lambda t: 'yes' if 'class CheckifyContext' in ck(t) else 'no')
row('assert_func signature', lambda t: re.sub(r'\s+', ' ', first(r'def assert_func\((.*?)\)', ck(t)) or 'absent')[16:60])
row('rule signature (cond_error_check)', lambda t: re.sub(r'\s+', ' ', first(r'def cond_error_check\((.*?)\)', ck(t)) or 'absent')[21:70])
row('next_code global counter', lambda t: 'yes' if re.search(r'^def next_code\(', ck(t), re.M) else 'no')
row('ErrorEffect defined in', lambda t: 'checkify.py' if re.search(r'^class ErrorEffect', ck(t), re.M) else ('effects.py' if re.search(r'^class ErrorEffect', text(t, 'jax/_src/effects.py'), re.M) else '?'))
row('ErrorEffect body sha', lambda t: digest(block(ck(t), r'^class ErrorEffect') or block(text(t, 'jax/_src/effects.py'), r'^class ErrorEffect')))
row('JaxException body sha', lambda t: digest(block(ck(t), r'^class JaxException')))
row('Error class body sha', lambda t: digest(block(ck(t), r'^class Error:')))
row('Error field order', lambda t: ','.join(re.findall(r'^  (_\w+): ', block(ck(t), r'^class Error:') or '', re.M)))
row('Error.get_exception sha', lambda t: digest(first(r'  def get_exception\(self\).*?(?=\n  def )', ck(t))))
row('update_error sha', lambda t: digest(block(ck(t), r'^def update_error')))
row('_check_error signature', lambda t: re.sub(r'\s+', ' ', first(r'def _check_error\((.*?)\)', ck(t)) or 'absent'))
row('check_discharge_rule sha', lambda t: digest(block(ck(t), r'^def check_discharge_rule')))
row('discharge remaps code namespace', lambda t: 'yes' if 'offset = ctx.current_code' in (block(ck(t), r'^def check_discharge_rule') or '') else 'no')
row('cond_error_check sha', lambda t: digest(block(ck(t), r'^def cond_error_check')))
row('scan_error_check sha', lambda t: digest(block(ck(t), r'^def scan_error_check')))
row('while_loop_error_check sha', lambda t: digest(block(ck(t), r'^def while_loop_error_check')))
row('scan rule params', lambda t: re.sub(r'\s+', ' ', first(r'def scan_error_check\((.*?)\)', ck(t)) or 'absent').split('*in_flat,')[-1].strip(' )'))
row('BatchedError sha', lambda t: digest(block(ck(t), r'^(@dataclasses.dataclass\n)?class BatchedError')))
row('get_traceback', lambda t: 'yes' if re.search(r'^def get_traceback', ck(t), re.M) else 'no')
row('check_p params', lambda t: re.sub(r'\s+', ' ', first(r'def check_abstract_eval\((.*?)\)', ck(t)) or 'absent')[24:])
row('public checkify exports sha', lambda t: digest(text(t, 'jax/experimental/checkify.py')))

pe = lambda tag: text(tag, 'jax/_src/interpreters/partial_eval.py')
row('pe.separate_consts', lambda t: 'yes' if re.search(r'^def separate_consts', pe(t), re.M) else 'no')
row('pe.convert_constvars_jaxpr (public re-export)', lambda t: 'yes' if 'convert_constvars_jaxpr' in text(t, 'jax/interpreters/partial_eval.py') else 'no')
core = lambda tag: text(tag, 'jax/_src/core.py')
row('class ClosedJaxpr separate', lambda t: 'yes' if re.search(r'^class ClosedJaxpr', core(t), re.M) else 'no (unified)')
row('get_bind_params returns', lambda t: re.sub(r'\s+', ' ', (first(r'  def get_bind_params\(self, params\):\n\s+return [^\n]+', core(t)) or 'absent')).split('return ')[-1])
row('xla_bridge.backends_are_initialized', lambda t: 'yes' if re.search(r'^def backends_are_initialized', text(t, 'jax/_src/xla_bridge.py'), re.M) else 'no')
row('jax.extend.backend exports', lambda t: ','.join(sorted(set(re.findall(r'^\s+(\w+) as \w+,', text(t, 'jax/extend/backend.py'), re.M)))) or '-')
loops = lambda tag: text(tag, 'jax/_src/lax/control_flow/loops.py')
row('scan_p params have num_carry', lambda t: 'yes' if re.search(r'num_carry=', loops(t)) else 'no')
row('scan_p params have ft_in', lambda t: 'yes' if re.search(r'ft_in=', loops(t)) else 'no')
row('while_p params', lambda t: 'cond_nconsts,cond_jaxpr,body_nconsts,body_jaxpr' if all(k in loops(t) for k in ('cond_nconsts=', 'cond_jaxpr=', 'body_nconsts=', 'body_jaxpr=')) else 'changed')
conds = lambda tag: text(tag, 'jax/_src/lax/control_flow/conditionals.py')
row('cond_p.bind(... branches=', lambda t: 'yes' if re.search(r'cond_p\.bind\([^)]*branches=', conds(t), re.S) else 'no')
ext = lambda tag: text(tag, 'jax/extend/core/__init__.py')
for name in ('Jaxpr', 'JaxprEqn', 'Primitive', 'jaxpr_as_fun', 'Literal', 'Var', 'ClosedJaxpr', 'Effect', 'new_jaxpr_eqn'):
    row(f'jax.extend.core.{name}', lambda t, name=name: 'yes' if re.search(rf'\b{name} as {name}\b', ext(t)) else 'no')
prims = lambda tag: text(tag, 'jax/extend/core/primitives.py')
for name in ('cond_p', 'while_p', 'scan_p', 'custom_jvp_call_p', 'jit_p', 'pjit_p'):
    row(f'jax.extend.core.primitives.{name}', lambda t, name=name: 'yes' if re.search(rf'\b{name} as {name}\b', prims(t)) else 'no')
row('jax.core.eval_jaxpr', lambda t: 'yes' if 'eval_jaxpr' in text(t, 'jax/core.py') else 'no')

width = max(len(name) for name, _ in rows)
print(' ' * width, *[tag.ljust(9) for tag in tags])
for name, values in rows:
    print(name.ljust(width), *[str(v)[:9].ljust(9) if len(str(v)) <= 9 else str(v) for v in values], sep=' ' if all(len(str(v)) <= 9 for v in values) else '\n    ')
