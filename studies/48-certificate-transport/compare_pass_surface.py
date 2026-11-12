"""Text check of the JAX surface the candidate pass touches, per release tag.
Reads fetched sources only. Usage: python -I compare_pass_surface.py <jaxsrc dir>
"""
import re, sys, pathlib

root = pathlib.Path(sys.argv[1])
tags = sorted((p.name for p in root.iterdir()), key=lambda t: tuple(int(x) for x in t.split('.')))


def text(tag, rel):
    path = root / tag / rel
    return path.read_text() if path.exists() else ''


def has(pattern, source, flags=re.M):
    return 'yes' if re.search(pattern, source, flags) else 'no'


checks = [
    ('JaxprEqn has ctx (JaxprEqnContext)', lambda t: has(r'^class JaxprEqnContext', text(t, 'jax/_src/core.py'))),
    ('JaxprEqnContext.manager', lambda t: has(r'def manager\(self\)', text(t, 'jax/_src/core.py'))),
    ('Jaxpr.replace', lambda t: has(r'^  def replace\(self, \*\*kwargs\)', text(t, 'jax/_src/core.py'))),
    ('Jaxpr.constvars property/field', lambda t: has(r'constvars', text(t, 'jax/_src/core.py'))),
    ('Jaxpr.consts attached to Jaxpr', lambda t: has(r'^  def with_consts\(', text(t, 'jax/_src/core.py'))),
    ('core.jaxpr_as_fun defined', lambda t: has(r'^def jaxpr_as_fun', text(t, 'jax/_src/core.py'))),
    ('cond_p params: only branches', lambda t: 'yes' if re.search(r'cond_p\.bind\(\s*index, \*ops,\s*branches=tuple\(\w+\)\s*,?\s*(\*\*params)?\)', text(t, 'jax/_src/lax/control_flow/conditionals.py'), re.S) else 'other'),
    ('scan_p bind has reverse/length/jaxpr/unroll', lambda t: 'yes' if all(k in text(t, 'jax/_src/lax/control_flow/loops.py') for k in ('reverse=reverse', 'length=length', 'jaxpr=jaxpr', 'unroll=unroll')) else 'no'),
    ('scan operand order consts,init,xs', lambda t: has(r'\*consts, \*in_state, \*in_carry, \*xs|\[\*consts, \*init_flat, \*xs_flat\]|consts \+ in_flat|\*consts, \*init_flat, \*xs_flat', text(t, 'jax/_src/lax/control_flow/loops.py'))),
    ('custom_vmap_call primitive name', lambda t: (re.search(r"Primitive\('(custom_vmap_call)'\)", text(t, 'jax/_src/custom_batching.py')) or [None, 'absent'])[1]),
    ('custom_vmap_call params', lambda t: ','.join(sorted(set(re.findall(r'custom_vmap_p\.bind\([^)]*?(call|rule|in_tree|out_tree)=', text(t, 'jax/_src/custom_batching.py'), re.S)))) or '?'),
    ('checkify.check_error public', lambda t: has(r'check_error as check_error', text(t, 'jax/experimental/checkify.py'))),
    ('checkify.user_checks public', lambda t: has(r'user_checks as user_checks', text(t, 'jax/experimental/checkify.py'))),
    ('eval_jaxpr(jaxpr, consts, *args)', lambda t: has(r'^def eval_jaxpr\(jaxpr: Jaxpr, consts, \*args', text(t, 'jax/_src/core.py'))),
    ('custom_jvp via custom_jvp_call_p bind', lambda t: has(r'custom_jvp_call_p\.bind\(', text(t, 'jax/_src/custom_derivatives.py') if (root / t / 'jax/_src/custom_derivatives.py').exists() else text(t, 'jax/_src/checkify.py'))),
    ('checkify handles custom JVP as hijax object', lambda t: has(r'CustomJVPTraced', text(t, 'jax/_src/checkify.py'))),
]
width = max(len(name) for name, _ in checks)
print(' ' * width, *[tag.ljust(9) for tag in tags])
for name, fn in checks:
    print(name.ljust(width), *[str(fn(tag))[:9].ljust(9) for tag in tags])
