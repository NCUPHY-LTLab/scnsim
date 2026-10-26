"""Reproduce documentation Plotly/SVG figures from published saved data.

Run from the repository root (requires NumPy, Plotly and Kaleido/Chrome):
    python scripts/docs_plot_presentation.py --source-dir examples/engineer/figures \
        --output-dir examples/engineer/figures --cases all

This presentation tool never imports SCNSim or executes a circuit calculation.
CSV columns and embedded Plotly arrays remain the numerical authority. HTML
uses Plotly's CDN; historical inputs and their manifests are never overwritten.
"""
from __future__ import annotations
import argparse
import base64
import csv
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import plotly.graph_objects as go

CASES = ('overview', 'chapter1', 'chapter3', 'chapter4-raw', 'chapter4-ptc', 'chapter7', 'chapter8')
COLORS = ('#246398', '#b65a27', '#487a55')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def values(value):
    """Decode Plotly's public typed-array JSON without changing numeric values."""
    if isinstance(value, dict) and 'bdata' in value:
        array = np.frombuffer(base64.b64decode(value['bdata']), dtype=value['dtype'])
        if value.get('shape'):
            array = array.reshape(tuple(int(n.strip()) for n in value['shape'].split(',')))
        return array
    return np.asarray(value)


def saved_figure(path):
    text = path.read_text()
    match = re.search(r'Plotly\.newPlot\(\s*"[^"]+"\s*,\s*', text)
    if match is None:
        raise ValueError(f'{path.name}: no saved JSON Plotly figure')
    decoder = json.JSONDecoder()
    data, end = decoder.raw_decode(text[match.end():])
    tail = text[match.end() + end:].lstrip()
    layout, _ = decoder.raw_decode(tail[1:].lstrip())
    return go.Figure(data=data, layout=layout)


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def column(rows, name):
    return np.asarray([float(row[name]) for row in rows])


def style(figure, label, component, magnitude_range=None):
    multiple = len(figure.data) > 1
    for index, trace in enumerate(figure.data):
        trace.update(xaxis='x', yaxis='y', line={'color': COLORS[index % len(COLORS)], 'width': 2},
                     marker={'color': COLORS[index % len(COLORS)], 'size': 7},
                     hovertemplate='%{x:.6g} GHz<br>%{y:.6g} ' +
                     ('dB' if component == 'magnitude' else 'deg') + '<extra></extra>')
    figure.update_layout(
        template='plotly_white', autosize=True, height=360 if multiple else 340,
        title={'text': f'{label} {component}', 'x': .02, 'xanchor': 'left', 'font': {'size': 18}},
        font={'family': 'Arial, sans-serif', 'size': 14, 'color': '#243344'},
        margin={'l': 78, 'r': 24, 't': 78 if multiple else 55, 'b': 62},
        showlegend=multiple,
        legend={'orientation': 'h', 'y': 1.03, 'yanchor': 'bottom', 'x': 1, 'xanchor': 'right'},
        paper_bgcolor='white', plot_bgcolor='white',
    )
    figure.update_xaxes(title_text='Frequency (GHz)', ticks='outside', showline=True,
                        linecolor='#6f7b85', gridcolor='#e5e9ed', zeroline=False,
                        automargin=True, title_standoff=12)
    figure.update_yaxes(title_text='Magnitude (dB)' if component == 'magnitude' else 'Phase (deg)',
                        ticks='outside', showline=True, linecolor='#6f7b85', gridcolor='#e5e9ed',
                        zeroline=False, automargin=True, title_standoff=12)
    if component == 'magnitude' and magnitude_range is not None:
        figure.update_yaxes(range=list(magnitude_range))
    return figure


def pair(stem, label, x, series, *, magnitude_range=None, markers=False):
    """Yield figures and their original arrays for exact output checks."""
    for component, index in (('magnitude', 1), ('phase', 2)):
        figure = go.Figure(data=[go.Scatter(x=x, y=row[index], name=row[0],
                                            mode='lines+markers' if markers else 'lines')
                                 for row in series])
        yield f'{stem}-{component}', style(figure, label, component, magnitude_range), \
            [(x, row[index]) for row in series]


def build_case(name, source):
    """Return output figures, exact input files, and human-readable channel mapping."""
    if name == 'overview':
        path = source / '01-s11-overview.csv'
        rows = read_csv(path)
        x = column(rows, 'frequency_Hz') * 1e-9
        z = column(rows, 'S11_real') + 1j * column(rows, 'S11_imag')
        figures = list(pair('chapter1-s11', 'S11', x,
                            [('S11', 20*np.log10(abs(z)), np.angle(z, deg=True))],
                            magnitude_range=(-.05, .05)))
        return figures, [path], 'Port 1 = signal_in. Existing one-execution baseline, 401 points.'
    if name == 'chapter1':
        path = source / '01-s11-data.csv'
        rows = read_csv(path)
        x = column(rows, 'frequency_GHz')
        series = [(label, column(rows, point+'_magnitude_dB'), column(rows, point+'_phase_deg'))
                  for point, label in (('baseline', '110 fF'), ('selected', '120 fF'))]
        return list(pair('01-s11-comparison', 'S11', x, series, magnitude_range=(-.05, .05))), \
            [path], 'Port 1 = signal_in. Baseline 110 fF and selected 120 fF; both stored series retained.'
    if name in ('chapter3', 'chapter4-raw', 'chapter4-ptc'):
        stem = {'chapter3': '03-direct-s21', 'chapter4-raw': '04-raw-direct-s21',
                'chapter4-ptc': '04-ptc-direct-s21'}[name]
        path = source / (stem+'.csv')
        rows = read_csv(path)
        x = column(rows, 'frequency_GHz')
        z = column(rows, 'real') + 1j*column(rows, 'imag')
        series = [('S21', column(rows, 'magnitude_dB'), np.angle(z, deg=True))]
        return list(pair(stem, 'S21', x, series, markers=True)), [path], \
            'S21: output Port 2 (feedline_out) from input Port 1 (feedline_in), mode (). Three saved samples only.'
    if name == 'chapter7':
        path = source / '07-response-points.csv'
        rows = read_csv(path)
        series = []
        x = None
        channel = 's_readout_head_from_readout_head'
        for point, label in (('baseline', 'Baseline'), ('length_1p8_mm', 'ℓ = 1.8 mm'),
                             ('capacitance_times_170_over_175', 'C × 170/175')):
            selected = [row for row in rows if row['point'] == point and row['trace'] == channel]
            if not selected or any(row['status'] != 'success' for row in selected):
                raise ValueError(f'{point}: missing successful saved S11 samples')
            point_x = column(selected, 'frequency_GHz')
            if x is None:
                x = point_x
            elif not np.array_equal(x, point_x):
                raise ValueError('Saved parameter points use different frequency grids')
            z = column(selected, 'real') + 1j*column(selected, 'imag')
            series.append((label, 20*np.log10(column(selected, 'magnitude')), np.angle(z, deg=True)))
        return list(pair('07-readout-head-reflection', 'S11', x, series, markers=True)), [path], \
            'Port 1 = readout_head; only original s_readout_head_from_readout_head channel. Baseline; length 1.8 mm; full C matrix ×170/175. Complete 16-channel CSV unchanged.'
    if name == 'chapter8':
        path = source / '08-report.html'
        original = saved_figure(path)
        if len(original.data) != 1:
            raise ValueError('Expected the one saved Chapter 8 magnitude trace')
        trace = original.data[0]
        metadata = trace.meta['scnsim']
        if metadata['y_axis']['component'] != 'magnitude' or metadata['unit'] != 'dB':
            raise ValueError('Expected the saved dB magnitude trace')
        x, y = values(trace.x), values(trace.y)
        figure = style(go.Figure(data=[trace]), 'S11', 'magnitude', original.layout.yaxis.range)
        return [('08-s11-magnitude', figure, [(x, y)])], [path], \
            'Port 1 = signal_in. Exact one saved magnitude trace; no phase or complex data reconstructed.'
    raise ValueError(f'Unknown case: {name}')


def generate(source, output, cases, *, png=False, verify_only=False):
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'operation': 'Saved-data presentation only; no SCNSim import or circuit execution.',
                'generator_sha256': sha256(Path(__file__)), 'cases': {}, 'assets': {}}
    for name in cases:
        figures, inputs, mapping = build_case(name, source)
        entry = {'sources': {path.name: sha256(path) for path in inputs}, 'mapping': mapping, 'figures': {}}
        for stem, figure, arrays in figures:
            for trace, (x, y) in zip(figure.data, arrays, strict=True):
                assert np.array_equal(values(trace.x), x, equal_nan=True)
                assert np.array_equal(values(trace.y), y, equal_nan=True)
            if verify_only:
                existing = saved_figure(output / (stem+'.html'))
                assert len(existing.data) == len(arrays)
                for trace, (x, y) in zip(existing.data, arrays, strict=True):
                    assert np.array_equal(values(trace.x), x, equal_nan=True)
                    assert np.array_equal(values(trace.y), y, equal_nan=True)
            else:
                figure.write_html(output / (stem+'.html'), include_plotlyjs='cdn', full_html=True,
                                  auto_open=False, config={'responsive': True, 'displaylogo': False})
                figure.write_image(output / (stem+'.svg'), format='svg', width=760, height=figure.layout.height)
                if png:
                    figure.write_image(output / (stem+'.png'), format='png', width=760, height=figure.layout.height)
            entry['figures'][stem] = {'point_counts': [len(x) for x, _ in arrays], 'x_exact': True,
                                     'y_exact': True, 'showlegend': bool(figure.layout.showlegend)}
            for extension in ('html', 'svg'):
                path = output / (stem+'.'+extension)
                manifest['assets'][path.name] = {'bytes': path.stat().st_size, 'sha256': sha256(path)}
        manifest['cases'][name] = entry
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=Path('examples/engineer/figures'))
    parser.add_argument('--output-dir', type=Path, default=Path('examples/engineer/figures'))
    parser.add_argument('--cases', nargs='+', choices=('all',)+CASES, default=['all'])
    parser.add_argument('--png', action='store_true', help='Also export inspection PNGs')
    parser.add_argument('--verify-only', action='store_true', help='Compare existing HTML arrays; do not export figures')
    args = parser.parse_args()
    cases = CASES if 'all' in args.cases else args.cases
    manifest = generate(args.source_dir, args.output_dir, cases, png=args.png, verify_only=args.verify_only)
    (args.output_dir/'docs-plot-assets.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'cases': list(cases), 'assets': len(manifest['assets']),
                      'asset_bytes': sum(row['bytes'] for row in manifest['assets'].values()),
                      'verify_only': args.verify_only}, indent=2))


if __name__ == '__main__':
    main()
