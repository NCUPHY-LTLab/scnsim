"""Call-local Jacobian overview assembly over selected detached saved blocks.

The Results owner resolves labels without loading unrelated values. This
module owns display conversion, compatible-unit grouping and Plotly layout,
never a solver, Result mutation, reader lease or persistent numeric cache.
"""
from __future__ import annotations

from html import escape, unescape
import json
from math import ceil
from textwrap import wrap

import numpy as np

from ... import units
from ...results.sensitivity import _angular_unit_power, _jacobian_blocks_for_quantities
from ...specs import OutputCoordinate
from .common import _metadata_value, _plotly, _presentation_source, _style
from .sensitivity import (
    _coordinate_record, _failure_text, _jacobian_display, _keyed_parameter_options,
    _normalization_records, _output_coordinate_labels, _output_quantity,
    _requested_normalization,
)


_HOVER = (
    "output=%{customdata[0]}<br>parameter=%{customdata[1]}<br>"
    "value=%{customdata[2]}<br>unit=%{customdata[3]}<br>"
    "status=%{customdata[4]}<br>original=%{customdata[5]}<br>"
    "conversion=%{customdata[6]}<br>reason=%{customdata[7]}<extra></extra>"
)


def _hover_text(value):
    """Wrap display text without truncating the recorded cause or identity."""
    return '<br>'.join(escape(line) for paragraph in str(value).split('\n')
                      for line in (wrap(paragraph,width=64,replace_whitespace=False,
                                        drop_whitespace=False,break_on_hyphens=False) or ['']))


def _hover(cell):
    return [_hover_text(cell[key]) for key in (
        'output_identity', 'parameter_identity', 'value_text', 'display_unit',
        'status', 'source_text', 'conversion_text', 'failure_text')]


def _display_rows(result, blocks, *, output_unit, output_units, parameter_units,
                  parameter_labels, normalized, input_scales, output_scales):
    # Every block comes from this single Result's authored wrt layout. Use
    # ParameterRef equality/identity, never its presentation label, for columns.
    parameters = tuple(blocks[0].parameters)
    columns = {parameter: index for index, parameter in enumerate(parameters)}
    coordinates = tuple(coordinate for block in blocks for coordinate in block.output_coordinates)
    if output_units is not None:
        if any(not isinstance(key, OutputCoordinate) for key in output_units):
            raise TypeError('output_units keys must be saved OutputCoordinate values')
        if any(not isinstance(value, str) for value in output_units.values()):
            raise TypeError('output_units values must be unit strings')
        if set(output_units) - set(coordinates):
            raise ValueError('output_units contains an output outside the selected quantities')
    parameter_options = _keyed_parameter_options(parameter_units, parameters=parameters,
        name='parameter_units', value_type=str)
    label_options = _keyed_parameter_options(parameter_labels, parameters=parameters,
        name='parameter_labels', value_type=str)
    scales = (_requested_normalization(result, parameters=parameters, coordinates=coordinates,
        input_scales=input_scales, output_scales=output_scales) if normalized else None)
    rows = []
    for block in blocks:
        quantity = _output_quantity(result, block)
        labels = _output_coordinate_labels(quantity, block)
        values, _, output_display_units, parameter_display_units, ratios = _jacobian_display(
            block, output_unit=output_unit, output_units=output_units,
            parameter_units_by_key=parameter_options, normalized=normalized,
            normalized_scales=scales)
        raw = (np.asarray(block.values).reshape((-1, len(block.parameters)))
               if block.status == 'complete' else None)
        for row, coordinate in enumerate(block.output_coordinates):
            cells = {}
            for local_column, parameter in enumerate(block.parameters):
                column = columns[parameter]
                source_output = block.row_units[row] if block.row_units else block.output_unit
                source_unit = units.registry.Unit(source_output) / units.registry.Unit(parameter.spec.si_unit)
                source_value = None if raw is None else raw[row, local_column].item()
                value = None if values is None else values[row, local_column].item()
                cells[column] = dict(
                    unit=ratios[row][local_column], value=value,
                    source_unit=str(source_unit), source_value=source_value,
                    initial_display_unit=str(ratios[row][local_column]),
                    output_display_unit=output_display_units[row],
                    parameter_display_unit=parameter_display_units[local_column],
                    status=block.status, failure=block.failure, failure_text=_failure_text(block),
                    output_coordinate=_coordinate_record(coordinate),
                    output_identity=f'{coordinate.quantity_label}/{coordinate.component}/{coordinate.index}',
                    parameter_identity=f'{parameter.definitions_id}.{parameter.id}',
                    parameter={'definitions_id':parameter.definitions_id,'parameter_id':parameter.id},
                    block_id=block.block_id, quantity_id=block.quantity_id,
                )
            rows.append(dict(label=labels[row], coordinate=_coordinate_record(coordinate), cells=cells))
    column_labels = [str(label_options.get((p.definitions_id,p.id),p.id)) for p in parameters]
    return rows, parameters, column_labels, scales


def _unit_groups(rows, *, normalized):
    groups = []
    for row_index,row in enumerate(rows):
        for column,cell in row['cells'].items():
            unit=cell['unit']
            group=next((g for g in groups if unit.is_compatible_with(g['unit'])
                        and _angular_unit_power(unit)==_angular_unit_power(g['unit'])),None)
            if group is None:
                group=dict(unit=unit,members=[]);groups.append(group)
            group['members'].append((row_index,column))
            cell['group']=groups.index(group)
            target=group['unit']
            if cell['value'] is not None:
                cell['value']=np.asarray(units.registry.Quantity(cell['value'],unit).to(target).magnitude).item()
            cell['display_unit']=f'{target:~P}'
            cell['value_text']=str(cell['value']) if cell['value'] is not None else 'unavailable'
            cell['source_text']=f"{cell['source_value']} {cell['source_unit']}" if cell['source_value'] is not None else 'unavailable'
            if normalized:
                cell['conversion_text']='explicit input/output normalization scales'
                cell['conversion']=dict(kind='explicit_normalization')
            else:
                factor=units.registry.Quantity(1,cell['source_unit']).to(target).magnitude
                cell['conversion_text']=f"{cell['source_unit']} -> {target}; multiply by {factor}"
                cell['conversion']=dict(kind='unit_conversion',source_unit=cell['source_unit'],
                    initial_display_unit=cell['initial_display_unit'],target_unit=str(target),factor=float(factor))
    return groups


def _cell_record(cell):
    return {key:_metadata_value(value) for key,value in cell.items() if key!='unit'}


def _panel_cells(rows,group):
    row_indices=sorted({row for row,_ in group['members']})
    columns=sorted({column for _,column in group['members']})
    members=set(group['members'])
    cells=[]
    for row in row_indices:
        line=[]
        for column in columns:
            if (row,column) in members:
                cell=rows[row]['cells'][column]
            else:
                # Blank in this panel means another derivative unit group,
                # never a zero derivative or an unavailable numerical output.
                original=rows[row]['cells'][column]
                cell={**original,'value':None,'value_text':'not applicable',
                      'display_unit':'','status':'not_applicable','failure_text':'different derivative unit group',
                      'source_text':f"{original['source_value']} {original['source_unit']}",
                      'conversion_text':'belongs to another unit panel'}
            line.append(cell)
        cells.append(line)
    return row_indices,columns,cells



def _table_default_height(figure, columns, headers, column_widths):
    """Budget the default canvas for every wrapped row, without truncating data.

    Plotly expands table rows for wrapped text but does not grow its default
    canvas. Estimate text wrapping from the styled font and proportional column
    widths; callers may still override Figure/export height explicitly.
    """
    margin=figure.layout.margin
    font_size=figure.layout.font.size or figure.layout.template.layout.font.size or 12
    available_width=(figure.layout.width or 700)-(margin.l or 0)-(margin.r or 0)
    total_width=sum(column_widths)
    character_widths=[max(1,int((available_width*weight/total_width-20)/(0.6*font_size)))
                      for weight in column_widths]

    def lines(text, width):
        return sum(max(1,len(wrap(unescape(part),width=width)))
                   for part in text.split('<br>'))

    header_lines=max(lines(text,width) for text,width in zip(headers,character_widths,strict=True))
    row_lines=[max(lines(column[row],width)
                   for column,width in zip(columns,character_widths,strict=True))
               for row in range(len(columns[0]))]
    # Line leading and cell padding are presentation space, not data limits.
    return ceil((margin.t or 0)+(margin.b or 0)
                +sum(1.3*font_size*count+16 for count in [header_lines,*row_lines]))


def jacobian_overview(result, *, quantities, style, output_unit, output_units,
                      parameter_units, parameter_labels, normalized, input_scales,
                      output_scales, theme):
    """One Figure, selected quantity order, and independent derivative-unit panels."""
    blocks=_jacobian_blocks_for_quantities(result,quantities)
    rows,parameters,column_labels,scales=_display_rows(result,blocks,
        output_unit=output_unit,output_units=output_units,parameter_units=parameter_units,
        parameter_labels=parameter_labels,normalized=normalized,input_scales=input_scales,
        output_scales=output_scales)
    groups=_unit_groups(rows,normalized=normalized)
    go,_,make_subplots=_plotly()
    meta=dict(kind='sensitivity_jacobian_overview',quantities=list(quantities),normalized=normalized,
        source=_presentation_source(result),
        parameters=[dict(definitions_id=p.definitions_id,parameter_id=p.id) for p in parameters],
        rows=[dict(label=row['label'],coordinate=row['coordinate'],
                   cells=[_cell_record(row['cells'][column]) for column in range(len(parameters))]) for row in rows],
        groups=[dict(unit=str(g['unit']),members=[list(position) for position in g['members']]) for g in groups])
    if scales is not None:meta['normalization_scales']=_normalization_records(scales)
    if not groups:
        figure=go.Figure()
        figure.add_annotation(text='No output coordinates in the selected saved quantities',
                              x=0.5,y=0.5,xref='paper',yref='paper',showarrow=False)
    elif style=='table':
        columns=[['<br>'.join(escape(part) for part in row['label'].split(' / ')) for row in rows]]
        for column in range(len(parameters)):
            columns.append([escape(f"{row['cells'][column]['value_text']} {row['cells'][column]['display_unit']}"
                                   +(f" ({row['cells'][column]['failure_text']})" if row['cells'][column]['status']!='complete' else '')) for row in rows])
        table_widths=[4,*[3 for _ in parameters]]
        table_headers=['output coordinate',*[escape(x) for x in column_labels]]
        figure=go.Figure(go.Table(columnwidth=table_widths,
                                 header=dict(values=table_headers),
                                 cells=dict(values=columns,align='left'),meta={'scnsim':meta}))
    else:
        figure=make_subplots(rows=len(groups),cols=1,
            subplot_titles=[escape(f"dq/dp ({g['unit']:~P})") for g in groups])
        legend_coordinates=set()
        has_bars=False
        for ordinal,group in enumerate(groups):
            panel=ordinal+1
            rr,cc,cells=_panel_cells(rows,group)
            selected_style=('bar' if len(rr)==1 else 'heatmap') if style=='auto' else style
            coloraxis='coloraxis' if panel==1 else f'coloraxis{panel}'
            yaxis=figure.layout['yaxis' if panel==1 else f'yaxis{panel}']
            domain=yaxis.domain
            figure.update_layout(**{coloraxis:dict(colorscale='RdBu',reversescale=True,cmid=0,
                colorbar=dict(title=f"{group['unit']:~P}",y=(domain[0]+domain[1])/2,len=domain[1]-domain[0]))})
            if selected_style=='heatmap':
                figure.add_trace(go.Heatmap(z=[[cell['value'] for cell in line] for line in cells],x=cc,y=rr,
                    coloraxis=coloraxis,customdata=[[_hover(cell) for cell in line] for line in cells],
                    hovertemplate=_HOVER,hoverongaps=True,meta={'scnsim':{'unit_group':ordinal}}),row=panel,col=1)
                unavailable=[(row,column,cell) for row,line in zip(rr,cells,strict=True)
                             for column,cell in zip(cc,line,strict=True) if cell['status']=='unavailable']
                if unavailable:
                    figure.add_trace(go.Scatter(x=[x[1] for x in unavailable],y=[x[0] for x in unavailable],
                        mode='markers',marker=dict(symbol='x-open',color='gray',size=10),name='unavailable',
                        customdata=[_hover(x[2]) for x in unavailable],hovertemplate=_HOVER,
                        showlegend=False),row=panel,col=1)
                figure.update_yaxes(tickmode='array',tickvals=rr,ticktext=[escape(rows[x]['label']) for x in rr],
                                    autorange='reversed',row=panel,col=1)
            else:
                has_bars=True
                # Bar traces do not reliably autoscale a shared coloraxis.
                # Use this panel's actual converted values, centered on zero.
                values=[cell['value'] for line in cells for cell in line
                        if cell['status']=='complete']
                if values:
                    extent=max(abs(value) for value in values)
                    color_limits=dict(cauto=False,cmin=-extent,cmax=extent)
                    if extent==0:
                        # An all-zero panel has no nonzero color range. Keep
                        # zero neutral without inventing a numerical extent.
                        color_limits.update(colorscale=[[0,'rgb(247,247,247)'],[1,'rgb(247,247,247)']],
                                            colorbar=dict(tickmode='array',tickvals=[0]))
                    figure.update_layout(**{coloraxis:color_limits})
                for row,line in zip(rr,cells,strict=True):
                    points=[(column,cell) for column,cell in zip(cc,line,strict=True) if cell['status']=='complete']
                    if points:
                        identity=json.dumps(rows[row]['coordinate'],sort_keys=True,separators=(',',':'))
                        figure.add_trace(go.Bar(x=[x[0] for x in points],y=[x[1]['value'] for x in points],
                            marker=dict(color=[x[1]['value'] for x in points],coloraxis=coloraxis),
                            name=escape(rows[row]['label']),offsetgroup=str(row),
                            legendgroup=identity,showlegend=identity not in legend_coordinates,
                            customdata=[_hover(x[1]) for x in points],hovertemplate=_HOVER),row=panel,col=1)
                        legend_coordinates.add(identity)
                missing=[f"{rows[row]['label']}: {cell['failure_text']}" for row,line in zip(rr,cells,strict=True)
                         for cell in line if cell['status']=='unavailable']
                if missing:
                    figure.add_annotation(text='Unavailable derivatives (hover for recorded reason)',
                        hovertext=_hover_text('; '.join(dict.fromkeys(missing))),
                        x=0,y=1,xref='x domain',yref='y domain',xanchor='left',yanchor='top',
                        showarrow=False,row=panel,col=1)
                figure.update_yaxes(title_text=f"dq/dp ({group['unit']:~P})",row=panel,col=1)
            figure.update_xaxes(tickmode='array',tickvals=cc,ticktext=[escape(column_labels[x]) for x in cc],
                                title_text='parameter coordinates',row=panel,col=1)
        figure.update_layout(barmode='group',height=max(400,260*len(groups)))
    figure.update_layout(meta={'scnsim':meta})
    figure=_style(figure,theme,title='Sensitivity Jacobian overview')
    # Explicit layout defaults prevent marker gray from producing a low-contrast
    # unavailable tooltip. Derive both colors from the selected Plotly theme;
    # callers retain the normal Figure layout/annotation override interfaces.
    template=figure.layout.template.layout
    hoverlabel=dict(bgcolor=template.font.color,font=dict(color=template.paper_bgcolor))
    figure.update_layout(hoverlabel=hoverlabel)
    for annotation in figure.layout.annotations:
        if annotation.hovertext:
            annotation.update(hoverlabel=hoverlabel)
    if groups and style=='table' and figure.layout.height is None:
        figure.update_layout(height=_table_default_height(figure,columns,table_headers,table_widths))
    if groups and style!='table' and has_bars:
        # Keep the shared target legend below all panels, away from colorbars.
        figure.update_layout(legend=dict(orientation='h',x=0,xanchor='left',y=-0.08,yanchor='top'),
                             margin=dict(r=112,b=220))
    return figure
