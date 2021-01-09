import re, shlex, base64, io

import sklearn
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
from wordcloud import WordCloud

from .reader import load_dir

def build_results(articles, chunks, levnames, unit):
    res = dict()

    nlev = len(levnames)
    if nlev == 0:
        # Just a bunch of plain files
        raise NotImplementedError()
    else:
        graph = px.density_heatmap(data_frame=articles, x=levnames[0],
                                   y=levnames[1] if 1 < nlev else None,
                                   facet_row=levnames[2] if 2 < nlev else None,
                                   facet_col=levnames[3] if 3 < nlev else None,
                                   title="Article Counts")
        res['articles_summary'] = pio.to_html(graph)

        graph = px.density_heatmap(data_frame=chunks, x=levnames[0],
                                   y=levnames[1] if 1 < nlev else None,
                                   facet_row=levnames[2] if 2 < nlev else None,
                                   facet_col=levnames[3] if 3 < nlev else None,
                                   title=f"Count of {unit}")
        res['chunks_summary'] = pio.to_html(graph)

    alltext = '\n\n'.join(chunks['Text'])
    cloud = WordCloud().generate(alltext)
    with io.BytesIO() as buffer:
        cloud.to_image().save(buffer, 'png')
        img = base64.b64encode(buffer.getvalue()).decode()
    res['wordcloud_all_img'] = img #f'<img src="data:image/png;base64,{img}">'

    res['chunks_table'] = chunks.to_html(table_id="chunks").replace(r'\n', '<br>')
    
    return res

def process(dname, path, info):
    #@TODO: Check for saved articles/chunks with same parameters
    
    level_filters = [info[key] for key in info if re.match('level[0-9]+_filter', key)]

    # We're just going to treat everything as a regex, so escape it if necessary.
    if not info['fterms']:
        fpat = None
    else:
        if info['fregex']:
            # @TODO: Don't assume a single regex filter term.
            fpat = info['fterms']
        else:
            # Hopefully these do not need to be human-readable, b/c
            # re.escape escapes many unnecessary characters...
            fterms = [re.escape(x) for x in shlex.split(info['fterms'])]
            main = '|'.join(fterms)
            fpat = f'\\b(?:{main})\\b'

        fpat = re.compile(fpat, flags=re.I if info['fcase'] else None)
        
    spat = re.compile(info['article_regex_splitter'], flags=re.M)

    articles_df, chunks_df = load_dir(path, level_filters, info['unit'], fpat, spat)

    colmap = {f'_level_{i}': name for i, name in enumerate(info['level_names'])}
    articles_df.rename(columns=colmap, inplace=True)
    chunks_df.rename(columns=colmap, inplace=True)

    chunks_df = chunks_df.reindex([*info['level_names'], 'Filename', 'Article ID', 'Text'], axis=1)


    #@TODO: Save articles/chunks after filtering, plus parameters used

    #@TODO: Save html results
    
    return build_results(articles_df, chunks_df, info['level_names'], info['unit'])
    
